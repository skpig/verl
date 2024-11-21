# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import warnings
import os
import logging
import hdfs_io
import ray
import torch
import torch.distributed

import verl.utils.torch_functional as verl_F
from single_controller.base import Worker
from single_controller.base.decorator import register, Dispatch
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, load_fsdp_grad, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.import_utils import import_external_libs
from verl.utils.debug import log_gpu_memory_usage
from torch.distributed.device_mesh import init_device_mesh
from verl.utils.model import compute_position_id_with_mask
import numpy as np

from alpha_seed.workers.hybrid_engine.hsdp import create_device_mesh
from alpha_seed.workers.hybrid_engine.fsdp_ulysses import (FSDPUlyssesShardingManager, ulysses_pad_and_slice_inputs)
from alpha_seed.workers.utils import rearrange_micro_batches
from dist_attn.ulysses.ops import slice_input_tensor, gather_outputs
from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size

from seed_models.utils.count_flops import FlopsCounter

from codetiming import Timer

from datetime import timedelta

from .initialize import get_device_init_context, create_init_fn
from ..utils import rearrange_micro_batches

logger = logging.getLogger(__file__)


@ray.remote
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config):
        super().__init__()
        if not torch.distributed.is_initialized():
            timeout = timedelta(minutes=int(os.getenv('NCCL_TIMEOUT', 60)))
            torch.distributed.init_process_group(backend="nccl", timeout=timeout)
        self.config = config

        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(config.fsdp_size, 'Reward')

        self.ulysses_sp_device_mesh = None
        sp_size = config.ulysses_sequence_parallel_size
        if sp_size > 1:
            self.ulysses_sp_device_mesh = init_device_mesh('cuda',
                                                           mesh_shape=(world_size // sp_size, sp_size),
                                                           mesh_dim_names=['dp', 'sp'])
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_sp_device_mesh)
        self.config.micro_batch_size //= world_size // sp_size

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_local_path,
                                                                 trust_remote_code=config.model.get(
                                                                     'trust_remote_code', False))
        self.tokenizer = AutoTokenizer.from_pretrained(local_path,
                                                       trust_remote_code=config.model.get('trust_remote_code', False))

        if self.rank == 0:
            print(f'Switch chat_template: {self._do_switch_chat_template}')

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_device_init_context(use_meta_tensor=True)

        use_rmpad = self.config.get('use_rmpad', False)
        if use_rmpad:
            # optimize the model via rmpad
            from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch
            assert apply_monkey_patch(config=model_config,
                                      verbose=self.rank == 0), f'Cannot find rmpad version of {model_config.model_type}'

        model_config.pad_token_id = self.tokenizer.pad_token_id

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # model_config.moe_implementation = 'group_gemm'  # Note that this is deprecated. Use seed-models stable
            setattr(model_config, '_moe_implementation', 'fused')
            setattr(model_config, 'classifier_dropout', 0.)
            reward_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch.bfloat16,
                                                                            attn_implementation='flash_attention_2',
                                                                            config=model_config,
                                                                            trust_remote_code=trust_remote_code)
            # with torch.no_grad():
            #     # set reward model score bias to zero
            #     if reward_module.score.bias is not None:
            #         reward_module.score.bias.zero_()
            reward_module.to(torch.bfloat16)
            if self.rank == 0:
                print(reward_module)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        cpu_offload = None
        if self.config.model.fsdp_config.param_offload:
            cpu_offload = CPUOffload(offload_params=True)

        fsdp_config = self.config.model.fsdp_config
        sharding_strategy_config = fsdp_config.get('sharding_strategy', 'FULL_SHARD')  # zero3
        sharding_strategy = getattr(ShardingStrategy, sharding_strategy_config)

        reward_module = FSDP(
            reward_module,
            param_init_fn=create_init_fn(reward_module),
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            device_mesh=self.device_mesh,
            sync_module_states=True,
            forward_prefetch=True,
            cpu_offload=cpu_offload)  # we always offload reward

        if self.rank == 0:
            print(model_config)

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        self.reward_module.eval()
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange

        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.config.get('use_rmpad', False):
                # 重新组合input_ids和attention_mask
                max_prompt_length = self.config['max_prompt_length']
                response_ids = micro_batch['input_ids'][:, max_prompt_length:].to(torch.int64)
                response_mask = micro_batch['attention_mask'][:, max_prompt_length:].to(torch.int64)
                reflection_nums = torch.zeros((response_mask.shape[0],))
                if self.config.get('use_last_response', False):
                    response_ids, response_mask, reflection_nums = self.get_last_response(response_ids, response_mask)

                prompt_ids = micro_batch['answer_input_ids'].to(torch.int64)
                input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
                prompt_mask = micro_batch['answer_attention_mask'].to(torch.int64)
                attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)

                batch, seqlen = input_ids.shape
                input_ids_rmpad, indices, cu_seqlens, _ = unpad_input(input_ids.unsqueeze(-1),
                                                                      attention_mask=attention_mask)  # (totol_nnz, 1)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                position_ids = compute_position_id_with_mask(attention_mask)
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # handle ulysses sequence parallelism
                sp_size = get_ulysses_sequence_parallel_world_size()
                total_s = input_ids_rmpad.size(1)
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size)

                output = self.reward_module(input_ids=input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False)

                # handle ulysses sequence parallelism
                if sp_size > 1:
                    output.logits = gather_outputs(output.logits, gather_dim=1, padding_dim=1, unpad_dim_size=total_s)

                rm_score = output.logits.squeeze(0).squeeze(-1)  # (total_nnz,)
                last_pos = cu_seqlens[1:] - 1
                rm_score = rm_score[last_pos]  # (bsz,)
                assert rm_score.shape == (batch,)
            else:
                raise NotImplementedError
            return rm_score, reflection_nums

    def get_last_response(self, response_ids, response_mask):
        bs = response_ids.shape[0]
        pad_token_id = self.tokenizer.pad_token_id
        new_response_ids = []
        reflection_nums = []
        for bi in range(bs):
            raw_resp_len = len(response_ids[bi])
            response_txt_i = self.tokenizer.decode(response_ids[bi])
            new_response_txt_i, reflection_num = self._reflect_postprocess(response_txt_i)
            new_response_ids_i = torch.tensor(self.tokenizer.encode(new_response_txt_i)).to(
                device=response_ids[bi].device, dtype=response_ids[bi].dtype)
            new_raw_resp_len = len(new_response_ids_i)
            if new_raw_resp_len >= raw_resp_len:
                if new_raw_resp_len > raw_resp_len:
                    print('new_response_txt_i', new_response_txt_i.replace(self.tokenizer.pad_token, ''))
                    print('response_txt_i', response_txt_i.replace(self.tokenizer.pad_token, ''))
                new_response_txt_i = response_txt_i
                new_response_ids_i = response_ids[bi]
            else:
                padding = torch.tensor([pad_token_id for _ in range(raw_resp_len - len(new_response_ids_i))
                                       ]).to(device=response_ids[bi].device, dtype=response_ids[bi].dtype)
                new_response_ids_i = torch.cat((new_response_ids_i, padding))
            new_response_ids.append(new_response_ids_i)
            reflection_nums.append(reflection_num)
        new_response_ids = torch.stack(new_response_ids).to(device=response_ids.device, dtype=response_ids.dtype)
        new_response_mask = new_response_ids.not_equal(pad_token_id).to(device=response_ids.device,
                                                                        dtype=response_mask.dtype)
        reflection_nums = torch.tensor(reflection_nums).to(device=response_ids.device, dtype=response_ids.dtype)
        return new_response_ids, new_response_mask, reflection_nums

    def _reflect_postprocess(self, input_text_i):

        def sep_reflect(input_text_i, reflect_start='<reflection>', reflect_end='</reflection>'):
            reflect_start_pos_list = []
            reflect_end_pos_list = []
            # Initialize variables to store the positions of the last reflect_start and reflect_end
            last_reflect_start = -1
            last_reflect_end = -1
            second_last_reflect_end = -1

            # Find all the positions of reflect_start and reflect_end
            current_position = 0

            while True:
                # Find the next reflect_start position
                reflect_start_pos = input_text_i.find(reflect_start, current_position)
                if reflect_start_pos == -1:
                    break  # No more reflect_start tokens
                last_reflect_start = reflect_start_pos
                reflect_start_pos_list += [last_reflect_start]
                current_position = reflect_start_pos + len(reflect_start)

            # Reset the current position for reflect_end search
            current_position = 0

            # Loop to find the last and second last reflect_end
            while True:
                reflect_end_pos = input_text_i.find(reflect_end, current_position)
                if reflect_end_pos == -1:
                    break  # No more reflect_end tokens
                last_reflect_end = reflect_end_pos
                reflect_end_pos_list += [last_reflect_end]
                current_position = reflect_end_pos + len(reflect_end)

            return reflect_start_pos_list, reflect_end_pos_list

        reflect_start = '<reflection>'
        reflect_end = '</reflection>'
        reflect_start_pos_list, reflect_end_pos_list = sep_reflect(input_text_i, reflect_start, reflect_end)
        if len(reflect_start_pos_list) == 0 or len(reflect_start_pos_list) != len(reflect_end_pos_list):
            new_input_text_i = input_text_i
        elif len(reflect_start_pos_list) == 1:
            new_input_text_i = input_text_i[:reflect_start_pos_list[0]] + input_text_i[reflect_end_pos_list[0] +
                                                                                       len(reflect_end):]
        else:
            last_reflect_end = reflect_end_pos_list[-1]
            second_last_reflect_end = reflect_end_pos_list[-2]
            last_reflect_start = reflect_start_pos_list[-1]
            new_input_text_i = input_text_i[second_last_reflect_end +
                                            len(reflect_end):last_reflect_start] + input_text_i[last_reflect_end +
                                                                                                len(reflect_end):]
        reflection_num = min(len(reflect_start_pos_list), len(reflect_end_pos_list))
        return new_input_text_i, reflection_num

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask'].to(torch.int64)
        position_ids = compute_position_id_with_mask(attention_mask)
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        assert NotImplementedError

        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    def norm(self, rm_score):
        rm_score = (rm_score - self.config["mean"]) / self.config["std"]
        return rm_score

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to('cuda')
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_data = data

        rm_data.batch = rm_data.batch.cuda()
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(rm_data)

            if self.config.use_dynamic_bsz:
                (micro_batches, num_micro_batches) = rearrange_micro_batches(batch=rm_data.batch,
                                                                             max_token_len=self.config.max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = rm_data.batch.split(self.config.micro_batch_size)
                num_micro_batches = len(micro_batches)

            output = []
            total_reflection_nums = []
            for i, micro_batch in enumerate(micro_batches):
                rm_score, reflection_nums = self._forward_micro_batch(micro_batch)
                # 归一化
                if i < num_micro_batches:
                    output.append(rm_score)
                    total_reflection_nums.append(reflection_nums)
            scores = torch.cat(output, dim=0)  # (batch_size)
            reflection_nums = torch.cat(total_reflection_nums, dim=0)
            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores, 'reflection_nums': reflection_nums})

            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # reset FSDP buffer after forward
        self.reward_module._handle.reshard(True)
        torch.cuda.empty_cache()
        return output
