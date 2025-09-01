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

import os
import logging
import ray
import torch
import torch.distributed

import verl.utils.torch_functional as verl_F
from mono_rl.single_controller import Worker
from mono_rl.single_controller import register, Dispatch
from mono_rl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask

from transformers import AutoTokenizer
from alpha_seed.workers.hybrid_engine.fsdp_gather import DataGatherManager
from verl.utils.seqlen_balancing import rearrange_micro_batches
from alpha_seed.utils import ndtimeline
from mono_rl.models.seed_models.parallel.collectives import get_memory
from mono_rl.worker.engine.fsdp.initialize import cleanup_local_tmp_folder_safetensors_files

from alpha_seed.utils.mono_rl.config import reward_config_to_mono_config
from mono_rl.worker.engine.fsdp.models.model import FSDPModel
from mono_rl.worker import Role
from datetime import timedelta

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
        self.role = "rm"

        sp_size = config.ulysses_sequence_parallel_size
        tp_size = config.tp_size
        world_size = torch.distributed.get_world_size()

        self.config.micro_batch_size //= (world_size // sp_size // tp_size)

        self._model_initialized = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, remove_safetensors_after_init=False):

        if self._model_initialized:
            return

        # The critic config of the alpha_seed DictConfig format
        from omegaconf import DictConfig
        as_config: DictConfig = self.config

        trust_remote_code = as_config.model.get('trust_remote_code', False)

        # Setup input_tokenizer which is not covered by the monorl engine
        input_tokenizer_local_path = None
        if as_config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(as_config.model.input_tokenizer)
            self.input_tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_local_path,
                                                                 trust_remote_code=trust_remote_code)

        if self.rank == 0:
            print(f'Switch chat_template: {self._do_switch_chat_template}')

        mono_config = reward_config_to_mono_config(as_config)
        mono_config.engine.fsdp.param_offload = True

        self.engine = FSDPModel(mono_config.engine)
        self.engine.init_model(build_optimizer=False)
        self.tokenizer = self.reward_engine.tokenizer

        # get the device meshes from monorl fsdp model engine and construct the data gather manager
        self.fsdp_mesh, self.tp_mesh, self.oe_mesh, self.sp_mesh, self.gather_mesh = self.engine.fsdp_mesh, self.engine.tp_mesh, self.engine.oe_mesh, self.engine.sp_mesh, self.engine.gather_mesh
        self.gather_manager = DataGatherManager(self.gather_mesh, self.sp_mesh)

        torch.cuda.empty_cache()
        ndtimeline.init_with_ray(self)
        self._model_initialized = True
        if remove_safetensors_after_init:
            cleanup_local_tmp_folder_safetensors_files(self.engine.model_config._name_or_path)

    def forward_micro_batch(self, micro_batch, response_length):
        max_prompt_length = self.config['max_prompt_length']
        response_ids = micro_batch['input_ids'][:, max_prompt_length:].to(torch.int64)
        response_mask = micro_batch['attention_mask'][:, max_prompt_length:].to(torch.int64)
        reflection_nums = torch.zeros((response_mask.shape[0],))
        if self.config.get('use_last_response', False):
            response_ids, response_mask, reflection_nums = self.get_last_response(response_ids, response_mask)

        prompt_ids = micro_batch['answer_input_ids'].to(torch.int64)
        micro_batch["input_ids"] = torch.cat([prompt_ids, response_ids], dim=-1)
        prompt_mask = micro_batch["answer_attention_mask"].to(torch.int64)
        micro_batch["attention_mask"] = torch.cat([prompt_mask, response_mask], dim=-1)

        output_td, _ = self.engine._forward_micro_batch(micro_batch, response_length, role=Role.Reward)
        rm_score = output_td["rm_scores"]  # (total_nnz,)
        assert rm_score.shape == (micro_batch["input_ids"].shape[0],)
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
        torch.cuda.reset_peak_memory_stats()
        data = data.to('cuda')
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_data = data

        rm_data.batch = rm_data.batch.cuda()
        response_length = data.batch['responses'].shape[-1]

        with self.gather_manager:
            rm_data = self.gather_manager.preprocess_data(rm_data)

            if self.config.use_dynamic_bsz:
                micro_batches, _ = rearrange_micro_batches(batch=rm_data.batch, max_token_len=self.config.max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = rm_data.batch.split(self.config.micro_batch_size)

            output = []
            total_reflection_nums = []
            for i, micro_batch in enumerate(micro_batches):
                rm_score, reflection_nums = self.forward_micro_batch(micro_batch, response_length=response_length)
                output.append(rm_score)
                total_reflection_nums.append(reflection_nums)
            scores = torch.cat(output, dim=0)  # (batch_size)
            reflection_nums = torch.cat(total_reflection_nums, dim=0)
            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores, 'reflection_nums': reflection_nums})

            output = self.gather_manager.postprocess_data(output)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        max_memory_allocated, max_memory_reserved = get_memory()
        output.meta_info.update({
            'memory/rm_max_allocated': max_memory_allocated,
            'memory/rm_max_reserved': max_memory_reserved
        })
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def do_ndtimeline_action(self, action, *args, **kwargs):
        ndtimeline.do_ndtimeline_action(action, *args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reinit(self, config):
        import gc
        if self._model_initialized:
            del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        self.__init__(config)
