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
Create a XPerfGPT Rollout
"""

from transformers import PreTrainedTokenizer
from verl import DataProto
import copy
from contextlib import contextmanager, nullcontext

from torch import nn
import tempfile
import json
import queue
import threading

from xperf_gpt.inference.session import InferenceSession

from pathlib import Path
import os
from unittest.mock import patch

import torch
import torch.distributed as dist

import torch.distributed
from torch.distributed.device_mesh import init_device_mesh

from contextlib import contextmanager
import logging

from alpha_seed.workers.xperf_rollout.utils import get_xperf_gpt_config
from alpha_seed.workers.xperf_rollout.utils.layout_convert_helper import init_meta
from alpha_seed.workers.streaming_service.xperf_model_prophet import XperfModelProphet
from alpha_seed.workers.xperf_rollout.utils.logits_manipulate import logits_manipulate_fn_core, logits_manipulate_fn_eta, logits_manipulate_fn_minp, logits_manipulate_fn_clip
from alpha_seed.utils.observility import get_profiler_context_wrapped, profile_step
from functools import partial

import ray

try:
    from verl.utils.debug.performance import NullProfileEnter
except:
    print('Cannot find profile utilities. Please use latest verl master')
    raise


@contextmanager
def logging_set_level(level: int = logging.WARNING):
    prev_level = logging.getLogger().level
    logging.getLogger().setLevel(level)
    try:
        yield
    finally:
        logging.getLogger().setLevel(prev_level)


def remove_nccl_files():
    cwd = os.getcwd()
    print(f'cwd: {cwd}')
    for p in Path(cwd).glob("xperf_gpt_nccl_file*"):
        print(f'Removing file {p.name}')
        p.unlink(missing_ok=True)


class AsyncXPerfGPTRollout(object):
    """
    This class creates a training framework agnostic XPerfGPTRollout.
    For weight binding, it will be implemented in the resharding manager.
    User has to pass a tp_device_mesh to create inference session. tp_device_mesh will also
    be used for weight binding, weight redistribution and data resharding. 
    If tp_device_mesh is None, we assume it is executed on a single GPU
    """

    def __init__(self, config, tokenizer, model_hf_config, is_standalone=False):
        self.config = config
        self.tokenizer = tokenizer
        if hasattr(config, 'profile'):
            self.profiler_context = get_profiler_context_wrapped(filename=config.profile.filename,
                                                                 profile_on_ranks=config.profile.profile_on_ranks,
                                                                 upload_to_mlx=config.profile.upload_to_mlx,
                                                                 enable=config.profile.enable,
                                                                 wait=1)
        else:
            self.profiler_context = nullcontext(NullProfileEnter())
        self.is_standalone = is_standalone
        self.async_remain_warmup_step = self.config.rollout_pool.get("warmup_step", 0)
        # auto infer rollout running config
        # off-policy rollout should disable paged attention, for maintaining FIFO order
        use_vllm = self.config.get('enable_paged_attention', True) and not is_standalone
        enable_cuda_graph = self.config.get('enable_cuda_graph', False)
        slot_block_size = self.config.get('slot_block_size', 1024)

        model_cfg = get_xperf_gpt_config(model_config=model_hf_config, tokenizer=tokenizer)
        model_cfg["quant_mode"] = self.config.get("quant_mode", "NO_QUANT")
        sched_cfg = {
            "max_sequence_length": config.prompt_length + config.response_length,
            "max_context_len": config.prompt_length,
            "vllm_block_size": slot_block_size,
        }
        tp_size = self.config.get('tensor_model_parallel_size', 1)
        use_ep = self.config.get('use_ep', False)

        xperf_prophet = XperfModelProphet(model_cfg, sched_cfg, tp_size)
        gpu_memory_utilization = self.config.get('gpu_memory_utilization', 0.7)
        if use_vllm:
            prophet_cfg = xperf_prophet.profile_available_vllm_cfg(gpu_memory_utilization=gpu_memory_utilization)
            max_batch_size = prophet_cfg["orca_max_batch_size"]
            max_ctx_batch_size = 8
            num_slots = prophet_cfg["vllm_num_slots"]
        else:
            prophet_cfg = xperf_prophet.profile_available_orca_cfg(gpu_memory_utilization=gpu_memory_utilization)
            max_batch_size = prophet_cfg["orca_max_batch_size"]
            max_ctx_batch_size = 8
            num_slots = max_batch_size

        # create a 2D device mesh
        if tp_size > 1:
            world_size = torch.distributed.get_world_size()
            gen_dp_size = world_size // tp_size

            print(f'world_size: {world_size}, gen_dp_size: {gen_dp_size}, tp_size: {tp_size}')
            self.device_mesh = init_device_mesh(device_type='cuda',
                                                mesh_shape=(gen_dp_size, tp_size),
                                                mesh_dim_names=('dp', 'tp'))
        else:
            self.device_mesh = None  # this is actually the whole world size. No need to have a device mesh for it.

        print("initializing xperf gpt...")
        print(
            f"use_vllm, enable_cuda_graph, sched_cfg, prophet_cfg, device {use_vllm}, {enable_cuda_graph}, {sched_cfg}, {prophet_cfg}, {enable_cuda_graph}, {os.getenv('CUDA_VISIBLE_DEVICES')}"
        )
        torch.manual_seed(9898)

        if config.get('enable_eot', False):
            bothink = tokenizer.convert_tokens_to_ids("<Begin_of_Thinking>")
            eothink = tokenizer.convert_tokens_to_ids("<End_of_Thinking>")
            boresponse = tokenizer.convert_tokens_to_ids("<Begin_of_Response>")
            eoresponse = tokenizer.convert_tokens_to_ids("<End_of_Response>")

            logits_manipulate_fn = partial(logits_manipulate_fn_core,
                                           manipulate_args={
                                               'eothink': eothink,
                                               'response_length': config.response_length,
                                               'soft_interval': config.get('soft_interval', 512),
                                               'summary_min_space': config.get('summary_min_space', 1024)
                                           })
        elif config.get("ban_eos", 'v0') != 'v0':
            if config['ban_eos'] == 'v1':
                eos_id = tokenizer.eos_token_id
            elif config['ban_eos'] == 'v2':
                eos_id = tokenizer.convert_tokens_to_ids('</')
            else:
                raise NotImplementedError(f'ban_eos {config["ban_eos"]} not supported')
            gen_start_ids = tokenizer.encode(f"{tokenizer.bos_token}assistant\n")

            def find_gen_start_ids(history_id, gen_start_ids):
                for i in range(len(history_id)):
                    if gen_start_ids == history_id[i:i + len(gen_start_ids)]:
                        return i + len(gen_start_ids)
                return None

            def logits_manipulate_fn_core(logits, history_ids, manipulate_args):
                eos_id = manipulate_args['eos_id']
                max_len = manipulate_args['max_len']
                gen_start_ids = manipulate_args['gen_start_ids']
                LARGE = max(1000.0, torch.max(logits) - torch.min(logits))
                ban_eos_list = []
                for history_id in history_ids:
                    gen_start_idx = find_gen_start_ids(history_id, gen_start_ids)
                    assert gen_start_idx is not None, history_id
                    ban_eos = (len(history_id) - gen_start_idx) < (0.5 * max_len)
                    ban_eos_list.append(ban_eos)
                ban_eos = torch.tensor(ban_eos_list, device=logits.device).float()
                logits[:, eos_id] -= LARGE * ban_eos
                return logits

            logits_manipulate_fn = partial(logits_manipulate_fn_core,
                                           manipulate_args={
                                               'eos_id': eos_id,
                                               'max_len': config.response_length,
                                               'gen_start_ids': gen_start_ids
                                           })
        elif config.train_generate_kwargs['min_p'] != -1:
            logits_manipulate_fn = partial(logits_manipulate_fn_minp,
                                           manipulate_args={
                                               'min_p': config.train_generate_kwargs.min_p,
                                           })
        elif config.train_generate_kwargs['eta_epsilon'] != -1:
            logits_manipulate_fn = partial(logits_manipulate_fn_eta,
                                           manipulate_args={
                                               'eta_epsilon': config.train_generate_kwargs.eta_epsilon,
                                           })
        elif config.train_generate_kwargs['logits_clamp'] != 0:
            logits_manipulate_fn = partial(logits_manipulate_fn_clip,
                                           manipulate_args={
                                               'logits_clamp': config.train_generate_kwargs.logits_clamp,
                                           })
        else:
            logits_manipulate_fn = None
        generate_kwargs = dict(max_new_tokens=config.response_length,
                               do_sample=config.train_generate_kwargs.do_sample,
                               top_k=config.train_generate_kwargs.top_k,
                               top_p=config.train_generate_kwargs.top_p,
                               temperature=config.train_generate_kwargs.temperature,
                               logits_manipulate_fn=logits_manipulate_fn)
        inference_sess = InferenceSession(num_slots=num_slots,
                                          max_batch_size=max_batch_size,
                                          max_length=config.prompt_length + config.response_length,
                                          slot_block_size=slot_block_size,
                                          use_vllm=use_vllm,
                                          vocab_tp=config.get('vocab_tp', False),
                                          enable_truncation=False,
                                          context_limit_bs=max_ctx_batch_size,
                                          enable_cuda_graph=enable_cuda_graph)
        inference_sess.max_off_policy_steps = self.config.get('max_off_policy_steps', 5)
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json") as f:
            json.dump(model_cfg, f)
            f.flush()
            global_rank = 0 if not dist.is_initialized() else dist.get_rank()
            tp_rank = 0 if self.device_mesh is None else self.device_mesh['tp'].get_local_rank()
            tp_size = 1 if self.device_mesh is None else self.device_mesh['tp'].size()

            print(f'tp_rank: {tp_rank}, tp_size: {tp_size}')

            assert tp_rank < tp_size

            # get a free port and addr
            from single_controller.base.worker import WorkerHelper
            worker_helper = WorkerHelper()
            if tp_rank == 0:
                free_port_addr = list(worker_helper.get_availale_master_addr_port())
            else:
                free_port_addr = [None, None]
            # broadcast port and addr in tp group
            if self.device_mesh is not None:
                tp_group = self.device_mesh['tp'].get_group()
                tp_src_rank = dist.get_global_rank(tp_group, group_rank=0)
                torch.distributed.broadcast_object_list(free_port_addr, src=tp_src_rank, group=tp_group)
            master_addr, master_port = free_port_addr[0], free_port_addr[1]

            with patch.dict(
                    os.environ, {
                        'RANK': str(tp_rank),
                        'WORLD_SIZE': str(tp_size),
                        'LOCAL_RANK': str(tp_rank),
                        'LOCAL_WORLD_SIZE': str(tp_size),
                        'MASTER_ADDR': master_addr,
                        'MASTER_PORT': master_port,
                        'XPERF_SESSION_SET_TORCH_DEVICE': '0'
                    }):
                local_world_size = 8  # TODO: hard code
                for start_rank in range(0, local_world_size, tp_size):
                    end_rank = start_rank + tp_size
                    if start_rank <= global_rank % local_world_size < end_rank:
                        print(
                            f'Global rank {global_rank}, tp_rank {tp_rank}, master_addr: {master_addr}, master_port: {master_port}'
                        )
                        with logging_set_level(self.config.get('logging_level', 'INFO')):
                            inference_sess.init_inference_engine(f.name,
                                                                 generate_kwargs,
                                                                 rank0_split=False,
                                                                 mp_size=tp_size,
                                                                 enable_metrics=True,
                                                                 use_ep=use_ep)
                    if dist.is_initialized() and tp_size > 1:
                        dist.barrier()
                        if tp_rank == 0:
                            # remove nccl_file
                            remove_nccl_files()
                        dist.barrier()

        if tp_size == 1:
            dist.barrier()

        self.inference_engine = inference_sess
        self.__init_sub_process()

        # offload to CPU
        init_meta(self.inference_engine.engine.module)
        torch.cuda.empty_cache()

    def __init_sub_process(self):
        os.environ["USE_SESSION_CACHE"] = "0"
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.process_thread = threading.Thread(target=self.generate, args=())
        self.process_thread.start()

    def set_rollout_callback_function(self, eos_callback_fn):
        self.inference_engine.set_callback_function(eos_callback_fn=eos_callback_fn)

    def reset_kv_cache(self):
        model = self.inference_engine.engine.module
        for i in range(model.num_layers):
            if (hasattr(model, "kv_mirror_layers")):
                if i + 1 in model.kv_mirror_layers:
                    mirror_layer = model.kv_mirror_imitated_layers[model.kv_mirror_layers.index(i + 1)] - 1
                    model.layers_impl[i].set_kv_cache(model.layers_impl[mirror_layer].get_kv_cache_2HBSD(
                        torch.bfloat16))

    def generate(self):
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
        while True:
            (query_pool, complete_ratio, generation_kwargs, prompt_meta_info) = self.input_queue.get(block=True)
            original_query_pool = copy.deepcopy(query_pool)
            self.inference_engine.set_generator_strategy(**generation_kwargs)
            with logging_set_level(self.config.get('logging_level', 'WARN')), self.profiler_context as p:
                try:
                    self.reset_kv_cache()
                    self.inference_engine.execute(query_pool,
                                                  complete_ratio=complete_ratio,
                                                  stop_event=self.stop_event if self.is_standalone else None,
                                                  prompt_meta_info=prompt_meta_info)
                    profile_step(p, None)
                except Exception as e:
                    if os.getenv('XPERF_DUMP_NAN', '1') == '1':
                        global_rank = 0 if not dist.is_initialized() else dist.get_rank()
                        tp_rank = 0 if self.device_mesh is None else self.device_mesh['tp'].get_local_rank()
                        tp_size = 1 if self.device_mesh is None else self.device_mesh['tp'].size()

                        save_model_name = f"{global_rank}_{tp_rank}_{tp_size}"
                        print("saving... inference engine ... ", f"{save_model_name}_model_engine")
                        torch.save(self.inference_engine.engine.module.layers_weight,
                                   f"{save_model_name}_model_engine_layers_weight.pt")
                        torch.save(self.inference_engine.engine.module.wte_weight,
                                   f"{save_model_name}_model_engine_wte_weight.pt")
                        torch.save(self.inference_engine.engine.module.lm_head_weight,
                                   f"{save_model_name}_model_engine_lm_head_weight.pt")
                        torch.save(self.inference_engine.engine.module.layernorm_weight,
                                   f"{save_model_name}_model_engine_layernorm_weight.pt")
                        torch.save(query_pool, f"{save_model_name}_query_pool.pt")
                        torch.save(self.inference_engine.get_inorder_responses(), f"{save_model_name}_output.pt")

                        from hdfs_io.hdfs_io import hcopy, hmkdir
                        dump_nan_dir = self.config.get("dump_nan", None)
                        if dump_nan_dir is None:
                            print("dump_nan config is not set, skip")
                            raise (e)
                        print(f"dump weights/tensors to {dump_nan_dir}")
                        hmkdir(self.config.get("dump_nan", None))
                        hcopy(f"{save_model_name}_model_engine_layers_weight.pt", self.config.get("dump_nan", None))
                        hcopy(f"{save_model_name}_model_engine_wte_weight.pt", self.config.get("dump_nan", None))
                        hcopy(f"{save_model_name}_model_engine_layernorm_weight.pt", self.config.get("dump_nan", None))
                        hcopy(f"{save_model_name}_model_engine_lm_head_weight.pt", self.config.get("dump_nan", None))
                        hcopy(f"{save_model_name}_query_pool.pt", self.config.get("dump_nan", None))
                        hcopy(f"{save_model_name}_output.pt", self.config.get("dump_nan", None))

                    raise (e)

            response_outputs = []
            is_finished = []
            for prompt, v in zip(original_query_pool, self.inference_engine.get_inorder_responses()):
                response_outputs.append((v.input_ids + v.new_token_ids)[len(prompt):])
                is_finished.append(v.is_finished)
            is_finished = torch.Tensor(is_finished)
            metrics = {}
            if hasattr(self.inference_engine.pp_scheduler,
                       "init_metrics") and self.inference_engine.pp_scheduler.enable_metrics:
                metrics = self.inference_engine.pp_scheduler.metrics
            self.inference_engine.empty_cache()
            self.output_queue.put((response_outputs, is_finished, metrics))

    def _get_output_from_queue(self):
        while True:
            try:
                return self.output_queue.get(timeout=1)
            except Exception:
                assert self.process_thread.is_alive()

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, is_async=False):

        complete_ratio = prompts.meta_info.get('complete_ratio', 1)

        prompt_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        batch_size = prompt_ids.shape[0]
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        off_policy_steps = prompts.batch["off_policy_steps"]
        first_non_one_indices = (prompt_ids != self.tokenizer.pad_token_id).int().argmax(dim=1)
        rmv_padding_prompt_ids = [row[index:].tolist() for row, index in zip(prompt_ids, first_non_one_indices)]

        # (zhangchi.usc1992) note, here we pass all the non_tensor_batch and meta_info to the inference engine as prompt_meta_info.
        prompt_meta_info = [{
            "off_policy_steps": off_policy_step
        } for off_policy_step in off_policy_steps.reshape(-1).tolist()]
        for key, value in prompts.non_tensor_batch.items():
            for i in range(batch_size):
                prompt_meta_info[i][key] = value[i]

        self.input_queue.put(
            (rmv_padding_prompt_ids, complete_ratio, prompts.meta_info['generation_kwargs'], prompt_meta_info))

        if is_async:
            yield
            # stop event
            if self.async_remain_warmup_step <= 0:
                self.stop_event.set()
            (response_outputs, is_finished, metrics) = self._get_output_from_queue()
            if self.async_remain_warmup_step <= 0:
                self.stop_event.clear()
            self.async_remain_warmup_step -= 1
        else:
            # complete_ratio or all prompts are finished
            (response_outputs, is_finished, metrics) = self._get_output_from_queue()

        # Note that the tokenizer may change at runtime
        tokenizer: PreTrainedTokenizer = self.tokenizer
        # remove warning
        tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True
        with patch.object(tokenizer, "padding_side", "right"):
            response_outputs = tokenizer.pad(dict(input_ids=response_outputs),
                                             padding="max_length",
                                             max_length=self.config.response_length,
                                             return_tensors="pt")

        response_ids = response_outputs["input_ids"].cuda().to(torch.int32)
        response_attention_mask = response_outputs["attention_mask"].cuda().to(torch.int8)
        attention_mask = torch.hstack((attention_mask, response_attention_mask))
        input_ids = torch.hstack((prompt_ids, response_ids))

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = {
            # 'prompts': prompt_ids,
            # 'responses': response_ids,
            'input_ids': input_ids.to(torch.int32),  # here input_ids become the whole sentences
            'attention_mask': attention_mask.to(torch.int8),
            'is_finished': is_finished.to(torch.int8),
            'off_policy_steps': off_policy_steps.to(torch.int8),
        }

        out = DataProto.from_dict(batch)
        metrics["off_policy_steps"] = off_policy_steps.tolist()
        out.meta_info["xperf_metrics"] = metrics
        yield out
