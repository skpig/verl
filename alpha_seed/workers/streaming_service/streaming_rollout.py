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

from verl import DataProto

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

try:
    from verl.utils.debug import get_profiler_context
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

    def __init__(self, config, tokenizer, model_hf_config):
        self.config = config
        self.profiler_context = get_profiler_context(filename=config.profile.filename,
                                                     profile_on_ranks=config.profile.profile_on_ranks,
                                                     default_hdfs_dir=config.profile.default_hdfs_dir,
                                                     upload_to_mlx=config.profile.upload_to_mlx,
                                                     enable=config.profile.enable)

        # auto infer rollout running config
        use_vllm = self.config.get('use_vllm', False)
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
        generate_kwargs = dict(max_new_tokens=config.response_length,
                               do_sample=config.train_generate_kwargs.do_sample,
                               top_k=config.train_generate_kwargs.top_k,
                               top_p=config.train_generate_kwargs.top_p,
                               temperature=config.train_generate_kwargs.temperature)

        inference_sess = InferenceSession(num_slots=num_slots,
                                          max_batch_size=max_batch_size,
                                          max_length=config.prompt_length + config.response_length,
                                          slot_block_size=slot_block_size,
                                          use_vllm=use_vllm,
                                          vocab_tp=False,
                                          context_limit_bs=max_ctx_batch_size,
                                          enable_cuda_graph=enable_cuda_graph)

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
                        inference_sess.init_inference_engine(f.name,
                                                             generate_kwargs,
                                                             rank0_split=False,
                                                             mp_size=tp_size,
                                                             enable_metrics=True)
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

    def generate(self):
        while True:
            (query_pool, complete_ratio, generation_kwargs) = self.input_queue.get(block=True)
            self.inference_engine.set_generator_strategy(**generation_kwargs)
            with logging_set_level(self.config.get('logging_level', 'WARN')):
                self.inference_engine.execute(query_pool, complete_ratio=complete_ratio, stop_event=self.stop_event)

            response_outputs = []
            is_finished = []
            for v in self.inference_engine.get_inorder_responses():
                response_outputs.append(v.new_token_ids)
                is_finished.append(v.is_finished)
            is_finished = torch.Tensor(is_finished)

            metrics = {}
            if hasattr(self.inference_engine.pp_scheduler,
                       "init_metrics") and self.inference_engine.pp_scheduler.enable_metrics:
                metrics = self.inference_engine.pp_scheduler.metrics
            self.inference_engine.empty_cache()
            self.output_queue.put((response_outputs, is_finished, metrics))

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, is_async=False):
        complete_ratio = prompts.meta_info.get('complete_ratio', 1)

        prompt_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        first_non_one_indices = (prompt_ids != 1).int().argmax(dim=1)
        rmv_padding_prompt_ids = [row[index:].tolist() for row, index in zip(prompt_ids, first_non_one_indices)]
        self.input_queue.put((rmv_padding_prompt_ids, complete_ratio, prompts.meta_info['generation_kwargs']))

        if is_async:
            yield
            # stop event
            self.stop_event.set()
            (response_outputs, is_finished, metrics) = self.output_queue.get()
            self.stop_event.clear()
        else:
            # complete_ratio or all prompts are finished
            (response_outputs, is_finished, metrics) = self.output_queue.get()

        tokenizer = self.inference_engine.tokenizer
        with patch.object(tokenizer, "padding_side", "right"):
            response_outputs = tokenizer.pad(dict(input_ids=response_outputs),
                                             padding="max_length",
                                             max_length=self.config.response_length,
                                             return_tensors="pt")

        response_ids = response_outputs["input_ids"].cuda()
        response_attention_mask = response_outputs["attention_mask"].cuda()
        attention_mask = torch.hstack((attention_mask, response_attention_mask))
        input_ids = torch.hstack((prompt_ids, response_ids))

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = {
            'prompts': prompt_ids,
            'responses': response_ids,
            'input_ids': input_ids,  # here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'is_finished': is_finished
        }

        out = DataProto.from_dict(batch)
        out.meta_info["xperf_metrics"] = metrics
        yield out
