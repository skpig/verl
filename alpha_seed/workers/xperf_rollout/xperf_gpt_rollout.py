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

from .utils import get_xperf_gpt_config
from .utils.weight_loader import offload_to_cpu, init_meta


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
        p.unlink()


class XPerfGPTRollout(object):
    """
    This class creates a training framework agnostic XPerfGPTRollout.
    For weight binding, it will be implemented in the resharding manager.
    User has to pass a tp_device_mesh to create inference session. tp_device_mesh will also
    be used for weight binding, weight redistribution and data resharding. 
    If tp_device_mesh is None, we assume it is executed on a single GPU
    """

    def __init__(self, config, tokenizer, model_hf_config):
        self.config = config
        tp_size = self.config.get('tensor_model_parallel_size', 1)

        num_kv_heads = model_hf_config.num_key_value_heads
        assert tp_size <= num_kv_heads, f'tp_size {tp_size} must not be larger than num_kv_heads {num_kv_heads}'

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

        generate_kwargs = dict(max_new_tokens=config.response_length,
                               do_sample=config.train_generate_kwargs.do_sample,
                               top_k=config.train_generate_kwargs.top_k,
                               top_p=config.train_generate_kwargs.top_p,
                               temperature=config.train_generate_kwargs.temperature)

        use_vllm = self.config.get('use_vllm', False)
        num_slots = self.config.get('num_slots', 256)
        slot_block_size = self.config.get('slot_block_size', 1024)
        enable_cuda_graph = self.config.get('enable_cuda_graph', False)

        print("initializing xperf gpt...")
        print(
            f"use_vllm, num_slots, slot_block_size, enable_cuda_graph {use_vllm}, {num_slots}, {slot_block_size}, {enable_cuda_graph}"
        )

        inference_sess = InferenceSession(num_slots=num_slots,
                                          max_batch_size=config.micro_batch_size,
                                          max_length=config.prompt_length + config.response_length,
                                          slot_block_size=slot_block_size,
                                          use_vllm=use_vllm,
                                          vocab_tp=False,
                                          context_limit_bs=8,
                                          enable_cuda_graph=enable_cuda_graph)
        xperf_config = get_xperf_gpt_config(model_config=model_hf_config, tokenizer=tokenizer)

        with tempfile.NamedTemporaryFile(mode='w', suffix=".json") as f:
            json.dump(xperf_config, f)
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

        # offload to CPU
        init_meta(self.inference_engine.engine.module)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        meta_info = prompts.meta_info
        num_bon = meta_info.get("num_bon", 1)
        timeout_seconds = self.config.get('timeout_seconds', 60 * 30)

        prompt_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # prompts
        tokenizer = self.inference_engine.tokenizer
        query_pool = tokenizer.batch_decode(prompt_ids.cpu())
        query_pool = [x.replace(tokenizer.pad_token, '') for x in query_pool]
        # print("infer... num queries.. {} num_bon.. {}".format(len(query_pool), num_bon))
        sampler = self.inference_engine.sampler

        generation_kwargs = prompts.meta_info['generation_kwargs']
        self.inference_engine.set_generator_strategy(**generation_kwargs)

        with logging_set_level(self.config.get('logging_level', 'WARN')):
            self.inference_engine.execute(query_pool, timeout=timeout_seconds, num_BoN=num_bon)

        response_outputs = dict(input_ids=[v.new_token_ids for v in self.inference_engine.get_inorder_responses()])
        metrics = {}
        if hasattr(self.inference_engine.pp_scheduler,
                   "init_metrics") and self.inference_engine.pp_scheduler.enable_metrics:
            metrics = self.inference_engine.pp_scheduler.metrics
        # empty kv cache
        self.inference_engine.empty_cache()

        with patch.object(tokenizer, "padding_side", "right"):
            response_outputs = tokenizer.pad(response_outputs,
                                             padding="max_length",
                                             max_length=self.config.response_length,
                                             return_tensors="pt")
        response_ids = response_outputs["input_ids"].cuda()
        response_attention_mask = response_outputs["attention_mask"].cuda()

        prompt_ids = prompt_ids.repeat(num_bon, 1)
        attention_mask = attention_mask.repeat(num_bon, 1)

        attention_mask = torch.hstack((attention_mask, response_attention_mask))
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        input_ids = torch.hstack((prompt_ids, response_ids))

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = {
            'prompts': prompt_ids,
            'responses': response_ids,
            'input_ids': input_ids,  # here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

        out = DataProto.from_dict(batch)
        out.meta_info["xperf_metrics"] = metrics
        return out
