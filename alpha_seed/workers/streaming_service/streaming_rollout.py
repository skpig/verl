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
import itertools
import uuid

from distlib.locators import Locator
from pydantic import UUID4
from transformers import PreTrainedTokenizer
from mono_rl import DataProto
import copy
from contextlib import contextmanager, nullcontext

import numpy as np
from torch import nn
import tempfile
import json
import queue
import threading
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from typing import AsyncGenerator, List
import asyncio
import xperf_gpt
from mono_rl.single_controller import Execute

from alpha_seed.workers.xperf_rollout.session import InferenceSession, StepProfiler, LoadMetric
from alpha_seed.workers.xperf_rollout.component.query import Query, AsyncQuery
from mono_rl.single_controller.base.worker import WorkerHelper, Worker

from pathlib import Path
import os
from unittest.mock import patch

import torch
import torch.distributed as dist

import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from mono_rl.single_controller import register, Dispatch

from contextlib import contextmanager
import logging

from alpha_seed.workers.xperf_rollout.utils import get_xperf_gpt_config
from alpha_seed.workers.xperf_rollout.utils.custom_xperf_convert_helper import XCustomInferenceModuleAdapter
from alpha_seed.workers.streaming_service.streaming_utils import is_multihost_model, DataPack, pack_to_dataproto, get_gpus_per_node
from alpha_seed.workers.xperf_rollout.utils.layout_convert_helper import offload_to_device
from alpha_seed.workers.xperf_rollout.utils.pooled_ucx_weights_communicator import UCXWeightsCommunicator
from alpha_seed.workers.xperf_rollout.utils.nccl_weights_communicator import NCCLWeightsCommunicator
from alpha_seed.workers.streaming_service.xperf_model_prophet import XperfModelProphet
from alpha_seed.workers.xperf_rollout.utils.logits_manipulate import logits_manipulate_fn_core, logits_manipulate_fn_eta, logits_manipulate_fn_minp, logits_manipulate_fn_clip
from alpha_seed.utils.observility import get_profiler_context_wrapped, profile_step
from functools import partial
import omegaconf
import dill

import ray

try:
    from mono_rl.utils.debug.performance import NullProfileEnter
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

    def __init__(self, config, role: str = 'rollout'):
        self.config = config
        self.role = role
        self.weights_loaded = threading.Event()  # 表示weights是否已经加载完毕，hybrid里回load/offload交替
        # 表示engine是否在gen loop里，hybrid模式如果weights offloaded，不应该在gen loop里，可以用这个event来判断状态
        self.gen_loop_exited = threading.Event()

    def initialize(self,
                   local_path=None,
                   is_standalone=False,
                   rank=None,
                   world_size=None,
                   master_addr=None,
                   master_port=None):
        from alpha_seed.utils.ckpt.hdfs import download_minimal_required_files

        local_path = download_minimal_required_files(local_path, False, torch.distributed.get_rank(),
                                                     torch.distributed.get_world_size())

        from transformers import AutoTokenizer, AutoConfig
        from omegaconf import OmegaConf
        print(f'local_path: {local_path}')

        from seed_models import P4Config, P5Config, P6Config
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=False)
        self.model_hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=False)
        self.is_standalone = is_standalone

        self.rank = rank
        self.world_size = world_size
        if master_addr is None:
            master_addr_port = list(WorkerHelper().get_availale_master_addr_port())
            master_addr = master_addr_port[0]
            master_port = master_addr_port[1]
        self.master_addr = master_addr
        self.master_port = master_port
        xperf_gpt.load_xperf_gpt()

    def setup_rollout(self):
        if hasattr(self.config, 'profile'):
            self.profiler_context = get_profiler_context_wrapped(filename=self.config.profile.filename,
                                                                 profile_on_ranks=self.config.profile.profile_on_ranks,
                                                                 upload_to_mlx=self.config.profile.upload_to_mlx,
                                                                 enable=self.config.profile.enable,
                                                                 wait=1)
        else:
            self.profiler_context = nullcontext(NullProfileEnter())
        self.async_remain_warmup_step = self.config.rollout_pool.get("warmup_step", 0)
        # auto infer rollout running config
        enable_paged_attn = self.config.get('enable_paged_attention', True) and not self.config.xperf_custom.enable
        enable_cuda_graph = self.config.get('enable_cuda_graph', False)
        slot_block_size = self.config.get('slot_block_size', 1024)

        model_cfg = get_xperf_gpt_config(model_config=self.model_hf_config, tokenizer=self.tokenizer)
        vision_cfg = None
        if 'vision_config' in model_cfg:
            text_cfg = model_cfg['text_config']
            vision_cfg = model_cfg['vision_config']
        else:
            text_cfg = model_cfg
        text_cfg["quant_mode"] = self.config.get("quant_mode", "NO_QUANT")
        sched_cfg = {
            "max_sequence_length": self.config.prompt_length + self.config.response_length,
            "max_context_len": self.config.prompt_length,
            "vllm_block_size": slot_block_size,
        }
        tp_size = self.config.get('tensor_model_parallel_size', 1)
        use_ep = self.config.get('use_ep', False)
        multi_host_tp = is_multihost_model(tp_size)

        # TODO(caisonghua): enable XperfModelProphet vit part later
        xperf_prophet = XperfModelProphet(text_cfg, sched_cfg, tp_size)
        gpu_memory_utilization = self.config.get('gpu_memory_utilization', 0.7)
        if enable_paged_attn:
            prophet_cfg = xperf_prophet.profile_available_vllm_cfg(gpu_memory_utilization=gpu_memory_utilization)
            max_batch_size = prophet_cfg["orca_max_batch_size"]
            max_ctx_batch_size = self.config.get("max_ctx_batch_size", 8)
            num_slots = prophet_cfg["vllm_num_slots"]
        else:
            prophet_cfg = xperf_prophet.profile_available_orca_cfg(gpu_memory_utilization=gpu_memory_utilization)
            max_batch_size = prophet_cfg["orca_max_batch_size"]
            max_ctx_batch_size = self.config.get("max_ctx_batch_size", 8)
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
            f"enable_paged_attn, enable_cuda_graph, sched_cfg, prophet_cfg, device {enable_paged_attn}, {enable_cuda_graph}, {sched_cfg}, {prophet_cfg}, {enable_cuda_graph}, {os.getenv('CUDA_VISIBLE_DEVICES')}"
        )
        torch.manual_seed(9898)

        logits_manipulate_fn = None
        generate_kwargs = dict(max_new_tokens=self.config.response_length,
                               do_sample=self.config.train_generate_kwargs.do_sample,
                               top_k=self.config.train_generate_kwargs.top_k,
                               top_p=self.config.train_generate_kwargs.top_p,
                               temperature=self.config.train_generate_kwargs.temperature,
                               logits_manipulate_fn=logits_manipulate_fn)
        step_profiler = StepProfiler(self.config.profile)
        if self.config.xperf_triton.enable:
            max_batch_size = self.config.xperf_triton.max_batch_size
            max_ctx_batch_size = self.config.xperf_triton.max_ctx_batch_size
            num_slots = max_batch_size
            if self.config.xperf_triton.use_paged_attn:
                num_slots = self.config.xperf_triton.num_slots

        inference_sess = InferenceSession(num_slots=num_slots,
                                          max_batch_size=max_batch_size,
                                          max_length=self.config.prompt_length + self.config.response_length,
                                          slot_block_size=slot_block_size,
                                          enable_paged_attn=enable_paged_attn,
                                          vocab_tp=self.config.get('vocab_tp', False),
                                          enable_truncation=False,
                                          context_limit_bs=max_ctx_batch_size,
                                          enable_cuda_graph=enable_cuda_graph,
                                          standalone=self.is_standalone,
                                          schedule_strategy=self.config.schedule_strategy,
                                          step_profiler=step_profiler)
        inference_sess.max_off_policy_steps = self.config.get('max_off_policy_steps', 5)
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json") as f:
            print(f"load xperf config ... {text_cfg}")
            json.dump(text_cfg, f)
            f.flush()
            global_rank = 0 if not dist.is_initialized() else dist.get_rank()
            tp_rank = 0 if self.device_mesh is None else self.device_mesh['tp'].get_local_rank()
            tp_size = 1 if self.device_mesh is None else self.device_mesh['tp'].size()

            print(f'tp_rank: {tp_rank}, tp_size: {tp_size}')

            assert tp_rank < tp_size

            # get a free port and addr
            from mono_rl.single_controller.base.worker import WorkerHelper
            worker_helper = WorkerHelper()
            if tp_rank == 0:
                free_port_addr = list(worker_helper.get_availale_master_addr_port())
            else:
                free_port_addr = [None, None]

            if is_multihost_model(self.config.get('tensor_model_parallel_size', 1)):
                self._set_multihost_env()
                free_port_addr[1] = int(os.getenv("PORT9", free_port_addr[1])) if tp_rank == 0 else None

            # broadcast port and addr in tp group
            if self.device_mesh is not None:
                tp_group = self.device_mesh['tp'].get_group()
                tp_src_rank = dist.get_global_rank(tp_group, group_rank=0)
                torch.distributed.broadcast_object_list(free_port_addr, src=tp_src_rank, group=tp_group)
                inference_sess.set_tp_group(tp_group)
            else:
                inference_sess.set_tp_group(None)

            master_addr, master_port = free_port_addr[0], free_port_addr[1]
            gpus_per_node = get_gpus_per_node()
            local_world_size = min(tp_size, gpus_per_node)
            with patch.dict(
                    os.environ, {
                        'RANK': str(tp_rank),
                        'WORLD_SIZE': str(tp_size),
                        'LOCAL_RANK': str(tp_rank % local_world_size),
                        'LOCAL_WORLD_SIZE': str(local_world_size),
                        'MASTER_ADDR': master_addr,
                        'MASTER_PORT': str(int(master_port) - 1),
                        'XPERF_SESSION_SET_TORCH_DEVICE': '0'
                    }):
                for start_rank in range(0, gpus_per_node, tp_size):
                    end_rank = start_rank + tp_size
                    if start_rank <= global_rank % local_world_size < end_rank:
                        print(
                            f'XPerf init Global rank {global_rank}, tp_rank {tp_rank}, master_addr: {master_addr}, master_port: {master_port}'
                        )
                        with logging_set_level(self.config.get('logging_level', 'INFO')):
                            xperf_custom_kwargs = {}
                            if self.config.xperf_custom.enable:
                                xperf_custom_kwargs['xperf_custom_backbone'] = self.config.xperf_custom.backbone
                                xperf_custom_kwargs['xperf_custom_preset'] = self.config.xperf_custom.preset

                            inference_sess.init_inference_engine(f.name,
                                                                 generate_kwargs,
                                                                 rank0_split=False,
                                                                 mp_size=tp_size,
                                                                 enable_metrics=True,
                                                                 use_ep=use_ep,
                                                                 tokenizer_path=self.tokenizer.name_or_path,
                                                                 multi_host_tp=multi_host_tp,
                                                                 use_xperf_custom=self.config.xperf_custom.enable,
                                                                 use_xperf_triton=self.config.xperf_triton.enable,
                                                                 vit_config=vision_cfg,
                                                                 xperf_triton_cfg=self.config.xperf_triton,
                                                                 **xperf_custom_kwargs)
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

        # offload to meta device
        if not self.is_standalone:
            offload_to_device(self.inference_engine.engine.module, "meta")
        torch.cuda.empty_cache()

    def add_inflight_query(self, query: Query) -> str:
        aq = AsyncQuery(query)
        # inference_engine is running on a different threads
        with self.inference_engine.update_weights_lock:
            self.inference_engine.pending.append(aq)
        return aq.id

    async def get_inflight_query(self, query_id: str) -> Query:
        with self.inference_engine.update_weights_lock:
            aq = self.inference_engine.pending.query_pool.pop(query_id)
        await aq.wait_until_done()
        _ = self.inference_engine.finished.pop(query_id, None)
        aq.query.input_embedding = None
        if aq.exception is not None:
            raise aq.exception
        return aq.query

    def get_all_queries(self, query_type: str) -> List[Query]:
        return self.inference_engine.get_all_queries(query_type=query_type, retain_finished=False)

    def get_load_metrics(self) -> LoadMetric:
        return self.inference_engine.get_load_metrics()

    def get_master_addr_port(self):
        return self.master_addr, self.master_port

    def _set_tuner_config(self):
        os.environ["USE_SESSION_CACHE"] = "0"
        if self.config.get("quant_mode", "NO_QUANT") == "WFP8":
            # set environment variables for tuner
            os.environ["XGPT_TUNER_ENABLE"] = os.getenv("XGPT_TUNER_ENABLE", "1")
            os.environ["XPERF_TUNER_ONLINE_PRIORITY"] = os.getenv("XPERF_TUNER_ONLINE_PRIORITY", "1")
            os.environ["XPERF_TUNER_ONLINE_VERSION"] = "2.0.0+xgpt"
            base_dir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
            tuning_path = os.path.join(base_dir, "xperf_rollout", "tuning")
            os.environ["XPERF_TUNER_CONFIG_LOAD_PATH"] = tuning_path

    def _set_multihost_env(self):
        os.environ["NCCL_SOCKET_IFNAME"] = os.getenv("NCCL_SOCKET_IFNAME", "eth0")
        os.environ["NCCL_IB_HCA"] = os.getenv("NCCL_IB_HCA", "^=mlx5_0")
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        os.environ["NCCL_IB_GID_INDEX"] = "3"
        os.environ["NCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_IB_TIMEOUT"] = "20"
        os.environ["NCCL_IB_RETRY_CNT"] = "7"
        os.environ["NCCL_MULTI_HOST"] = "1"

    def __init_sub_process(self):
        self._set_tuner_config()
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
        if self.config.mode == "server" or self.role == "rollout_server":
            self.stop_event.set()
        # rollout_server 等于 standalone rollout + elastic模式
        use_async_gen = self.config.mode == "server" or self.role == "rollout_server"
        self.process_thread = threading.Thread(target=self.async_generate if use_async_gen else self.generate,
                                               name="streaming-rollout-background-generate")
        self.process_thread.start()

    def set_rollout_callback_function(self, eos_callback_fn):

        def make_eos_call_back_fn(eos_callback_fn, device_mesh):
            from alpha_seed.workers.xperf_rollout.component.query import Query

            def tp_eos_callback_fn(query: Query):
                if device_mesh is None:
                    tp_rank = 0
                else:
                    tp_rank = device_mesh['tp'].get_local_rank()

                if tp_rank == 0:
                    # only happens on tp rank zero
                    eos_callback_fn(query)

            return tp_eos_callback_fn

        tp_eos_callback_fn = make_eos_call_back_fn(eos_callback_fn, self.device_mesh)
        self.inference_engine.set_callback_function(eos_callback_fn=tp_eos_callback_fn)

    def reset_status(self):
        model = self.inference_engine.engine.module
        if isinstance(model, XCustomInferenceModuleAdapter):
            return
        for i in range(model.num_layers):
            if (hasattr(model, "kv_mirror_layers")):
                if i + 1 in model.kv_mirror_layers:
                    mirror_layer = model.kv_mirror_imitated_layers[model.kv_mirror_layers.index(i + 1)] - 1
                    model.layers_impl[i].set_kv_cache(model.layers_impl[mirror_layer].get_kv_cache_2HBSD(
                        torch.bfloat16))
        if hasattr(self.inference_engine.engine.module, "_prepare_yarn_embedding"):
            self.inference_engine.engine.module._prepare_yarn_embedding()

    def _dump_context(self):
        if os.getenv('XPERF_DUMP_NAN', '0') == '1':
            if isinstance(self.inference_engine.engine.module, XCustomInferenceModuleAdapter):
                print("XPerf custom engine dump weights not supported, skipping...")
                return
            from hdfs_io.hdfs_io import hcopy, hmkdir
            dump_nan_dir = self.config.get("dump_nan", None)
            if dump_nan_dir is None:
                print("dump_nan config is not set, skip")
                return
            global_rank = 0 if not dist.is_initialized() else dist.get_rank()
            tp_rank = 0 if self.device_mesh is None else self.device_mesh['tp'].get_local_rank()
            tp_size = 1 if self.device_mesh is None else self.device_mesh['tp'].size()
            save_model_name = f"{global_rank}_{tp_rank}_{tp_size}"
            print("saving... inference engine ... ", f"{save_model_name}_model_engine")
            torch.save(self.inference_engine.engine.module.layers_weight,
                       f"{save_model_name}_model_engine_layers_weight.pt")
            torch.save(self.inference_engine.engine.module.wte_weight, f"{save_model_name}_model_engine_wte_weight.pt")
            torch.save(self.inference_engine.engine.module.lm_head_weight,
                       f"{save_model_name}_model_engine_lm_head_weight.pt")
            torch.save(self.inference_engine.engine.module.layernorm_weight,
                       f"{save_model_name}_model_engine_layernorm_weight.pt")
            torch.save(self.inference_engine.get_inorder_responses(), f"{save_model_name}_output.pt")
            print(f"dump weights/tensors to {dump_nan_dir}")
            hmkdir(self.config.get("dump_nan", None))
            hcopy(f"{save_model_name}_model_engine_layers_weight.pt", self.config.get("dump_nan", None))
            hcopy(f"{save_model_name}_model_engine_wte_weight.pt", self.config.get("dump_nan", None))
            hcopy(f"{save_model_name}_model_engine_layernorm_weight.pt", self.config.get("dump_nan", None))
            hcopy(f"{save_model_name}_model_engine_lm_head_weight.pt", self.config.get("dump_nan", None))
            hcopy(f"{save_model_name}_output.pt", self.config.get("dump_nan", None))

    def generate(self):
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
        while True:
            (query_pool, complete_ratio, generation_kwargs, prompt_meta_info) = self.input_queue.get(block=True)
            original_query_pool = copy.deepcopy(query_pool)
            self.inference_engine.set_generator_strategy(**generation_kwargs)
            with logging_set_level(self.config.get('logging_level', 'WARN')), self.profiler_context as p:
                try:
                    self.reset_status()
                    self.gen_loop_exited.clear()
                    self.inference_engine.execute(query_pool,
                                                  complete_ratio=complete_ratio,
                                                  stop_event=self.stop_event if self.is_standalone else None,
                                                  prompt_meta_info=prompt_meta_info)
                    profile_step(p, None)
                    self.gen_loop_exited.set()
                except Exception as e:
                    self._dump_context()
                    raise (e)

            response_outputs = []
            response_log_probs = []
            response_probs_gt_threshold_num = []
            response_probs_lt_threshold_sum = []
            is_finished = []
            off_policy_steps = []
            model_output_masks = []
            query_metrics = []
            extra_data = []
            for prompt, v in zip(original_query_pool, self.inference_engine.get_inorder_responses()):
                response_output_ids = (v.input_ids + v.new_token_ids)[len(prompt):]
                response_outputs.append(response_output_ids)
                response_log_probs.append(v.new_token_log_probs)
                response_probs_gt_threshold_num.append(v.probs_gt_threshold_num)
                response_probs_lt_threshold_sum.append(v.probs_lt_threshold_sum)
                is_finished.append(v.is_finished)
                off_policy_steps.append([-1] * len(v.new_token_log_probs))
                model_output_masks.append(v.model_output_mask)
                query_metrics.append(v.metrics)
                extra_data.append(v.extra_data)

            metrics = {}
            if hasattr(self.inference_engine.infer_scheduler,
                       "init_metrics") and self.inference_engine.infer_scheduler.enable_metrics:
                metrics = self.inference_engine.infer_scheduler.metrics
            query_metrics_dict = dict()
            for q_metrics in query_metrics:
                for key, val in q_metrics.items():
                    if key not in query_metrics_dict:
                        query_metrics_dict[key] = val
                    if type(val) != type(query_metrics_dict[key]):
                        continue
                    query_metrics_dict[key] += val
            metrics.update(query_metrics_dict)
            self.inference_engine.empty_cache()
            data_pack = DataPack(response_outputs=response_outputs,
                                 response_log_probs=response_log_probs,
                                 response_probs_gt_threshold_num=response_probs_gt_threshold_num,
                                 response_probs_lt_threshold_sum=response_probs_lt_threshold_sum,
                                 response_model_output_mask=model_output_masks,
                                 this_turn_off_policy_steps=off_policy_steps,
                                 is_finished=is_finished,
                                 extra_data=extra_data,
                                 metrics=metrics)
            self.output_queue.put(data_pack)

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
        off_turn_off_policy_steps = prompts.batch["off_policy_steps"]
        first_non_one_indices = (prompt_ids != self.tokenizer.pad_token_id).int().argmax(dim=1)
        rmv_padding_prompt_ids = [row[index:].tolist() for row, index in zip(prompt_ids, first_non_one_indices)]
        generation_kwargs = prompts.meta_info['generation_kwargs']

        # (zhangchi.usc1992) note, here we pass all the non_tensor_batch and meta_info to the inference engine as prompt_meta_info.
        prompt_meta_info = [{
            "off_policy_steps": max(off_policy_step),
            "generation_kwargs": generation_kwargs,
        } for off_policy_step in off_turn_off_policy_steps.tolist()]
        for key, value in prompts.non_tensor_batch.items():
            for i in range(batch_size):
                prompt_meta_info[i][key] = value[i]

        self.input_queue.put((rmv_padding_prompt_ids, complete_ratio, generation_kwargs, prompt_meta_info))

        if is_async:
            yield
            # stop event
            if self.async_remain_warmup_step <= 0:
                self.stop_event.set()
            data_pack = self._get_output_from_queue()
            if self.async_remain_warmup_step <= 0:
                self.stop_event.clear()
            self.async_remain_warmup_step -= 1
        else:
            # complete_ratio or all prompts are finished
            data_pack = self._get_output_from_queue()

        out = pack_to_dataproto(prompts=prompts, data_pack=data_pack, config=self.config, tokenizer=self.tokenizer)
        yield out

    def async_generate(self):
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
        while True:
            # 这里先等rollout的weights确定load好了再进入engine的循环，避免在hybrid engine里提前进入engine循环
            # 触发到_should_terminate里的tensor all reduce导致和actor model初始化互相死锁
            self.weights_loaded.wait()
            with logging_set_level(self.config.get('logging_level', 'WARN')), self.profiler_context as p:
                try:
                    self.reset_status()
                    self.gen_loop_exited.clear()
                    self.inference_engine.async_execute(self.stop_event)
                    profile_step(p, None)
                    self.gen_loop_exited.set()
                except Exception as e:
                    self._dump_context()
                    raise (e)


from omegaconf import DictConfig


@ray.remote
class RemoteAsyncXPerfGPTRollout(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        if not torch.distributed.is_initialized():
            from datetime import timedelta
            timeout = timedelta(minutes=int(os.getenv('NCCL_TIMEOUT', 60)))
            torch.distributed.init_process_group(backend="nccl", timeout=timeout)
        self.config = config
        self.rollout_actor = AsyncXPerfGPTRollout(config=self.config.rollout, role=role)
        self._weights_loaded = threading.Event()
        self._hybrid_rollout_addrs = None

    def _stop_engine(self):
        if self.rollout_actor.stop_event.is_set():
            return

        with self.rollout_actor.inference_engine.update_weights_lock:
            self.rollout_actor.stop_event.set()

        # wait until completely stopped
        self.rollout_actor.gen_loop_exited.wait()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def ready(self):
        # worker是否ready可以接受请求(model compute相关)
        # 子类继承这个方法自定义就绪判断，例如需要额外初始化model的
        return self._weights_loaded.is_set()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def will_be_destroyed(self):
        self._stop_engine()
        self.rollout_actor.reset_status()
        self.rollout_actor.inference_engine.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def add_inflight_queries(self, queries: List[Query]):
        ret = []
        for q in queries:
            qid = self.rollout_actor.add_inflight_query(q)
            ret.append(qid)
        return ret

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def get_inflight_query(self, query_id) -> Query:
        return await self.rollout_actor.get_inflight_query(query_id)

    # 只在dp_size=1的情况下调用，所以这里rank0执行即可
    @register(execute_mode=Execute.RANK_ZERO, blocking=True)
    def get_all_queries(self, query_type: str):
        return self.rollout_actor.get_all_queries(query_type)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=True)
    def get_load_metrics(self) -> LoadMetric:
        return self.rollout_actor.get_load_metrics()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self, *args, **kwargs):
        self.rollout_actor.initialize(self.config.model.path, True)
        self.rollout_actor.setup_rollout()
        weights_communicator = self.config.rollout.weights_communicator
        CommunicatorCls = UCXWeightsCommunicator if weights_communicator == "ucx" else NCCLWeightsCommunicator
        self.weights_communicator = CommunicatorCls(inference_engine=self.rollout_actor.inference_engine,
                                                    standalone=self.rollout_actor.is_standalone,
                                                    device_mesh=self.rollout_actor.device_mesh)
        # build device mesh
        self.master_address = os.getenv('MASTER_ADDR', 'localhost')
        self.master_port = os.getenv('MASTER_PORT', '12345')

        print(f'Master address: {self.master_address}, Master port: {self.master_port}')

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def setup_as_client(self, role, source_addresses, hybrid_rollout_addrs: List[str]):
        self._hybrid_rollout_addrs = hybrid_rollout_addrs
        # connect to weight source after model initialized
        source_address = source_addresses[self.rank]
        self.weights_communicator.setup_as_client(role, source_address)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def setup_standalone_worker_comm(self, hybrid_master_address, standalone_master_address, port, role):
        self.weights_communicator.setup_standalone_worker_comm(hybrid_master_address, standalone_master_address, port,
                                                               role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_eos_callback_fn(self, eos_callback_fn):
        self.rollout_actor.set_rollout_callback_function(eos_callback_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_master_addr(self):
        key = "standalone_master_addr" if self.rollout_actor.is_standalone else "hybrid_master_addr"
        out = DataProto.from_dict(tensors={'mock': torch.tensor([[0]])}, meta_info={key: self.master_address})
        return out

    # caller 自己去wait这个non-blocking
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def stop_server_before_weights_update_non_blocking(self):
        self._stop_engine()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def stop_server_before_weights_update(self):
        self._stop_engine()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def update_standalone_worker(self, role):
        offload_to_device(self.rollout_actor.inference_engine.engine.module, "cuda")
        self.weights_communicator.wait_for_setup_completed()
        with self.rollout_actor.inference_engine.update_weights_lock:
            self.weights_communicator.update_standalone_worker(role)
        self._weights_loaded.set()
        self.rollout_actor.weights_loaded.set()

    # group 内任意一个rank发送结束信号即可
    @register(execute_mode=Execute.RANK_ZERO)
    def update_standalone_worker_end(self):
        self.weights_communicator.update_standalone_worker_end(self._hybrid_rollout_addrs)

    # caller 自己去wait这个non-blocking
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def restart_server_after_weights_update_non_blocking(self):
        with self.rollout_actor.inference_engine.update_weights_lock:
            self.rollout_actor.reset_status()
            self.rollout_actor.stop_event.clear()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def restart_server_after_weights_update(self):
        with self.rollout_actor.inference_engine.update_weights_lock:
            self.rollout_actor.reset_status()
            self.rollout_actor.stop_event.clear()
