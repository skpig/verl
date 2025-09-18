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
import time
import traceback

from alpha_seed.workers.xperf_rollout.utils.base_weights_communicator import WeightsCommunicator, WeightsRankInfo
from mono_rl import DataProto
import copy
from contextlib import contextmanager, nullcontext

import numpy as np
from torch import nn
import tempfile
import json
import queue
import threading
from typing import List, Type, Optional
import xperf_gpt
from mono_rl.single_controller import Execute

from alpha_seed.workers.xperf_rollout.session import InferenceSession, StepProfiler, LoadMetric
from alpha_seed.workers.xperf_rollout.component.query import Query, AsyncQuery
from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManagerRegisterCenter
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
from alpha_seed.workers.streaming_service.streaming_utils import is_multihost_model, DataPack, pack_to_dataproto, get_gpus_per_node, get_gpu_support_nvlink
from alpha_seed.workers.xperf_rollout.utils.layout_convert_helper import offload_to_device
from alpha_seed.workers.xperf_rollout.utils.pooled_ucx_weights_communicator import UCXWeightsCommunicator, \
    WeightsUpdatingInterrupt
from alpha_seed.workers.xperf_rollout.utils.nccl_weights_communicator import NCCLWeightsCommunicator
from alpha_seed.workers.streaming_service.xperf_model_prophet import XperfModelProphet
from alpha_seed.utils.observility import get_profiler_context_wrapped, profile_step
from mono_rl.models.seed_models.modeling_vlm import add_pixel_values_to_inflight_query
from alpha_seed.workers.xperf_rollout.profiler.visualizer import visualize_metrics
from mono_rl.utils.dataset.dist_data_util import get_dist_data_manager, get_local_inputs
import omegaconf

import ray

try:
    from mono_rl.utils.debug.performance import NullProfileEnter
except:
    print('Cannot find profile utilities. Please use latest verl master')
    raise

logger = logging.getLogger(__name__)


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
        self.stop_event = threading.Event()
        self.exit_event = threading.Event()
        self.is_async_generate = self.config.mode == "server" or self.role == "rollout_server"
        self.process_thread = None
        self._process_thread_last_error = None
        self._process_thread_last_tb = None
        self._request_managers = {}
        self._update_ref = None

        # 初始状态下不在loop里，先set
        self.gen_loop_exited.set()

    def switch_mode(self, to_async: bool):
        """
        Switch the background generation thread between async_generate and generate.
        Will stop the current thread and start a new one.

        :param to_async: If True, switch to async_generate; else use generate.
        """
        print(f"[Switch] Switching to {'async_generate' if to_async else 'generate'} mode...")
        if self.process_thread and self.process_thread.is_alive():
            self.exit_event.set()
            self.stop_event.set()
            self.process_thread.join(timeout=5)
            if self.process_thread.is_alive():
                print("[Switch] process_thread did not exit cleanly")
            else:
                print("[Switch] Previous process_thread stopped successfully")

        new_target = self.async_generate if to_async else self.generate
        self.process_thread = threading.Thread(target=new_target, name="streaming-rollout-background-generate")
        self.exit_event.clear()
        self.process_thread.start()
        print("[Switch] New process_thread started")

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

        from seed_models import P4Config, P5Config, P6Config, M8Config

        # tokenizer_path = os.path.join(local_path, 'tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=False)
        self.model_hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=False)
        self.vit_model_path = local_path
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
        text_cfg["dynamic_quant"] = self.config.get("use_fp8_attention", False)
        if "W4A8" in text_cfg["quant_mode"]:
            text_cfg["attn_use_w8"] = True
            text_cfg["has_output_quant"] = False
            text_cfg["use_perrank_qscale"] = False
            text_cfg["use_perexpert_qscale"] = True
            text_cfg["has_kv_qscale"] = False
            text_cfg["m8_optimal_fusion"] = True
            if text_cfg["quant_mode"] == "W4A8C8":
                text_cfg["has_kv_qscale"] = True
        sched_cfg = {
            "max_sequence_length": self.config.prompt_length + self.config.response_length,
            "max_context_len": self.config.prompt_length,
            "vllm_block_size": slot_block_size,
        }
        tp_size = self.config.get('tensor_model_parallel_size', 1)
        use_ep = self.config.get('use_ep', False)
        use_vocab_tp = self.config.get('vocab_tp', False)
        use_mtp = self.config.get('use_mtp', False)
        use_fp8_attention = self.config.get("use_fp8_attention", False)
        use_custom_allreduce = tp_size <= 8 and get_gpu_support_nvlink()
        multi_host_tp = is_multihost_model(tp_size)

        # TODO(caisonghua): enable XperfModelProphet vit part later
        xperf_prophet = XperfModelProphet(text_cfg, sched_cfg, tp_size)
        gpu_memory_utilization = self.config.get('gpu_memory_utilization', 0.7)
        if enable_paged_attn:
            prophet_cfg = xperf_prophet.profile_available_vllm_cfg(gpu_memory_utilization=gpu_memory_utilization)
            max_batch_size = prophet_cfg["orca_max_batch_size"]
            num_slots = prophet_cfg["vllm_num_slots"]
        else:
            prophet_cfg = xperf_prophet.profile_available_orca_cfg(gpu_memory_utilization=gpu_memory_utilization)
            max_batch_size = prophet_cfg["orca_max_batch_size"]
            num_slots = max_batch_size
        max_ctx_batch_size = self.config.get("max_ctx_batch_size", 8)

        # create a 2D device mesh
        world_size = torch.distributed.get_world_size()
        gen_dp_size = world_size // tp_size

        print(f'world_size: {world_size}, gen_dp_size: {gen_dp_size}, tp_size: {tp_size}')
        self.device_mesh = init_device_mesh(device_type='cuda',
                                            mesh_shape=(gen_dp_size, tp_size),
                                            mesh_dim_names=('dp', 'tp'))

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
            num_slots = max_batch_size
            if enable_paged_attn and self.config.num_slots:
                num_slots = self.config.num_slots

        inference_sess = InferenceSession(num_slots=num_slots,
                                          max_batch_size=max_batch_size,
                                          max_length=self.config.prompt_length + self.config.response_length,
                                          slot_block_size=slot_block_size,
                                          enable_paged_attn=enable_paged_attn,
                                          vocab_tp=use_vocab_tp,
                                          enable_truncation=False,
                                          context_limit_bs=max_ctx_batch_size,
                                          enable_cuda_graph=enable_cuda_graph,
                                          standalone=self.is_standalone,
                                          schedule_strategy=self.config.schedule_strategy,
                                          step_profiler=step_profiler,
                                          vit_use_xperf_gpt=self.config.vit_use_xperf_gpt,
                                          vit_use_dp=self.config.vit_use_dp,
                                          dp_vit_batching_step=self.config.dp_vit_batching_step,
                                          prefix_cache_impl=self.config.prefix_cache_impl,
                                          prefix_cache_slot_num=self.config.prefix_cache_slot_num,
                                          prefix_cache_max_length=self.config.prefix_cache_max_length,
                                          enable_mtp_decoding=use_mtp)
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
                        'XPERF_SESSION_SET_TORCH_DEVICE': '0',
                        'XPERF_CUSTOM_ALL_REDUCE': str(int(use_custom_allreduce)),
                        'XPERF_CUSTOM_ALL_REDUCE_BUFFER_SIZE': "536870912"
                    }):
                for start_rank in range(0, gpus_per_node, tp_size):
                    end_rank = start_rank + tp_size
                    if start_rank <= global_rank % local_world_size < end_rank:
                        print(
                            f'XPerf init Global rank {global_rank}, tp_rank {tp_rank}, master_addr: {master_addr}, master_port: {master_port}'
                        )
                        with logging_set_level(self.config.get('logging_level', 'INFO')):
                            inference_sess.init_inference_engine(
                                session_config_path=f.name,
                                generation_config=generate_kwargs,
                                # Model loading parameters
                                tokenizer_path=self.tokenizer.name_or_path,
                                # XPerf custom parameters
                                use_xperf_custom=self.config.xperf_custom.enable,
                                xperf_custom_backbone=self.config.xperf_custom.backbone
                                if self.config.xperf_custom.enable else None,
                                xperf_custom_preset=self.config.xperf_custom.preset
                                if self.config.xperf_custom.enable else None,
                                # XPerf triton parameters
                                use_xperf_triton=self.config.xperf_triton.enable,
                                xperf_triton_cfg=omegaconf.OmegaConf.to_container(self.config.xperf_triton,
                                                                                  resolve=True),
                                # Distributed/parallel settings
                                mp_size=tp_size,
                                use_ep=use_ep,
                                multi_host_tp=multi_host_tp,
                                rank0_split=False,
                                # Output and monitoring settings
                                enable_metrics=True,
                                # Vision model settings
                                vit_config=vision_cfg,
                                vit_model_cfg_path=self.vit_model_path,
                                use_fp8_attention=use_fp8_attention,
                            )
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
        self.dist_data_manager = get_dist_data_manager()
        torch.cuda.empty_cache()

    def add_inflight_query(self, query: Query) -> str:
        aq = AsyncQuery(query)
        # inference_engine is running on a different threads
        with self.inference_engine.update_weights_lock:
            self.inference_engine.pending.append(aq)
        return aq.id

    # abort some queries that no longer necessary to run on this engine
    def abort_queries(self, query_ids: List[str], not_after: float):
        # try to revoke from pending, running, paused and waiting list
        # 将要abort的放进去，后面等待engine自己内部的循环同步点abort
        with self.inference_engine.update_weights_lock:
            self.inference_engine.abort(query_ids, not_after)

    def get_valid_history_ids(self):
        """
        Get all history ids of the queries that have been processed by this rollout engine.
        This is used to check if the query has been processed before.
        """
        assert self._process_thread_last_error is None, \
            (f"process thread has error({self._process_thread_last_error}), please check the traceback in log.\n"
             f"{self._process_thread_last_tb}")
        return self.inference_engine.get_valid_history_ids()

    def update_queries(self, query_type: str, engine_id: str, wg_name: str):
        assert self._process_thread_last_error is None, \
            (f"process thread has error({self._process_thread_last_error}), please check the traceback in log.\n"
             f"{self._process_thread_last_tb}")
        queries = self.inference_engine.get_all_queries(query_type=query_type)
        if len(queries) > 0:
            if self._update_ref is not None:
                ray.wait([self._update_ref])  # 等上一轮的更新完
            if query_type not in self._request_managers:
                self._request_managers[query_type] = RequestManagerRegisterCenter.get(query_type)
            request_manager = self._request_managers[query_type]
            self._update_ref = request_manager.update_intermediate_queries.remote(queries, engine_id, wg_name,
                                                                                  time.time())

    def get_load_metrics(self) -> LoadMetric:
        return self.inference_engine.get_load_metrics()

    def get_master_addr_port(self):
        return self.master_addr, self.master_port

    def _set_tuner_config(self):
        os.environ["USE_SESSION_CACHE"] = "0"
        # set environment variables for tuner
        os.environ["XGPT_TUNER_ENABLE"] = os.getenv("XGPT_TUNER_ENABLE", "1")
        os.environ["XPERF_TUNER_ONLINE_PRIORITY"] = os.getenv("XPERF_TUNER_ONLINE_PRIORITY", "1")
        quant_mode = self.config.get("quant_mode", "NO_QUANT")
        if quant_mode in ["W4A8", "W4A8C8"]:
            os.environ["XPERF_TUNER_ONLINE_VERSION"] = "2.1.4+xgpt"
            base_dir = os.path.normpath(os.path.dirname(os.path.dirname(__file__)))
            tuning_path = os.path.join(base_dir, "xperf_rollout", "tuning", "w4a8")
            os.environ["XPERF_TUNER_CONFIG_LOAD_PATH"] = tuning_path
        else:
            # TODO: config does not take effect
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
        if self.is_async_generate:
            self.stop_event.set()
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        # rollout_server 等于 standalone rollout + elastic模式
        self.process_thread = threading.Thread(target=self.async_generate if self.is_async_generate else self.generate,
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
            torch.save(self.inference_engine.engine.module.weights, f"{save_model_name}_model_engine_weights.pt")
            torch.save(self.inference_engine.get_inorder_responses(), f"{save_model_name}_output.pt")
            print(f"dump weights/tensors to {dump_nan_dir}")
            hmkdir(self.config.get("dump_nan", None))
            hcopy(f"{save_model_name}_model_engine_weights.pt", self.config.get("dump_nan", None))
            hcopy(f"{save_model_name}_output.pt", self.config.get("dump_nan", None))

    def _process_log_probs(self, prompts, data_pack):
        if self.is_async_generate:
            self.switch_mode(True)
        original_prompt_ids = prompts.batch['prompts']  # (bs, prompt_length)
        first_non_one_indices = (original_prompt_ids != self.tokenizer.pad_token_id).int().argmax(dim=1)
        rmv_padding_original_prompt_ids = [
            row[index:].tolist() for row, index in zip(original_prompt_ids, first_non_one_indices)
        ]
        max_new_tokens = prompts.meta_info.get('generation_kwargs').get('max_new_tokens', self.config.response_length)
        log_prob_lists = data_pack.response_log_probs
        padded_rollout_policy_log_probs = torch.full((len(log_prob_lists), max_new_tokens), fill_value=-100.0)
        for i, log_probs in enumerate(log_prob_lists):
            response_log_probs = log_probs[len(rmv_padding_original_prompt_ids[i]):]
            assert (len(response_log_probs) <= max_new_tokens)
            padded_rollout_policy_log_probs[i, :len(response_log_probs)] = torch.tensor(response_log_probs)
        prompts.batch["rollout_policy_log_probs"] = padded_rollout_policy_log_probs.bfloat16()
        return prompts

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, is_async=False, mode="rollout"):
        if mode == "log_probs" and self.is_async_generate:
            self.switch_mode(False)

        complete_ratio = prompts.meta_info.get('complete_ratio', 1)
        prompt_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        batch_size = prompt_ids.shape[0]
        # left-padded attention_mask
        off_turn_off_policy_steps = prompts.batch["off_policy_steps"]
        first_non_one_indices = (prompt_ids != self.tokenizer.pad_token_id).int().argmax(dim=1)
        rmv_padding_prompt_ids = [row[index:].tolist() for row, index in zip(prompt_ids, first_non_one_indices)]
        generation_kwargs = prompts.meta_info['generation_kwargs']

        prompt_meta_info = [{
            "off_policy_steps": max(off_policy_step) + 1,
            "generation_kwargs": generation_kwargs,
            "mode": mode,
        } for off_policy_step in off_turn_off_policy_steps.tolist()]

        # GRM
        grm_keys = ['grm_pre_ids', 'grm_post_ids']
        grm_data = {key: prompts.batch.get(key, None) for key in grm_keys}
        for i in range(batch_size):
            current_meta = prompt_meta_info[i]  # 获取当前元数据引用
            for key in grm_keys:
                data = grm_data[key]
                if data is not None:  # 当且仅当数据存在时赋值
                    current_meta[key] = data[i]

        batch_size = len(prompts)
        if 'image_data_ref' in prompts.non_tensor_batch:
            image_data = get_local_inputs(prompts.non_tensor_batch, 'image_data_ref', self.dist_data_manager)
        for key, value in prompts.non_tensor_batch.items():
            for i in range(batch_size):
                prompt_meta_info[i][key] = value[i]
                if key == 'image_data_ref':
                    prompt_meta_info[i]['image_data'] = image_data[i]
        for i in range(batch_size):
            prompt_meta_info[i]['validate'] = prompts.meta_info.get('validate', False)
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
        if mode == "log_probs":
            prompts = self._process_log_probs(prompts, data_pack)
            yield prompts
            return

        out = pack_to_dataproto(prompts=prompts, data_pack=data_pack, config=self.config, tokenizer=self.tokenizer)
        yield out

    def async_generate(self):
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
        while (not self.exit_event.is_set()):
            if not self.weights_loaded.wait(timeout=1):
                time.sleep(0.05)
                continue
            if self.stop_event.is_set():
                time.sleep(0.05)
                continue
            with logging_set_level(self.config.get('logging_level', 'WARN')), self.profiler_context as p:
                try:
                    self.reset_status()
                    self.gen_loop_exited.clear()
                    # server mode: rollout only
                    self.inference_engine.switch_inference_mode("rollout")
                    self.inference_engine.async_execute(self.stop_event)
                    profile_step(p, None)
                    self.gen_loop_exited.set()
                except Exception as e:
                    self._dump_context()
                    self._process_thread_last_error = e
                    self._process_thread_last_tb = traceback.format_exc()
                    raise (e)

    def heartbeat(self):
        if self._process_thread_last_tb:
            raise RuntimeError("async_generate crashed") from self._process_thread_last_error
        if not self.process_thread.is_alive():
            raise RuntimeError("async_generate thread is not alive")

    def generate(self):
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
        while (not self.exit_event.is_set()):
            try:
                item = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue
            (query_pool, complete_ratio, generation_kwargs, prompt_meta_info) = item
            original_query_pool = copy.deepcopy(query_pool)
            self.inference_engine.set_generator_strategy(**generation_kwargs)
            with logging_set_level(self.config.get('logging_level', 'WARN')), self.profiler_context as p:
                try:
                    self.reset_status()
                    self.gen_loop_exited.clear()
                    self.inference_engine.switch_inference_mode(prompt_meta_info[0]['mode'])
                    self.inference_engine.execute(query_pool,
                                                  complete_ratio=complete_ratio,
                                                  stop_event=self.stop_event if self.is_standalone else None,
                                                  prompt_meta_info=prompt_meta_info)
                    profile_step(p, None)
                    self.gen_loop_exited.set()
                except Exception as e:
                    self._dump_context()
                    self._process_thread_last_error = e
                    self._process_thread_last_tb = traceback.format_exc()
                    raise (e)

            response_outputs = []
            response_log_probs = []
            is_finished = []
            off_policy_steps = []
            model_output_masks = []
            query_metrics = []
            extra_data = []
            for prompt, v in zip(original_query_pool, self.inference_engine.get_inorder_responses()):
                response_output_ids = (v.input_ids + v.new_token_ids)[len(prompt):]
                response_outputs.append(response_output_ids)
                response_log_probs.append(v.log_probs)
                is_finished.append(v.is_finished)
                off_policy_steps.append([-1] * len(v.log_probs))
                model_output_masks.append(v.model_output_mask)
                query_metrics.append(v.metrics)
                extra_data.append(v.extra_data)
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
            visualize_metrics(metrics)

            if self.config.recommend_standalone_usage.enable:
                dec_bs = metrics['dec_bs']
                max_bs = max(dec_bs)
                max_bs_idx = len(dec_bs) - dec_bs[::-1].index(max_bs)
                cumulative_latency = np.cumsum([0] + metrics["per_token_latency"])
                # compute max kv_util for complete ratio range for 0.00, 0.01, 0.02, ..., 0.99
                metrics['max_kv_util_for_complete_ratio'] = {}
                metrics['hybrid_latency_with_complete_ratio'] = {}
                for i in range(100):
                    complete_ratio = i / 100
                    bs = max_bs * (1 - complete_ratio)
                    bs_idx = next((idx + max_bs_idx for idx, val in enumerate(dec_bs[max_bs_idx:]) if val <= bs), None)
                    if bs_idx is not None:
                        metrics['max_kv_util_for_complete_ratio'][complete_ratio] = max(
                            metrics['kv_cache_utils'][bs_idx:])
                        metrics['hybrid_latency_with_complete_ratio'][
                            complete_ratio] = cumulative_latency[bs_idx] / 1000

            self.inference_engine.empty_cache()
            data_pack = DataPack(response_outputs=response_outputs,
                                 response_log_probs=response_log_probs,
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
                assert self._process_thread_last_error is None, \
                    (f"process thread has error({self._process_thread_last_error}), please check the traceback in log.\n"
                     f"{self._process_thread_last_tb}")


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
            timeout = timedelta(seconds=int(os.getenv('NCCL_TIMEOUT', 3600)))
            torch.distributed.init_process_group(backend="nccl", timeout=timeout)
        self.config = config
        self.role = role
        self.rollout_actor = AsyncXPerfGPTRollout(config=self.config.rollout, role=role)
        self._weights_loaded = threading.Event()
        self._hybrid_rollout_info = None
        self._stable_standalone_rollout_addrs = None  # stable的实例也会作为server，给elastic rollout提供参数
        self.weights_communicator: WeightsCommunicator = None

    def _stop_engine(self):
        if self.rollout_actor.stop_event.is_set():
            return

        with self.rollout_actor.inference_engine.update_weights_lock:
            self.rollout_actor.stop_event.set()

        # wait until completely stopped
        self.rollout_actor.gen_loop_exited.wait()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def initialized(self):
        return self.weights_communicator is not None and self.weights_communicator.has_setup

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def ready(self):
        # worker是否ready可以接受请求(model compute相关)
        # 子类继承这个方法自定义就绪判断，例如需要额外初始化model的
        return self._weights_loaded.is_set()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def will_be_destroyed(self):
        self._stop_engine()
        self.rollout_actor.reset_status()
        self.rollout_actor.inference_engine.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def add_inflight_queries(self, queries: List[Query]):
        ret = []
        queries = add_pixel_values_to_inflight_query(queries, self.rollout_actor.dist_data_manager)
        for q in queries:
            qid = self.rollout_actor.add_inflight_query(q)
            ret.append(qid)
        return ret

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def add_inflight_queries_non_blocking(self, queries: List[Query]):
        return self.add_inflight_queries(queries)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def abort_queries(self, query_ids: List[str], not_after: float):
        # 只abort那些在abort_before之前分到engine的
        self.rollout_actor.abort_queries(query_ids, not_after)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def abort_queries_non_blocking(self, query_ids: List[str], not_after: float):
        return self.abort_queries(query_ids, not_after)

    # 只在dp_size=1的情况下调用，所以这里rank0执行即可
    @register(execute_mode=Execute.RANK_ZERO, blocking=True)
    def get_history_ids(self):
        return self.rollout_actor.get_valid_history_ids()

    @register(execute_mode=Execute.RANK_ZERO, blocking=False)
    def get_history_ids_async(self):
        return self.rollout_actor.get_valid_history_ids()

    # 只在dp_size=1的情况下调用，所以这里rank0执行即可
    @register(execute_mode=Execute.RANK_ZERO, blocking=True)
    def update_queries(self, query_type: str, engine_id: str, wg_name: str):
        self.rollout_actor.update_queries(query_type, engine_id, wg_name)

    @register(execute_mode=Execute.RANK_ZERO, blocking=True)
    def get_load_metrics(self) -> LoadMetric:
        return self.rollout_actor.get_load_metrics()

    @register(execute_mode=Execute.RANK_ZERO, blocking=False)
    def get_load_metrics_async(self) -> LoadMetric:
        return self.rollout_actor.get_load_metrics()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self, *args, **kwargs):
        self.rollout_actor.initialize(self.config.model.path, True)
        self.rollout_actor.setup_rollout()
        weights_communicator = self.config.rollout.weights_communicator
        enable_aiomonitor = self.config.rollout.weights_communicator_enable_aiomonitor
        CommunicatorCls = UCXWeightsCommunicator if weights_communicator == "ucx" else NCCLWeightsCommunicator
        self.weights_communicator = CommunicatorCls(inference_engine=self.rollout_actor.inference_engine,
                                                    standalone=self.rollout_actor.is_standalone,
                                                    device_mesh=self.rollout_actor.device_mesh,
                                                    enable_aiomonitor=enable_aiomonitor)

        # save nccl master addr and port
        self.master_address = os.getenv('MASTER_ADDR', 'localhost')
        self.master_port = os.getenv('MASTER_PORT', '12345')
        print(f'Master address: {self.master_address}, Master port: {self.master_port}')

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def setup_as_client(self, role, source_addresses_by_tp: List[List[WeightsRankInfo]],
                        hybrid_rollout_info: List[WeightsRankInfo]):
        self._hybrid_rollout_info = hybrid_rollout_info
        # connect to weight source after model initialized
        tp_size = self.config.rollout.tensor_model_parallel_size
        # 按tp取整个dp group切片，% tp_size为了避免self是一个整个world包含多个dp(elastic下只会有1个dp，非elastic有多个dp)
        tp_rank = self.rank % tp_size
        source_info_this_tp = [dp[tp_rank] for dp in source_addresses_by_tp]
        self.weights_communicator.setup_as_client(role, source_info_this_tp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def setup_as_relay(self, ifname=None):
        ucx_address, oob_address = self.weights_communicator.setup_as_server(ifname)
        return WeightsRankInfo(
            rank=0,  # 对于standalone rollout来说，因为每个dp都是独立的，所以这里rank和dp_rank暂时为0算
            dp_rank=0,
            tp_rank=self.rank,  # 只有一个dp，所以就是tp rank
            ip=self._get_node_ip(),
            worker_name=self._get_actor_name(),
            ucx_address=ucx_address,
            oob_address=oob_address,
            pid=os.getpid(),
        )

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
        self.weights_communicator.on_will_start_update()
        self._stop_engine()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def stop_server_before_weights_update(self):
        self.weights_communicator.on_will_start_update()
        self._stop_engine()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def update_standalone_worker(self, role):
        offload_to_device(self.rollout_actor.inference_engine.engine.module, "cuda")
        try:
            with self.rollout_actor.inference_engine.update_weights_lock:
                self.weights_communicator.update_standalone_worker(role)
            self._weights_loaded.set()
            self.rollout_actor.weights_loaded.set()
        except WeightsUpdatingInterrupt as e:
            logger.debug(f"weights update interrupt {role=}")

    # group 内任意一个rank发送结束信号即可
    @register(execute_mode=Execute.RANK_ZERO)
    def update_standalone_worker_end(self):
        self.weights_communicator.update_standalone_worker_end(self._hybrid_rollout_info)

    # caller 自己去wait这个non-blocking
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def restart_server_after_weights_update_non_blocking(self):
        return self.restart_server_after_weights_update()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def restart_server_after_weights_update(self):
        # 对于elastic rollout，update_standalone_worker并不阻塞，所以在restart时务必等update完了
        # TODO(lixiang): 对于elastic的，可以放到后台线程去跑，提早返回
        self.weights_communicator.update_standalone_worker_wait()
        with self.rollout_actor.inference_engine.update_weights_lock:
            self.rollout_actor.reset_status()
            self.rollout_actor.stop_event.clear()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def do_ndtimeline_action(self, action, *args, **kwargs):
        # 需要实现的一个接口方法，但现在没什么要做的事情，所以先返回空
        return

    @register(execute_mode=Execute.RANK_ZERO, blocking=True)
    def return_metrics(self):
        metrics = self.rollout_actor.inference_engine.infer_scheduler.metrics
        visualize_metrics(metrics)
        return metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def empty_engine_cache(self):
        self.rollout_actor.inference_engine.empty_cache(only_clear_metrics=True)


# for type annotation convenience
def _unwrap_ray_remote(cls) -> Type[RemoteAsyncXPerfGPTRollout]:
    if hasattr(cls, '__ray_actor_class__'):
        cls = cls.__ray_actor_class__
    return cls


@ray.remote
class ElasticAsyncXPerfGPTRollout(_unwrap_ray_remote(RemoteAsyncXPerfGPTRollout)):

    def __init__(self, config: DictConfig, role: str, hybrid_rollout_info: List[WeightsRankInfo]):
        super().__init__(config, role)
        self.hybrid_rollout_info = hybrid_rollout_info
        self._elastic_has_setup = threading.Event()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_and_setup(
            self,
            weights_source_by_dp_group: List[List[WeightsRankInfo]],  # [[tp0,...], [tp0,...], ...]
            setup_relay: bool,
            intermediately_update_weights: bool = False):
        """
        放在这里统一setup，返回一个ObjectRef，让调用方一次性等待整个初始化完成
        :param weights_source_by_dp_group: 要连上的拉取weight的server address, 按 [dp0[tp0,...], dp1[tp0,...], ...] 分组
        :param setup_relay: 是否要设置为relay提供别的worker拉参数
        :param intermediately_update_weights: 初始化完之后是否立即拉一次参数，适用于elastic的场景
        :return: relay address, worker group返回的则是整个tp group的address，如果setup_relay=False，则返回空字符串
        """
        self.init_model()
        self.setup_as_client(self.role, weights_source_by_dp_group, self.hybrid_rollout_info)

        relay_info: Optional[WeightsRankInfo] = None
        if setup_relay:
            relay_info = self.setup_as_relay()

        if intermediately_update_weights:
            self.update_standalone_worker(self.role)
            self.restart_server_after_weights_update()

        self._elastic_has_setup.set()
        return relay_info

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def initialized(self):
        return self._elastic_has_setup.is_set()
