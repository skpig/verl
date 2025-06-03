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
Contains a resharding manager that binds weights from FSDP zero3 to XPerfGPT
"""

from .base import BaseShardingManager
import gc

import numpy as np
import os
from unittest.mock import patch
import warnings

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType, ShardedStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.torch_functional import broadcast_dict_tensor, allgather_dict_tensors
from verl.utils.debug import log_gpu_memory_usage

from alpha_seed.workers.xperf_rollout.session import InferenceSession
from alpha_seed.workers.xperf_rollout.utils.pooled_ucx_weights_communicator import UCXWeightsCommunicator
from alpha_seed.workers.xperf_rollout.utils.nccl_weights_communicator import NCCLWeightsCommunicator

import torch
import torch.distributed

from torch.distributed._tensor import DTensor

from verl import DataProto

from alpha_seed.workers.xperf_rollout.utils.layout_convert_helper import offload_to_device, load_to_cuda
from alpha_seed.workers.xperf_rollout.utils.weight_loader import get_xperf_gpt_weight_bind_fn
import logging

logger = logging.getLogger(__file__)


class ActorXPerfGPTShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 model_config,
                 inference_engine: InferenceSession,
                 device_mesh: DeviceMesh,
                 standalone=False,
                 only_bind_once=False,
                 backend='fsdp',
                 weights_communicator="nccl"):
        super().__init__()
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.model_config = model_config
        self.weights_communicator = weights_communicator

        # here standalone means standalone validator or standalone validator
        self.standalone = standalone
        self.bind_fn = get_xperf_gpt_weight_bind_fn(model_config,
                                                    self.inference_engine.engine.module.quant_mode,
                                                    is_custom_xperf=self.inference_engine.is_xperf_custom,
                                                    is_xperf_triton=self.inference_engine.is_xperf_triton,
                                                    backend=backend)

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None
        # broadcast random states across tp group

        # True for generation only scenarios, we don't need to update weights, only call bind_fn for once
        self.only_bind_once = only_bind_once
        self._bind_fn_called = False
        CommunicatorCls = UCXWeightsCommunicator if self.weights_communicator == "ucx" else NCCLWeightsCommunicator
        self.weights_communicator = CommunicatorCls(inference_engine=self.inference_engine,
                                                    standalone=self.standalone,
                                                    device_mesh=self.device_mesh)

    def release_param_and_cache(self):
        """Release the GPU memory occupied by xperf parameter and cache"""
        device = "meta" if not self.only_bind_once else "cpu"
        offload_to_device(tp_model=self.inference_engine.engine.module, device=device)
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After release_param_and_cache')

    def _get_actor_state_dict(self):
        raise NotImplementedError

    def __enter__(self):
        # standalone worker does not need to do this
        if self.standalone:
            offload_to_device(self.inference_engine.engine.module, "cuda")
            return
        # gather full state_dict in CPU
        if (not self.only_bind_once) or (not self._bind_fn_called):
            # materialize to cuda if tensors are on meta device
            state_dict = self._get_actor_state_dict()
            offload_to_device(self.inference_engine.engine.module, "cuda")
            # prepare the state_dict into a format for xperf_gpt
            if self.model_config.model_type == 'seed_vl':
                self.bind_fn(self.inference_engine.engine.module,
                             self.inference_engine.vit_engine,
                             state_dict=state_dict,
                             device_mesh=self.device_mesh)
            else:
                self.bind_fn(self.inference_engine.engine.module, state_dict=state_dict, device_mesh=self.device_mesh)

            self._bind_fn_called = True
        else:
            load_to_cuda(tp_model=self.inference_engine.engine.module)

        # important: need to manually set the random states of each tp to be identical. Otherwise, xperf_gpt will hang
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            # print("setting random states...", self.gen_random_states)
            torch.cuda.set_rng_state(self.gen_random_states)

        if self.inference_engine.is_xperf_triton:
            self.inference_engine.engine.module.enter()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.inference_engine.is_xperf_triton:
            self.inference_engine.engine.module.exit()
        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        # only support to release xperf weight and kv cache
        # right after generation when there is no standalone workers
        if (not self.standalone):
            device = "meta" if not self.only_bind_once else "cpu"
            offload_to_device(tp_model=self.inference_engine.engine.module, device=device)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        AllGather data from tp region
        """

        if self.device_mesh is not None:
            tp_size = self.device_mesh['tp'].size()
            group = self.device_mesh['tp'].get_group()

            prev_device = data.batch.device
            data.batch = data.batch.cuda(device=torch.cuda.current_device())
            data.batch = allgather_dict_tensors(data.batch.contiguous(), size=tp_size, group=group, dim=0)
            data.batch = data.batch.to(prev_device)

            # all gather non_tensor_batch
            all_non_tensor_batch = [None for _ in range(tp_size)]
            torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=group)
            data.non_tensor_batch = {
                k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch
            }

        data.check_consistency()
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        # tp_group = self.tp_device_mesh.get_group()
        # tp_src_rank = torch.distributed.get_global_rank(tp_group, group_rank=0)
        if self.device_mesh is not None:
            tp_size = self.device_mesh['tp'].size()
            assert tp_size > 1
            # broadcast_dict_tensor(data.batch,
            #                       src=tp_src_rank,
            #                       group=tp_group)
            dp_rank = torch.distributed.get_rank()
            # dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]

        data.check_consistency()
        return data


class FSDPXPerfGPTShardingManager(ActorXPerfGPTShardingManager):

    def _get_actor_state_dict(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # note FSDP module is already set state dict type at initialization
            state_dict = self.module.state_dict()
        return state_dict


def normalize_key(name, layer_name, layer_offset):
    """
    name is the param name. layer_name is typically "layer", 
    """
    if layer_name in name:  # belong to an intermediate layer
        split_name = name.split('.')
        # find the num next to split_name
        for i, name in enumerate(split_name):
            if name == layer_name:
                break
        layer_num_idx = i + 1
        # check the name
        assert len(split_name) >= layer_num_idx + 1, f'split_name = {split_name}'
        assert split_name[layer_num_idx].isdigit(), f'split_name = {split_name}'
        # increment layer_num_idx by layer_offset
        split_name[layer_num_idx] = str(int(split_name[layer_num_idx]) + layer_offset)
        name = '.'.join(split_name)  # weight name in inference_tp_model

    return name


class MegatronXPerfGPTShardingManager(ActorXPerfGPTShardingManager):

    def _get_actor_state_dict(self):
        # from verl.utils.model import normalize_pp_vpp_params
        from megatron.training import unwrap_model
        from megatron.model import DistributedDataParallel, Float16Module
        from megatron.core.distributed import DistributedDataParallel as MultiPrecisionDDP

        all_state_dict = {}

        has_non_cuda_param = False

        valid_start_str = ['transformer.ln_f', 'transformer.h', 'transformer.wte.weight']
        # module = self.module
        # convert the state dict
        for module in self.module:
            # normalize names
            state_dict = module.state_dict()
            # remove duplicate keys
            keys = list(state_dict.keys())
            for key in keys:
                is_valid = False
                for start_str in valid_start_str:
                    if key.startswith(start_str):
                        is_valid = True
                        break

                if not is_valid:
                    state_dict.pop(key)

            unwrapped_module = unwrap_model(module,
                                            module_instances=(DistributedDataParallel, Float16Module,
                                                              MultiPrecisionDDP))
            start_layer_idx = unwrapped_module.transformer.h.layers[0].layer_number - 1

            for key, param in state_dict.items():
                normalized_key = normalize_key(key, 'layers', start_layer_idx)
                assert normalized_key not in all_state_dict
                all_state_dict[normalized_key] = param

                if not param.is_cuda:
                    has_non_cuda_param = True
                    print(f'param {key} is not on cuda')

        assert not has_non_cuda_param

        return all_state_dict
