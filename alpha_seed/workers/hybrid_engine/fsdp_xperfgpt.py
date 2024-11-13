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

import os
from unittest.mock import patch
import warnings

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType, ShardedStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.torch_functional import broadcast_dict_tensor, allgather_dict_tensors

from xperf_gpt.inference.session import InferenceSession

import torch
import torch.distributed

from torch.distributed._tensor import DTensor

from verl import DataProto

from alpha_seed.workers.xperf_rollout.utils.weight_loader import offload_to_cpu, get_xperf_gpt_weight_bind_fn


class FSDPXPerfGPTShardingManager(BaseShardingManager):

    def __init__(self, module: FSDP, model_config, inference_engine: InferenceSession, device_mesh: DeviceMesh,
                 standalone):
        super().__init__()
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.model_config = model_config
        self.standalone = standalone
        self.world_size_offset = 0
        self.world_size = torch.distributed.get_world_size()

        self.bind_fn = get_xperf_gpt_weight_bind_fn(model_config)

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

    def setup_standalone_rollout_comm(self, hybrid_master_address, standalone_master_address):
        assert (hybrid_master_address is not None)
        assert (standalone_master_address is not None)
        self.world_size_offset = len(hybrid_master_address)
        master_address = hybrid_master_address[0].meta_info["hybrid_master_addr"]
        # breakpoint()
        # hack for standalone
        self.rank = torch.distributed.get_rank() + (0 if not self.standalone else self.world_size_offset)
        self.hybrid_world_size = len(hybrid_master_address)
        self.standalone_world_size = len(standalone_master_address)
        self.world_size = self.hybrid_world_size + self.standalone_world_size

        print("world_size ", self.world_size, " rank ", self.rank, " master_addr ", master_address)
        with patch.dict(
                os.environ,
            {
                'RANK': str(self.rank),
                'WORLD_SIZE': str(self.world_size),
                'LOCAL_RANK': str(self.rank % 8),
                'LOCAL_WORLD_SIZE': str(min(8, self.world_size)),
                'MASTER_ADDR': master_address,
                'MASTER_PORT': "12333",  # find a free port
            }):
            self.nccl_layer = torch.classes.XGPT.NCCLPrimitive()
            self.nccl_layer.init("standalone", self.world_size, self.rank, "tcp", 0)

    def __enter__(self):
        # gather full state_dict in CPU
        from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

        # TODO: optimize this. Since state_dict is a copy, there are actually two copies in the GPU memory
        # We need to switch to FSDP2 to handle this.
        cfg = ShardedStateDictConfig(offload_to_cpu=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.module, StateDictType.SHARDED_STATE_DICT, cfg):
                state_dict = self.module.state_dict()

        # prepare the state_dict into a format for xperf_gpt
        self.bind_fn(self.inference_engine.engine.module, state_dict=state_dict, device_mesh=self.device_mesh)
        if hasattr(self.inference_engine.pp_scheduler, "init_cuda_graph"):
            self.inference_engine.pp_scheduler.init_cuda_graph()
        # important: need to manually set the random states of each tp to be identical. Otherwise, xperf_gpt will hang
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            # print("setting random states...", self.gen_random_states)
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        AllGather data from tp region
        """

        if self.device_mesh is not None:
            tp_size = self.device_mesh['tp'].size()
            group = self.device_mesh['tp'].get_group()

            data.batch = allgather_dict_tensors(data.batch.contiguous(), size=tp_size, group=group, dim=0)

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
        return data

    def update_standalone_rollout(self):

        def _update_xperf_model(comm_fn, comm_rank):
            layernorm_weight = self.inference_engine.engine.module.layernorm_weight.cuda()
            lm_head_weight = self.inference_engine.engine.module.lm_head_weight.cuda()
            wte_weight = self.inference_engine.engine.module.wte_weight.cuda()
            comm_fn(layernorm_weight, comm_rank)
            comm_fn(lm_head_weight, comm_rank)
            comm_fn(wte_weight, comm_rank)
            self.inference_engine.engine.module.layernorm_weight = layernorm_weight
            self.inference_engine.engine.module.lm_head_weight = lm_head_weight
            self.inference_engine.engine.module.wte_weight = wte_weight

            layers_weight = self.inference_engine.engine.module.layers_weight
            for layer, layer_weight in enumerate(layers_weight):
                for i, weight in enumerate(layer_weight):
                    if isinstance(weight, torch.Tensor):
                        weight = weight.cuda()
                        comm_fn(weight, comm_rank)
                        self.inference_engine.engine.module.layers_weight[layer][i] = weight
            self.inference_engine.current_steps = 0

        if self.standalone:
            from_rank = self.rank % self.hybrid_world_size
            _update_xperf_model(self.nccl_layer.recv, from_rank)
        else:
            for i in range((self.world_size - 1) // self.hybrid_world_size):
                to_rank = self.rank + self.hybrid_world_size + i * self.hybrid_world_size
                if to_rank < self.world_size:
                    _update_xperf_model(self.nccl_layer.send, to_rank)

        if not self.standalone:
            # offload to CPU
            offload_to_cpu(tp_model=self.inference_engine.engine.module)
            # set to train
            self.module.train()
        else:
            # restore random states
            if self.device_mesh is not None:
                torch.cuda.set_rng_state(self.gen_random_states)
