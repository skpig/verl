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

from seed_models import P5ForCausalLM, P5Config


class FSDPXPerfGPTShardingManager(BaseShardingManager):

    def __init__(self, module: FSDP, model_config: P5Config, inference_engine: InferenceSession,
                 device_mesh: DeviceMesh):
        super().__init__()
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.model_config = model_config

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

    def __enter__(self):
        # gather full state_dict in CPU
        from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

        # TODO: optimize this. Since state_dict is a copy, there are actually two copies in the GPU memory
        # We need to switch to FSDP2 to handle this.
        cfg = ShardedStateDictConfig(offload_to_cpu=False)
        with FSDP.state_dict_type(self.module, StateDictType.SHARDED_STATE_DICT, cfg):
            state_dict = self.module.state_dict()

        # prepare the state_dict into a format for xperf_gpt
        self.bind_fn(self.inference_engine.engine.module, state_dict=state_dict, device_mesh=self.device_mesh)
        if hasattr(self.inference_engine.pp_scheduler, "init_cuda_graph"):
            self.inference_engine.pp_scheduler.init_cuda_graph()
        # important: need to manually set the random states of each tp to be identical. Otherwise, xperf_gpt will hang
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        # offload to CPU
        offload_to_cpu(tp_model=self.inference_engine.engine.module)
        # set to train
        self.module.train()
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
