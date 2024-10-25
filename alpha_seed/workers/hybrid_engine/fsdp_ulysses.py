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

import random
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.torch_functional import allgather_dict_tensors
import numpy as np

import torch
import torch.distributed

from verl import DataProto


class FSDPUlyssesShardingManager(BaseShardingManager):

    def __init__(self, device_mesh: DeviceMesh):
        super().__init__()
        self.device_mesh = device_mesh
        self.seed_offset = 12345

    def __enter__(self):
        if self.device_mesh is not None:
            dp_rank = self.device_mesh['dp'].get_local_rank()
            self.prev_torch_seed = torch.seed()
            torch.manual_seed(dp_rank + self.seed_offset)
            self.prev_torch_cuda_state = torch.cuda.get_rng_state()
            torch.cuda.manual_seed(dp_rank + self.seed_offset)
            self.prev_random_state = random.getstate()
            random.seed(dp_rank + self.seed_offset)
            self.prev_np_state = np.random.get_state()
            np.random.seed(dp_rank + self.seed_offset)
            self.seed_offset += 1

    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.manual_seed(self.prev_torch_seed)
            torch.cuda.set_rng_state(self.prev_torch_cuda_state)
            random.setstate(self.prev_random_state)
            np.random.set_state(self.prev_np_state)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        AllGather data from sp region
        """
        if self.device_mesh is not None:
            sp_size = self.device_mesh['sp'].size()
            group = self.device_mesh['sp'].get_group()
            data.batch = allgather_dict_tensors(data.batch.contiguous(), size=sp_size, group=group, dim=0)
            # all gather non_tensor_batch
            all_non_tensor_batch = [None for _ in range(sp_size)]
            torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=group)
            data.non_tensor_batch = {
                k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch
            }
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        if self.device_mesh is not None:
            sp_size = self.device_mesh['sp'].size()
            sp_rank = self.device_mesh['sp'].get_local_rank()
            data = data.chunk(chunks=sp_size)[sp_rank]
        return data
