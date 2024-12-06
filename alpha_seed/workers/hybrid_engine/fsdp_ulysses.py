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
from typing import Optional
from .base import BaseShardingManager

import random
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.torch_functional import allgather_dict_tensors
from dist_attn.ulysses.parallel_states import set_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_group
from dist_attn.ulysses.ops import slice_input_tensor
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
            # change to use model-specific sp group
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self.device_mesh['sp'].get_group())
            # setup seed
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
            # revert to previous sp group
            set_ulysses_sequence_parallel_group(self.prev_sp_group)
            # revert to previous seed
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

            prev_device = data.batch.device
            data.batch = data.batch.cuda(device=torch.cuda.current_device())
            data.batch = allgather_dict_tensors(data.batch.contiguous(), size=sp_size, group=group, dim=0)
            data.batch = data.batch.to(prev_device)
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


def ulysses_pad_and_slice_inputs(input_ids_rmpad: torch.Tensor, position_ids_rmpad: Optional[torch.Tensor],
                                 sp_size: int):
    """
    Pad and slice input_ids to be divisible by sp_size
    Pad position_ids to be divisible by sp_size.

    Note both input_ids_rmpad and position_ids_rmpad will be padded,
    but only input_ids will be sliced.

    The is the utility of pre-forward for ulysses sequence parallelism

    Args:
        input_ids_rmpad: shape of [bsz, seqlen]
        position_ids_rmpad: shape of [bsz, seqlen], where bsz must be 1
        sp_size (int): ulysses sequence parallelism size

    Returns:
        torch.Tensor: padded and sliced input_ids
        torch.Tensor: padded and sliced position_ids
        int: pad size 
    """
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.size(0) == 1
        assert input_ids_rmpad.size(1) == position_ids_rmpad.size(1)
    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0
    _, total_s = input_ids_rmpad.shape
    pad_size = (sp_size - total_s % sp_size) % sp_size
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=0)
        if position_ids_rmpad is not None:
            pad_pos_ids = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
    # we don't need to slice position ids
    # if position_ids_rmpad is not None:
    #     position_ids_rmpad = slice_input_tensor(position_ids_rmpad, dim=1, padding=False)
    return input_ids_rmpad, position_ids_rmpad, pad_size
