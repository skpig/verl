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
from typing import Any
import numpy as np

import torch
import torch.distributed

from verl import DataProto


class DataGatherManager(BaseShardingManager):

    def __init__(self, gather_mesh: DeviceMesh, sp_mesh: DeviceMesh):
        super().__init__()
        dp_gather_mesh = gather_mesh._parent_mesh
        assert dp_gather_mesh.ndim == 2
        self.dp_mesh = dp_gather_mesh['dp']
        self.gather_mesh = gather_mesh
        self.sp_mesh = sp_mesh
        self.seed_offset = 12345

    def __enter__(self):
        if self.sp_mesh.size() > 1:
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self.sp_mesh.get_group())
        if self.dp_mesh.size() > 1:
            # set seed
            dp_rank = self.dp_mesh.get_local_rank()
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
        if self.sp_mesh.size() > 1:
            # revert to previous sp group
            set_ulysses_sequence_parallel_group(self.prev_sp_group)
        if self.dp_mesh.size() > 1:
            # revert to previous seed
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.manual_seed(self.prev_torch_seed)
            torch.cuda.set_rng_state(self.prev_torch_cuda_state)
            random.setstate(self.prev_random_state)
            np.random.set_state(self.prev_np_state)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        AllGather data from gather_mesh region
        """
        if self.gather_mesh.size() > 1:
            gather_size = self.gather_mesh.size()
            group = self.gather_mesh.get_group()

            prev_device = data.batch.device
            data.batch = data.batch.cuda(device=torch.cuda.current_device())
            data.batch = allgather_dict_tensors(data.batch.contiguous(), size=gather_size, group=group, dim=0)
            data.batch = data.batch.to(prev_device)
            # all gather non_tensor_batch
            all_non_tensor_batch = [None for _ in range(gather_size)]
            torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=group)
            data.non_tensor_batch = {
                k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch
            }
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        if self.gather_mesh.size() > 1:
            gather_size = self.gather_mesh.size()
            gather_rank = self.gather_mesh.get_local_rank()
            data = data.chunk(chunks=gather_size)[gather_rank]
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


def gather_outpus_and_unpad(x: torch.Tensor,
                            gather_dim: int,
                            unpad_dim: int = None,
                            padding_size: int = 0,
                            grad_scaler: bool = True,
                            sp_size: int = 1):
    group = get_ulysses_sequence_parallel_group()
    if group == None:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    if unpad_dim is not None:
        assert isinstance(padding_size, int), 'padding size is not given or is not an integer'
        if padding_size == 0:
            return x
        x = _unpad_tensor(x, unpad_dim, padding_size)
    return x


class Gather(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any,
                group: torch.distributed.ProcessGroup,
                local_tensor: torch.Tensor,
                gather_dim: int,
                grad_scaler: bool = True,
                async_op=False) -> torch.Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = torch.distributed.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = torch.distributed.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (None, grad_output.split(ctx.part_size,
                                        dim=ctx.gather_dim)[ctx.sp_rank].contiguous(), None, None, None, None)


def all_gather_tensor(local_tensor: torch.Tensor,
                      group: Optional[torch.distributed.ProcessGroup] = None,
                      async_op: bool = False):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = torch.distributed.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    torch.distributed.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


def _unpad_tensor(x: torch.Tensor, dim: int, padding_size: int) -> torch.Tensor:
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]
