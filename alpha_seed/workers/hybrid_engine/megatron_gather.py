"""
Data gather for Mariana 3D parallelism
"""

from typing import Optional
from .base import BaseShardingManager

import random
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.torch_functional import allgather_dict_tensors
import numpy as np

import torch
import torch.distributed

from verl import DataProto

from megatron.core import parallel_state as mpu


class MegatronDataGatherManager(BaseShardingManager):

    def __init__(
        self,
    ):
        super().__init__()
        # TODO(zhangchi.usc1992): support per-model 3D device mesh online switch.
        # We may also need to patch mariana global variables such as get_args and environment

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        AllGather data from tp/pp region
        """
        group = mpu.get_model_parallel_group()
        group_size = mpu.get_tensor_model_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size()

        if group_size > 1:
            prev_device = data.batch.device
            data.batch = data.batch.cuda(device=torch.cuda.current_device())
            data.batch = allgather_dict_tensors(data.batch.contiguous(), size=group_size, group=group, dim=0)
            data.batch = data.batch.to(prev_device)
            # all gather non_tensor_batch
            all_non_tensor_batch = [None for _ in range(group_size)]
            torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=group)
            data.non_tensor_batch = {
                k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch
            }
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        group = mpu.get_model_parallel_group()
        group_size = mpu.get_tensor_model_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size()
        local_rank = torch.distributed.get_rank(group=group)

        if group_size > 1:
            data = data.chunk(chunks=group_size)[local_rank]
        return data
