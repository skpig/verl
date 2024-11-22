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

import torch.distributed
from torch.distributed.device_mesh import init_device_mesh


def create_device_mesh(fsdp_size, role):
    world_size = torch.distributed.get_world_size()

    if fsdp_size > 0 and (world_size // fsdp_size) > 1:
        # if dp_size > 1, use HSDP
        assert world_size % fsdp_size == 0, "world_size must be divisible by fsdp_size"
        dp_size = world_size // fsdp_size
        device_mesh = init_device_mesh('cuda', mesh_shape=(dp_size, fsdp_size), mesh_dim_names=['dp', 'fsdp'])
        print(f"Using HSDP {dp_size} {device_mesh} for {role}")
    else:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
        print(f"using FSDP {world_size} {device_mesh} for {role}")
    return device_mesh