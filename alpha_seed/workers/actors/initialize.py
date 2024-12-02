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
Meta device initialization with FSDP

Example usage:

```python
init_context = get_device_init_context(use_meta_tensor=True)

with meta_device_init():
    model = ... # create model

shard_states = parallel_load_safetensors(ckpt_path)
model = FSDP(
    model,
    param_init_fn=parallel_init_fsdp_fn(model, shard_states),
    ...
)
```

This will initialize model's parameters on meta device.
In FSDP, the parameters will be loaded from shard states and
broadcasted to each rank.

This module speeds up the loading process and
addresses the issues of shared parameters in HF accelerate.
"""
from typing import Callable, Dict
from contextlib import contextmanager
import torch
import torch.distributed as dist
import torch.nn as nn
import os
import json
import math
from safetensors.torch import load_file
import hdfs_io
import itertools
from verl.utils.fs import copy_local_path_from_hdfs


def create_init_fn(module: torch.nn.Module) -> Callable:
    """
    Get tensor materialization function with the support of shared parameters and buffers

    Args:
        module: the whole model that may include shared parameters / buffers

    Returns:
        Callable: module -> module initialization method to materialize meta-device tensors on GPU
    """
    states = {}
    for _, param in module.named_parameters(remove_duplicate=False):
        states.setdefault(param, []).append(param)
    # remove standalone parameters and buffers
    states = {s: ts for s, ts in states.items() if len(ts) > 1}
    materialized_states = {}

    def init_fn(sub_mod: torch.nn.Module):
        # Note(zhiqi.0) we can't use nn.Module._apply or nn.Module.to_empty() because it will create new parameters,
        # breaking the connection of shared parameters;
        # We also can't use torch.utils.swap_tensors becuase tensors are on different devices;
        # Instead, we can only reset sub_mod._parameters to make it work
        device = torch.cuda.current_device()
        for name, state in sub_mod.named_parameters(recurse=False):
            if state in states:
                if state not in materialized_states:
                    materialized_states[state] = torch.nn.Parameter(torch.empty_like(state.data, device=device),
                                                                    requires_grad=state.requires_grad)
                materialize_state = materialized_states[state]
            else:
                materialize_state = torch.nn.Parameter(torch.empty_like(state.data, device=device),
                                                       requires_grad=state.requires_grad)
            sub_mod._parameters[name] = materialize_state
        for name, buffer in sub_mod.named_buffers(recurse=False):
            assert not buffer.is_meta, f"find buffer {name} to be meta"
        return sub_mod

    return init_fn


@contextmanager
def meta_device_init():
    """
    Create model parameters with meta device.

    Note buffers in model will still be initialized in default device (e.g., CPU),
    since the buffers can be non-persistent and filled with expected values that can
    NOT be captured in meta device.
    """
    device = torch.device("meta")
    old_register_parameter = nn.Module.register_parameter
    registered = set()

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        # we will skip register shared parameters as it
        # is already registered previously
        if param is not None and param not in registered:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            registered.add(module._parameters[name])

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        registered.clear()
        nn.Module.register_parameter = old_register_parameter


def get_device_init_context(use_meta_tensor=True):
    cpu_init_weights = lambda: torch.device('cpu')
    if use_meta_tensor:
        if dist.get_rank() == 0:
            init_context = cpu_init_weights
        else:
            init_context = meta_device_init
    else:
        init_context = cpu_init_weights
    return init_context


def parallel_load_safetensors(filepath):

    # copy from hdfs into local filepath
    if filepath.startswith("hdfs://"):
        filepath = copy_local_path_from_hdfs(filepath)

    index_file = os.path.join(filepath, "model.safetensors.index.json")
    index = json.load(open(index_file, "rb"))
    safetensors2param = {}
    for param_name, filename in index["weight_map"].items():
        safetensors2param.setdefault(filename, []).append(param_name)

    total_files = len(safetensors2param)
    ckpt_chunks = sorted(safetensors2param.keys())
    world_size = dist.get_world_size()
    size = int(math.ceil(total_files / world_size))
    ckpt_chunks = list(map(lambda x: ckpt_chunks[x * size:x * size + size], list(range(world_size))))

    shard_states = {}
    device = torch.cuda.current_device()
    for rank, files in enumerate(ckpt_chunks):
        if rank == dist.get_rank():
            for file in files:
                file = os.path.join(filepath, file)
                states = load_file(file, device=device)
                # print(f"rank {rank} loading {file}...")
                shard_states.update(states)
        else:
            for file in files:
                for param_name in safetensors2param[file]:
                    shard_states[param_name] = rank
    return shard_states


def parallel_init_fsdp_fn(module: torch.nn.Module, shard_states: Dict[str, torch.nn.Parameter]):

    state2fqn = {}
    for name, state in itertools.chain(module.named_parameters(remove_duplicate=False),
                                       module.named_buffers(remove_duplicate=False)):
        state2fqn.setdefault(state, []).append(name)
    # remove standalone parameters and buffers
    shared = set(s for s, names in state2fqn.items() if len(names) > 1)
    materialized_states = {}

    @torch.no_grad()
    def create_and_sync_state(param_name, state, is_param):
        assert param_name in shard_states, f"{param_name} not loaded"
        device = torch.cuda.current_device()
        if is_param:
            param = torch.nn.Parameter(torch.empty_like(state.data, device=device), requires_grad=state.requires_grad)
        else:  # buffer
            param = torch.empty_like(state.data, device=device)
        loaded = shard_states[param_name]
        if isinstance(loaded, (torch.nn.Parameter, torch.Tensor)):
            param.data.copy_(loaded.data)
            dist.broadcast(param.data, src=dist.get_rank())
        else:
            assert isinstance(loaded, int)  # the rank that holds the state
            dist.broadcast(param.data, src=loaded)
        shard_states.pop(param_name)
        del loaded
        return param

    def init_fn(sub_mod: torch.nn.Module):
        param_and_buffers = tuple(sub_mod.named_parameters(recurse=False)) + tuple(sub_mod.named_buffers(recurse=False))
        # param_and_buffers = sorted(sub_mod.named_parameters(recurse=False), key=lambda x: x[0])
        for name, state in param_and_buffers:
            is_param = name in sub_mod._parameters
            fqn = state2fqn[state].pop(0)
            # non-persistent buffers will not be saved in state dict, we can safely skip it
            if (not is_param) and fqn not in shard_states:
                if state.is_meta:
                    raise RuntimeError(
                        f"find a non-persistent buffer ({fqn}) initiated with device meta. "
                        "Such buffer is not saved in checkpoint and user should guarantee to init in CPU / GPU device.")
                continue
            if state in shared:
                if state not in materialized_states:
                    materialized_states[state] = create_and_sync_state(fqn, state, is_param)
                else:
                    if fqn in shard_states:
                        shard_states.pop(fqn)
                materialize_state = materialized_states[state]
            else:
                materialize_state = create_and_sync_state(fqn, state, is_param)
            if is_param:
                sub_mod._parameters[name] = materialize_state
            else:
                sub_mod._buffers[name] = materialize_state
        # for debug
        # if len(shard_states) == 0: print("clear")
        return sub_mod

    return init_fn
