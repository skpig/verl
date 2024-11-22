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

with init_context():
    model = ... # create model and load checkpoint

model = FSDP(
    model,
    param_init_fn=create_init_fn(model),
    sync_module_states=True,
    ...
)
```

This will initialize model with CPU checkpoint **only** at rank 0, while
others are initialized with meta device. In FSDP, the parameters will
be broadcasted to each rank.

This module addresses the issues of shared parameters in HF accelerate.
"""
from typing import Callable
from contextlib import contextmanager
import functools
import torch
import torch.distributed as dist
import torch.nn as nn


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
        if dist.get_rank() == 0:
            return sub_mod

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
        return sub_mod

    return init_fn


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn

    with init_on_device(device=torch.device("meta")):
        tst = nn.Linear(100, 100)  # on `meta` device
    ```
    """

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    registered = set()

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            if param in registered:
                return
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            registered.add(module._parameters[name])

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            if buffer in registered:
                return
            module._buffers[name] = module._buffers[name].to(device)
            registered.add(module._buffers[name])

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):

        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        registered.clear()
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def get_device_init_context(use_meta_tensor=True):
    cpu_init_weights = lambda: torch.device('cpu')
    if use_meta_tensor:
        if dist.get_rank() == 0:
            init_context = cpu_init_weights
        else:
            init_context = functools.partial(init_on_device, torch.device("meta"))
    else:
        init_context = cpu_init_weights
    return init_context
