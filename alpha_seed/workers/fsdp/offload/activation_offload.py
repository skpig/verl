################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import itertools
from typing import Any, Dict, Iterable, Tuple, Optional
import torch
import torch.distributed
import torch.nn as nn
import logging
import math
from ..fully_shard import FSDP
from ..initialize import singleton
from torch.utils._pytree import tree_flatten

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ActOffloadPolicy:
    """
    Activation offload policy for ``apply_activation_offload``

    Args:
        offload_size: (min_numel, max_numel) denotes for the tensor numel
            for offload. ``None`` denotes that no limit for minimal numel or maximal numel.
        offload_last_layer: whether to offload the last layer's output.
        buffer_size_gb: the size of cpu buffer in GB.
        buffer_dtype: the dtype of cpu buffer.
        dynamic_buffer_resize: whether to resize cpu buffer dynamically. Default ``True``.
    """

    # offload_size (min_numel, max_numel), None in
    offload_size: Tuple[Optional[int], Optional[int]] = (1024 * 1024, None)
    offload_last_layer: bool = False
    buffer_size_gb: int = 20
    buffer_dtype: torch.dtype = torch.bfloat16
    dynamic_buffer_resize: bool = True
    pin_memory: bool = False


@singleton
@dataclass
class _ActCPUBuffer:
    buffer_size_gb: int = 0
    buffer_dtype: torch.dtype = torch.bfloat16
    dynamic_buffer_resize: bool = True
    pin_memory: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.buffer_dtype, torch.dtype)
        assert self.buffer_size_gb >= 0
        self.buffer_dtype = torch.bfloat16
        log_msg = f"allocating pinned cpu buffer for activation offload: {self.buffer_size_gb} GB"
        logger.info(log_msg)
        self.cpu_buffer = torch.empty(
            self.buffer_size_gb * 1024 * 1024 * 1024 // self.buffer_dtype.itemsize,
            dtype=self.buffer_dtype,
            device=torch.device("cpu"),
            pin_memory=self.pin_memory,
        )
        self.buffer_ofst: int = 0
        self.expected_cpu_buffer_numel: int = 0
        self.outside_cpu_buffer_memory: int = 0

    def get_cpu_tensor(self, x: torch.Tensor) -> torch.Tensor:
        cpu_device = torch.device("cpu")
        if x.device == cpu_device:
            return x
        numel = x.numel()

        may_use_cpu_buffer = (x.dtype == self.cpu_buffer.dtype) and (x.layout == torch.strided)
        if may_use_cpu_buffer:
            self.expected_cpu_buffer_numel += numel

        can_use_cpu_buffer = may_use_cpu_buffer and (self.buffer_ofst + numel <= self.cpu_buffer.size(0))
        if can_use_cpu_buffer:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("using cpu buffer to offload tensor %s (%s), buffer_ofst: %s", x.size(), x.dtype,
                             self.buffer_ofst)
            x_cpu = self.cpu_buffer[self.buffer_ofst:self.buffer_ofst + numel].view(x.size())
            x_cpu.as_strided_(x.size(), x.stride())
            self.buffer_ofst += numel
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("using adhoc cpu tensor to offload %s (%s)", x.size(), x.dtype)
            x_cpu = torch.empty_like(x, device=cpu_device, pin_memory=True)
            x_cpu.as_strided_(x.size(), x.stride())
            self.outside_cpu_buffer_memory += x.numel() * x.dtype.itemsize
        return x_cpu

    def reset_buffer(self):
        # resize buffer is necessary
        if self.dynamic_buffer_resize:
            size_gb = math.ceil(self.expected_cpu_buffer_numel * self.cpu_buffer.dtype.itemsize / (1024**3))
            numel = int(size_gb) * (1024**3) // self.cpu_buffer.dtype.itemsize
            if numel > self.cpu_buffer.size(0):
                logging.info("resize offload cpu buffer to %s GB.", size_gb)
                self.cpu_buffer.untyped_storage().resize_(0)
                self.cpu_buffer = torch.empty(
                    numel,
                    dtype=self.buffer_dtype,
                    device=torch.device("cpu"),
                    pin_memory=self.pin_memory,
                )
        # clear offset pointer
        self.buffer_ofst = 0
        self.expected_cpu_buffer_numel = 0
        self.outside_cpu_buffer_memory = 0


class _ActOffload(torch.autograd.graph.saved_tensors_hooks):

    def __init__(self, offload_policy: ActOffloadPolicy, verbose: bool = False):
        """
        Activation offload hooks.

        Args:
            offload_policy (ActOffloadPoilcy): configuration for activation offload
            verbose (bool): whether to turn on debug log. Default ``false``.
        """
        super().__init__(self.offload_pack, self.offload_unpack)
        if not isinstance(offload_policy, ActOffloadPolicy):
            raise TypeError(f"expected offload_policy to be of type ActOffloadPolicy, but got {type(offload_policy)}")
        if verbose:
            logger.setLevel(logging.DEBUG)
        self.offload_layers = []
        self.current_layer = []
        self.offload_min_numel = offload_policy.offload_size[0]
        self.offload_max_numel = offload_policy.offload_size[1]
        self.offload_last_layer: bool = offload_policy.offload_last_layer
        # act offload stream should have higher priority than optimizer offload stream
        self.offload_stream = torch.cuda.Stream(-1)
        # init cpu buffer
        self.cpu_buffer = _ActCPUBuffer(buffer_size_gb=offload_policy.buffer_size_gb,
                                        buffer_dtype=offload_policy.buffer_dtype,
                                        dynamic_buffer_resize=offload_policy.dynamic_buffer_resize,
                                        pin_memory=offload_policy.pin_memory)
        self.offload_gpu_memory: int = 0

    def layer_offload(self, inputs=None, offload: bool = False):
        if inputs is None:
            inputs = []
        current_layer_load = list(set(inputs) - set(self.current_layer))
        current_layer_offload = list(set(self.current_layer) - set(inputs))

        # load current layer input to gpu
        if len(current_layer_load) > 0:
            torch.cuda.current_stream().wait_stream(self.offload_stream)
            for act in current_layer_load:
                if isinstance(act, torch.Tensor) and hasattr(act, "is_offload") and act.is_offload:
                    x_cpu = act.cpu_data
                    # NOTE: incorrect when `non_blocking=True`
                    x_gpu = x_cpu.to("cuda", non_blocking=False)
                    act.data = x_gpu
                    act.is_offload = False
                    self.offload_gpu_memory -= act.numel() * act.dtype.itemsize

        # offload previous layer to cpu
        if offload and len(current_layer_offload) > 0:
            for act in current_layer_offload:
                if not act.is_offload and not hasattr(act, "should_not_offload"):
                    self.offload_gpu_memory += act.numel() * act.dtype.itemsize
                    act.data = torch.empty((0,), device="cuda", dtype=act.dtype)
                    act.is_offload = True
            self.offload_layers.append(self.current_layer)
            self.current_layer = []

    def layer_prefetch(self, layer):
        main_stream = torch.cuda.current_stream()
        for x in layer:
            if x.is_offload and x.cpu_data is not None:
                x_cpu = x.cpu_data
                with torch.cuda.stream(self.offload_stream):
                    x_gpu = x_cpu.to("cuda", non_blocking=True)
                    x_gpu.record_stream(main_stream)
                x.prefetch_data = x_gpu
                x.is_prefetch = True

    def offload_pack(self, x: torch.Tensor):
        # note: this is to compose with the case of
        # `torch.utils.checkpoint.checkpoint(..., reentrant=False)`
        if not x.is_cuda:
            return x
        # skip already offloaded tensor
        if hasattr(x, "is_offload") and x.is_offload:
            return x
        numel = x.numel()
        min_numel = numel if self.offload_min_numel is None else self.offload_min_numel
        max_numel = numel if self.offload_max_numel is None else self.offload_max_numel
        if not isinstance(x, nn.Parameter) and min_numel <= numel <= max_numel and x.requires_grad:
            self.current_layer.append(x)
            x_cpu = self.cpu_buffer.get_cpu_tensor(x)
            self.offload_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.offload_stream):
                x_cpu.copy_(x.data, non_blocking=True)
                x.data.record_stream(self.offload_stream)
            x.cpu_data = x_cpu
            x.is_offload = False
            x.is_prefetch = False
            x.layer_idx = len(self.offload_layers)
        return x

    def offload_unpack(self, x):
        if hasattr(x, "is_offload"):
            if x.is_offload and not x.is_prefetch:
                self.prefetch_last()
            assert x.is_prefetch or (not x.is_offload), "x is offloaded but not prefetch"
            x.cpu_data = None
            if hasattr(x, "prefetch_data") and x.prefetch_data is not None:
                x.data = x.prefetch_data
                x.prefetch_data = None
            if not x.layer_idx > len(self.offload_layers):
                # if so, this is the first object in this layer
                torch.cuda.current_stream().wait_stream(self.offload_stream)
                if len(self.offload_layers) > 0:
                    last_layer = self.offload_layers.pop(-1)
                    self.layer_prefetch(last_layer)
        return x

    def __enter__(self):
        assert len(self.current_layer) == 0, "current offloading cache is not empty"
        self.offload_gpu_memory = 0
        self.cpu_buffer.reset_buffer()
        super().__enter__()

    def __exit__(self, *args: object):
        super().__exit__()
        # offload last layers
        # if we do not offload last layer, we need to reset it
        if not self.offload_last_layer:
            self.current_layer = []
        else:
            self.layer_offload(offload=True)

    def mark_not_offload(self, obj):
        if isinstance(obj, Iterable):
            for i in obj:
                i.should_not_offload = True
        else:
            obj.should_not_offload = True

    def prefetch_last(self):
        if len(self.offload_layers) > 0:
            self.layer_prefetch(self.offload_layers.pop(-1))
            torch.cuda.current_stream().wait_stream(self.offload_stream)


def _register_act_offload_hook(model: torch.nn.Module, layer_cls: Tuple[type], offload_context: _ActOffload):
    """
    Register layer offload hook to the module
    Register layer hook to restore offloaded inputs to gpu.

    Args:
        model (torch.nn.Module)
        block_cls (Tuple[type]): module classes that will apply activation offload
    """
    cnt = 0

    def pre_forward_hook(module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        # check whether this hook is the first hook of the ``module``
        first_hook_fn = module._forward_pre_hooks[next(iter(module._forward_pre_hooks))]
        if first_hook_fn is not pre_forward_hook:
            raise RuntimeError(
                "Activation offload requires its module pre_forward hook to be the first hook. "
                "This error indicates user registered other hooks with ``prepend=True`` during the model training. "
                "Please try to register them with ``prepend=False`` or disable activation offload.")
        (flatten_args, _), (flatten_kwargs, _) = tree_flatten(args), tree_flatten(kwargs)
        inputs = []
        for t in itertools.chain(flatten_args, flatten_kwargs):
            if isinstance(t, torch.Tensor):
                inputs.append(t)
        offload_context.layer_offload(inputs, offload=isinstance(module, layer_cls))

    for module in model.modules():
        if module is model:
            continue
        # NOTE: we must use ``prepend=True`` because previous pre-hooks may change the input tensors
        # e.g., cast tensor dtype in fsdp, therefore losing the offload information
        module.register_forward_pre_hook(pre_forward_hook, with_kwargs=True, prepend=True)
        if isinstance(module, layer_cls):
            cnt += 1

    return cnt


def apply_activation_offload(model: nn.Module,
                             block_cls: Iterable[type] = None,
                             offload_policy: ActOffloadPolicy = None,
                             verbose: bool = False) -> int:
    """
    Apply activation offload to the model. The activation offload can be applied
    both before or after FSDP wrap.

    Warnings:
        - If recompute is also enabled with activation offload, it is recommended
        to use `use_reentrant=False` with activation offload. If user needs to
        use `use_reentrant=True`, please apply monkey patch of recompute with:

        ```
        from vescale.parallel.fsdp2.extension.recompute import patch_recompute
        patch_recompute(model, use_reentrant)
        ```

        - If none of the ``cls`` matches within the model, RuntimeError will be raised.

    Example:
        >>> from vescale.parallel.fsdp2.extension.act_offload import apply_activation_offload, ActOffloadPolicy
        >>> model = ...
        >>> offload_policy = ActOffloadPolicy(buffer_size_gb=20, )
        >>> apply_activation_offload(model, cls=(nn.Linear,), offload_policy=offload_policy, verbose=False)
        >>> ...
        >>> loss = model(inputs)
        >>> loss.backward()
        >>> ...

    Args:
        model (torch.nn.Module)
        block_cls (Iterable[type]): ``nn.Module`` classes that will happen offload. This controls
            the granularity of memory offload happened. Default ``None``, which will apply to the ``FSDPModule``s.
        offload_policy (ActOffloadPolicy): policy for activation offload. Default ``None``, which will use
            the default act offload policy.
        verbose (bool): whether to print debug logs at logging.DEBUG level. Default ``False``.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected model to be torch.nn.Module but got {type(model)}")

    block_cls = {FSDP} if block_cls is None else set(block_cls)

    if offload_policy is None:
        offload_policy = ActOffloadPolicy()
    act_offload_context = _ActOffload(offload_policy=offload_policy, verbose=verbose)

    is_initialized = False

    def enter_act_offload(module: torch.nn.Module, input):
        # register forward hooks
        # NOTE: we can only register hooks right before model forward to guarantee that
        # the act_offload hooks must be the first hook called during module forward.
        # Otherwise, earlier hooks may change the input tensors (e.g., tensor.float()) and the offload information
        # attached on that tensor will be lost at the entry point.
        nonlocal is_initialized
        if not is_initialized:
            registered = _register_act_offload_hook(model, tuple(block_cls), act_offload_context)
            if registered == 0:
                raise RuntimeError("activation offload doesn't take effect, because no module type "
                                   f"can be matched to register offload hook (block cls: {block_cls})")
            is_initialized = True

        if torch.is_grad_enabled():
            act_offload_context.__enter__()

    def exit_act_offload(module: torch.nn.Module, input, output):
        if torch.is_grad_enabled():
            act_offload_context.__exit__()

    # register act offload hooks for forward
    # in this way user can directly call `model(input).backward()` with out additional context
    model.register_forward_pre_hook(enter_act_offload, prepend=True)
    model.register_forward_hook(exit_act_offload, prepend=False)


from torch.utils.checkpoint import (
    check_backward_validity,
    _infer_device_type,
    _get_autocast_kwargs,
    _get_device_module,
    get_device_states,
    set_device_states,
    detach_variable,
)
from torch.distributed.fsdp._common_utils import (_get_module_fsdp_state_if_fully_sharded_module, _module_handle)
from torch.distributed.fsdp._runtime_utils import (
    _pre_backward_hook,
    _post_backward_hook,
)
import contextlib


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(ctx.device)
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)

        # patch code, remove the extra allgather with use_reentrant + ckpt
        if not isinstance(ctx.run_function, torch.nn.Module):
            ctx.patch_module = ctx.run_function.__self__
        else:
            ctx.patch_module = ctx.run_function
        state = _get_module_fsdp_state_if_fully_sharded_module(ctx.patch_module)
        if state:
            handle = _module_handle(state, ctx.patch_module)
            if handle:
                handle._needs_pre_backward_unshard = True
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("When use_reentrant=True, torch.utils.checkpoint is incompatible"
                               " with .grad() or passing an `inputs` parameter to .backward()."
                               " To resolve this error, you can either set use_reentrant=False,"
                               " or call .backward() without passing the `inputs` argument.")
        # patch code, remove the extra allgather with use_reentrant + ckpt
        handle = None
        state = _get_module_fsdp_state_if_fully_sharded_module(ctx.patch_module)
        if state:
            handle = _module_handle(state, ctx.patch_module)
            if handle:
                _pre_backward_hook(state, ctx.patch_module, handle, None)

        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors
        device_module = _get_device_module(ctx.device)

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
            detached_inputs = detach_variable(tuple(inputs))

            device_autocast_ctx = torch.amp.autocast(device_type=ctx.device, **
                                                     ctx.device_autocast_kwargs) if torch.amp.is_autocast_available(
                                                         ctx.device) else contextlib.nullcontext()
            with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(
                    **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("none of output has requires_grad=True,"
                               " this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)

        # patch code, remove the extra allgather with use_reentrant + ckpt
        if handle:
            _post_backward_hook(state, handle, None)

        return (None, None) + grads
