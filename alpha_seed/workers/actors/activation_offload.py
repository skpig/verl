from collections.abc import Iterable
from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn as nn

act_offload_supported_layer_classes = [
    "seed_models.models.p6.modeling_p6.P6DecoderLayer",
    "seed_models.models.p7.modeling_p7.P7DecoderLayer",
    "seed_models.models.m8.modeling_m8.M8DecoderLayer",
    "seed_models.models.p6dense.modeling_p6d.P6DenseDecoderLayer"
    "seed_models.models.deepseek_v3.modeling_deepseek.DeepseekV3DecoderLayer"
    "torch.nn.modules.linear.Linear",
    "liger_kernel.transformers.rms_norm.LigerRMSNorm",
]


class ActOffload(torch.autograd.graph.saved_tensors_hooks):

    def __init__(self,
                 module,
                 layer_classes,
                 offload_threshold=10 * 1024 * 1024,
                 offload_upbound=None,
                 offload_last_layer=False):
        super().__init__(self.offload_pack, self.offload_unpack)
        self.offload_layers = []
        self.current_layer = []
        self.offload_threshold = offload_threshold
        self.offload_upbound = offload_upbound
        self.is_hook = False
        self.offload_last_layer = offload_last_layer
        self.offload_stream = torch.cuda.Stream()
        self.register_layer_offload_hook(module, layer_classes)

    def layer_offload(self):
        # offload previous layer to cpu
        if len(self.current_layer) > 0:
            for act in self.current_layer:
                if not act.is_offload and not hasattr(act, 'should_not_offload'):
                    act.data = torch.empty((1,), device='cuda', dtype=act.dtype)
                    act.is_offload = True
            self.offload_layers.append(self.current_layer)
            self.current_layer = []

    def layer_prefetch(self, layer):
        main_stream = torch.cuda.current_stream()
        for x in layer:
            if x.is_offload:
                x_cpu = x.cpu_data
                with torch.cuda.stream(self.offload_stream):
                    x_gpu = torch.empty_like(x_cpu, device='cuda')
                    x_gpu.copy_(x_cpu, non_blocking=True)
                    x_gpu.record_stream(main_stream)
                x.prefetch_data = x_gpu
                x.is_prefetch = True

    def offload_pack(self, x):
        if not isinstance(x, nn.Parameter) and x.numel() >= self.offload_threshold and (
                self.offload_upbound is None or x.numel() <= self.offload_upbound) and x.requires_grad:
            self.current_layer.append(x)
            x_cpu = torch.empty(x.data.size(), device=torch.device('cpu'), dtype=x.data.dtype, pin_memory=True)
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
        if hasattr(x, 'is_offload'):
            if x.is_offload and not x.is_prefetch:
                self.prefetch_last()
            assert x.is_prefetch or (not x.is_offload), "x is offloaded but not prefetch"
            x.cpu_data = None
            if hasattr(x, "prefetch_data"):
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
        super().__enter__()

    def __exit__(self, *args: object):
        super().__exit__()
        # offload last layers
        # if we do not offload last layer, we need to reset it
        if not self.offload_last_layer:
            self.current_layer = []
        else:
            self.layer_offload()

    def register_layer_offload_hook(self, module, layer_classes):
        for _, child in module.named_children():
            c = type(child)
            module_full_name = c.__module__ + '.' + c.__qualname__
            if module_full_name in layer_classes:
                child.register_forward_hook(lambda module, _in, _out: self.layer_offload())
            else:
                self.register_layer_offload_hook(child, layer_classes)

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


class ActNoOffload(nullcontext):

    def __init__(self):
        super().__init__()
        self.current_layer = []

    def mark_not_offload(self, obj):
        pass

    def prefetch_last(self):
        pass


def get_offload_context(enable,
                        module,
                        offload_threshold=1 * 1024 * 1024,
                        offload_upbound=None,
                        offload_last_layer=False):
    if enable:
        return ActOffload(module,
                          layer_classes=act_offload_supported_layer_classes,
                          offload_threshold=offload_threshold,
                          offload_upbound=offload_upbound,
                          offload_last_layer=offload_last_layer)
    else:
        return ActNoOffload()


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
