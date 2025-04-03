from typing import Tuple, Union, Optional, Dict
import types
import functools
import warnings
import torch
from torch.distributed._tensor.placement_types import Placement
from vescale import CPUOffloadPolicy, DeviceMesh, FSDPModule, MixedPrecisionPolicy, OffloadPolicy
from vescale.parallel.fsdp2 import fully_shard as vescale_fully_shard
from vescale.parallel.fsdp2.extension.act_offload import apply_activation_offload
from vescale.initialize.hf_utils import parallel_load_safetensors, parallel_init_module_fn
from vescale.parallel.fsdp2.extension.state_dict import eager_init_optimizer

from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name
from alpha_seed.utils.observility.training_stats import MetricsTorchDispatchMode, metrics_context_fn
from contextlib import nullcontext
from torch.utils.checkpoint import noop_context_fn


def fully_shard(model: PreTrainedModel,
                block_cls: Union[type, str],
                fsdp_mesh: DeviceMesh,
                tp_plan: Optional[Dict[str, Placement]] = None,
                tp_mesh: DeviceMesh = None,
                tp_outside: bool = None,
                recompute: bool = False,
                act_offload: bool = False,
                param_offload: bool = False,
                weights: str = None,
                enable_training_stats: bool = False,
                ignored_modules: Tuple = None,
                fsdp_kwargs: dict = None,
                act_offload_kwargs: dict = None) -> Tuple[FSDPModule, Optional[MetricsTorchDispatchMode]]:
    """
    Create FSDP/HSDP withx tensor parallelism extension.

    By default, the model will use mixed-precision training (i.e., BF16 for forward/backward 
    and FP32 for gradient reduction).

    Warnings:
        * tensor parallelism is not supported yet

    Args:
        model (PreTrainedModel): A HuggingFace model initialized on a meta device.
        block_cls (type | str): A unit (usually a decoder layer class) used to wrap into an FSDPModule.
        fsdp_mesh (DeviceMesh): A 1D or 2D device mesh for FSDP or HSDP.
        tp_plan (Dict[str, Placement] | None): A tensor parallelism plan specifying the shard placement of parameter names. 
            The names can be simplified, and all full parameter names containing the specified name will be matched and parallelized.
        tp_mesh (DeviceMesh | None): A 1D tensor parallelism device mesh when tp_plan is specified.
        tp_outside (bool): Whether the tp_mesh is the outermost device mesh dimension relative to the fsdp mesh.
        recompute (bool): Whether to apply recomputation using the HuggingFace interface.
        act_offload (bool): Whether to apply activation offloading.
        param_offload (bool): Whether to apply parameter CPU offloading.
        weights (str or None): The checkpoint file path. Default is None, meaning the parameters will be initialized with random values.
        enable_training_state (bool): Whether to enable training state tracking.
        ignored_modules (tuple): Modules to be ignored in mixed precision.
        fsdp_kwargs (dict): Other keyword arguments for FSDP initialization passed to FSDP.
        act_offload_kwargs (dict): Special keyword arguments for activation offloading.

    """
    # parallelize model
    if tp_mesh.size() > 1:
        raise NotImplementedError("vescale fully_shard not supported tensor parallelism")

    if enable_training_stats:
        assert recompute, f"Detected training stats or act_offload is enabled, must open gradient checkpointing"

    # apply recompute for each layer
    metrics_context = MetricsTorchDispatchMode() if enable_training_stats else nullcontext()
    if recompute:
        if not isinstance(model, PreTrainedModel):
            raise RuntimeError(f"Recompute only works with HF PreTrainedModel")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                'use_reentrant':
                    False,
                "context_fn":
                    functools.partial(metrics_context_fn, metrics_context
                                     ) if enable_training_stats else noop_context_fn,
            })

    # set mixed precision
    mp_config = dict(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    if ignored_modules is not None:
        raise NotImplementedError("vescale fully_shard not supported ignored modules")
    mixed_precision_policy = MixedPrecisionPolicy(**mp_config)

    # set module wrap class
    if isinstance(block_cls, str):
        block_cls = get_module_class_from_name(model, block_cls)
    if not issubclass(block_cls, torch.nn.Module):
        raise NotImplementedError(f"block cls must be subclass of torch.nn.Module, but got {block_cls}")

    # set cpu offload
    offload_policy = CPUOffloadPolicy() if param_offload else OffloadPolicy()

    fully_shard_fn = functools.partial(vescale_fully_shard,
                                       mesh=fsdp_mesh,
                                       reshard_after_forward=True,
                                       mp_policy=mixed_precision_policy,
                                       offload_policy=offload_policy,
                                       params_stored_in_dtensor=False)

    # load pretrained weights
    shards = parallel_load_safetensors(weights) if weights else {}
    module_materialize_fn = parallel_init_module_fn(model, shards)

    # wrap to fsdp + prefetch
    last_fsdp_module = None
    for module in model.modules():
        if isinstance(module, block_cls):
            module_materialize_fn(module)
            module: FSDPModule = fully_shard_fn(module)
            if last_fsdp_module is not None:
                last_fsdp_module.set_modules_to_forward_prefetch([module])
                module.set_modules_to_backward_prefetch([last_fsdp_module])
            last_fsdp_module = module
    module_materialize_fn(model)
    model: FSDPModule = fully_shard_fn(model)
    model.set_reshard_after_backward(True)
    model._set_unshard_async_op(True)

    for name, module in model.named_modules():
        assert not hasattr(module, "_tp_mesh"), f"{name} already gots _tp_mesh field"
        module._tp_mesh = tp_mesh

    if len(shards) > 0:
        warnings.warn(
            "detected some parameter is not loaded in the model. Ignore this warning if you changed the model structure."
        )
        shards.clear()

    # enable activation offload
    if act_offload:
        if act_offload_kwargs is None:
            act_offload_kwargs = {}
        apply_activation_offload(model,
                                 buffer_size_gb=act_offload_kwargs.get('buffer_size', 40),
                                 buffer_dtype=mp_config["param_dtype"])

    return model, metrics_context


def register_dtensor_hook(model: FSDPModule, optim: torch.optim.Optimizer):
    """
    Regsiter dtensor state dict patch to FSDPModule
    """
    model.state_dict = types.MethodType(functools.partial(FSDPModule.state_dict, dtensor=True), model)
    optim.state_dict = staticmethod(
        functools.partial(FSDPModule.optim_state_dict, model, optim, dtensor=True),
    )
    optim.load_state_dict = staticmethod(
        functools.partial(FSDPModule.optim_state_dict_to_load, model, optim, load_directly=True),
    )
