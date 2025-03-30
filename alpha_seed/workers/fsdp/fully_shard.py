"""
torch FSDP with Tensor Parallelism, memory offload extension.
"""
from typing import Union, Optional, Dict, Tuple
import types
import warnings
import functools
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed._tensor.placement_types import Placement
from .extensions import register_dtensor_save_hook, parallelize_module
from .initialize import parallel_load_safetensors, parallel_init_fsdp_fn
from .offload.activation_offload import get_offload_context
from .clip_grad_norm import clip_grad_norm_

from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name
from alpha_seed.utils.observility.training_stats import MetricsTorchDispatchMode, metrics_context_fn
from contextlib import nullcontext
from torch.utils.checkpoint import noop_context_fn


def fully_shard(
    model: PreTrainedModel,
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
    act_offload_kwargs: dict = None,
) -> FSDP:
    """
    Create FSDP/HSDP with tensor parallelism extension.

    Args:
        model (PreTrainedModel): HuggingFace model that is initialized in meta device
        block_cls (type | str): unit (usually a decoder layer class)
            that is used to wrap into FSDPModule
        fsdp_mesh (DeviceMesh): 1-dim or 2-dim device mesh for FSDP or HSDP, can only be 1-dim or 2-d
    """
    # parallelize model
    if tp_plan is not None:
        assert tp_mesh is not None
        tp_plan = parallelize_module(model, tp_plan, tp_mesh)

    assert not (enable_training_stats and act_offload), f"act offload and training stats can not be enabled together"

    if enable_training_stats or act_offload:
        assert recompute, f"Detected training stats or act_offload is enabled, must open gradient checkpointing"

    # apply recompute for each layer
    metrics_context = MetricsTorchDispatchMode() if enable_training_stats else nullcontext()
    if recompute:
        use_reentrant = act_offload
        if not isinstance(model, PreTrainedModel):
            raise RuntimeError(f"Recompute only works with HF PreTrainedModel")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                'use_reentrant':
                    use_reentrant,
                "context_fn":
                    functools.partial(metrics_context_fn, metrics_context) if (
                        enable_training_stats and not use_reentrant) else noop_context_fn,
            })

    # set mixed precision
    mp_config = dict(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    if ignored_modules is not None:
        mp_config['_module_classes_to_ignore'] = tuple(ignored_modules)
    mixed_precision = MixedPrecision(**mp_config)

    # set module wrap class
    if isinstance(block_cls, str):
        block_cls = get_module_class_from_name(model, block_cls)
    if not issubclass(block_cls, torch.nn.Module):
        raise NotImplementedError(f"block cls must be subclass of torch.nn.Module, but got {block_cls}")

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(block_cls,),
    )
    # set fsdp/hsdp sharding strategy
    if fsdp_mesh.ndim > 1 and fsdp_mesh.size() > 1:
        strategy = ShardingStrategy.HYBRID_SHARD
    else:
        strategy = ShardingStrategy.FULL_SHARD

    # set cpu offload
    offload = None
    if param_offload:
        offload = CPUOffload(offload_params=True)

    # load pretrained weights
    shards = parallel_load_safetensors(weights) if weights else {}
    init_fn = parallel_init_fsdp_fn(model, shards)

    # wrap to fsdp
    model: FSDP = FSDP(model,
                       use_orig_params=True,
                       param_init_fn=init_fn,
                       auto_wrap_policy=auto_wrap_policy,
                       sharding_strategy=strategy,
                       mixed_precision=mixed_precision,
                       cpu_offload=offload,
                       forward_prefetch=True,
                       sync_module_states=False,
                       device_id=torch.cuda.current_device(),
                       device_mesh=fsdp_mesh)
    if len(shards) > 0:
        warnings.warn(
            "detected some parameter is not loaded in the model. Ignore this warning if you changed the model structure."
        )
        shards.clear()

    # enable activation offload
    if act_offload:
        if act_offload_kwargs is None:
            act_offload_kwargs = {}
        context = get_offload_context(True, model, **act_offload_kwargs)

        def enter_act_offload(module: torch.nn.Module, input):
            if torch.is_grad_enabled():
                context.__enter__()

        def exit_act_offload(module: torch.nn.Module, input, output):
            if torch.is_grad_enabled():
                context.__exit__()

        model.register_forward_pre_hook(enter_act_offload, prepend=True)
        model.register_forward_hook(exit_act_offload, prepend=False)

    # default use sharded state dict for checkpoint save
    FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT)

    # register dtensor-based hook for tensor parallelism
    register_dtensor_save_hook(model, tp_plan, tp_outside)
    # tensor parallelism requires customized clip grad norm
    if tp_mesh is not None:
        model.clip_grad_norm_ = types.MethodType(clip_grad_norm_, model)

    return model, metrics_context
