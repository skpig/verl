from typing import Tuple, Union, Optional, Dict
import types
import functools
import warnings
import torch
from torch.distributed._tensor.placement_types import Placement
from vescale import CPUOffloadPolicy, DeviceMesh, FSDPModule, MixedPrecisionPolicy, OffloadPolicy
from vescale.parallel.fsdp2.extension.spmd import fully_shard_with_mp
from vescale.parallel.fsdp2.extension.recompute import patch_recompute
from vescale.parallel.fsdp2.extension.act_offload import apply_activation_offload, ActOffloadPolicy
from vescale.initialize.hf_utils import parallel_load_safetensors, parallel_init_module_fn
from vescale.parallel.mp import parallelize_module
from vescale.plan.module_parallel import ModuleParallelPlan, ParallelType
from vescale.dtensor.placement_types import Replicate as veReplicate
from vescale.dtensor.placement_types import Shard as veShard
import fnmatch

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
                oe_mesh: DeviceMesh = None,
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

    # parallelize module
    assert fsdp_mesh._parent_mesh is not None
    global_mesh = fsdp_mesh._parent_mesh
    default_mp_mesh = global_mesh[tuple(n for n in global_mesh.mesh_dim_names if n not in fsdp_mesh.mesh_dim_names)]
    assert default_mp_mesh.size() == 1
    # -- set up mesh
    named_mesh = {ParallelType.MP: default_mp_mesh}
    if tp_mesh is not None:
        named_mesh["EP"] = tp_mesh
    if oe_mesh is not None:
        named_mesh["OE"] = oe_mesh
    # -- set up plan
    plan_name_count = {key: 0 for key in tp_plan}
    spmd_plan = ModuleParallelPlan()
    for fqn, _ in model.named_parameters():
        for plan_name, placement in tp_plan.items():
            if fnmatch.fnmatch(fqn, plan_name):
                placement = veShard(placement.dim) if placement.is_shard() else veReplicate()
                # FIXME(zhiqi.0): make this not to be that hack
                mesh_name = "OE" if "over_encoded" in plan_name else "EP"
                spmd_plan.shard_tensor(fqn, [
                    placement,
                ], mesh=mesh_name)
                print(f"add shard tensor ({fqn=}): {placement} for {mesh_name}")
                plan_name_count[plan_name] += 1
                break
        else:
            spmd_plan.shard_tensor(fqn, [veReplicate()], mesh=ParallelType.MP)
    # -- check whether all the plan keys are used
    missing_keys = {key for key in plan_name_count if plan_name_count[key] == 0}
    if len(missing_keys) > 0:
        raise RuntimeError(f"Cannot find sharding plans of {missing_keys} in model")
    # partition model
    model = parallelize_module(model, named_mesh, spmd_plan, init_only=True)

    # set cpu offload
    offload_policy = CPUOffloadPolicy() if param_offload else OffloadPolicy()
    fully_shard_fn = functools.partial(fully_shard_with_mp,
                                       mesh=fsdp_mesh,
                                       reshard_after_forward=True,
                                       mp_policy=mixed_precision_policy,
                                       offload_policy=offload_policy,
                                       verbose=True)

    # load pretrained weights
    shards = parallel_load_safetensors(weights, device="cpu") if weights else {}
    module_materialize_fn, _, _ = parallel_init_module_fn(model, shards, pad_state=True)

    # wrap to fsdp + prefetch
    last_fsdp_modules = None
    for module in model.modules():
        if isinstance(module, block_cls):
            module_materialize_fn(module)
            modules: Tuple[FSDPModule] = fully_shard_fn(module)
            if last_fsdp_modules is not None:
                last_fsdp_modules[0].set_modules_to_forward_prefetch(modules)
                modules[0].set_modules_to_backward_prefetch(last_fsdp_modules)
            last_fsdp_modules = modules
    module_materialize_fn(model)
    fully_shard_fn(model)
    model.set_reshard_after_backward(True)
    model._set_unshard_async_op(True)

    # apply recompute for each layer
    if enable_training_stats:
        assert recompute, f"Detected training stats or act_offload is enabled, must open gradient checkpointing"
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
        patch_recompute(model)

    # compatible to legacy parallelism
    for name, module in model.named_modules():
        assert not hasattr(module, "_tp_mesh"), f"{name} already gots _tp_mesh field"
        module._tp_mesh = tp_mesh
    for param in model.parameters():
        if hasattr(param, "_spec"):
            param.mesh = param._spec.mesh

    if len(shards) > 0:
        warnings.warn(
            "detected some parameter is not loaded in the model. Ignore this warning if you changed the model structure."
        )
        shards.clear()

    # enable activation offload
    if act_offload:
        if act_offload_kwargs is None:
            act_offload_kwargs = {}
        policy = ActOffloadPolicy(buffer_size_gb=act_offload_kwargs.get('buffer_size', 32))
        apply_activation_offload(model, offload_policy=policy, ignored_block_cls=(torch.nn.Embedding, torch.nn.Dropout))

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
