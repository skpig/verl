from typing import Dict, Union

import torch
from torch.distributed._tensor import Shard, Replicate, DeviceMesh
from dataclasses import dataclass

import logging

logger = logging.getLogger(__file__)


@dataclass
class TPSpec:
    """
    Tensor parallelism specification.
    """
    mesh: DeviceMesh
    shard: Union[Shard, Replicate]
    fqn: str

    def __post_init__(self):
        assert self.mesh.ndim == 1


def apply_parallel_plan(model: torch.nn.Module, config, tp_mesh: DeviceMesh) -> Dict[str, TPSpec]:
    """
    Apply tensor parallelism to the model.
    """
    make_plan_fn = None
    if config.model_type == 'seed_m8':
        from ..modeling_m8 import make_m8_plan
        make_plan_fn = make_m8_plan
    if config.model_type == "deepseek_v3":
        from ..modeling_ds import make_dsv3_plan
        make_plan_fn = make_dsv3_plan

    if make_plan_fn is None:
        assert tp_mesh.size() == 1, f"tensor parallelism is not support for model: {config.model_type}"
        return {}
    plan: Dict = make_plan_fn()
    parallelize_module(model, plan, tp_mesh)
    # get shard plan
    shard_plan = {}
    for fqn, param in model.named_parameters(remove_duplicate=False):
        if hasattr(param, "_spec"):
            shard_plan[fqn] = param._spec
    return shard_plan


@torch.no_grad()
def parallelize_module(model: torch.nn.Module, plan: Dict[str, Shard], tp_mesh: DeviceMesh):
    assert tp_mesh.ndim == 1
    tp_size = tp_mesh.size()
    tp_rank = tp_mesh.get_local_rank()
    logger.info(f"tp rank [{tp_rank}]: initializing tensor parallelism with size: {tp_size}")

    param2fqn = {}
    for fqn, param in model.named_parameters(remove_duplicate=False):
        assert not hasattr(param, '_spec'), f"torch.nn.Parameter._spec is reserved for parallelization"
        param2fqn.setdefault(param, []).append(fqn)

    cnts = {plan_name: 0 for plan_name in plan.keys()}

    for sub_module in model.modules():
        for name, param in sub_module.named_parameters(recurse=False):
            fqn = param2fqn[param][0]
            if hasattr(param, "_spec"):  # already sharded
                continue
            for plan_name, shard in plan.items():
                if plan_name in fqn:
                    assert len(param2fqn[param]) == 1, \
                        f"{fqn} is a shared parameter, which doesn't support sharding now"
                    assert param.size(shard.dim) % tp_size == 0
                    # print(f"{fqn} is sharded by {shard}")
                    chunk = torch.chunk(param.data, chunks=tp_size, dim=shard.dim)[tp_rank]
                    param = torch.nn.Parameter(chunk, requires_grad=param.requires_grad)
                    param._spec = TPSpec(mesh=tp_mesh, shard=shard, fqn=fqn)
                    sub_module._parameters[name] = param
                    cnts[plan_name] += 1
                    # sub_module.register_parameter(name, param)
                    break
            else:  # not specified by plan
                param._spec = TPSpec(mesh=tp_mesh, shard=Replicate(), fqn=fqn)

    for plan_name, cnt in cnts.items():
        if cnt == 0:
            raise RuntimeError(f"{plan_name} is not found in model")

    # make shure all parameters are sharded
    for param in model.parameters():
        assert hasattr(param, "_spec"), f"Internal Error: {param} is omitted"

    return model
