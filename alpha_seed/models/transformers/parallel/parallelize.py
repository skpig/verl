from typing import Dict
import torch
from torch.distributed._tensor import DeviceMesh
from alpha_seed.workers.fsdp.extensions import parallelize_module, TPSpec

import logging

logger = logging.getLogger(__file__)


def apply_parallel_plan(model: torch.nn.Module, config, tp_mesh: DeviceMesh) -> Dict[str, TPSpec]:
    """
    Apply tensor parallelism to the model.
    """
    make_plan_fn = None
    if config.model_type == 'seed_m8' or \
            (hasattr(config, "text_config") and config.text_config.model_type == 'seed_m8'):
        from ..modeling_m8 import make_m8_plan
        make_plan_fn = make_m8_plan
    if config.model_type == "deepseek_v3":
        from ..modeling_ds import make_dsv3_plan
        make_plan_fn = make_dsv3_plan
    if "P6Dense" in config.architectures[0]:
        from ..modeling_p6d import make_p6d_plan
        make_plan_fn = make_p6d_plan

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
