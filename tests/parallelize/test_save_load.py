import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, CPUOffload
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from alpha_seed.workers.actors.initialize import create_mesh, parallel_load_safetensors, parallel_init_fsdp_fn, meta_device_init

from alpha_seed.workers.actors.checkpoint.extensions import register_dtensor_save_hook
from tests.hybrid_engine.utils import print_each_rank
from alpha_seed.models.transformers.ops import clip_grad_norm_

import numpy as np
import random

from ..launch import torchrun
from functools import partial

import os
from .test_parallel_init import DummyModel, tp_plan, MLP
from alpha_seed.models.transformers.parallel.parallelize import parallelize_module

os.environ['NCCL_DEBUG'] = '0'


def build_model(fsdp_size: int, tp_size: int):
    torch.manual_seed(42)

    meshes = create_mesh(fsdp_size, tp_size, 1)
    fsdp_mesh, tp_mesh = meshes[:2]

    model = DummyModel().cuda()

    shard_plan = None
    if tp_size > 1:
        parallelize_module(model, tp_plan, tp_mesh)
        shard_plan = {}
        for fqn, param in model.named_parameters(remove_duplicate=False):
            if hasattr(param, "_spec"):
                shard_plan[fqn] = param._spec
                assert shard_plan[fqn].mesh is not None

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    cpu_offload = CPUOffload(offload_params=False)
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(MLP,),
    )
    strategy = ShardingStrategy.HYBRID_SHARD if fsdp_mesh.ndim > 1 and fsdp_mesh.size(
    ) > 1 else ShardingStrategy.FULL_SHARD

    model = FSDP(model,
                 use_orig_params=True,
                 auto_wrap_policy=auto_wrap_policy,
                 sharding_strategy=strategy,
                 mixed_precision=mixed_precision,
                 cpu_offload=cpu_offload,
                 forward_prefetch=True,
                 sync_module_states=False,
                 device_id=torch.cuda.current_device(),
                 device_mesh=fsdp_mesh)

    register_dtensor_save_hook(model, shard_plan)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, optimizer, meshes


def train_one_step(model, optimizer):
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (1, 128))

    optimizer.zero_grad()
    output = model(input_ids)
    loss = torch.sum(output)

    gnorm = clip_grad_norm_(model, max_norm=1.0)
    optimizer.step()
    if dist.get_rank() == 0:
        print(f"loss: {loss}, gnorm: {gnorm}")
    return loss, gnorm


def save_v1(model, optimizer, folder):
    if dist.get_rank() == 0:
        os.makedirs(folder, exist_ok=True)
    dist.barrier()
    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        rng = {
            'cpu': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        }
        state = {"model": model_state_dict, "optimizer": optimizer_state_dict, "rng": rng}
    filename = f"model_optim_rank_{dist.get_rank()}.pt"
    filepath = os.path.join(folder, filename)
    torch.save(state, filepath)
    print_each_rank(f"finished saving checkpoint to {filepath}")


def load_v1(model, optimizer, folder):
    filename = f"model_optim_rank_{dist.get_rank()}.pt"
    filepath = os.path.join(folder, filename)
    state = torch.load(filepath, weights_only=False)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = state["model"]
        optimizer_state_dict = state["optimizer"]
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    rng = state["rng"]
    torch.cuda.random.set_rng_state(rng['cuda'])
    torch.random.set_rng_state(rng['cpu'])
    np.random.set_state(rng['numpy'])
    random.setstate(rng['random'])
    print_each_rank(f"finished loading checkpoint from {filepath}.")


def model_save_load_fsdp_tp(fsdp_size: int, tp_size: int):

    model, optim, meshes = build_model(fsdp_size=fsdp_size, tp_size=tp_size)

    iter_res = []

    iter_res.append(train_one_step(model, optim))
    # save
    save_v1(model, optim, "/tmp/ckpt")
    # train 2 steps
    iter_res.append(train_one_step(model, optim))
    iter_res.append(train_one_step(model, optim))
    # load
    load_v1(model, optim, "/tmp/ckpt")
    # train 2 steps
    iter_res.append(train_one_step(model, optim))
    iter_res.append(train_one_step(model, optim))

    # bitwise resume correct
    assert torch.allclose(iter_res[-1][0], iter_res[-3][0], rtol=0, atol=0)
    assert torch.allclose(iter_res[-1][1], iter_res[-3][1], rtol=0, atol=0)
    assert torch.allclose(iter_res[-2][0], iter_res[-4][0], rtol=0, atol=0)
    assert torch.allclose(iter_res[-2][1], iter_res[-4][1], rtol=0, atol=0)


test_model_save_load_fsdp_v1 = partial(torchrun, 4, model_save_load_fsdp_tp, 4, 1)
test_model_save_load_hsdp_v1 = partial(torchrun, 4, model_save_load_fsdp_tp, 2, 1)
test_model_save_load_fsdp_tp_v1 = partial(torchrun, 4, model_save_load_fsdp_tp, 2, 2)
test_model_save_load_hsdp_tp_v1 = partial(torchrun, 8, model_save_load_fsdp_tp, 2, 2)
