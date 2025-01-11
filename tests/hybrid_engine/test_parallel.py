"""
torchrun --nproc_per_node=$ARNOLD_WORKER_GPU --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID \
    --master_addr=$ARNOLD_WORKER_0_HOST --master_port=12321 \
    tests/hybrid_engine/test_parallel.py \
    --model m8 --fsdp-size 4 --tp-size 2 --sp-size 2 --offload \
    2>&1 | tee log.txt

torchrun --nproc_per_node=4 --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID \
    --master_addr=$ARNOLD_WORKER_0_HOST --master_port=12321 \
    tests/hybrid_engine/test_parallel.py \
    --model m8 --fsdp-size 4 --tp-size 2 --sp-size 2 --offload \
    2>&1 | tee log.txt
"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os

os.environ['TRITON_CACHE_MANAGER'] = 'triton.runtime.cache:RemoteCacheManager'
os.environ['TRITON_REMOTE_CACHE_BACKEND'] = 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend'
os.environ['BPEX_NO_WARN_ON_UNTUNED_CASE'] = '1'
os.environ['NCCL_DEBUG'] = '0'

# make deterministic

deterministic = True

if deterministic:
    os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.flags(deterministic=True)

import warnings
import seed_models
from seed_models.utils.count_flops import FlopsCounter
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, CPUOffload
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision
from transformers import AutoConfig, AutoModelForCausalLM
from alpha_seed.workers.actors.initialize import create_mesh, parallel_load_safetensors, parallel_init_fsdp_fn, meta_device_init
from alpha_seed.workers.hybrid_engine.fsdp_gather import ulysses_pad_and_slice_inputs
from dist_attn.ulysses.parallel_states import set_ulysses_sequence_parallel_group
from dist_attn.ulysses.ops import gather_outputs
from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch
from alpha_seed.workers.actors.checkpoint.extensions import register_dtensor_save_hook
from alpha_seed.models.transformers.parallel import apply_parallel_plan
from alpha_seed.models.transformers.ops import clip_grad_norm_

from verl.utils.debug import get_profiler_context
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from tests.hybrid_engine.utils import prepare_data, print_each_rank, ref_loss_fn
import time
from tqdm import trange
import numpy as np
import random
import argparse

p7_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/20241123/1118a2_2.5b'
# m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/25B_MoE_SFT29_32k_bsz6_lr2e5_tp4_hf'
m8_path = 'hdfs://harunava/home/byte_data_seed_us/hdd_va/user/zhiqi.0/rlhf/m8_2B5_sft'


def init_model(model_path: str, fsdp_size: int, tp_size: int, sp_size: int, offload: bool):

    meshes = create_mesh(fsdp_size, tp_size, sp_size)
    fsdp_mesh, tp_mesh = meshes[:2]
    model_path = copy_local_path_from_hdfs(model_path)

    with meta_device_init(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = AutoConfig.from_pretrained(model_path)
        setattr(config, '_moe_implementation', 'fused')
        setattr(config, 'embd_pdrop', 0.0)
        setattr(config, 'attention_dropout', 0.0)
        setattr(config, 'resid_pdrop', 0.0)

        # monkey patch
        apply_monkey_patch(config)
        model = AutoModelForCausalLM.from_config(config=config,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="flash_attention_2")
        # enable recompute
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})

        nparams = sum(p.numel() for p in model.parameters())
        print_each_rank(f"number of parameters before parallelization: {nparams / (1e9):.2f}B")

        shard_plan = apply_parallel_plan(model, config, tp_mesh)

        nparams = sum(p.numel() for p in model.parameters())
        print_each_rank(f"number of parameters after parallelization: {nparams / (1e9):.2f}B")

    print_each_rank(f"After init from HF model: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    cpu_offload = CPUOffload(offload_params=offload)
    auto_wrap_policy = get_fsdp_wrap_policy(module=model)

    shards = parallel_load_safetensors(model_path)
    init_fn = parallel_init_fsdp_fn(model, shards)
    strategy = ShardingStrategy.HYBRID_SHARD if fsdp_mesh.ndim > 1 and fsdp_mesh.size(
    ) > 1 else ShardingStrategy.FULL_SHARD

    model = FSDP(model,
                 use_orig_params=True,
                 param_init_fn=init_fn,
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
    print_each_rank(f"After FSDP init: memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    return model, optimizer, meshes


@torch.no_grad()
def gather_inputs(tensor: torch.Tensor, gather_mesh, gather_dim: int):
    return DTensor.from_local(tensor, gather_mesh, [
        Shard(gather_dim),
    ]).full_tensor()


def train(model, optimizer, meshes, steps: int = 20, profile_to_mlx: bool = False):

    fsdp_mesh, tp_mesh, sp_mesh, gather_mesh = meshes
    if sp_mesh.size() > 1:
        set_ulysses_sequence_parallel_group(sp_mesh.get_group())

    flops_counter = FlopsCounter(model.config)
    world_size = dist.get_world_size()

    profile_context = get_profiler_context(
        filename=f"profile_{dist.get_rank()}",
        profile_on_ranks=[0],
        upload_to_mlx=profile_to_mlx,
        default_hdfs_dir="/opt/tiger/alpha-seed/profile",
        enable=True,
        wait=10,
        warmup=2,
        active=1,
    )

    bar = trange(steps, total=steps, disable=dist.get_rank() != 0)

    for step in range(steps):
        torch.manual_seed(42)

        if step == 2:
            save_checkpoint(model, optimizer, "checkpoint/")
        if step == 5:
            load_checkpoint(model, optimizer, "checkpoint/")

        input_ids, input_ids_rolled, masks, position_ids = prepare_data()

        with profile_context as prof:

            torch.cuda.synchronize()
            start = time.time()

            if sp_mesh.size() > 1:
                input_ids = gather_inputs(input_ids, gather_mesh, gather_dim=1)
                input_ids_rolled = gather_inputs(input_ids_rolled, gather_mesh, gather_dim=0)
                position_ids = gather_inputs(position_ids, gather_mesh, gather_dim=1)
                masks = gather_inputs(masks, gather_mesh, gather_dim=0)
                unpad_size = input_ids.size(1)
                input_ids, position_ids, _ = ulysses_pad_and_slice_inputs(input_ids, position_ids, sp_mesh.size())

            optimizer.zero_grad()
            # forward
            output = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
            if sp_mesh.size() > 1:
                output.logits = gather_outputs(output.logits, gather_dim=1, padding_dim=1, unpad_dim_size=unpad_size)
            loss, _ = ref_loss_fn(output, input_ids_rolled, masks)
            # backward
            loss.backward()

            gnorm = clip_grad_norm_(model, max_norm=1.0)
            optimizer.step()

            torch.cuda.synchronize()
            span = time.time() - start

            # metrics
            ntokens = masks.sum().item() * world_size / gather_mesh.size()
            estimated_flops, promised_flops = flops_counter.estimate_flops([ntokens], span)
            mfu = round(estimated_flops / (promised_flops * world_size), 3)
            memory = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            bar.set_postfix({
                'loss': loss.item(),
                'ntokens': ntokens,
                'mfu': mfu,
                'memory(GB)': memory,
                'gnorm': gnorm.item()
            })
            bar.update()
            prof.step()


def save_checkpoint(model, optimizer, folder):
    if dist.get_rank() == 0:
        os.makedirs(folder, exist_ok=True)
    dist.barrier()
    print_each_rank(f"start saving checkpoint to {folder}...")
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = model.state_dict()
        # check model state dict
        for name, tensor in model_state_dict.items():
            if isinstance(tensor, DTensor):
                assert len(tensor.placements) >= 2, f"{name}: {tensor.placements}"
                if isinstance(tensor.placements[-1], Shard):
                    assert tensor.placements[-1] != tensor.placements[-2]
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


def load_checkpoint(fsdp_model: FSDP, optimizer: torch.optim.Optimizer, folder):
    print_each_rank(f"start loading checkpoint from {folder}...")
    filename = f"model_optim_rank_{dist.get_rank()}.pt"
    filepath = os.path.join(folder, filename)
    state = torch.load(filepath, weights_only=False)
    with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = state["model"]
        optimizer_state_dict = state["optimizer"]
        fsdp_model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    rng = state["rng"]
    torch.cuda.random.set_rng_state(rng['cuda'])
    torch.random.set_rng_state(rng['cpu'])
    np.random.set_state(rng['numpy'])
    random.setstate(rng['random'])
    print_each_rank(f"finished loading checkpoint from {filepath}.")


def test_performance(model_path: str,
                     fsdp_size: int,
                     tp_size: int,
                     sp_size: int,
                     offload: bool,
                     profile_to_mlx: bool = False):

    model, optimizer, device_mesh = init_model(model_path, fsdp_size, tp_size, sp_size, offload)
    train(model, optimizer, device_mesh, profile_to_mlx=profile_to_mlx)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="m8", choices=['m8', 'p7'])
    parser.add_argument("--offload", action='store_true', default=False)
    parser.add_argument("--fsdp-size", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--sp-size", type=int, default=2)
    parser.add_argument("--profile-to-mlx", action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    model_path = {
        "m8": m8_path,
        "p7": p7_path,
    }

    test_performance(model_path[args.model],
                     fsdp_size=args.fsdp_size,
                     tp_size=args.tp_size,
                     sp_size=args.sp_size,
                     offload=args.offload,
                     profile_to_mlx=args.profile_to_mlx)
    dist.destroy_process_group()
