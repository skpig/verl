"""
torchrun --nproc_per_node=$ARNOLD_WORKER_GPU --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID \
    --master_addr=$ARNOLD_WORKER_0_HOST --master_port=12321 \
    -m tests.hybrid_engine.test_parallel \
    --model hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/m10_680m_new \
    --strategy vescale-fsdp2 \
    --tp-size 4 \
    --sp-size 8 \
    --grad-accum 1 \
    --max-token 8192 \
    --seqlen 8192 \
    --ce-loss-fusion \
    --act-offload \
    --optim-offload \
    --profile-to-mlx \
    2>&1 | tee fsdp2.txt
"""
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os

os.environ['TRITON_CACHE_MANAGER'] = 'triton.runtime.cache:RemoteCacheManager'
os.environ['TRITON_REMOTE_CACHE_BACKEND'] = 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend'
os.environ['BPEX_NO_WARN_ON_UNTUNED_CASE'] = '1'
os.environ['NCCL_DEBUG'] = '0'

# make deterministic

deterministic = False

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
from torch.distributed.fsdp import StateDictType
from transformers import AutoConfig, AutoModelForCausalLM
from alpha_seed.workers.fsdp.initialize import meta_device_init
from alpha_seed.workers.hybrid_engine.fsdp_gather import ulysses_pad_and_slice_inputs
from dist_attn.ulysses.parallel_states import set_ulysses_sequence_parallel_group
from dist_attn.ulysses.ops import gather_outputs
from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch, get_parallel_plan
from alpha_seed.workers.fsdp.offload import offload_fsdp_optimizer, load_fsdp_optimizer
from alpha_seed.trainer.optim import get_optimizer_from_config

from mono_rl.utils.debug import get_profiler_context, MemoryProfiler
from verl.utils.fs import copy_local_path_from_hdfs
from tests.hybrid_engine.utils import print_each_rank
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
import time
from tqdm import trange
import numpy as np
import random
import argparse
import verl.utils.torch_functional as verl_F

# p7_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/20241123/1118a2_2.5b'
# m8_path = 'hdfs://harunava/home/byte_data_seed_us/hdd_va/user/zhiqi.0/rlhf/m8_2B5_sft'
# m10_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/m10_680m_new'


def init_model(model_path: str, fsdp_size: int, tp_size: int, oe_size: int, sp_size: int, optimizer_type: str):

    if args.strategy == 'vescale-fsdp2':
        from alpha_seed.workers.vescale.initialize import create_mesh
    else:
        from alpha_seed.workers.fsdp.initialize import create_mesh
    meshes = create_mesh(fsdp_size, tp_size, oe_size, sp_size)
    fsdp_mesh, tp_mesh, oe_mesh = meshes[:3]
    model_path = copy_local_path_from_hdfs(model_path)

    with meta_device_init(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = AutoConfig.from_pretrained(model_path)
        setattr(config, '_moe_implementation', 'fused')
        setattr(config, 'embd_pdrop', 0.0)
        setattr(config, 'attention_dropout', 0.0)
        setattr(config, 'resid_pdrop', 0.0)

        # shrink layer
        # setattr(config, 'num_hidden_layers', 10)
        # setattr(config, "kv_mirror_imitated_layers", list(range(0, 2)))
        # setattr(config, "kv_mirror_layers", list(range(8, 10)))
        # setattr(config, "pre_post_layernorm_layers", list(range(0, 10)))

        # monkey patch
        apply_monkey_patch(config, strategy=args.strategy)
        model = AutoModelForCausalLM.from_config(config=config,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="flash_attention_2")

    print_each_rank(f"After init from HF model: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    if args.strategy == 'fsdp':
        from alpha_seed.workers.fsdp import fully_shard
    elif args.strategy == 'vescale-fsdp2':
        from alpha_seed.workers.vescale import fully_shard
        from alpha_seed.workers.vescale.fully_shard import register_dtensor_hook
        from vescale.parallel.fsdp2.extension.optimizer_offload import apply_optimizer_offload, OptimizerOffloadPolicy

    model, _ = fully_shard(model=model,
                           block_cls=model._no_split_modules[0],
                           fsdp_mesh=fsdp_mesh,
                           tp_plan=get_parallel_plan(config, tp_mesh, strategy=args.strategy),
                           tp_mesh=tp_mesh,
                           oe_mesh=oe_mesh,
                           recompute=True,
                           act_offload=args.act_offload,
                           param_offload=args.param_offload,
                           weights=model_path)

    optim_config = {
        "type": optimizer_type,
        "lr": 1e-4,
        "betas": [0.9, 0.95],
    }
    from omegaconf import DictConfig
    optimizer = get_optimizer_from_config([param for param in model.parameters() if param.requires_grad],
                                          DictConfig(optim_config))

    if args.strategy == 'fsdp':
        if args.optim_offload:
            optimizer.register_step_pre_hook(
                lambda optim, args, kwargs: load_fsdp_optimizer(optim, torch.cuda.current_device()))
            optimizer.register_step_post_hook(lambda optim, args, kwargs: offload_fsdp_optimizer(optim))
    elif args.strategy == 'vescale-fsdp2':
        register_dtensor_hook(model, optimizer)
        if args.optim_offload:
            policy = OptimizerOffloadPolicy(
                gpu_reserved_size=0,
                overlap_with_forward=False,
            )
            apply_optimizer_offload(model,
                                    optimizer,
                                    get_seqlen_fn=lambda args, kwargs: kwargs["input_ids"].numel(),
                                    offload_policy=policy)

    print_each_rank(f"After FSDP init: memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    torch.cuda.reset_peak_memory_stats()
    return model, optimizer, meshes


def init_random_data(seqlen: int, max_token: int):

    num_seqs = max_token // seqlen
    assert num_seqs >= 1
    seqs = [seqlen] * num_seqs + [max_token % seqlen]
    input_ids, position_ids = [], []
    device = torch.cuda.current_device()
    for seq in seqs:
        input_ids += [torch.randint(0, 8192, size=(seq,), dtype=torch.long, device=device)]
        position_ids += [torch.arange(seq, dtype=torch.long, device=device)]
    input_ids = torch.concat(input_ids).unsqueeze(0)
    assert input_ids.size(1) == max_token
    position_ids = torch.concat(position_ids).unsqueeze(0)
    input_ids_rolled = torch.roll(input_ids, shifts=-1, dims=1).squeeze(0)
    masks = torch.ones_like(input_ids).squeeze(0)
    return input_ids, input_ids_rolled, masks, position_ids, seqs


@torch.no_grad()
def gather_inputs(tensor: torch.Tensor, gather_mesh, gather_dim: int):
    return DTensor.from_local(tensor, gather_mesh, [
        Shard(gather_dim),
    ]).full_tensor()


def train(model, optimizer, meshes, steps: int = 20, profile_to_mlx: bool = False):

    fsdp_mesh, tp_mesh, oe_mesh, sp_mesh, gather_mesh, _ = meshes
    if sp_mesh.size() > 1:
        set_ulysses_sequence_parallel_group(sp_mesh.get_group())

    flops_counter = FlopsCounter(model.config)
    world_size = dist.get_world_size()

    profile_context = get_profiler_context(
        filename=f"rank{dist.get_rank()}.seqlen.{args.seqlen}",
        profile_on_ranks=[0],
        upload_to_mlx=profile_to_mlx,
        default_hdfs_dir="/opt/tiger/alpha-seed/profile",
        enable=True,
        wait=10,
        warmup=2,
        active=1,
    )

    memory_profiler = MemoryProfiler(filename='./memory',
                                     enable=torch.distributed.get_rank() == 0,
                                     upload_to_mlx=profile_to_mlx,
                                     wait=2 * args.grad_accum - 1,
                                     active=1)

    bar = trange(steps, total=steps, disable=dist.get_rank() != 0)
    for step in range(steps):

        if args.test_save_load:
            if step == 2:
                save_checkpoint(model, optimizer, "checkpoint/")
            if step == 5:
                load_checkpoint(model, optimizer, "checkpoint/")

        input_ids, input_ids_rolled, masks, position_ids, seqs = init_random_data(args.seqlen, args.max_token)

        with profile_context as prof:

            torch.manual_seed(gather_mesh.get_local_rank())
            torch.cuda.synchronize()
            start = time.time()

            if gather_mesh.size() > 1:
                input_ids = gather_inputs(input_ids, gather_mesh, gather_dim=1)
                input_ids_rolled = gather_inputs(input_ids_rolled, gather_mesh, gather_dim=0)
                position_ids = gather_inputs(position_ids, gather_mesh, gather_dim=1)
                masks = gather_inputs(masks, gather_mesh, gather_dim=0)
                unpad_size = input_ids.size(1)
                input_ids, position_ids, _ = ulysses_pad_and_slice_inputs(input_ids, position_ids, sp_mesh.size())
                labels, _, _ = ulysses_pad_and_slice_inputs(input_ids_rolled.unsqueeze(0), None, sp_mesh.size())
                labels = labels.squeeze(0)

            # add mtp labels
            mtp_labels = []  # mimic, not real
            if hasattr(model.config, 'mtp_mode'):
                mtp_n_heads = model.config.mtp_n_heads
                for i in range(1, mtp_n_heads):
                    input_ids_rolled_mtp = torch.roll(input_ids, shifts=-i - 1, dims=1)
                    mtp_labels.append(input_ids_rolled_mtp)

            # forward
            if args.ce_loss_fusion:
                output = model(input_ids=input_ids,
                               position_ids=position_ids,
                               use_cache=False,
                               labels=labels,
                               temperature=1.0,
                               fuse_lm_head_ce_loss=True,
                               mtp_labels=mtp_labels)
                log_probs = output.loss
            else:
                output = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
                logits = output.logits.squeeze(0)
                log_probs = cross_entropy_loss(logits, input_ids_rolled, inplace_backward=True)[0]
            if sp_mesh.size() > 1:
                log_probs = gather_outputs(log_probs, gather_dim=0, padding_dim=0, unpad_dim_size=unpad_size)
            loss = verl_F.masked_mean(log_probs, masks)

            # backward
            loss.backward()

            gnorm = 0.0
            if (step + 1) % args.grad_accum == 0:
                gnorm = model.clip_grad_norm_(max_norm=1.0).item()
                optimizer.step()
                optimizer.zero_grad()
                if args.strategy == 'vescale-fsdp2':
                    model.zero_grad()
                if isinstance(model, FSDP):
                    for module in FSDP.fsdp_modules(model):
                        module._flat_param.grad = None

            torch.cuda.synchronize()
            span = time.time() - start

            # metrics
            global_seqs = seqs * world_size
            estimated_flops, promised_flops = flops_counter.estimate_flops(global_seqs, span)
            mfu = round(estimated_flops / (promised_flops * world_size), 3)
            memory_reserve = round(torch.cuda.max_memory_reserved() / (1024**3), 3)
            memory_alloc = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            malloc_retries = torch.cuda.memory_stats()['num_alloc_retries']
            bar.set_postfix({
                'loss': loss.item(),
                'ntokens': sum(global_seqs),
                'mfu': mfu,
                'memory(GB)': f"({memory_alloc}/{memory_reserve})",
                'gnorm': gnorm,
                'mretry': malloc_retries,
            })
            bar.update()
            prof.step()
            memory_profiler.step()


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
                     oe_size: int,
                     sp_size: int,
                     profile_to_mlx: bool = False,
                     optimizer_type: str = 'adam'):

    model, optimizer, device_mesh = init_model(model_path, fsdp_size, tp_size, oe_size, sp_size, optimizer_type)
    train(model, optimizer, device_mesh, profile_to_mlx=profile_to_mlx)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--strategy", type=str, choices=['fsdp', 'vescale-fsdp2'], default='fsdp')
    parser.add_argument("--fsdp-size", type=int, default=-1)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--oe-size", type=int, default=1)
    parser.add_argument("--sp-size", type=int, default=2)
    parser.add_argument("--profile-to-mlx", action='store_true', default=False)
    parser.add_argument("--test-save-load", action='store_true', default=False)
    parser.add_argument("--max-token", type=int, default=16384, help="max token length (total) for a batch")
    parser.add_argument("--seqlen", type=int, default=16384, help="sequence length for a device (before tp / sp)")
    parser.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation times")
    parser.add_argument("--optimizer-type", type=str, default="adam", help="optimizer type, default: adam")
    parser.add_argument("--optim-offload", action='store_true', default=False)
    parser.add_argument("--param-offload", action='store_true', default=False)
    parser.add_argument("--act-offload", action='store_true', default=False)
    parser.add_argument("--ce-loss-fusion",
                        action='store_true',
                        default=False,
                        help="use ce loss fusion to reduce memory")
    args = parser.parse_args()
    print(args)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    test_performance(args.model,
                     fsdp_size=args.fsdp_size,
                     tp_size=args.tp_size,
                     oe_size=args.oe_size,
                     sp_size=args.sp_size,
                     profile_to_mlx=args.profile_to_mlx,
                     optimizer_type=args.optimizer_type)
    dist.destroy_process_group()
