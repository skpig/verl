"""
torchrun --nproc_per_node=8 scripts/checkpoint/convert_hf_to_omnistore.py \
    --hf-path hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf \
    --omnistore-path /opt/tiger/alpha-seed/omnistore-ckpt

torchrun --nproc_per_node=8 scripts/checkpoint/convert_hf_to_omnistore.py \
    --hf-path hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf \
    --omnistore-path hdfs://haruna/home/byte_data_seed/lf_lq/user/zhiqi.0/rlhf/omnistore-p6-3b3-sft
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'

import seed_models  # noqa
import argparse
import hdfs_io
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig, AutoModelForCausalLM

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision, CPUOffload
from alpha_seed.workers.actors.initialize import parallel_load_safetensors, parallel_init_fsdp_fn, meta_device_init, create_init_fn
from omnistore import FSDPCheckpointer

import time


def convert_hf_to_omnistore(hdfs_or_local_path: str, save_path: str):
    """
    Convert huggingface checkpoint to omnistore checkpoint.
    """
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

    if hdfs_or_local_path.startswith("hdfs://"):
        hf_local_path = copy_local_path_from_hdfs(hdfs_or_local_path)
    else:
        hf_local_path = hdfs_or_local_path

    if dist.get_rank() == 0:
        if save_path.startswith(f"hdfs://"):
            hdfs_io.makedirs(save_path)
        else:
            os.makedirs(save_path)

        for filename in os.listdir(hf_local_path):
            # skip model checkpoint
            if filename.endswith(".safetensors") or filename == "model.safetensors.index.json":
                continue
            src_file = os.path.join(hf_local_path, filename)
            print(f"copying {filename} into {save_path}...")
            hdfs_io.copy(src_file, save_path)

    dist.barrier()

    with meta_device_init():
        config = AutoConfig.from_pretrained(hf_local_path, trust_remote_code=True)
        setattr(config, '_moe_implementation', 'fused')
        if dist.get_rank() == 0:
            print(f'Model config: {config}')
        model = AutoModelForCausalLM.from_config(config=config,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="flash_attention_2")
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    auto_wrap_policy = get_fsdp_wrap_policy(module=model)

    shards = parallel_load_safetensors(hf_local_path)
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        param_init_fn=parallel_init_fsdp_fn(model, shards),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        sync_module_states=False,
        device_id=torch.cuda.current_device(),
        device_mesh=device_mesh,
    )

    torch.cuda.synchronize()
    ckpt = {'model': model}
    print(f"omnistore start saving checkpoint...")
    FSDPCheckpointer.save(save_path, ckpt)


def test_load(omni_path: str):
    """
    Test load omnistore checkpoint.
    """
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

    if omni_path.startswith("hdfs://"):
        omni_path = copy_local_path_from_hdfs(omni_path)

    dist.barrier()
    start = time.time()

    with meta_device_init():
        config = AutoConfig.from_pretrained(omni_path)
        setattr(config, '_moe_implementation', 'fused')
        model = AutoModelForCausalLM.from_config(config,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="flash_attention_2")

    print(f"memory: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
    meta_create_time = time.time() - start
    print(f"meta create time: {meta_create_time} seconds")
    dist.barrier()

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    auto_wrap_policy = get_fsdp_wrap_policy(module=model)
    model = FSDP(
        model,
        param_init_fn=create_init_fn(model),
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        sync_module_states=False,
        device_id=torch.cuda.current_device(),
        device_mesh=device_mesh,
    )

    torch.cuda.synchronize()
    dist.barrier()
    print(f"after wrapping to FSDP module, time {time.time() - start:.2f} seconds")

    FSDPCheckpointer.load(omni_path, {'model': model})
    torch.cuda.synchronize()
    print(f"finish loading parameters, time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path",
                        type=str,
                        required=True,
                        help="hdfs or local (folder) path of huggingface checkpoint")
    parser.add_argument("--omnistore-path",
                        type=str,
                        required=True,
                        help="hdfs or local (folder) path to save omnistore checkpoint")
    parser.add_argument(f"--test-load-only",
                        action="store_true",
                        default=False,
                        help="only test load omnistore checkpoint")
    args = parser.parse_args()
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    if not args.test_load_only:
        convert_hf_to_omnistore(args.hf_path, args.omnistore_path)
    test_load(args.omnistore_path)
    dist.destroy_process_group()
