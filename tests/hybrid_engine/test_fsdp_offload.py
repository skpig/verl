"""
PYTHONPATH=.:$PYTHONPATH torchrun --nproc_per_node=8 tests/hybrid_engine/test_fsdp_offload.py
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'

import seed_models  # noqa
import warnings

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision

from torch.distributed.device_mesh import init_device_mesh
from alpha_seed.workers.actors.initialize import parallel_load_safetensors, parallel_init_fsdp_fn, meta_device_init

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, load_fsdp_param_and_grad

from tests.hybrid_engine.utils import prepare_data, print_each_rank

p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
p6dense_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf'
p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'
m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/25B_MoE_SFT29_32k_bsz6_lr2e5_tp4_hf'


def get_model():

    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    model_path = copy_local_path_from_hdfs(p6_path)

    with meta_device_init(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = AutoConfig.from_pretrained(model_path)
        setattr(config, '_moe_implementation', 'fused')
        model = AutoModelForCausalLM.from_config(config=config,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="flash_attention_2")

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    auto_wrap_policy = get_fsdp_wrap_policy(module=model)

    shards = parallel_load_safetensors(model_path)
    actor_module_fsdp = FSDP(model,
                             use_orig_params=False,
                             param_init_fn=parallel_init_fsdp_fn(model, shards),
                             auto_wrap_policy=auto_wrap_policy,
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             mixed_precision=mixed_precision,
                             sync_module_states=False,
                             device_id=torch.cuda.current_device(),
                             device_mesh=device_mesh)
    return actor_module_fsdp


def test_offload_and_load():

    model = get_model()
    curr_memory = torch.cuda.memory_allocated()
    print_each_rank(f"after init model: {curr_memory / (1024 ** 3):.2f} GB")

    offload_fsdp_param_and_grad(model)
    offload_memory = torch.cuda.memory_allocated()
    print_each_rank(f"after offloading model: {offload_memory / (1024 ** 3):.2f} GB")

    load_fsdp_param_and_grad(model, torch.cuda.current_device())
    load_memory = torch.cuda.memory_allocated()
    print_each_rank(f"after loading model: {load_memory / (1024 ** 3):.2f} GB")

    assert abs(load_memory - curr_memory) / (1024**3) < 0.1
    assert offload_memory / (1024**3) < 0.1


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    torch.cuda.set_device(dist.get_rank())
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

    test_offload_and_load()
    dist.destroy_process_group()
