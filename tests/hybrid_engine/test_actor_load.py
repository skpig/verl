"""
torchrun --nproc_per_node=8 tests/hybrid_engine/test_actor_load.py
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['USE_SESSION_CACHE'] = '0'

import seed_models  # noqa

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision

from torch.distributed.device_mesh import init_device_mesh
from alpha_seed.workers.actors.initialize import parallel_load_safetensors, parallel_init_fsdp_fn

dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
torch.cuda.set_device(dist.get_rank())
device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
p6dense_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf'
p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'
m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/25B_MoE_SFT29_32k_bsz6_lr2e5_tp4_hf'

model_path = copy_local_path_from_hdfs(p6_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"

dist.barrier()
import time

start = time.time()

with torch.device('meta'):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="flash_attention_2",
                                                 _moe_implementation='fused')
    config = model.config

meta_create_time = time.time() - start
print(f"meta create time: {meta_create_time} seconds")
dist.barrier()

mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

auto_wrap_policy = get_fsdp_wrap_policy(module=model)
print(auto_wrap_policy)

# TODO: add transformer policy
shards = parallel_load_safetensors(model_path)

load_shard_time = time.time() - start
print(f"until load shard time: {load_shard_time} seconds")
print(f"{dist.get_rank()}: loaded torch memory: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

actor_module_fsdp = FSDP(model,
                         use_orig_params=True,
                         param_init_fn=parallel_init_fsdp_fn(model, shards),
                         auto_wrap_policy=auto_wrap_policy,
                         sharding_strategy=ShardingStrategy.FULL_SHARD,
                         mixed_precision=mixed_precision,
                         sync_module_states=False,
                         device_id=torch.cuda.current_device(),
                         device_mesh=device_mesh)

torch.cuda.synchronize()
end = time.time()
print(f"{dist.get_rank()}: lafter init: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
print(f"init time: {end - start} seconds")
dist.destroy_process_group()
