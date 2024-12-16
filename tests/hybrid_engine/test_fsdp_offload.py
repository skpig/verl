"""
PYTHONPATH=.:$PYTHONPATH torchrun --nproc_per_node=2 tests/hybrid_engine/test_fsdp_offload.py
"""

# disbale recompute warning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os

os.environ['TRITON_CACHE_MANAGER'] = 'triton.runtime.cache:RemoteCacheManager'
os.environ['TRITON_REMOTE_CACHE_BACKEND'] = 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend'
os.environ['BPEX_NO_WARN_ON_UNTUNED_CASE'] = '1'
os.environ['NCCL_DEBUG'] = 'warn'

# make deterministic
os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import seed_models  # noqa
import warnings
import functools

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
from alpha_seed.workers.actors.offload import offload_fsdp_model_to_cpu, load_fsdp_model_to_gpu

from tests.hybrid_engine.utils import prepare_data, print_each_rank, to_random, ref_loss_fn
import gc

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.flags(deterministic=True)

p6_400m_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhiqi.0/rlhf/p6_400m_sft'
p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
p6dense_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf'
p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'
m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/25B_MoE_SFT29_32k_bsz6_lr2e5_tp4_hf'


def get_model(use_orig_params: bool):

    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    model_path = copy_local_path_from_hdfs(p6_400m_path)

    with meta_device_init(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = AutoConfig.from_pretrained(model_path)
        setattr(config, '_moe_implementation', 'fused')
        model = AutoModelForCausalLM.from_config(config=config,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="flash_attention_2")
        # model.gradient_checkpointing_enable()

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    auto_wrap_policy = get_fsdp_wrap_policy(module=model)

    shards = parallel_load_safetensors(model_path)
    actor_module_fsdp = FSDP(model,
                             use_orig_params=use_orig_params,
                             param_init_fn=parallel_init_fsdp_fn(model, shards),
                             auto_wrap_policy=auto_wrap_policy,
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             mixed_precision=mixed_precision,
                             sync_module_states=False,
                             device_id=torch.cuda.current_device(),
                             device_mesh=device_mesh)
    optimizer = torch.optim.AdamW(actor_module_fsdp.parameters(), lr=1e-4)
    return actor_module_fsdp, optimizer


def train(model: FSDP, optimizer, offload: bool, steps=3):

    input_ids, input_ids_rolled, masks, position_ids = prepare_data()
    if offload:
        offload_fsdp_model_to_cpu(model)
    losses, gnorms = [], []
    for _ in range(steps):
        rand_input_ids, rand_input_ids_rolled, _, _ = to_random(input_ids, input_ids_rolled, masks, position_ids)

        if offload:
            load_fsdp_model_to_gpu(model)

        optimizer.zero_grad()
        output = model(input_ids=rand_input_ids, position_ids=position_ids)
        loss, _ = ref_loss_fn(output, rand_input_ids_rolled, masks)
        loss.backward()
        gnorm = model.clip_grad_norm_(max_norm=1.0)
        optimizer.step()

        losses.append(loss)
        gnorms.append(gnorm)

        if offload:
            offload_fsdp_model_to_cpu(model)
    return losses, gnorms


def test_offload_and_load(use_orig_params: bool = False):

    model, _ = get_model(use_orig_params=use_orig_params)
    curr_memory = torch.cuda.memory_allocated()
    print_each_rank(f"after init model: {curr_memory / (1024 ** 3):.2f} GB")

    offload_fsdp_model_to_cpu(model)
    offload_memory = torch.cuda.memory_allocated()
    print_each_rank(f"after offloading model: {offload_memory / (1024 ** 3):.2f} GB")

    load_fsdp_model_to_gpu(model)
    load_memory = torch.cuda.memory_allocated()
    print_each_rank(f"after loading model: {load_memory / (1024 ** 3):.2f} GB")

    assert abs(load_memory - curr_memory) / (1024**3) < 0.1
    assert offload_memory / (1024**3) < 0.1


def test_offload_and_load_correctness(use_orig_params: bool = False):

    torch.manual_seed(42)
    model, optimizer = get_model(use_orig_params=use_orig_params)
    ref_losses, ref_gnorms = train(model, optimizer, offload=False)

    torch.manual_seed(42)
    model, optimizer = get_model(use_orig_params=use_orig_params)
    offload_losses, offload_gnorms = train(model, optimizer, offload=True)

    for idx in range(len(ref_losses)):
        ref_loss, real_loss = ref_losses[idx], offload_losses[idx]
        torch.testing.assert_close(ref_loss,
                                   real_loss,
                                   atol=1e-5,
                                   rtol=1e-6,
                                   msg=f"loss {idx}, {ref_loss} vs. {real_loss}")
        ref_gnorm, real_gnorm = ref_gnorms[idx], offload_gnorms[idx]
        torch.testing.assert_close(ref_gnorm,
                                   real_gnorm,
                                   atol=1e-5,
                                   rtol=1e-6,
                                   msg=f"gnorm {idx}, {ref_gnorm} vs. {real_gnorm}")


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    torch.cuda.set_device(dist.get_rank())
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

    test_fns = [
        functools.partial(test_offload_and_load, use_orig_params=True),
        functools.partial(test_offload_and_load, use_orig_params=False),
        functools.partial(test_offload_and_load_correctness, use_orig_params=True),
        functools.partial(test_offload_and_load_correctness, use_orig_params=False),
    ]

    for test_fn in test_fns:
        test_fn()

        if dist.get_rank() == 0:
            print(f"{test_fn.func.__name__} passed")

        gc.collect()
        torch.cuda.empty_cache()

    dist.destroy_process_group()
