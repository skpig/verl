"""
torchrun --nproc_per_node=8 tests/hybrid_engine/test_autotuner.py
"""
import os

os.environ["TRITON_CACHE_MANAGER"] = "triton.runtime.cache:RemoteCacheManager"
os.environ["TRITON_REMOTE_CACHE_BACKEND"] = "alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend"
os.environ["NCCL_DEBUG"] = "0"
os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

from alpha_seed.tuner.auto_tuner import AutoTuner, GpusPerNode, Env, HaveNVLink
import torch
import torch.distributed as dist
import yaml

dist.init_process_group(backend='nccl')
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

mem_margin = 0.2  # we only use 20% GPU memory for constraints test
export_path = "./out.yaml"

ngpus = dist.get_world_size()
gpu_type = torch.cuda.get_device_name().split()[-1]
env = Env(
    gpu_type,
    min(ngpus // GpusPerNode.get(gpu_type, 8), 1),
    min(GpusPerNode.get(gpu_type, 8), ngpus),
    ngpus,
    torch.cuda.get_device_properties(0).total_memory / (1024**3) * (1 - mem_margin),
    HaveNVLink.get(gpu_type, True),
)

tuner = AutoTuner(
    model_path="hdfs://haruna/home/byte_data_seed/lf_lq/user/zhiqi.0/models/m8_680m_sft",
    max_seqlen=8192,
    env=env,
)

tuner.search(constraints=None, export_path=export_path, plot_file="search.png")
with open(export_path, "r") as f:
    recipe = yaml.safe_load(f)

assert recipe["actor_rollout_ref"]["actor"]["act_offload"] is True
assert recipe["actor_rollout_ref"]["actor"]["fsdp_config"]["param_offload"] is False
assert recipe["actor_rollout_ref"]["actor"]["profile"]["enable"] is True

dist.destroy_process_group()
