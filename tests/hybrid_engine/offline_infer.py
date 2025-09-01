"""
We instantiate a mariana models. Wrap it using FSDP with FULL_SHARD. Then, feed the weights from FSDP model to XPerfGPT and perform generation
using TP
"""

import os
import numpy as np

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['USE_SESSION_CACHE'] = '0'

import seed_models  # noqa

from seed_models import M8ForCausalLM
from seed_models.models.m8.modeling_m8 import M8TopkCapGate

import warnings

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from mono_rl import DataProto

import torch
import torch.distributed
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision

from torch.distributed.device_mesh import init_device_mesh

from hdfs_io import hput

local_rank, rank, world_size = initialize_global_process_group()

device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
p6dense_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf'
p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'
m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/25B_MoE_SFT29_32k_bsz6_lr2e5_tp4_hf'
p6_path_qwen = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/qwen2.5_32b_v3.1.2_o1-mini-monologue_241201_hf'
m8_15b_path = 'hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/yueyu/1222a1_v12/checkpoints/global_step_47/actor/huggingface'

from verl.utils.model import print_model_size, update_model_config
from verl.utils.torch_dtypes import PrecisionType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
    CPUOffload
from torch import optim

local_path = copy_local_path_from_hdfs(m8_15b_path)

# note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
# TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=False)
torch_dtype = torch.float32

# override model kwargs
actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=False)

override_config_kwargs = {
    'bos_token_id': tokenizer.bos_token_id,
    'eos_token_id': tokenizer.eos_token_id,
    'pad_token_id': tokenizer.pad_token_id,
}
override_config_kwargs.update({})
update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
setattr(actor_model_config, '_moe_implementation', 'fused')
if rank == 0:
    print(f'Model config after override: {actor_model_config}')

# optimize the model via rmpad
from mono_rl.models.seed_models.monkey_patch import apply_monkey_patch
assert apply_monkey_patch(config=actor_model_config,
                          verbose=rank == 0), f'Cannot find rmpad version of {actor_model_config.model_type}'

from mono_rl.worker.engine.fsdp.initialize import parallel_init_fsdp_fn, parallel_load_safetensors, meta_device_init

enable_gradient_checkpointing = True

with meta_device_init(), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    actor_module = AutoModelForCausalLM.from_config(actor_model_config,
                                                    torch_dtype=torch_dtype,
                                                    attn_implementation='flash_attention_2',
                                                    trust_remote_code=False)
    # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
    actor_module.to(torch_dtype)

    if enable_gradient_checkpointing:
        actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if rank == 0:
            print(actor_module)
            print('Enable actor gradient checkpointing')
            if hasattr(actor_module, 'transformer'):
                model = actor_module.transformer
            elif hasattr(actor_module, 'model'):
                model = actor_module.model
            else:
                model = None
            if model is not None:
                print(f'{model.gradient_checkpointing=}, {model.training=}, {model._gradient_checkpointing_func=}')
torch.distributed.barrier()

if rank == 0:
    print_model_size(actor_module)

param_dtype = torch.bfloat16
reduce_dtype = torch.float32
buffer_dtype = torch.float32

mixed_precision = MixedPrecision(param_dtype=param_dtype,
                                 reduce_dtype=reduce_dtype,
                                 buffer_dtype=buffer_dtype,
                                 _module_classes_to_ignore=[M8TopkCapGate])

auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=None)

if rank == 0:
    print(f'wrap_policy: {auto_wrap_policy}')

cpu_offload = None

# if role == 'actor':
#     if self.config.actor.fsdp_config.param_offload:
#         cpu_offload = CPUOffload(offload_params=True)
# elif role == 'ref':
#     if self.config.ref.fsdp_config.param_offload:
#         cpu_offload = CPUOffload(offload_params=True)

# we only support ZeRO3 of hybrid DP+FSDP or full FSDP
if device_mesh.ndim == 1:
    sharding_strategy = ShardingStrategy.FULL_SHARD
elif device_mesh.ndim == 2:
    sharding_strategy = ShardingStrategy.HYBRID_SHARD
else:
    raise NotImplementedError(f"get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")

# TODO: add transformer policy
actor_module_fsdp = FSDP(actor_module,
                         param_init_fn=parallel_init_fsdp_fn(actor_module, parallel_load_safetensors(local_path)),
                         use_orig_params=False,
                         auto_wrap_policy=auto_wrap_policy,
                         device_id=local_rank,
                         sharding_strategy=sharding_strategy,
                         mixed_precision=mixed_precision,
                         sync_module_states=False,
                         forward_prefetch=True,
                         device_mesh=device_mesh,
                         cpu_offload=cpu_offload)

from alpha_seed.workers.ppo_actor import DataParallelPPOActor
from omegaconf import OmegaConf
import yaml

config_yaml = """
fsdp_config:
  param_offload: True
  wrap_policy:
    # transformer_layer_cls_to_wrap: None
    min_num_params: 0
log_prob_micro_batch_size: 128
ulysses_sequence_parallel_size: 1
fsdp_size: -1
use_dynamic_bsz: True
max_token_len: -1
ema: 1.0
use_rmpad: True
"""

config = OmegaConf.create(yaml.safe_load(config_yaml))

actor = DataParallelPPOActor(config=config, actor_module=actor_module_fsdp)

# read input data
data_path = copy_local_path_from_hdfs(
    'hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/yueyu/1222a1_v12/train_batch/train_batch_48.pt')

# split into data into chunks
data = DataProto.load_from_disk(data_path)
data = DataProto(batch=data.batch, non_tensor_batch=data.non_tensor_batch, meta_info=data.meta_info)
data.check_consistency()
data.meta_info['use_dynamic_bsz'] = True
data.meta_info['max_token_len'] = 2048 + 16384
data.meta_info['temperature'] = 1
# get the current chunk for dp
data_chunk = data.chunk(world_size)[rank]
# perform infer
entropy, log_probs = actor.compute_log_prob(data_chunk)

# allgather
tensor_list = [torch.empty_like(log_probs) for _ in range(world_size)]
torch.distributed.all_gather(tensor_list=tensor_list, tensor=log_probs)

# dump to disk
if rank == 0:
    torch.save(log_probs, 'logprobs_0103.pt')
    hput('logprobs_0103.pt', 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/debug')

# update gate_ema. For m8
with torch.inference_mode():
    for i in range(actor_module_fsdp.module.config.num_hidden_layers):
        gate = actor_module_fsdp.module.transformer.h[i].module.mlp.moe.gate
        with FSDP.summon_full_params(gate, rank0_only=False, offload_to_cpu=False):
            gate.module.update_gate_ema()

torch.distributed.barrier()

# now we compute log prob again on the same data
entropy, log_probs = actor.compute_log_prob(data_chunk)

# allgather
tensor_list = [torch.empty_like(log_probs) for _ in range(world_size)]
torch.distributed.all_gather(tensor_list=tensor_list, tensor=log_probs)

# dump to disk
if rank == 0:
    torch.save(log_probs, 'logprobs_after_ema_0103.pt')
    hput('logprobs_after_ema_0103.pt', 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/debug')
