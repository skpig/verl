"""
We instantiate a mariana models. Wrap it using FSDP with FULL_SHARD. Then, feed the weights from FSDP model to XPerfGPT and perform generation
using TP
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'

import seed_models  # noqa

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

import torch
import torch.distributed
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision

from torch.distributed.device_mesh import init_device_mesh

local_rank, rank, world_size = initialize_global_process_group()

device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

model_path = copy_local_path_from_hdfs(
    'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf')
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"

with torch.device('cpu'):
    # model = mariana_models.P5ForCausalLM(config=config)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="flash_attention_2",
    )

    config = model.config

mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

auto_wrap_policy = get_fsdp_wrap_policy(module=model)
print(auto_wrap_policy)

# TODO: add transformer policy
actor_module_fsdp = FSDP(
    model,
    use_orig_params=True,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
    mixed_precision=mixed_precision,
    sync_module_states=False,
    device_id=torch.cuda.current_device(),
    device_mesh=device_mesh)

from alpha_seed.workers.xperf_rollout.xperf_gpt_rollout import XPerfGPTRollout
from alpha_seed.workers.hybrid_engine.fsdp_xperfgpt import FSDPXPerfGPTShardingManager

from omegaconf import OmegaConf

rollout_config = OmegaConf.create({
    'prompt_length': 256,
    'response_length': 256,
    'micro_batch_size': 128,
    'tensor_model_parallel_size': 4,
    'train_generate_kwargs': {
        'do_sample': False,
        'top_k': 0,
        'top_p': 1.,
        'temperature': 1.,
    }
})

import xperf_gpt

xperf_gpt.load_xperf_gpt()

rollout = XPerfGPTRollout(config=rollout_config, tokenizer=tokenizer, model_hf_config=config)
sharding_manager = FSDPXPerfGPTShardingManager(module=actor_module_fsdp,
                                               model_config=config,
                                               inference_engine=rollout.inference_engine,
                                               device_mesh=rollout.device_mesh)

from verl.utils.model import create_random_mask, compute_position_id_with_mask
from verl import DataProto

prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

chat = [{'role': 'user', 'content': prompt}]

sentences = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

input_data = tokenizer(sentences, return_tensors='pt').to('cuda')

input_ids = input_data['input_ids']
attention_mask = input_data['attention_mask']
position_ids = compute_position_id_with_mask(attention_mask)

data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

data = DataProto.from_dict(data, meta_info={'generation_kwargs': rollout_config.train_generate_kwargs})

with sharding_manager:
    data = sharding_manager.preprocess_data(data)
    output = rollout.generate_sequences(data)
    output = sharding_manager.postprocess_data(output)

output_ids = output.batch['input_ids']

text_out = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(text_out)
