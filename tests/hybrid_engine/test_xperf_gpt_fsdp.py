"""
We instantiate a mariana models. Wrap it using FSDP with FULL_SHARD. Then, feed the weights from FSDP model to XPerfGPT and perform generation
using TP
"""

import os
import numpy as np

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['USE_SESSION_CACHE'] = '0'

import seed_models  # noqa

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

import torch
import torch.distributed
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision

from torch.distributed.device_mesh import init_device_mesh

local_rank, rank, world_size = initialize_global_process_group()

device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
p6dense_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf'
# p6dense_path1 = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/models/p6dense-0.5B-Instruct'
p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'
m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf'
p6_path_qwen = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/qwen2.5_32b_v3.1.2_o1-mini-monologue_241201_hf'

from verl.utils.seed import CHAT_TEMPLATE

model_path = copy_local_path_from_hdfs(m8_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"

tokenizer.chat_template = CHAT_TEMPLATE

with torch.device('cpu'):
    config = AutoConfig.from_pretrained(model_path)
    setattr(config, '_moe_implementation', 'fused')

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="flash_attention_2",
                                                 config=config)

    config = model.config
    print(config)

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

from alpha_seed.workers.streaming_service.streaming_rollout import AsyncXPerfGPTRollout
from alpha_seed.workers.hybrid_engine.fsdp_xperfgpt import FSDPXPerfGPTShardingManager

from omegaconf import OmegaConf

rollout_config = OmegaConf.create({
    'prompt_length': 256,
    'response_length': 2048,
    'micro_batch_size': 128,
    'tensor_model_parallel_size': 2,
    'train_generate_kwargs': {
        'do_sample': True,
        'top_k': 0,
        'top_p': 1.,
        'temperature': 1.,
        'min_p': -1,
        'eta_epsilon': 0
    },
    'enable_paged_attention': False
})

import xperf_gpt

xperf_gpt.load_xperf_gpt()

from xperf_gpt.inference.session import Query


def make_eos_call_back_fn(device_mesh):

    def eos_callback_fn(query: Query):
        if device_mesh is None:
            tp_rank = 0
        else:
            tp_rank = device_mesh['tp'].get_local_rank()

        if tp_rank == 0:
            print(f'Rank: {torch.distributed.get_rank()}, {query.meta_info}')

    return eos_callback_fn


rollout = AsyncXPerfGPTRollout(config=rollout_config, tokenizer=tokenizer, model_hf_config=config)
sharding_manager = FSDPXPerfGPTShardingManager(module=actor_module_fsdp,
                                               model_config=config,
                                               inference_engine=rollout.inference_engine,
                                               device_mesh=rollout.device_mesh)

eos_callback_fn = make_eos_call_back_fn(rollout.device_mesh)
rollout.set_rollout_callback_function(eos_callback_fn=eos_callback_fn)

from verl import DataProto

prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

chat = [{'role': 'user', 'content': prompt}]

sentences = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

input_data = tokenizer(sentences, return_tensors='pt').to('cuda')

input_ids = input_data['input_ids']
attention_mask = input_data['attention_mask']
off_policy_steps = torch.tensor([0]).to('cuda')

data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'off_policy_steps': off_policy_steps}

non_tensors = {
    'oj_feature': np.array([f'rank_{torch.distributed.get_rank()}' for i in range(input_ids.shape[0])], dtype=object)
}

data = DataProto.from_dict(data,
                           non_tensors=non_tensors,
                           meta_info={'generation_kwargs': rollout_config.train_generate_kwargs})

with sharding_manager:
    data = sharding_manager.preprocess_data(data)
    output = next(rollout.generate_sequences(data))
    output = sharding_manager.postprocess_data(output)

output_ids = output.batch['input_ids']

if torch.distributed.get_rank() == 0:
    text_out = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
    print(text_out[0].replace(tokenizer.pad_token, ''))

    # from IPython import embed
    # embed()

torch.distributed.barrier()
