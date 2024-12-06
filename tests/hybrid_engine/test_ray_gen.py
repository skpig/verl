"""
Test the generation inside ray
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'

from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup

from omegaconf import OmegaConf

from verl.utils.fs import copy_local_path_from_hdfs

from transformers import AutoTokenizer

import torch

import ray

if __name__ == '__main__':
    p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
    p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'
    m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/25B_MoE_SFT29_32k_bsz6_lr2e5_tp4_hf'

    config = OmegaConf.create({
        'actor': {
            'fsdp_size': -1,
            'ulysses_sequence_parallel_size': 1
        },
        'rollout': {
            'name': 'xperf_gpt',
            'prompt_length': 256,
            'response_length': 256,
            'micro_batch_size': 128,
            'log_prob_micro_batch_size': 128,
            'tensor_model_parallel_size': 4,
            'train_generate_kwargs': {
                'do_sample': False,
                'top_k': 0,
                'top_p': 1.,
                'temperature': 1.,
            },
            'enable_paged_attention': False
        },
        'ref': {
            'ulysses_sequence_parallel_size': 1
        },
        'model': {
            'path': p6_path
        }
    })

    resource_pool = RayResourcePool(process_on_nodes=[4], use_gpu=True)
    ray_cls_with_init = RayClassWithInitArgs(cls=AsyncActorRolloutRefWorker, config=config, role='rollout')
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    ray.get(wg.init_model())

    from verl import DataProto

    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

    chat = [{'role': 'user', 'content': prompt}]

    model_path = copy_local_path_from_hdfs(p6_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    sentences = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    input_data = tokenizer(sentences, return_tensors='pt').to('cuda')

    input_ids = input_data['input_ids']
    attention_mask = input_data['attention_mask']
    off_policy_steps = torch.tensor([0]).to('cuda')

    data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'off_policy_steps': off_policy_steps}

    data = DataProto.from_dict(data, meta_info={'generation_kwargs': config.rollout.train_generate_kwargs})

    data = data.repeat(4)
    output = wg.generate_sequences(data)

    output_ids = output.batch['input_ids']

    text_out = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

    for i in range(len(text_out)):
        print(text_out[i].replace(tokenizer.pad_token, ''))
