from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import create_random_mask, compute_position_id_with_mask

from seed_models.integrations.liger_kernel.liger_utils import apply_liger_kernel_to_p7, apply_liger_kernel_to_p6

import seed_models

p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'

model_path = copy_local_path_from_hdfs(p7_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"

apply_liger_kernel_to_p6()

with torch.device('cpu'):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2",
                                                 _moe_implementation='fused')

    config = model.config

model = model.cuda()

print(model)
print(model.config)

prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

chat = [{'role': 'user', 'content': prompt}]

sentences = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

input_data = tokenizer(sentences, return_tensors='pt').to('cuda')

input_ids = input_data['input_ids']
attention_mask = input_data['attention_mask']

data = {'input_ids': input_ids}

output = model.generate(**data, max_new_tokens=512, do_sample=False, top_p=0.7, use_cache=True)

text_out = tokenizer.batch_decode(output, skip_special_tokens=False)
print(text_out[0].replace(tokenizer.pad_token, ''))
