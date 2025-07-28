from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import create_random_mask, compute_position_id_with_mask

from seed_models.integrations.liger_kernel.liger_utils import apply_liger_kernel_to_p7, apply_liger_kernel_to_p6

import seed_models

# p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/models/p6dense-0.5B-Instruct'
p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'
m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/25B_MoE_SFT29_32k_bsz6_lr2e5_tp4_hf'
m10_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/m10_680m_new'

model_path = copy_local_path_from_hdfs(m10_path)
print(model_path)
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

prompts = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
]

outputs = []

for prompt in prompts:
    chat = [{'role': 'user', 'content': prompt}]

    sentences = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    input_data = tokenizer(sentences, return_tensors='pt').to('cuda')

    input_ids = input_data['input_ids']
    attention_mask = input_data['attention_mask']

    data = {'input_ids': input_ids}

    output = model.generate(**data, max_new_tokens=256, do_sample=False, top_p=0.7, use_cache=True)

    outputs.append(output[0])

text_outs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
for text_out in text_outs:
    print(text_out.replace(tokenizer.pad_token, ''))

# create outputs[0] and outputs[1] with position ids. compute the output logits

input_ids = torch.cat(outputs, dim=0)
cu_seqlens = torch.cumsum(torch.tensor([0] + [output.shape[0] for output in outputs]), dim=0).cuda()
position_ids = torch.cat([torch.arange(output.shape[0]) for output in outputs]).cuda()
model_output = model(input_ids=input_ids, position_ids=position_ids)

logits = model_output.logits[0]  # [total_nnz, vocab_size]
mtp_logits = model_output.mtp_logits[0][0]

# compute ground truth with unpad format

input_ids_rolled = torch.roll(input_ids, shifts=-1)

input_ids_rolled_rolled = torch.roll(input_ids_rolled, shifts=-1)

pretrain_loss = torch.nn.functional.cross_entropy(logits, input_ids_rolled, reduction='none')  # (total_nnz)
pretrain_loss[cu_seqlens[1:] - 1] = 0
mean_loss = torch.sum(pretrain_loss) / (pretrain_loss.shape[0] - 2)

mtp_loss = torch.nn.functional.cross_entropy(mtp_logits, input_ids_rolled_rolled, reduction='none')  # (total_nnz)
mtp_loss[cu_seqlens[1:] - 1] = 0
mtp_loss[cu_seqlens[1:] - 2] = 0
mean_mtp_loss = torch.sum(mtp_loss) / (mtp_loss.shape[0] - 4)

print(mean_loss)
"""
Load a hf model, generate several sequences, and print the mtp acceptance ratio. Have to done in rmpad format
"""
