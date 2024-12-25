from transformers import AutoModelForCausalLM, Qwen2Config, AutoTokenizer, AutoModelForTokenClassification, Qwen2ForTokenClassification

import torch
from verl.utils.fs import copy_local_path_from_hdfs

local_path = copy_local_path_from_hdfs(
    'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/models/Qwen2.5-0.5B-Instruct')

model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16)
config: Qwen2Config = model.config
tokenizer = AutoTokenizer.from_pretrained(local_path)

assert isinstance(config, Qwen2Config)

from seed_models import P6DenseConfig, P6DenseForCausalLM

p6d_config = P6DenseConfig(vocab_size=config.vocab_size,
                           hidden_size=config.hidden_size,
                           intermediate_size=config.intermediate_size,
                           num_hidden_layers=config.num_hidden_layers,
                           num_attention_heads=config.num_attention_heads,
                           num_key_value_heads=config.num_key_value_heads,
                           max_position_embeddings=config.max_position_embeddings,
                           attention_bias=True,
                           rope_theta=config.rope_theta,
                           rope_scaling=config.rope_scaling,
                           mlp_bias=False,
                           bos_token_id=tokenizer.bos_token_id,
                           eos_token_id=tokenizer.eos_token_id)

p6d_model = AutoModelForCausalLM.from_config(p6d_config, torch_dtype=torch.bfloat16)

state_dict = model.state_dict()
p6d_state_dict = p6d_model.state_dict()

for i in range(p6d_config.num_hidden_layers):
    key = f'model.layers.{i}.self_attn.o_proj.bias'
    o_bias = torch.zeros_like(p6d_state_dict[key])
    state_dict[key] = o_bias

p6d_model.load_state_dict(state_dict)

# save to local disk
saved_local_path = '/opt/tiger/p6dense-0.5B-Instruct'
p6d_model.save_pretrained(saved_local_path, max_shard_size='500MB')
tokenizer.save_pretrained(saved_local_path)

saved_local_path = '/opt/tiger/p6dense-0.5B-Instruct_rm'
p6d_config.num_labels = 1
p6d_config.hidden_dropout = 0
p6d_rm_model = AutoModelForTokenClassification.from_config(p6d_config)
p6d_rm_model.load_state_dict(state_dict, strict=False)
p6d_rm_model.save_pretrained(saved_local_path, max_shard_size='500MB')
tokenizer.save_pretrained(saved_local_path)
