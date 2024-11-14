# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we only support seed_models
"""

from transformers import PreTrainedTokenizer


def get_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    """
    Get a config file given model
    """
    if model_config.model_type == 'seed_p4':
        return _get_p4_xperf_gpt_config(model_config, tokenizer)
    elif model_config.model_type == "seed_p5":
        return _get_p5_xperf_gpt_config(model_config, tokenizer)
    elif model_config.model_type == 'seed_p6':
        return _get_p6_xperf_gpt_config(model_config, tokenizer)
    elif model_config.model_type == 'seed_p6dense':
        return _get_p6dense_xperf_gpt_config(model_config, tokenizer)
    else:
        raise NotImplementedError(f'Unsupported model {model_config.model_type}')


def _get_p4_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import P4Config
    assert isinstance(model_config, P4Config)
    xperf_config = dict(
        dtype="bfloat16",
        model_name="GPT2LMHeadModel",
        vocab_size=model_config.vocab_size,
        max_position_embeddings=model_config.max_position_embeddings,
        embed_dim=model_config.hidden_size,
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        has_ptb=model_config.parallel_transformer_block,
        has_mqa=model_config.num_attention_heads != model_config.num_key_value_heads,
        rope_mode="default",
        rope_base=int(model_config.rope_theta),
        # TODO: handle window_size config
        # TODO: orca should not rely on tokenizer
        tokenizer_path=tokenizer.name_or_path,
    )
    return xperf_config


def _get_p5_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import P5Config
    assert isinstance(model_config, P5Config)
    config = model_config
    xperf_config = {
        "dtype": "bfloat16",
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "embed_dim": config.hidden_size,
        "hidden_size": config.hidden_size,
        "num_heads": config.num_attention_heads,
        "num_layers": config.num_hidden_layers,
        "num_kv_heads": config.num_key_value_heads,
        "has_ptb": config.parallel_transformer_block,
        "has_mqa": config.num_attention_heads != config.num_key_value_heads,
        "moe_ffn_internal_dim": int(config.intermediate_size * config.moe_expert_eq_dim_factor),
        "moe_expert_num": config.moe_num_expert,
        "moe_topk": config.moe_topk,
        "is_exp_moe": False,
        "moe_ffn_has_bias": False,
        "model_name": "GPT2LMHeadModelMoe" if config.moe_num_expert else "GPT2LMHeadModel",
        "is_meta": False,
        "has_k_layernorm": config.use_key_layernorm,
        "tokenizer_path": tokenizer.name_or_path,
        "rope_mode": "default",
        "rope_base": int(config.rope_theta)
    }
    return xperf_config


def _get_p6_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import P6Config
    assert isinstance(model_config, P6Config)
    config = model_config
    xperf_config = {
        "dtype": "bfloat16",
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "embed_dim": config.hidden_size,
        "hidden_size": config.hidden_size,
        "num_heads": config.num_attention_heads,
        "num_layers": config.num_hidden_layers,
        "num_kv_heads": config.num_key_value_heads,
        "has_ptb": False,
        "has_mqa": config.num_attention_heads != config.num_key_value_heads,
        "moe_ffn_internal_dim": int(config.intermediate_size),
        "moe_expert_num": config.moe_num_expert,
        "moe_topk": config.moe_topk,
        "is_exp_moe": False,
        "moe_ffn_has_bias": False,
        "model_name": "GPT2LMHeadModelMoe" if config.moe_num_expert else "GPT2LMHeadModel",
        "is_meta": True,
        "has_k_layernorm": config.use_key_layernorm,
        "has_context_layernorm": config.use_context_groupnorm,
        "tokenizer_path": tokenizer.name_or_path,
        "has_mlp_gate": True,
        "rope_mode": config.rope_scaling['rope_type'],
        "rope_base": int(config.rope_theta),
        "rope_scale": config.rope_scaling['factor']
    }
    return xperf_config


def _get_p6dense_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import P6DenseConfig
    assert isinstance(model_config, P6DenseConfig)
    config = model_config
    xperf_config = {
        "model_name": "SeedLLaMAForCausalLM",
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_size": config.hidden_size,
        "ffn_internal_dim": config.intermediate_size,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": config.num_key_value_heads,
        "num_layers": config.num_hidden_layers,
        "gqa_weights_layout": "AABB",
        "quant_mode": "NO_QUANT",
        "is_meta": True,
        "dtype": "bfloat16",
        "tokenizer_path": tokenizer.name_or_path,
        "rope_mode": config.rope_scaling['rope_type'],
        "rope_base": int(config.rope_theta),
        "rope_scale": config.rope_scaling['factor']
    }
    return xperf_config
