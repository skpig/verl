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

from functools import partial
from transformers import PreTrainedTokenizer
import torch


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
    elif model_config.model_type == 'seed_p7':
        return _get_p7_xperf_gpt_config(model_config, tokenizer)
    elif model_config.model_type == 'seed_m8':
        return _get_m8_xperf_gpt_config(model_config, tokenizer)
    elif model_config.model_type == 'seed_vl':
        return _get_vl_xperf_gpt_config(model_config, tokenizer)
    elif model_config.model_type == 'deepseek_v3':
        return _get_dsv3_xperf_gpt_config(model_config, tokenizer)
    elif model_config.model_type == 'seed_m10':
        return _get_m10_xperf_gpt_config(model_config, tokenizer)
    elif model_config.model_type == 'seed_m11':
        return _get_m11_xperf_gpt_config(model_config, tokenizer)
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
        "is_meta": True,
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
        "rope_scale": int(config.rope_scaling['factor'])
    }
    return xperf_config


def _get_p6dense_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import P6DenseConfig
    assert isinstance(model_config, P6DenseConfig)
    config = model_config

    if config.rope_scaling is None:
        config.rope_scaling = {'rope_type': 'default', 'factor': 1}

    xperf_config = {
        "model_name": "SeedLLaMAForCausalLM",
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "embed_dim": config.hidden_size,
        "hidden_size": config.hidden_size,
        "ffn_internal_dim": config.intermediate_size,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": config.num_key_value_heads,
        "num_layers": config.num_hidden_layers,
        "gqa_weights_layout": "AABB",
        "quant_mode": "NO_QUANT",
        "is_meta": True,
        "dtype": "bfloat16",
        "has_mlp_bias": config.mlp_bias,
        "has_attn_bias": config.attention_bias,
        "rms_norm_eps": config.rms_norm_eps,
        "tokenizer_path": tokenizer.name_or_path,
        "rope_mode": config.rope_scaling['rope_type'],
        "rope_base": int(config.rope_theta),
        "rope_scale": int(config.rope_scaling['factor']),
        "q_head_times": 1 if not hasattr(config, 'query_head_scale_factor') else config.query_head_scale_factor,
    }
    if (head_dim := getattr(config, 'head_dim', None)) is not None:
        xperf_config['head_dim'] = head_dim
    if getattr(config, 'use_qk_rmsnorm', False):
        xperf_config['querynorm'] = True
        xperf_config['keynorm'] = True

    return xperf_config


def _get_p7_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import P7Config
    assert isinstance(model_config, P7Config)
    config = model_config
    xperf_config = {
        "dtype":
            "bfloat16",
        "vocab_size":
            config.vocab_size,
        "max_position_embeddings":
            config.max_position_embeddings,
        "embed_dim":
            config.hidden_size,
        "hidden_size":
            config.hidden_size,
        "num_heads":
            config.num_attention_heads,
        "q_head_times":
            config.query_head_scale_factor,
        "num_layers":
            config.num_hidden_layers,
        "num_kv_heads":
            config.num_key_value_heads,
        "has_mqa":
            config.num_attention_heads != config.num_key_value_heads,
        "moe_ffn_internal_dim":
            int(config.intermediate_size),
        "moe_expert_num":
            config.moe_num_expert,
        "moe_topk":
            config.moe_topk,
        "is_exp_moe":
            False,
        "share_expert_num":
            int(config.share_expert_num),
        "moe_ffn_has_bias":
            False,
        "model_name":
            "GPT2LMHeadModelMoe" if config.moe_num_expert else "GPT2LMHeadModel",
        "is_meta":
            True,
        "has_k_layernorm":
            config.use_key_layernorm,
        "has_context_layernorm":
            config.use_context_groupnorm,
        "has_attn_bias":
            config.attention_bias,
        "use_rmsnorm":
            True,
        "tokenizer_path":
            tokenizer.name_or_path,  # donot download from huggingface
        "has_mlp_gate":
            True,
        "rope_mode":
            config.rope_scaling['rope_type'],
        "rope_base":
            int(config.rope_theta),
        "rope_scale":
            int(config.rope_scaling['factor']),
        "rope_cut":
            True,
        "rope_cut_head_dim":
            config.rope_scaling["rope_cut_head_dim"],
        "rope_percentage":
            config.rope_scaling["rope_cut_head_dim"] / (config.hidden_size // config.num_attention_heads),
        "window_size":
            config.sliding_window,
        "gqa_weights_layout":
            "AABB",
    }

    return xperf_config


def _get_m8_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import M8Config
    assert isinstance(model_config, M8Config)
    config = model_config
    xperf_config = {
        "dtype":
            "bfloat16",
        "vocab_size":
            config.vocab_size,
        "max_position_embeddings":
            config.max_position_embeddings,
        "embed_dim":
            config.hidden_size,
        "hidden_size":
            config.hidden_size,
        "num_heads":
            config.num_attention_heads,
        "q_head_times":
            config.query_head_scale_factor,
        "num_layers":
            config.num_hidden_layers,
        "num_kv_heads":
            config.num_key_value_heads,
        "has_mqa":
            config.num_attention_heads != config.num_key_value_heads,
        "moe_ffn_internal_dim":
            int(config.intermediate_size),
        "moe_expert_num":
            config.moe_num_expert,
        "moe_topk":
            config.moe_topk,
        "is_exp_moe":
            False,
        "share_expert_num":
            int(config.share_expert_num),
        "moe_ffn_has_bias":
            False,
        "model_name":
            "GPT2LMHeadModelMoe" if config.moe_num_expert else "GPT2LMHeadModel",
        "is_meta":
            True,
        "has_k_layernorm":
            config.use_key_layernorm,
        "has_context_layernorm":
            config.use_context_groupnorm,
        "has_attn_bias":
            config.attention_bias,
        "use_rmsnorm":
            True,
        "tokenizer_path":
            tokenizer.name_or_path,  # donot download from huggingface
        "has_mlp_gate":
            True,
        "rope_mode":
            config.rope_scaling['rope_type'],
        "rope_base":
            int(config.rope_theta),
        "rope_scale":
            int(config.rope_scaling['factor']),
        "rope_cut":
            True,
        "rope_cut_head_dim":
            config.rope_scaling["rope_cut_head_dim"],
        "rope_percentage":
            config.rope_scaling["rope_cut_head_dim"] / (config.hidden_size // config.num_attention_heads),
        "window_size":
            config.sliding_window,
        "gqa_weights_layout":
            "AABB",
        "residual_post_ln_layers": [layer_idx + 1 for layer_idx in config.pre_post_layernorm_layers],
        "kv_mirror_imitated_layers": [layer_idx + 1 for layer_idx in config.kv_mirror_imitated_layers],
        "kv_mirror_layers": [layer_idx + 1 for layer_idx in config.kv_mirror_layers],
    }

    return xperf_config


def _get_vl_p6_xperf_vision_config(vision_config):
    assert vision_config.model_type == 'seed_vision_model'
    xperf_config = {
        "drop_path_rate":
            0,
        "freeze_vit":
            True,
        "img_size":
            448,
        "navit_anyres":
            True,
        "min_pixels":
            3136,
        "max_pixels":
            4014080,
        "multicrop_anyres":
            False,
        "num_img_token":
            256,
        "use_cls_token":
            False,
        "use_grad_checkpoint":
            True,
        "vit_model":
            "seed_vit_2b",
        "vit_precision":
            "bf16",
        "vit_pth":
            "hdfs://haruna/home/byte_data_seed/ssd_lq/iccv/vit/seed_vit_2b_stage0_448.pt",
        "vit_model_path":
            "/opt/tiger/test_m8_vl_xperf/7cf6546654e84dbbde5ec5e59510419f/megatron_ckpt/visual_megatron_states.pt",
        "projector_hidden_dim":
            3072,
        "projector_embed_dim":
            3072,
        "patch_size":
            14
    }
    same_keys = []
    for key in xperf_config:
        if hasattr(vision_config, key):
            same_keys.append(key)
        else:
            print(f'====== "{key}":{xperf_config[key]} not exists in vision config')
    import torch
    xperf_vision_config = {"vit_model": "seed_vit_2b", "navit_anyres": True, "vit_precision": "bf16"}
    vision_keys = [
        'drop_path_rate', 'freeze_vit', 'img_size', 'min_pixels', 'max_pixels', 'multicrop_anyres', 'num_img_token',
        'use_cls_token', 'use_grad_checkpoint', 'projector_hidden_dim', 'projector_embed_dim', 'patch_size'
    ]
    for key in vision_keys:
        xperf_vision_config[key] = getattr(vision_config, key)
    return xperf_vision_config


def _get_vl_m8_xperf_vision_config(vision_config):
    assert vision_config.model_type == 'seed_vision_model'
    vision_keys = [
        "drop_path_rate", "freeze_vit", "img_size", "max_pixels", "min_pixels", "num_img_token", "use_cls_token",
        "use_grad_checkpoint", "use_navit", "projector_hidden_dim", "projector_embed_dim", 'patch_size', "qkv_bias"
    ]
    xperf_vision_config = {"vit_precision": "bf16"}
    xperf_vision_config['vit_model'] = getattr(vision_config, 'vit_model') if hasattr(vision_config,
                                                                                      'vit_model') else None
    for key in vision_keys:
        xperf_vision_config[key] = getattr(vision_config, key)
    if hasattr(vision_config, 'transformer_config'):
        xperf_vision_config['transformer_config'] = getattr(vision_config, 'transformer_config')
        xperf_vision_config['transformer_config']['norm_layer'] = partial(torch.nn.LayerNorm, eps=1e-6)

    # avoid init fail in xperf_gpt
    xperf_vision_config["vit_model_path"] = None
    return xperf_vision_config


def _get_vl_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import SeedVLConfig
    assert isinstance(model_config, SeedVLConfig)
    if model_config.text_config.architectures[0] == "M8ForCausalLM":
        llm_config = _get_m8_xperf_gpt_config(model_config.text_config, tokenizer)
        vision_config = _get_vl_m8_xperf_vision_config(model_config.vision_config)
    elif model_config.text_config.architectures[0] == "P6ForCausalLM":
        llm_config = _get_p6_xperf_gpt_config(model_config.text_config, tokenizer)
        vision_config = _get_vl_p6_xperf_vision_config(model_config.vision_config)
    else:
        raise RuntimeError(f"Unsupported model type {model_config.text_config.architectures[0]}")
    return {"text_config": llm_config, "vision_config": vision_config}


def _get_dsv3_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import DeepseekV3Config
    assert isinstance(model_config, DeepseekV3Config)
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
        "kv_lora_rank": config.kv_lora_rank,
        "q_lora_rank": config.q_lora_rank,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "v_head_dim": config.v_head_dim,
        "ffn_internal_dim": config.intermediate_size,
        "moe_ffn_internal_dim": config.moe_intermediate_size,
        "moe_expert_num": config.n_routed_experts,
        "moe_topk": config.num_experts_per_tok,
        "n_group": config.n_group,
        "topk_group": config.topk_group,
        "share_expert_num": config.n_shared_experts,
        "model_name": "DeepSeekV3Model",
        "is_meta": True,
        "use_flash2": True,
        "has_output_quant": False,
        "has_kv_qscale": False,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_scaling": {
            "beta_fast": config.rope_scaling["beta_fast"],
            "beta_slow": config.rope_scaling["beta_slow"],
            "factor": config.rope_scaling["factor"],
            "mscale": config.rope_scaling["mscale"],
            "mscale_all_dim": config.rope_scaling["mscale_all_dim"],
            "original_max_position_embeddings": config.rope_scaling["original_max_position_embeddings"],
            "type": config.rope_scaling["type"],
        },
        "first_k_dense_replace": config.first_k_dense_replace,
        "rope_theta": config.rope_theta,
        "quant_mode": "NO_QUANT",
        "routed_scaling_factor": config.routed_scaling_factor,
        "weight_block_size": [128, 128]
    }
    return xperf_config


def _get_m10_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import M10Config
    assert isinstance(model_config, M10Config)
    config = model_config
    xperf_config = {
        "dtype":
            "bfloat16",
        "vocab_size":
            config.vocab_size,
        "max_position_embeddings":
            config.max_position_embeddings,
        "embed_dim":
            config.hidden_size,
        "hidden_size":
            config.hidden_size,
        "num_heads":
            config.num_attention_heads,
        "q_head_times":
            config.query_head_scale_factor,
        # "num_layers":
        #     config.num_hidden_layers + config.mtp_n_heads - 1,
        "num_layers":
            config.num_hidden_layers,
        "num_kv_heads":
            config.num_key_value_heads,
        "has_mqa":
            config.num_attention_heads != config.num_key_value_heads,
        "moe_ffn_internal_dim":
            int(config.intermediate_size),
        "moe_expert_num":
            config.moe_num_expert,
        "moe_topk":
            config.moe_topk,
        "is_exp_moe":
            False,
        "share_expert_num":
            int(config.moe_share_expert_num),
        "moe_ffn_has_bias":
            False,
        "model_name":
            "GPT2LMHeadModelMoe" if config.moe_num_expert else "GPT2LMHeadModel",
        "is_meta":
            True,
        "has_k_layernorm":
            config.use_key_norm,
        "has_context_layernorm":
            config.use_context_groupnorm,
        "has_attn_bias":
            config.attention_bias,
        "use_rmsnorm":
            True,
        "tokenizer_path":
            tokenizer.name_or_path,  # donot download from huggingface
        "has_mlp_gate":
            True,
        "rope_mode":
            config.rope_scaling['rope_type'],
        "rope_base":
            int(config.rope_theta),
        "rope_scale":
            int(config.rope_scaling['factor']),
        "rope_cut":
            True,
        "rope_cut_head_dim":
            config.rope_scaling["rope_cut_head_dim"],
        "rope_percentage":
            config.rope_scaling["rope_cut_head_dim"] / (config.hidden_size // config.num_attention_heads),
        "window_size":
            config.sliding_window,
        "gqa_weights_layout":
            "AABB",
        "attn_input_after_norm":
            True,
        "attn_residual_after_norm":
            True,
        "ffn_input_after_norm":
            True,
        "ffn_residual_after_norm":
            False,
        "special_norm_pos_config":
            "_{1:{\"attn_residual_after_norm\":False}}_",
        "mtp_n_heads":
            config.mtp_n_heads,
        "querynorm":
            config.use_query_norm,
        "keynorm":
            config.use_key_norm,
        "valuenorm":
            False,
        "contextnorm":
            config.use_context_groupnorm,
        "attn_outputnorm":
            config.use_attention_output_norm,
        "ffn_outputnorm":
            True,
    }

    return xperf_config


def _get_m11_xperf_gpt_config(model_config, tokenizer: PreTrainedTokenizer):
    from seed_models import M11Config
    assert isinstance(model_config, M11Config)
    config = model_config
    xperf_config = {
        "dtype":
            "bfloat16",
        "embed_dim":
            config.hidden_size,
        "gqa_weights_layout":
            "AABB",
        "has_attn_bias":
            config.attention_bias,
        "has_context_layernorm":
            config.use_context_groupnorm,
        "has_k_layernorm":
            config.use_key_norm,
        "has_mlp_gate":
            True,
        "hidden_size":
            config.hidden_size,
        "is_exp_moe":
            False,
        "is_meta":
            True,
        "kv_mirror_imitated_layers": [],
        "kv_mirror_layers": [],
        "max_position_embeddings":
            config.max_position_embeddings,
        "model_name":
            "GPT2LMHeadModelMoe" if config.moe_num_expert else "GPT2LMHeadModel",
        "moe_expert_num":
            config.moe_num_expert,
        "moe_ffn_has_bias":
            False,
        "moe_ffn_internal_dim":
            int(config.intermediate_size),
        "vocab_size":
            config.vocab_size,
        "num_heads":
            config.num_attention_heads,
        "q_head_times":
            config.query_head_scale_factor,
        # "num_layers":
        #     config.num_hidden_layers + config.mtp_n_heads - 1,
        "num_layers":
            config.num_hidden_layers,
        "num_kv_heads":
            config.num_key_value_heads,
        "has_mqa":
            config.num_attention_heads != config.num_key_value_heads,
        "moe_topk":
            config.moe_topk,
        "share_expert_num":
            int(config.moe_share_expert_num),
        "use_rmsnorm":
            True,
        "tokenizer_path":
            tokenizer.name_or_path,  # donot download from huggingface
        "rope_mode":
            config.rope_scaling['rope_type'],
        "rope_base":
            int(config.rope_theta),
        "rope_scale":
            int(config.rope_scaling['factor']),
        "rope_cut":
            True,
        "rope_cut_head_dim":
            config.rope_scaling["rope_cut_head_dim"],
        "rope_percentage":
            config.rope_scaling["rope_cut_head_dim"] / (config.hidden_size // config.num_attention_heads),
        "window_size":
            config.sliding_window,
        "attn_input_after_norm":
            True,
        "attn_residual_after_norm":
            True,
        "ffn_input_after_norm":
            True,
        "ffn_residual_after_norm":
            False,
        "special_norm_pos_config":
            "_{1:{\"attn_residual_after_norm\":False}}_",
        "mtp_n_heads":
            config.mtp_n_heads,
        "querynorm":
            config.use_query_norm,
        "keynorm":
            config.use_key_norm,
        "valuenorm":
            False,
        "contextnorm":
            config.use_context_groupnorm,
        "attn_outputnorm":
            config.use_attention_output_norm,
        "ffn_outputnorm":
            False,
        "over_enc_embed_dim":
            config.over_enc_embed_dim,
        "over_enc_fuse_all":
            config.over_enc_fuse_all,
        "over_enc_vocab_size":
            config.over_enc_vocab_size,
        "over_enc_vocab_stride":
            config.over_enc_vocab_stride,
        "over_enc_m":
            config.vwn_m,
        "over_enc_n_in":
            config.vwn_n_in,
        "over_enc_n_out":
            config.vwn_n_out
    }

    return xperf_config
