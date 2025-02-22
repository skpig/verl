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

import os
from typing import Optional, Tuple, Union, List

import seed_models
import torch
import torch.nn.functional as F

from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from bumi.function.flash_cross_entropy import FlashCrossEntropy

from seed_models.models.deepseek_v3.modeling_deepseek import (
    DeepseekV3FusedMoE,
    DeepseekV3FlashAttention2,
    apply_rotary_pos_emb,
    _flash_attention_forward,
    repeat_kv,
)

from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import gather_seq_scatter_heads, gather_heads_scatter_seq


def make_dsv3_plan():
    plan = {}
    return plan


def flash_attn2_forward(
    self: DeepseekV3FlashAttention2,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    assert "padding_mask" not in kwargs
    assert past_key_value is None
    assert not output_attentions
    assert not use_cache

    sp_size = get_ulysses_sequence_parallel_world_size()
    assert self.num_heads % sp_size == 0

    # [bsz, qlen, hidden]
    bsz, q_len, _ = hidden_states.size()

    assert self.q_lora_rank is not None
    # [bsz, qlen, hidden] -> [bsz, qlen, q_lora_rank]
    # [bsz, qlen, q_lora_rank] -> [bsz, qlen, q_lora_rank]
    # [bsz, qlen, q_lora_rank] -> [bsz, qlen, head * qhdim]
    q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q.stat_meta = {"name": f"layer_{self.layer_idx}.attn.q"}
    # [bsz, qlen, head * qhdim] -> [bsz, head, qlen, qhdim]
    q = q.view(bsz, q_len, -1, self.q_head_dim).transpose(1, 2)

    # =============== ulysses sp region ==================
    if sp_size > 1:
        q = gather_seq_scatter_heads(q, seq_dim=2, head_dim=1)
    full_qlen = q.size(2)
    # =============== ulysses sp region ==================

    # [bsz, head, qlen, qhdim]
    #   -> [bsz, head, qlen, qk_nope_head_dim]
    #   -> [bsz, head, qlen, qk_rope_head_dim]
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    # [bsz, qlen, hidden] -> [bsz, qlen, kv_lora_rank + qk_rope_head_dim (512+64)]
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    # [bsz, qlen, kv_lora_rank + qk_rope_head_dim (512+64)]
    #   -> [bsz, qlen, kv_lora_rank]
    #   -> [bsz, qlen, qk_rope_head_dim]
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    # [bsz, qlen, qk_rope_head_dim (64)] -> [bsz, 1, qlen, qk_rope_head_dim (64)]
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

    # =============== ulysses sp region ==================
    if sp_size > 1:
        k_pe = repeat_kv(k_pe, sp_size)
        k_pe = gather_seq_scatter_heads(k_pe, seq_dim=2, head_dim=1)
    # =============== ulysses sp region ==================

    # [bsz, qlen, kv_lora_rank] -> [bsz, head, qlen, qk_node_head_dim + v_head_dim]
    kv = (self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(bsz, q_len, -1,
                                                                  self.qk_nope_head_dim + self.v_head_dim).transpose(
                                                                      1, 2))

    # =============== ulysses sp region ==================
    if sp_size > 1:
        kv = gather_seq_scatter_heads(kv, seq_dim=2, head_dim=1)
    # =============== ulysses sp region ==================

    # [bsz, 1, qlen, qk_node_head_dim + v_head_dim]
    #  -> [bsz, head, qlen, qk_node_head_dim]
    #  -> [bsz, head, qlen, v_head_dim]
    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    kv_seq_len = value_states.shape[-2]
    # assert full_q_len == kv_seq_len

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)  # position_ids

    # [bsz, head, qlen, qhdim]
    local_heads = q_pe.size(1)
    query_states = k_pe.new_empty(bsz, local_heads, kv_seq_len, self.q_head_dim)  # q_len
    query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

    key_states = k_pe.new_empty(bsz, local_heads, kv_seq_len, self.q_head_dim)  # q_len
    key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

    if self.q_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (DeepseekV3RMSNorm handles it correctly)
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        # Handle the case where the model is quantized
        if hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        elif torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = (self.q_proj.weight.dtype if self.q_lora_rank is None else self.q_a_proj.weight.dtype)
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length=full_qlen,
        position_ids=position_ids,
        is_causal=self.is_causal,
        dropout=dropout_rate,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        softmax_scale=self.softmax_scale,
        training=self.training,
    )
    if self.q_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, :self.v_head_dim]

    attn_output = attn_output.reshape(bsz, full_qlen, -1).contiguous()
    # assert attn_output.size(-1) == self.num_heads // sp_size * self.v_head_dim

    # =============== ulysses sp region ==================
    if sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)
    # =============== ulysses sp region ==================

    attn_output = self.o_proj(attn_output)

    return attn_output, None, None
