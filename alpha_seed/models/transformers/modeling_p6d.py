# TODO(lijiahao.plus): port this file back to verl

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

import torch
from typing import Tuple
import logging

from transformers.cache_utils import Cache
from typing import Optional

from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import gather_seq_scatter_heads, gather_heads_scatter_seq, gather_outputs

import torch.nn.functional as F

from seed_models.models.p6dense.modeling_p6d import (
    P6DenseFlashAttention2,
    _flash_attention_forward,
    apply_rotary_pos_emb,
)

logger = logging.getLogger(__file__)


def flash_attn2_rmpad_forward(
    self: P6DenseFlashAttention2,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.IntTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    max_seqlen: int = None,
) -> Tuple[torch.Tensor, None, None]:
    assert cu_seqlens is None
    assert not output_attentions
    assert (not past_key_value) and (not use_cache)
    sp_size = get_ulysses_sequence_parallel_world_size()
    bsz, q_len, _ = hidden_states.size()  # q_len = seq_length / sp_size

    query_states = self.q_proj(hidden_states)  # (batch_size, seq_length / sp_size, num_heads * head_size)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if sp_size > 1:
        # (batch_size, num_head / sp_size, seq_length, head_size)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)
        # the position_ids and max_seqlen is required to be global for flash attention
        # TODO: optimize this, no need to allgather at each layer
        position_ids = gather_outputs(position_ids, gather_dim=1)
        max_seqlen = position_ids.max().item() + 1
        assert position_ids.size(1) == query_states.size(2)
    full_q_len = query_states.size(2)  # full_q_len = seq_length

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}.")

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # (batch_size, seq_length, num_head / sp_size, head_size)
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        full_q_len,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
        training=self.training,
        layer_number=self.layer_idx,
        max_seqlen=max_seqlen,
    )

    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()  # rmpad mode has no pad_input
    if sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, None
