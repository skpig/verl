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

import inspect
import torch

from transformers.models.qwen2.modeling_qwen2 import Cache

from transformers.cache_utils import Cache
from typing import Optional

from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, rearrange
from flash_attn.layers.rotary import apply_rotary_emb
from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import gather_seq_scatter_heads, gather_heads_scatter_seq

import torch.nn.functional as F


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# use flash-attn rotary embeddings with rmpad
# cos/sin shoudl be: (seq_length, rotary_dim / 2)
def apply_rotary_pos_emb_rmpad_flash(q, k, cos, sin, cu_seqlens, max_seqlen):
    q_embed = apply_rotary_emb(q,
                               cos,
                               sin,
                               interleaved=False,
                               inplace=False,
                               cu_seqlens=cu_seqlens,
                               max_seqlen=max_seqlen)
    k_embed = apply_rotary_emb(k,
                               cos,
                               sin,
                               interleaved=False,
                               inplace=False,
                               cu_seqlens=cu_seqlens,
                               max_seqlen=max_seqlen)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int, interleaved_kv_shared: bool = True) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    total_nnz, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    if interleaved_kv_shared:
        hidden_states = hidden_states[:, None, :, :].expand(total_nnz, n_rep, num_key_value_heads, head_dim)
    else:
        hidden_states = hidden_states[:, :, None, :].expand(total_nnz, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(total_nnz, num_key_value_heads * n_rep, head_dim)


def flash_attn2_rmpad_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs  # for compatibility
):
    assert past_key_value is None
    assert not output_attentions
    assert not use_cache

    bsz, total_nnz, _ = hidden_states.size()
    assert bsz == 1

    hidden_states = hidden_states.squeeze(0)  # (total_nnz, hidden_size)

    # this matches with hidden_states
    indices, cu_seqlens, max_seqlen_in_batch = _get_unpad_data(attention_mask)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(total_nnz, self.num_heads, self.head_dim)
    key_states = key_states.view(total_nnz, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(total_nnz, self.num_key_value_heads, self.head_dim)

    # repeat k/v heads if n_kv_heads < n_heads
    interleaved_kv_shared = getattr(self.config, 'interleaved_kv_shared', False)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    if get_ulysses_sequence_parallel_world_size() > 1:
        key_states = repeat_kv(key_states, self.num_key_value_groups, interleaved_kv_shared)
        value_states = repeat_kv(value_states, self.num_key_value_groups, interleaved_kv_shared)
        # TODO: can we perform all2all before repeat_kv to reduce communication? need to handle interleave/non-interleave kv scenarios
        qkv_states = torch.stack((query_states, key_states, value_states), dim=0)
        qkv_states = gather_seq_scatter_heads(qkv_states, seq_dim=1, head_dim=2)
        query_states, key_states, value_states = qkv_states[0], qkv_states[1], qkv_states[2]

    # note: there are two ways to implement rotary_emb, one that returns [seqlen, rotary_dim].
    # another directly returns [bsz, seqlen, rotary_dim] for each token

    cos, sin = self.rotary_emb(value_states, seq_len=max_seqlen_in_batch)
    cos, sin = cos[:, :cos.shape[1] // 2], sin[:, :sin.shape[1] // 2]  # flash attn only needs half
    query_states, key_states = apply_rotary_pos_emb_rmpad_flash(query_states, key_states, cos, sin, cu_seqlens,
                                                                max_seqlen_in_batch)

    if hasattr(self.config, 'use_key_layernorm') and self.config.use_key_layernorm:
        key_states = self.key_layernorm(key_states)
        # in fsdp training mode, the norm will be autocasted to float32
        if key_states.dtype != query_states.dtype:
            key_states = key_states.to(query_states.dtype)

    if get_ulysses_sequence_parallel_world_size() <= 1:
        key_states = repeat_kv(key_states, self.num_key_value_groups, interleaved_kv_shared)
        value_states = repeat_kv(value_states, self.num_key_value_groups, interleaved_kv_shared)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        assert torch.is_autocast_enabled()
        target_dtype = torch.get_autocast_gpu_dtype()
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    use_sliding_windows = (getattr(self.config, "sliding_window", None) is not None and
                           max_seqlen_in_batch > self.config.sliding_window[self.layer_idx])
    if use_sliding_windows:
        window_size = (self.config.sliding_window[self.layer_idx], self.config.sliding_window[self.layer_idx])
    else:
        window_size = (-1, -1)

    attn_output = flash_attn_varlen_func(query_states,
                                         key_states,
                                         value_states,
                                         cu_seqlens_q=cu_seqlens,
                                         cu_seqlens_k=cu_seqlens,
                                         max_seqlen_q=max_seqlen_in_batch,
                                         max_seqlen_k=max_seqlen_in_batch,
                                         dropout_p=dropout_rate,
                                         softmax_scale=None,
                                         causal=True,
                                         window_size=window_size)

    if get_ulysses_sequence_parallel_world_size() > 1:
        attn_output = gather_heads_scatter_seq(attn_output.unsqueeze(0), seq_dim=1, head_dim=2).squeeze(0)

    if hasattr(self.config, 'use_context_groupnorm') and self.config.use_context_groupnorm:
        attn_output = self.context_norm(attn_output)

    # TODO: adapt for TP
    attn_output = attn_output.reshape(total_nnz, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    attn_output = attn_output.unsqueeze(0)

    return attn_output, None, None
