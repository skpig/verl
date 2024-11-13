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
import warnings
from typing import Tuple
import logging

from transformers.models.qwen2.modeling_qwen2 import Cache
import torch.distributed as dist

from transformers.cache_utils import Cache
from typing import Optional

from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import gather_seq_scatter_heads, gather_heads_scatter_seq, gather_outputs

import torch.nn.functional as F

logger = logging.getLogger(__file__)


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


def flash_attn2_rmpad_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.IntTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    max_seqlen: int = None,
    **kwargs,
):
    assert (past_key_value is None) and (not use_cache)
    assert cu_seqlens is None
    from seed_models.models.p6.modeling_p6 import (apply_rotary_pos_emb, _flash_attention_forward,
                                                   _flash_supports_window_size, repeat_kv)
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")
    sp_size = get_ulysses_sequence_parallel_world_size()
    bsz, q_len, _ = hidden_states.size()  # q_len = seqlen/sp

    query_states = self.q_proj(hidden_states)  # bsz, seqlen/sp, hidden
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # bsz, nhead, seqlen/sp, hdim
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if cu_seqlens is None:
        kv_seq_len = key_states.shape[-2]
    else:
        kv_seq_len = cu_seqlens.diff().max().item()

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    # bsz, nhead, seqlen/sp, hdim
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    use_sliding_windows = (_flash_supports_window_size and getattr(self.config, "sliding_window", None) is not None and
                           kv_seq_len > self.config.sliding_window[self.layer_idx])

    if not _flash_supports_window_size:
        logger.warning_once(
            "The current flash attention version does not support sliding window attention, for a more memory"
            " efficient implementation make sure to upgrade flash-attn library.")

    if self.config.use_key_layernorm:
        key_states = self.key_layernorm(key_states)
        # in fsdp training mode, the norm will be autocasted to float32
        if key_states.dtype != query_states.dtype:
            key_states = key_states.to(query_states.dtype)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups, self.interleaved_kv_shared)
    value_states = repeat_kv(value_states, self.num_key_value_groups, self.interleaved_kv_shared)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # ulysses region
    # [bsz, nhead, seqlen/sp, hdim] -> [bsz, nhead/sp, seqlen, ,hdim]
    if sp_size > 1:
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)
        # the position_ids and max_seqlen is required to be global for flash attention
        # TODO: optimize this, no need to allgather at each layer
        position_ids = gather_outputs(position_ids, gather_dim=1)
        max_seqlen = position_ids.max().item() + 1
    full_qlen = query_states.size(2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
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

    # Reashape to the expected shape for Flash Attention
    # [bsz, seqlen, nhead/sp, hdim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length=full_qlen,
        position_ids=position_ids,
        is_causal=self.is_causal,
        dropout=dropout_rate,
        cu_seqlens=cu_seqlens,
        sliding_window=self.config.sliding_window[self.layer_idx] if use_sliding_windows else None,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        training=self.training,
        layer_number=self.layer_idx,
        max_seqlen=max_seqlen,
    )

    # [bsz, seqlen, nhead/sp, hdim] -> [bsz, seqlen/sp, nhead, hdim]
    attn_output = attn_output.reshape(bsz, full_qlen, -1, self.head_dim).contiguous()

    if sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    if self.config.use_context_groupnorm:
        attn_output = self.context_norm(attn_output)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
