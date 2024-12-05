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
import warnings
from typing import Tuple
import logging

from transformers.models.qwen2.modeling_qwen2 import Cache
import torch.distributed as dist

from transformers.cache_utils import Cache

from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import gather_seq_scatter_heads, gather_heads_scatter_seq, gather_outputs

import torch.nn.functional as F

from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

logger = logging.getLogger(__file__)

from .modeling_flash_attention_utils import _flash_attention_forward, _flash_supports_window_size

logger = logging.getLogger(__file__)

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
    from seed_models.models.p6.modeling_p6 import (apply_rotary_pos_emb, repeat_kv)
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


def p6_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.IntTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    output_aux_losses: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    fuse_lm_head_ce_loss: Optional[bool] = None,
    temperature: Optional[float] = None,
) -> Union[Tuple, MoeCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        fuse_lm_head_ce_loss: (`bool`, *optional*):
            Whether to fuse the loss computation of lm_head and cross entropy loss.
        temperature: (`float`, *optional*):
            Temperature for softmax CE loss. Only effective if fuse_lm_head_ce_loss is True.
    Returns:

    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_router_logits = (output_router_logits
                            if output_router_logits is not None else self.config.output_router_logits)
    output_aux_losses = output_aux_losses if output_aux_losses is not None else self.config.output_aux_losses

    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        output_aux_losses=output_aux_losses,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    if fuse_lm_head_ce_loss:
        assert labels is not None
        try:
            from bumi.function.flash_cross_entropy import FlashCrossEntropy
        except ImportError:
            scm = 'pip3 install http://luban-source.byted.org/repository/scm/seed.speech.bumi_1.7.0.0.tar.gz'
            raise ImportError(f"Please install bumi via {scm}")
        if temperature is not None:
            hidden_states.div_(temperature)
        # 2 means recompute the logits in the backward pass
        recompute_level = 2
        hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        # TODO(haibin.lin): accuracy metric is not necessarily needed, to be optimized further
        compute_accuracy = True
        # this gives better precision alignment with the torch implementation, with potentially lower precision with bf16 casts
        align_precision = True
        loss, _ = FlashCrossEntropy.apply(hidden_states_2d.bfloat16(), self.lm_head.weight, labels, recompute_level,
                                          compute_accuracy, align_precision)
        logits = None
    else:
        logits = self.lm_head(hidden_states)
        assert temperature is None
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            if cu_seqlens is not None:
                # Mask the last token of each sequence to torch.CrossEntropyLoss ignore_index, default is -100
                shift_labels[cu_seqlens[1:-1] - 1] = -100
            elif position_ids is not None:
                position_ids_ = position_ids.flatten()
                indices_q = torch.arange(position_ids_.size(0), device=position_ids_.device, dtype=torch.int32)
                cu_seq_lens = torch.cat((
                    indices_q[position_ids_ == 0],
                    torch.tensor(position_ids_.size(), device=position_ids_.device, dtype=torch.int32),
                ))
                shift_labels[cu_seq_lens[1:-1] - 1] = -100

            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fct(shift_logits, shift_labels)

    aux_loss = None
    if output_aux_losses:
        # aux_loss: (Union[`torch.Tensor`, Tuple[torch.Tensor]), should be a tuple of model.config.num_hidden_layers tensors of aux_loss
        aux_losses = outputs.aux_losses
        compute_device = aux_losses[0].device
        aux_loss = sum(layer_aux_loss.to(compute_device) for layer_aux_loss in aux_losses)

        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

    if not return_dict:
        output = (logits,) + outputs[1:]
        if output_aux_losses:
            output = (aux_loss,) + output
        return (loss,) + output if loss is not None else output

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )
