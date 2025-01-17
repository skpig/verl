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
    apply_rotary_pos_emb,
)
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_flash_attention_utils import _flash_attention_forward

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
    if sp_size > 1:
        if position_ids.size(0) != 1:
            raise RuntimeError(f"You are using an old version of seed models, please upgrade to the latest one.")
    bsz, q_len, _ = hidden_states.size()  # q_len = seq_length / sp_size

    query_states = self.q_proj(hidden_states)  # (batch_size, seq_length / sp_size, num_heads * head_size)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if sp_size > 1:
        # (batch_size, num_head / sp_size, seq_length, head_size)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)
        assert position_ids.size(1) == query_states.size(2), \
            f"got seqlen mismatches: {query_states.size(2)} != {position_ids.size(1)}"
    full_q_len = query_states.size(2)  # full_q_len = seq_length

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states.stat_meta = {"name": f"layer_{self.layer_idx}.P6DenseFlashAttention2.query_states"}
    key_states.stat_meta = {"name": f"layer_{self.layer_idx}.P6DenseFlashAttention2.key_states"}
    value_states.stat_meta = {"name": f"layer_{self.layer_idx}.P6DenseFlashAttention2.value_states"}

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


def p6d_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.IntTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    fuse_lm_head_ce_loss: Optional[bool] = None,
    temperature: Optional[float] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, P6DenseForCausalLM

    >>> model = P6DenseForCausalLM.from_pretrained("path/to/P6Dense")
    >>> tokenizer = AutoTokenizer.from_pretrained("path/to/P6Dense")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
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
            hidden_states = hidden_states / temperature
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
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
