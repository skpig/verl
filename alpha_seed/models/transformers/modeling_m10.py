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

import os
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from bumi.function.flash_cross_entropy import FlashCrossEntropy

import seed_models
from seed_models.models.m10.modeling_m10 import (apply_rotary_pos_emb, repeat_kv, Cache, M10FusedMoeBlock,
                                                 M10FlashAttention2, M10ForCausalLM)
from .modeling_flash_attention_utils import _flash_attention_forward, _flash_supports_window_size

from alpha_seed.models.transformers.ops.memory_efficient_ops import (compute_chunked_entropy_logprobs,
                                                                     compute_chunked_mtp_acceptance_ratio)
from alpha_seed.models.transformers import AlphaSeedMoeCausalLMOutputWithPast

import torch
from torch.distributed._tensor import Shard
from torch.distributed.device_mesh import DeviceMesh

from .parallel.collectives import allreduce_identity, identity_allreduce
from .ops.group_gemm_ep import FusedMoeExpertFunctionEP
from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size

from dist_attn.ulysses.ops import gather_seq_scatter_heads, gather_heads_scatter_seq

from .utils import _check_version

import logging

logger = logging.getLogger(__file__)


def make_m10_plan():
    plan = {
        # attention block (TP)
        "k_proj": Shard(0),
        "q_proj": Shard(0),
        "v_proj": Shard(0),
        "o_proj": Shard(1),
        # moe experts (EP)
        "moe.experts.gate_proj": Shard(0),
        "moe.experts.up_proj": Shard(0),
        "moe.experts.down_proj": Shard(0),
        # moe shared experts (TP)
        "moe.shared_experts.gate_proj": Shard(0),
        "moe.shared_experts.up_proj": Shard(0),
        "moe.shared_experts.down_proj": Shard(1),
        # TODO: support Megatron sequence parallelism
    }
    return plan


def make_m10_plan_fsdp2():
    _check_version()
    plan = {
        # moe experts (EP)
        "*.moe.experts.gate_proj": Shard(0),
        "*.moe.experts.up_proj": Shard(0),
        "*.moe.experts.down_proj": Shard(0),
    }
    return plan



def flash_attn2_rmpad_forward(
    self: M10FlashAttention2,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.IntTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    max_seqlen: int = None,
    use_tp = True,
    **kwargs,
):
    assert (not use_cache) and (not past_key_value)
    kwargs.pop("flash_attn_kwargs", None)
    kwargs.pop("gradient_checkpointing", None)
    assert len(kwargs) == 0, f"kwargs should be empty. But Got {kwargs.keys()}"
    assert position_embeddings is not None
    assert cu_seqlens is None
    assert max_seqlen is not None
    assert self.q_proj.bias is None and self.k_proj.bias is None and \
           self.v_proj.bias is None and self.o_proj.bias is None
    if position_ids.size(0) != 1:
        raise RuntimeError(f"You are using an old version of seed models, please upgrade to the latest one.")

    if use_tp:
        tp_mesh: DeviceMesh = self._tp_mesh
    else:
        tp_mesh = None

    tp_size = 1 if tp_mesh is None else tp_mesh.size()
    tp_group = None if tp_mesh is None else tp_mesh.get_group()
    sp_size = get_ulysses_sequence_parallel_world_size()
    assert self.num_query_heads % (tp_size * sp_size) == 0

    # ============== tensor parallel region ================
    if tp_size > 1:
        hidden_states = identity_allreduce(hidden_states, tp_group, "tp-iar")
    # ============== tensor parallel region ================

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # =============== ulysses sp region ==================
    if sp_size > 1:
        local_kv_heads = key_states.size(1)
        if sp_size > local_kv_heads:
            assert sp_size % local_kv_heads == 0
            n_repeat = sp_size // local_kv_heads
            key_states = repeat_kv(key_states, n_repeat)
            value_states = repeat_kv(value_states, n_repeat)

        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)
    full_qlen = query_states.size(2)
    # =============== ulysses sp region ==================

    dtype = query_states.dtype
    if self.config.use_query_norm:
        query_states = self.q_norm(query_states)
        # in fsdp training mode, the norm will be autocasted to float32
        if query_states.dtype != dtype:
            query_states = query_states.to(dtype)

    if self.config.use_key_norm:
        key_states = self.k_norm(key_states)
        # in fsdp training mode, the norm will be autocasted to float32
        if key_states.dtype != dtype:
            key_states = key_states.to(dtype)

    if self.rope_cut:
        query_nope_states, query_states = torch.split(query_states,
                                                      [self.head_dim - self.rope_cut_head_dim, self.rope_cut_head_dim],
                                                      dim=-1)
        key_nope_states, key_states = torch.split(key_states,
                                                  [self.head_dim - self.rope_cut_head_dim, self.rope_cut_head_dim],
                                                  dim=-1)

    kv_seq_len = key_states.shape[-2]
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if self.rope_cut:
        query_states = torch.concat([query_nope_states, query_states], dim=-1)
        key_states = torch.concat([key_nope_states, key_states], dim=-1)

    use_sliding_windows = (_flash_supports_window_size and getattr(self.config, "sliding_window", None) is not None and
                           kv_seq_len > self.config.sliding_window[self.layer_idx])

    if not _flash_supports_window_size:
        logger.warning_once(
            "The current flash attention version does not support sliding window attention, for a more memory"
            " efficient implementation make sure to upgrade flash-attn library.")

    dropout_rate = 0.0 if not self.training else self.attention_dropout

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

    query_states.stat_meta = {"name": f"layer_{self.layer_idx}.M10FlashAttention2.query_states"}
    key_states.stat_meta = {"name": f"layer_{self.layer_idx}.M10FlashAttention2.key_states"}
    value_states.stat_meta = {"name": f"layer_{self.layer_idx}.M10FlashAttention2.value_states"}
    # Reashape to the expected shape for Flash Attention
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
        varlen=True,
    )

    if self.config.use_context_groupnorm:
        attn_output = self.context_norm(attn_output)
    # ============== ulysses sp region ===================
    if sp_size > 1:
        attn_output = attn_output.reshape(bsz, full_qlen, -1, self.head_dim).contiguous()
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)
    # ============== ulysses sp region ===================
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    # ============== tensor parallel region ================
    if tp_size > 1:
        attn_output = allreduce_identity(attn_output, tp_group, "tp-ari")
    # ============== tensor parallel region ================

    if self.config.use_attention_output_norm:
        attn_output = self.o_norm(attn_output)

    attn_output = self.resid_dropout(attn_output)
    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def flash_attn2_rmpad_forward_fsdp2(
    self: M10FlashAttention2,
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
    return flash_attn2_rmpad_forward(self,
                                     hidden_states,
                                     attention_mask,
                                     position_ids,
                                     cu_seqlens,
                                     past_key_value,
                                     output_attentions,
                                     use_cache,
                                     position_embeddings,
                                     max_seqlen,
                                     use_tp=False,
                                     **kwargs)


def _fused_moe_ep_forward(
    self: M10FusedMoeBlock,
    hidden_states: torch.Tensor,
    output_aux_losses: Optional[bool] = None,
):
    """Patched moe forward function to support EP"""

    ep_mesh: DeviceMesh = self._tp_mesh
    ep_group = None if ep_mesh is None else ep_mesh.get_group()
    ep_size = 1 if ep_mesh is None else ep_mesh.size()

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # MOE Step 1: compute each token's weight for all experts.
    # router_logits shape (batch_size * sequence_len, num_experts)
    # router_weights shape: (batch_size * sequence_len, topk)
    routing_weights, router_logits, aux_loss, _, selected_experts = self.gate(hidden_states, output_aux_losses)

    # MOE Step 2: compute experts with group gemm.
    routing_weights = routing_weights.bfloat16()
    hidden_states = hidden_states.bfloat16()
    final_hidden_states, handle = FusedMoeExpertFunctionEP.apply(
        self.num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        self.experts.gate_proj,
        self.experts.up_proj,
        self.experts.down_proj,
        ep_group,
    )

    # MOE Step 3: compute with shared experts
    if ep_size > 1:
        hidden_states = identity_allreduce(hidden_states, group=ep_group, name="ep-iar")

    experts_share_states = self.shared_experts(hidden_states)

    if ep_size > 1:
        experts_share_states = allreduce_identity(experts_share_states, group=ep_group, name="ep-ari")

    final_hidden_states = final_hidden_states + experts_share_states

    # reshape output to input shape
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits, aux_loss


from seed_kernels.transformers.functional import seed_fused_moe


def fused_moe_block_forward(
    self: M10FusedMoeBlock,
    hidden_states: torch.Tensor,
    output_aux_losses: Optional[bool] = None,
) -> torch.Tensor:

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # MOE Step 1: compute each token's weight for all experts.
    # router_logits shape (batch_size * sequence_len, num_experts)
    routing_weights, router_logits, aux_loss, _, selected_experts = self.gate(hidden_states, output_aux_losses)

    # MOE Step 2: compute experts with group gemm + shared experts.
    final_hidden_states = seed_fused_moe(
        self.num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        self.experts.gate_proj,
        self.experts.up_proj,
        self.experts.down_proj,
        ep_group=self.experts.gate_proj.mesh.get_group(),
        shared_fc1_1_weight=self.shared_experts.gate_proj,
        shared_fc1_2_weight=self.shared_experts.up_proj,
        shared_fc2_weight=self.shared_experts.down_proj,
        ep_implementation="bumi",
    )

    # reshape output to input shape
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    return final_hidden_states, router_logits, aux_loss


from seed_models.models.m10.modeling_m10 import M10Model, M10MoeModelOutput
from dist_attn.ulysses.ops import gather_outputs, slice_input_tensor


def m10_model_forward(
    self: M10Model,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.IntTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    output_aux_losses: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_router_logits = (output_router_logits
                            if output_router_logits is not None else self.config.output_router_logits)
    output_aux_losses = output_aux_losses if output_aux_losses is not None else self.config.output_aux_losses
    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        if (cu_seqlens is not None or position_ids is not None) and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        if (cu_seqlens is not None or position_ids is not None) and inputs_embeds.dim() == 1:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    max_seqlen = None
    assert position_ids is not None
    position_ids = position_ids.view(batch_size, -1).long()
    max_seqlen = position_ids.max().item() + 1

    # here, we assume the input_ids are spliited in sequence parallel

    # allgather from sequence parallel region
    input_ids_full = gather_outputs(input_ids, gather_dim=1, padding_dim=0, unpad_dim_size=0)  # (1, total_nnz)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids_full)

    inputs_embeds = self.embd_dropout(inputs_embeds)

    # roll inputs_embeds for mtp_heads and slice into sequence parallel region
    mtp_embeds_lst = []
    for mtp_idx in range(1, self.mtp_n_heads):
        mtp_embed = torch.roll(inputs_embeds, shifts=-mtp_idx, dims=-2)  # (1, total_nnz, hidden_size)
        # slice into sequence parallel region first to avoid peak memory usage
        mtp_embed = slice_input_tensor(mtp_embed, dim=1, padding=False)
        mtp_embeds_lst.append(mtp_embed)

    # slice inputs_embeds
    inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, padding=False)

    if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError("You are attempting to perform batched generation with padding_side='right'"
                             " this may lead to unexpected behaviour for Flash Attention version of M10. Make sure to "
                             " call `tokenizer.padding_side  = 'left'` before tokenizing the input. ")

    assert self._attn_implementation in [
        "flash_attention_2",
        "eager",
        "native-sparse",
    ], "Only support flash_attention_2 and eager implementation for M10"
    assert cu_seqlens is None or (self._attn_implementation == "flash_attention_2" or self._attn_implementation
                                  == "native-sparse"), "`seqlens` is only supported in flash_attention_2"

    assert self._attn_implementation == 'flash_attention_2'

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_embedding(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_router_logits = () if output_router_logits else None
    all_aux_losses = () if output_aux_losses else None
    next_decoder_cache = None

    # mtp hidden states
    all_mtp_hidden_states = ()

    for i in range(self.config.num_hidden_layers - (self.mtp_n_heads - 1)):
        decoder_layer = self.model["layers"][i]
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                cu_seqlens,
                past_key_values,
                output_attentions,
                output_router_logits,
                output_aux_losses,
                use_cache,
                position_embeddings,
                max_seqlen,
                True,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                output_aux_losses=output_aux_losses,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                max_seqlen=max_seqlen,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_router_logits:
            all_router_logits += (layer_outputs[-2 if output_aux_losses else -1],)

        if output_aux_losses:
            all_aux_losses += (layer_outputs[-1],)

    if self.mtp_mode is not None:
        for mtp_idx in range(1, self.mtp_n_heads):
            decoder_layer = self.model["layers"][self.config.num_hidden_layers + mtp_idx - self.mtp_n_heads]
            mtp_embs = self.model["mtp_embs"][mtp_idx]
            mtp_ce_norms = self.model["mtp_ce_norms"][mtp_idx]

            # mtp embedding projection
            """
            input_embeds [1, seqlen, hidden_dim]
            roll so that tokens < n predict n
            input_ids = [a,b,c,d,e,f,g] cu_seqlens = [0,3,7]
            mtp_input_ids 1 = [b,c,_,e,f,g,_] cu_seqlens = [0,3,7]
            mtp_input_ids 2 = [c,_,_,f,g,_,_] cu_seqlens = [0,3,7]
            """

            mtp_embeds = mtp_embeds_lst[mtp_idx - 1]
            mtp_hidden_states = mtp_embs(hidden_states, mtp_embeds)

            mtp_outputs = decoder_layer(
                mtp_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                output_aux_losses=output_aux_losses,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                max_seqlen=max_seqlen,
            )

            mtp_hidden_states = mtp_outputs[0]

            # mtp head norm
            mtp_hidden_states = mtp_ce_norms(mtp_hidden_states)

            all_mtp_hidden_states += (mtp_hidden_states,)

            if output_attentions:
                all_self_attns += (mtp_outputs[1],)

            if output_router_logits:
                all_router_logits += (mtp_outputs[-2 if output_aux_losses else -1],)

            if output_aux_losses:
                all_aux_losses += (mtp_outputs[-1],)

    if self.mtp_mode is not None:
        hidden_states = self.model["mtp_ce_norms"][0](hidden_states)
    else:
        hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return M10MoeModelOutput(
        last_hidden_state=hidden_states,
        mtp_hidden_states=all_mtp_hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_logits,
        aux_losses=all_aux_losses,
    )


def m10_casual_lm_forward(self: M10ForCausalLM,
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
                          compute_entropy: Optional[bool] = False,
                          mtp_labels: Optional[torch.LongTensor] = None,
                          **kwargs) -> Union[Tuple, MoeCausalLMOutputWithPast]:
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

    loss = entropy = log_probs = None
    if fuse_lm_head_ce_loss:
        assert labels is not None
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

        # this is training mode. TODO(zhangchi.usc1992) add mtp head training
        acceptance_matrix = None

    else:
        # loss computation is skipped
        entropy, log_probs = compute_chunked_entropy_logprobs(self, hidden_states, labels, temperature, compute_entropy)
        logits = None

        # this is pure inference mode
        # we compute mtp acceptance ratio here
        acceptance_matrix = compute_chunked_mtp_acceptance_ratio(self,
                                                                 all_mtp_hidden_states=outputs.mtp_hidden_states,
                                                                 all_mtp_labels=mtp_labels)

    # add mtp here
    assert len(mtp_labels) == self.config.mtp_n_heads - 1
    # compute mtp log_probs

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

    return AlphaSeedMoeCausalLMOutputWithPast(loss=loss,
                                              aux_loss=aux_loss,
                                              logits=logits,
                                              past_key_values=outputs.past_key_values,
                                              hidden_states=outputs.hidden_states,
                                              attentions=outputs.attentions,
                                              router_logits=outputs.router_logits,
                                              entropy=entropy,
                                              log_probs=log_probs,
                                              acceptance_matrix=acceptance_matrix)
