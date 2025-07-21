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
from alpha_seed.models.transformers import *
from alpha_seed.models.transformers.ops.memory_efficient_ops import compute_chunked_entropy_logprobs

from seed_models.models.m8.modeling_m8 import (
    apply_rotary_pos_emb,
    repeat_kv,
    KVMirrorManagerHook,
    KVMirrorManager,
    Cache,
    M8FusedMoeBlock,
)
from .modeling_flash_attention_utils import _flash_attention_forward, _flash_supports_window_size

import torch
from torch.distributed._tensor import Shard
from torch.distributed.device_mesh import DeviceMesh

from .parallel.collectives import allreduce_identity, identity_allreduce
from .ops.group_gemm_ep import FusedMoeExpertFunctionEP
from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size

from dist_attn.ulysses.ops import gather_seq_scatter_heads, gather_heads_scatter_seq

import logging

logger = logging.getLogger(__file__)


def make_m8_plan():
    plan = {
        # attention block (TP)
        "k_proj": Shard(0),
        "q_proj": Shard(0),
        "v_proj": Shard(0),
        "o_proj": Shard(1),
        # moe experts (EP)
        "moe.experts.fc1_1": Shard(0),
        "moe.experts.fc1_2": Shard(0),
        "moe.experts.fc2": Shard(0),
        # moe shared experts (TP)
        "moe.experts_share.fc1_1": Shard(0),
        "moe.experts_share.fc1_2": Shard(0),
        "moe.experts_share.fc2": Shard(1),
        # TODO: support Megatron sequence parallelism
    }
    return plan


def make_m8_plan_fsdp2():
    plan = {
        # moe experts (EP)
        "*.moe.experts.fc1_1": Shard(0),
        "*.moe.experts.fc1_2": Shard(0),
        "*.moe.experts.fc2": Shard(0),
        # TODO: support Megatron sequence parallelism
    }
    return plan


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
    gradient_checkpointing: bool = False,
    **kwargs,
):
    assert (not use_cache) and (not past_key_value)
    flash_attn_kwargs = kwargs.pop("flash_attn_kwargs", None)
    assert len(kwargs) == 0
    assert position_embeddings is not None
    assert cu_seqlens is None

    if max_seqlen is None:
        assert flash_attn_kwargs is not None
        # from seed_models.utils.modeling_flash_attention_utils import GPUFlashAttentionKwargs
        # assert isinstance(flash_attn_kwargs, GPUFlashAttentionKwargs)
        # this leads to raise TypeError('TypedDict does not support instance and class checks')
        if 'max_seqlen_q' not in flash_attn_kwargs:
            # this means that the sequence is just a single seq
            max_seqlen = position_ids.max() + 1
        else:
            max_seqlen = flash_attn_kwargs['max_seqlen_q']

    assert max_seqlen is not None
    assert self.q_proj.bias is None and self.k_proj.bias is None and \
           self.v_proj.bias is None and self.o_proj.bias is None
    if position_ids.size(0) != 1:
        raise RuntimeError(f"You are using an old version of seed models, please upgrade to the latest one.")

    tp_mesh: DeviceMesh = self._tp_mesh
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

    if self.config.use_key_layernorm:
        key_states = self.key_layernorm(key_states)
        # in fsdp training mode, the norm will be autocasted to float32
        if key_states.dtype != query_states.dtype:
            key_states = key_states.to(query_states.dtype)

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

    args = [
        key_states,
        value_states,
        self.layer_idx,
        self.kv_mirror_layers,
        self.kv_mirror_imitated_layers,
        query_states.device,
        gradient_checkpointing,
    ]

    is_recent_seed_models = hasattr(self, "is_first_forward_in_recompute")
    if is_recent_seed_models:
        args.append(self.is_first_forward_in_recompute)

    key_states, value_states = KVMirrorManagerHook.apply(*args)

    if is_recent_seed_models and gradient_checkpointing:
        self.is_first_forward_in_recompute = not self.is_first_forward_in_recompute

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

    query_states.stat_meta = {"name": f"layer_{self.layer_idx}.M8FlashAttention2.query_states"}
    key_states.stat_meta = {"name": f"layer_{self.layer_idx}.M8FlashAttention2.key_states"}
    value_states.stat_meta = {"name": f"layer_{self.layer_idx}.M8FlashAttention2.value_states"}
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
    attn_output = self.resid_dropout(attn_output)
    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _fused_moe_ep_forward(
    self: M8FusedMoeBlock,
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
        self.experts.fc1_1,
        self.experts.fc1_2,
        self.experts.fc2,
        ep_group,
    )

    # MOE Step 3: compute with shared experts
    if ep_size > 1:
        hidden_states = identity_allreduce(hidden_states, group=ep_group, name="ep-iar")

    experts_share_states = self.experts_share(hidden_states)

    if ep_size > 1:
        experts_share_states = allreduce_identity(experts_share_states, group=ep_group, name="ep-ari")

    final_hidden_states = final_hidden_states + experts_share_states

    # reshape output to input shape
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits, aux_loss


def release_m8_kv_mirror(self):
    KVMirrorManager.activations_dict.clear()
    KVMirrorManager.activations_grad_dict.clear()


def m8_casual_lm_forward(
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
    compute_entropy: Optional[bool] = False,
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
    loss = entropy = logits = log_probs = None
    hidden_states = outputs[0]
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
    else:
        # loss computation is skipped
        entropy, log_probs = compute_chunked_entropy_logprobs(self, hidden_states, labels, temperature, compute_entropy)

    aux_loss = None
    if output_aux_losses:
        # aux_loss: (Union[`torch.Tensor`, Tuple[torch.Tensor]), should be a tuple of model.config.num_hidden_layers tensors of aux_loss
        aux_losses = outputs.aux_losses
        compute_device = aux_losses[0].device
        aux_loss = sum(layer_aux_loss.to(compute_device) for layer_aux_loss in aux_losses)

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
                                              log_probs=log_probs)
