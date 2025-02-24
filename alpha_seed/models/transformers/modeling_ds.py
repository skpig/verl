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

from transformers.modeling_outputs import CausalLMOutputWithPast
from bumi.function.flash_cross_entropy import FlashCrossEntropy

from seed_models.models.deepseek_v3.modeling_deepseek import (
    DeepseekV3MLP,
    DeepseekV3FusedMoE,
    DeepseekV3FlashAttention2,
    DeepseekV3ForCausalLM,
    apply_rotary_pos_emb,
    _flash_attention_forward,
    repeat_kv,
)

from torch.distributed._tensor import Shard
from torch.distributed.device_mesh import DeviceMesh
from .parallel.collectives import allreduce_identity, identity_allreduce
from .ops.group_gemm_ep import FusedMoeExpertFunctionEP

from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import gather_seq_scatter_heads, gather_heads_scatter_seq


def make_dsv3_plan():
    plan = {
        # mla (TP)
        "q_b_proj": Shard(0),
        "kv_b_proj": Shard(0),
        "o_proj": Shard(1),
        # moe experts (EP)
        "fc1_1": Shard(0),
        "fc1_2": Shard(0),
        "fc2": Shard(0),
        # moe shared experts / first layer (TP)
        "gate_proj": Shard(0),
        "up_proj": Shard(0),
        "down_proj": Shard(1),
    }
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

    tp_mesh: DeviceMesh = self._tp_mesh
    tp_size = 1 if tp_mesh is None else tp_mesh.size()
    tp_group = None if tp_mesh is None else tp_mesh.get_group()
    sp_size = get_ulysses_sequence_parallel_world_size()
    assert self.num_heads % (sp_size * tp_size) == 0

    # [bsz, qlen, hidden]
    bsz, q_len, _ = hidden_states.size()

    assert self.q_lora_rank is not None
    # [bsz, qlen, hidden] -> [bsz, qlen, q_lora_rank]
    # [bsz, qlen, q_lora_rank] -> [bsz, qlen, q_lora_rank]
    x = self.q_a_layernorm(self.q_a_proj(hidden_states))
    # ============== tensor parallel region ================
    if tp_size > 1:
        x = identity_allreduce(x, tp_group, "tp-iar")
    # ============== tensor parallel region ================
    # [bsz, qlen, q_lora_rank] -> [bsz, qlen, head * qhdim]
    q = self.q_b_proj(x)
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

    x = self.kv_a_layernorm(compressed_kv)
    # ============== tensor parallel region ================
    if tp_size > 1:
        x = identity_allreduce(x, tp_group, "tp-iar")
    # ============== tensor parallel region ================
    # [bsz, qlen, kv_lora_rank] -> [bsz, head, qlen, qk_node_head_dim + v_head_dim]
    kv = self.kv_b_proj(x).view(bsz, q_len, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)

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
    # ============== tensor parallel region ================
    if tp_size > 1:
        attn_output = allreduce_identity(attn_output, tp_group, "tp-ari")
    # ============== tensor parallel region ================

    return attn_output, None, None


def moe_ep_forward(self: DeepseekV3FusedMoE, hidden_states: torch.Tensor):

    ep_mesh: DeviceMesh = self._tp_mesh
    ep_group = None if ep_mesh is None else ep_mesh.get_group()

    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight = self.gate(hidden_states)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    topk_weight = topk_weight.bfloat16()
    hidden_states = hidden_states.bfloat16()
    y, handle = FusedMoeExpertFunctionEP.apply(
        self.n_routed_experts,
        topk_weight,
        topk_idx,
        hidden_states,
        self.fc1_1,
        self.fc1_2,
        self.fc2,
        ep_group,
    )
    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(identity)

    return y


def mlp_tp_forward(self: DeepseekV3MLP, x: torch.Tensor):

    tp_mesh: DeviceMesh = self._tp_mesh
    tp_group = None if tp_mesh is None else tp_mesh.get_group()
    tp_size = 1 if tp_mesh is None else tp_mesh.size()

    if tp_size > 1:
        x = identity_allreduce(x, tp_group, "tp-iar")

    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    if tp_size > 1:
        down_proj = allreduce_identity(down_proj, tp_group, "tp-ari")

    return down_proj


def deepseek_v3_casual_lm_forward(
    self: DeepseekV3ForCausalLM,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    fuse_lm_head_ce_loss: Optional[bool] = None,
    temperature: Optional[float] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    This monkey patch adds `fuse_lm_head_ce_loss` to fuse
    lm_head with cross-entropy loss computation.
    """
    output_attentions = (output_attentions if output_attentions is not None else self.config.output_attentions)
    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    if fuse_lm_head_ce_loss:
        assert labels is not None
        if temperature is not None:
            hidden_states = hidden_states / temperature
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
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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
