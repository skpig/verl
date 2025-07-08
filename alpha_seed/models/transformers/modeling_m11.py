from typing import Optional, Tuple, List, Union
import seed_models
from alpha_seed.models.transformers import AlphaSeedMoeCausalLMOutputWithPast
from alpha_seed.models.transformers.ops.memory_efficient_ops import compute_chunked_entropy_logprobs
from seed_models.models.m11.modeling_m11 import (apply_rotary_pos_emb, repeat_kv, Cache, M11FlashAttention2,
                                                 M11FusedMoeBlock, M11ForCausalLM)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from .modeling_flash_attention_utils import _flash_attention_forward, _flash_supports_window_size
from bumi.function.flash_cross_entropy import FlashCrossEntropy
import torch.nn.functional as F

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import Shard
from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import slice_input_tensor

from dist_attn.ulysses.ops import gather_seq_scatter_heads, gather_heads_scatter_seq
from packaging.version import Version

import logging

logger = logging.getLogger(__file__)

from functools import partial
from seed_kernels.transformers.functional import seed_fused_moe


def _check_version():
    import vescale
    if Version(vescale.__version__) < Version("0.2.19"):
        raise RuntimeError(f"vescale version must be >= 0.2.19, but got {vescale.__version__}. "
                           "Please install through pip3 install byted-vescale==0.2.19")
    import triton
    if Version(triton.__version__) < Version("3.3.0"):
        raise RuntimeError(f"triton version must be >= 3.3.0, but got {triton.__version__}. "
                           "Please install through pip3 install triton==3.3.0")


def make_m11_plan():

    _check_version()
    plan = {
        # oe
        "*over_encoded_embeddings.embedding_list.0.weight": Shard(0),
        # moe experts (EP)
        "*.moe.experts.gate_proj": Shard(0),
        "*.moe.experts.up_proj": Shard(0),
        "*.moe.experts.down_proj": Shard(0),
    }
    return plan


def flash_attn2_rmpad_forward(
    self: M11FlashAttention2,
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
    assert (not use_cache) and (not past_key_value)
    assert "padding_mask" not in kwargs
    assert past_key_value is None
    assert position_embeddings is not None
    assert max_seqlen is not None
    assert cu_seqlens is None

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # =============== ulysses sp region ==================
    sp_size = get_ulysses_sequence_parallel_world_size()
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

    kv_seq_len = max_seqlen

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    # rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
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
        layer_idx=self.layer_idx,
        max_seqlen=max_seqlen,
    )

    if self.config.use_context_groupnorm:
        attn_output = self.context_groupnorm(attn_output)
    # ============== ulysses sp region ===================
    if sp_size > 1:
        attn_output = attn_output.reshape(bsz, full_qlen, -1, self.head_dim).contiguous()
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)
    # ============== ulysses sp region ===================

    attn_output = attn_output.reshape(bsz, q_len, self.num_query_heads * self.head_dim).contiguous()
    attn_output = self.o_proj(attn_output)

    if self.config.use_attention_output_norm:
        attn_output = self.o_norm(attn_output)

    attn_output = self.resid_dropout(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def fused_moe_block_forward(
    self: M11FusedMoeBlock,
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


class RowParallelShardedEmbedding(torch.nn.Module):

    def __init__(self, _num_embeddings, embedding_dim, config):
        super().__init__()
        # padding num_embeddings to be divisible by world_size
        self.config = config
        self.max_tp_size = 128
        self.num_embeddings = (_num_embeddings + self.max_tp_size - 1) // self.max_tp_size * self.max_tp_size
        self.embedding_dim = embedding_dim
        self.stride = self.num_embeddings // self.max_tp_size
        # TODO: random init via weight initializer
        self.weight = torch.nn.Parameter(torch.empty(self.num_embeddings, embedding_dim), requires_grad=True)
        from .ops.oe import AllGatherReduceScatterEmbedding
        # optionally
        # from .ops.oe import AllToAllEmbedding
        self._embedding_function = AllGatherReduceScatterEmbedding

    def forward(self, input):
        tp_mesh: Optional[DeviceMesh] = getattr(self.weight, "mesh", None)
        tp_group = tp_mesh.get_group() if tp_mesh is not None else None
        return self._embedding_function.apply(tp_group, input, self.weight)


class M11DevFusedCatOverEncodingEmbeddingTP(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.over_encoding_stride = max(self.config.over_enc_vocab_stride)
        assert max(self.config.over_enc_embed_dim) == min(self.config.over_enc_embed_dim)
        self.over_encoding_embed_dim = self.config.over_enc_embed_dim[0]
        if self.over_encoding_embed_dim < 0:
            self.over_encoding_embed_dim = self.config.hidden_size

        self.register_buffer('over_encoding_vocab_stride',
                             torch.tensor(self.config.over_enc_vocab_stride),
                             persistent=False)  # [n]
        self.register_buffer('over_encoding_vocab_size',
                             torch.tensor(self.config.over_enc_vocab_size),
                             persistent=False)  # [n]
        self.register_buffer(
            'over_encoding_vocab_offset',
            torch.cumsum(torch.tensor([0] + self.config.over_enc_vocab_size[:-1]), 0),
            persistent=False,
        )  # [n]

        self.embedding_list = torch.nn.ModuleList([
            RowParallelShardedEmbedding(sum(self.config.over_enc_vocab_size),
                                        self.over_encoding_embed_dim,
                                        config=config)
        ])
        hidden_input_dim = (self.config.hidden_size * self.config.vwn_n_in //
                            self.config.vwn_m if config.use_vwn else self.config.hidden_size)

        self.emb_proj = torch.nn.Linear(
            self.over_encoding_embed_dim * len(self.config.over_enc_vocab_size) + self.config.hidden_size,
            hidden_input_dim,
            bias=False,
        )

    def forward(self, x, input_ids, position_ids):
        sp_size = get_ulysses_sequence_parallel_world_size()
        if sp_size > 1:
            position_ids = slice_input_tensor(position_ids, dim=1, padding=False)
        padded_input_ids = F.pad(input_ids, (self.over_encoding_stride, 0, 0, 0), "constant", 0)
        padded_input_ids[padded_input_ids <= 1] = 0
        # prepare over-enc input ids
        b, s = input_ids.shape
        over_enc_ids = input_ids.clone().unsqueeze(-1)
        vocab_size = 1
        for i in range(1, self.over_encoding_stride):
            _ids = padded_input_ids[:, -i - s:-i, None]
            vocab_size = (self.config.vocab_size * vocab_size) % self.over_encoding_vocab_size
            over_enc_ids = over_enc_ids + _ids * vocab_size * (i < self.over_encoding_vocab_stride) * (
                position_ids.unsqueeze(-1) >= i)
            over_enc_ids %= self.over_encoding_vocab_size
        over_enc_ids += self.over_encoding_vocab_offset
        # get over-enc embeddings
        over_enc_ids = over_enc_ids[:, -s:]
        over_embs = self.embedding_list[0](over_enc_ids)  # [b, s, n, dim]
        x = torch.cat([x, over_embs.flatten(2).contiguous()], dim=-1).contiguous()
        x = self.emb_proj(x)
        return x


# NOTE(zhiqi.0): we have no other choice to replace the original class
# since this requires some changes in module init
seed_models.models.m11.modeling_m11.FusedCatOverEncodingEmbedding = M11DevFusedCatOverEncodingEmbeddingTP


def m11_casual_lm_forward(
    self: M11ForCausalLM,
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

    Returns:

    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_router_logits = (output_router_logits
                            if output_router_logits is not None else self.config.output_router_logits)
    output_aux_losses = output_aux_losses if output_aux_losses is not None else self.config.output_aux_losses

    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

    return AlphaSeedMoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        entropy=entropy,
        log_probs=log_probs,
    )
