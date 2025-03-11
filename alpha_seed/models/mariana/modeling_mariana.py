"""
Contains base modeling for mariana megatron model
"""

import torch
from torch import nn
import logging

from mariana.models.text.gpt2_megatron import MegatronGPT2LMHeadModel as PretrainMegatronGPT2LMHeadModel
from mariana.models.layers.embedding import RotaryEmbedding

from mariana.models.text.config import ModelConfig, TrainConfig, MegatronConfig

from verl.utils.megatron import sequence_parallel as sp_utils

from megatron.core import tensor_parallel


def convert_gate_to_fp32(gpt):
    for i in range(len(gpt)):
        if hasattr(gpt[i].module.module, 'gpt_model'):
            layers = gpt[i].module.module.gpt_model.transformer.h.layers
        else:
            layers = gpt[i].module.module.transformer.h.layers
        for j in range(len(layers)):
            if (hasattr(layers[j], 'mlp') and getattr(layers[j].mlp, 'moe', None)
                    is not None):  # fix error caused by no-op layers & non-moe layers
                layers[j].mlp.moe.gate.wg_ema.data = layers[j].mlp.moe.gate.wg_ema.data.float()
                layers[j].mlp.moe.gate.ce_ema.data = layers[j].mlp.moe.gate.ce_ema.data.float()
                layers[j].mlp.moe.gate.cal_weights.data = layers[j].mlp.moe.gate.cal_weights.data.float()
                if hasattr(layers[j].mlp.moe.gate, 'wg_v'):
                    layers[j].mlp.moe.gate.wg_v.data = layers[j].mlp.moe.gate.wg_v.data.float()
                if hasattr(layers[j].mlp.moe.gate, 'wg_k'):
                    layers[j].mlp.moe.gate.wg_k.data = layers[j].mlp.moe.gate.wg_k.data.float()
                if hasattr(layers[j].mlp.moe.gate, 'wg_q'):
                    layers[j].mlp.moe.gate.wg_q.data = layers[j].mlp.moe.gate.wg_q.data.float()
                if hasattr(layers[j].mlp.moe.gate, 'wg'):
                    layers[j].mlp.moe.gate.wg.data = layers[j].mlp.moe.gate.wg.data.float()
    logging.info("Converted moe gate parameters and buffers to FP32.")


class MarianaForCausalLM(PretrainMegatronGPT2LMHeadModel):
    """
    We rewrite the forward function of pretrained CausalLM so that the model only outputs logits without other
    redundant outputs. This can represent any seed models. So, no need to distinguish version (e.g., p7, m8, m9.)
    """

    def __init__(self, model_config: ModelConfig, megatron_config: MegatronConfig, pre_process=True, post_process=True):
        super().__init__(model_config, megatron_config, pre_process, post_process)
        self.rotary_embedding = RotaryEmbedding(
            self.model_config.hidden_size // self.model_config.n_head,
            max_seq_len=self.model_config.max_position_embeddings,
            rope_scale=self.model_config.rope_scale,
            base=self.model_config.rope_base,
            mode=self.model_config.rope_mode,
            distributed_sequence_parallel_size=self.megatron_config.distributed_sequence_parallel_size,
            context_parallel_size=self.megatron_config.get("context_parallel_size", 1),
            rope_cut=self.model_config.rope_cut,
            rope_cut_head_dim=self.model_config.rope_cut_head_dim,
            rope_force_fp32=self.model_config.rope_force_fp32)

    def _forward_model(self, batch: dict[str, torch.Tensor]):
        enable_dsp = self.megatron_config.sequence_data_parallel_size > 1
        enable_sp = self.megatron_config.sequence_parallel
        # calc rope related logic
        host_seqlens = batch.get("host_seqlens", None)
        max_seq_len = batch.get("max_seq_len", None)
        cu_seqlens = batch.get("cu_seqlens", None)
        padded_seq_len = batch.get("padded_seq_len", None)
        seq_lens_start_end = batch.get("seq_lens_start_end", None)

        # TODO(zhangchi.usc1992): how should we pass s_max?
        self.rotary_embedding.generate_pos_embs(host_seqlens,
                                                host_seqlens.device,
                                                s_max=padded_seq_len,
                                                seq_lens_start_end=seq_lens_start_end)
        hidden_states, activation_stats, _ = self.transformer(
            input_ids=batch["input_ids"],  # pad to tp size
            position_ids=batch.get("position_ids", None),  # use this. remove sin/cos
            cu_seqlens=cu_seqlens,  # do not pad tp region
            cu_seqlens_q=batch.get("cu_seqlens_q", None),  # dsp
            cu_seqlens_splited=batch.get("cu_seqlens_splited", None),  # dsp
            max_s=batch.get("max_s", None),  # must have
            total_s=batch.get("total_s", None),  # total_seqlen without pad in tp region
            total_s_splited=batch.get("total_s_splited", None),  # dsp/cp
            seq_lens_start_end=seq_lens_start_end,  # rope. useless
            batch_offset_q=batch.get("batch_offset_q", None),  # dsp, useless
            seq_offset_q=batch.get("seq_offset_q", None),  # dsp, useless
            host_seqlens=host_seqlens,  # give None, useless
            max_seq_len=max_seq_len,  # useless
            cos_embs_indices=self.rotary_embedding.cos_embs,  # TODO: remove this
            sin_embs_indices=self.rotary_embedding.sin_embs  # TODO: remove this
        )
        return hidden_states

    def _forward_head(self, hidden_states):
        return self.lm_logits(hidden_states)

    def forward(self, batch: dict[str, torch.Tensor]):
        hidden_states = self._forward_model(batch=batch)

        if self.post_process:
            lm_logits = self._forward_head(hidden_states=hidden_states)
            out = {'logits': lm_logits}
        else:
            hidden_states = hidden_states * 1.0  # ?
            out = {
                'hidden_states': hidden_states,
            }

        return out


class MarianaForTokenClassification(MarianaForCausalLM):
    # TODO(zhangchi.usc1992): add value model. The head should be named as score_head
    def __init__(self, model_config: TrainConfig, megatron_config: MegatronConfig, pre_process=True, post_process=True):
        super().__init__(model_config, megatron_config, pre_process, post_process)
        self.score_head = nn.Linear(in_features=self.model_config.hidden_size, out_features=1, bias=True)
        sp_utils.mark_parameter_as_sequence_parallel(self.score_head.weight)
        sp_utils.mark_parameter_as_sequence_parallel(self.score_head.bias)

    def _forward_head(self, hidden_states):
        """hidden_states: [total_nnz_padded // tp, 1, hidden_states]
        """
        values = self.score_head(hidden_states)  # [total_nnz_padded // tp, 1, 1]
        values = torch.squeeze(values, dim=-1)  # [total_nnz_padded // tp, 1]
        # all gather from sequence parallel region, first dim
        values = tensor_parallel.gather_from_sequence_parallel_region(
            values, tensor_parallel_output_grad=False)  # [total_nnz_padded, 1]
        return values
