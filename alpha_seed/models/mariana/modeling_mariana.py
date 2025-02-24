"""
Contains base modeling for mariana megatron model
"""

import torch
import logging

from mariana.models.text.gpt2_megatron import MegatronGPT2LMHeadModel as PretrainMegatronGPT2LMHeadModel
from mariana.models.layers.embedding import RotaryEmbedding

from mariana.models.text.config import TrainConfig, MegatronConfig


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

    def forward(self, batch: dict[str, torch.Tensor]):
        enable_dsp = self.megatron_config.sequence_data_parallel_size > 1
        enable_sp = self.megatron_config.sequence_parallel
        # calc rope related logic
        host_seqlens = batch.get("host_seqlens", None)
        max_seq_len = batch.get("max_seq_len", None)
        cu_seqlens = batch.get("cu_seqlens", None)
        padded_seq_len = batch.get("padded_seq_len", None)
        seq_lens_start_end = batch.get("seq_lens_start_end", None)

        # self.rotary_embedding.generate_pos_embs(host_seqlens, host_seqlens.device, s_max=padded_seq_len, seq_lens_start_end=seq_lens_start_end)
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
            cos_embs_indices=None,  # TODO: remove this
            sin_embs_indices=None  # TODO: remove this
        )

        if self.post_process:
            lm_logits = self.lm_logits(hidden_states)
            # shift_logits = lm_logits  # [sum_token, 1, vocab_dim/tp]
            # shift_labels = shift_labels.unsqueeze(1).contiguous()  # [sum_token] -> [sum_token, 1]

            # if self.megatron_config.sequence_parallel or enable_dsp:
            #     shift_logits = shift_logits[:shift_labels.shape[0]]  # rm seq-para pad

            # shift_logits_float = shift_logits.float()
            # del lm_logits, hidden_states, shift_logits
            # # NOTE sp is all-gathered before cross_entropy, but dsp is all-gathered in ce.
            # loss = tensor_parallel.vocab_parallel_cross_entropy(shift_logits_float, shift_labels, padded_seq_len=padded_seq_len, unpad_seq_len=unpad_seq_len)

            # # Padded back loss
            # total_seq_len = word_idx.shape[0]
            # batch_size = cu_seqlens.size(0) - 1
            # total_seq_len = word_idx.shape[0]
            # loss = pad_input(loss[:total_seq_len], word_idx, batch_size, max_seq_len).squeeze(2)  # [sum_token, 1] -> [bs, seqlen-1]
            # loss = loss[:, :-1]
            # logprobs = -loss

            # Note: `clone` here is to make share logprobs._base is None.
            # so that it can be pseudo-freed in Megatron schedule.
            # See megatron/schedules_multiple_forward.py:deallocate_output_tensor
            # Deallocating a tensor view will not actually free its GPU storage.
            # logprobs = logprobs.clone()
            out = {
                # 'loss': loss,
                # 'output': logprobs,
                # 'logprobs': logprobs
                'logits': lm_logits
            }
        else:
            hidden_states = hidden_states * 1.0  # ?
            out = {
                'output': hidden_states,
            }

        return out


class MarianaForTokenClassification(PretrainMegatronGPT2LMHeadModel):
    # TODO(zhangchi.usc1992): add value model. The head should be named as score_head
    pass
