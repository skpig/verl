from typing import Callable, Dict, List, Tuple, Union
import logging
import torch
from xperf_gpt.utils.quant_config import QuantOption, get_quant_option


class XperfModelProphet:

    def __init__(
        self,
        model_cfg: Dict,
        sched_cfg: Dict,
        mp_size: int,
        force_use_full_cache: bool = False,
        orca_max_bs_clamp: int = 2048,
        orca_max_context_bs_clamp: int = 2048,
    ):
        self.model_cfg = model_cfg
        self.sched_cfg = sched_cfg
        self.model_type = model_cfg.get("model_name", "GPT2LMHeadModel")
        self.supported_model_types = [
            "GPT2LMHeadModel", "GPT2LMHeadModelMoe", "LlamaForCausalLM", "SeedLLaMAForCausalLM"
        ]
        if self.model_type not in self.supported_model_types:
            raise ValueError(f"XperfModelProphet not support [{self.model_type}] yet, can't auto configure scheduler.")

        self.max_position_embeddings = model_cfg.get("max_position_embeddings", sched_cfg["max_sequence_length"])
        self.has_ptb = model_cfg.get("has_ptb", False)
        self.quant_type = get_quant_option(model_cfg.get("quant_mode", "NO_QUANT"))
        self.hidden_size = model_cfg["hidden_size"]
        self.window_size = model_cfg.get("window_size", [])
        self.vocab_size = model_cfg["vocab_size"]
        self.max_sequence_length = sched_cfg["max_sequence_length"]
        self.num_heads = model_cfg["num_heads"]
        self.num_kv_heads = model_cfg.get("num_kv_heads", self.num_heads)
        self.num_layers = model_cfg["num_layers"]
        self.kv_num_layers = self.num_layers
        if "kv_mirror_imitated_layers" in model_cfg:
            self.kv_num_layers -= len(model_cfg["kv_mirror_imitated_layers"])
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = model_cfg.get("ffn_internal_dim",
                                               self.hidden_size * 5 - self.num_kv_heads * self.head_dim)
        self.moe_ffn_internal_dim = model_cfg.get("moe_ffn_internal_dim", 0)
        self.moe_expert_num = model_cfg.get("moe_expert_num", 0)
        self.moe_topk = model_cfg.get("moe_topk", 0)
        self.is_exp_moe = model_cfg.get("is_exp_moe", False)
        self.has_mlp_gate = model_cfg.get("has_mlp_gate", False)

        self.mp_size = mp_size
        self.force_use_full_cache = force_use_full_cache
        self.orca_max_bs_clamp = orca_max_bs_clamp
        self.orca_max_context_bs_clamp = orca_max_context_bs_clamp
        self.page_attn_slots_clamp = orca_max_bs_clamp * self.max_sequence_length

        self.ctx_seg_len = min(self.max_sequence_length, sched_cfg["max_context_len"])

    def get_model_size(self) -> int:
        # roughly estimate, ignore the remainder(norm_beta, bias...)
        element_size_map = {
            QuantOption.NoQuant: 2,
            QuantOption.W8_PerChannelSymm: 1,
            QuantOption.W8C8_PerChannelSymm: 1,
            QuantOption.W4_ChannelGroupAsymm: 0.5,
            QuantOption.W4C8_ChannelGroupAsymm: 0.5,
            QuantOption.W8A8_PerChannelSymm: 1,
            QuantOption.FP8_W8_PerTensor: 1,
        }
        element_sz = element_size_map[self.quant_type]

        def get_wpe_wte_weight_size():
            wpe_weight_size = self.max_position_embeddings * self.hidden_size * 2
            wte_weight_size = self.vocab_size * self.hidden_size * 2
            return wpe_weight_size + wte_weight_size

        def get_lm_head_weight_size():
            return self.vocab_size * self.hidden_size

        def get_attn_weight_size():
            # attn
            c_attn_weight = element_sz * self.hidden_size * (self.num_kv_heads * 2 + self.num_heads) * self.head_dim
            c_proj_weight = element_sz * self.hidden_size * self.num_heads * self.head_dim

            return c_attn_weight + c_proj_weight

        def calc_gpt2_model_size():
            # ffn
            ffn_weight = element_sz * self.hidden_size * self.intermediate_size * 2

            weights_sz_per_layer = get_attn_weight_size() + ffn_weight

            return (weights_sz_per_layer * self.num_layers // self.mp_size + get_wpe_wte_weight_size() +
                    get_lm_head_weight_size())

        def calc_gpt2moe_model_size():
            if self.is_exp_moe:
                # exp moe ffn
                exp_moe_ffn_weight = (element_sz * sum(
                    [eg * self.moe_ffn_internal_dim // 2**i * self.hidden_size for i, eg in enumerate([2, 4, 8, 16])]) *
                                      2)
                weights_sz_per_layer = get_attn_weight_size() + exp_moe_ffn_weight
            else:
                # normal moe ffn
                normal_moe_ffn_weight = (element_sz * self.moe_expert_num * self.moe_ffn_internal_dim *
                                         self.hidden_size * (3 if self.has_mlp_gate else 2))
                weights_sz_per_layer = get_attn_weight_size() + normal_moe_ffn_weight

            return (weights_sz_per_layer * self.num_layers // self.mp_size + get_wpe_wte_weight_size() +
                    get_lm_head_weight_size())

        def calc_llama_model_size():
            # ffn
            ffn_weight = element_sz * self.hidden_size * self.intermediate_size * 3

            weights_sz_per_layer = get_attn_weight_size() + ffn_weight

            return (weights_sz_per_layer * self.num_layers // self.mp_size + get_wpe_wte_weight_size() +
                    get_lm_head_weight_size())

        model_size_calculator = {
            "GPT2LMHeadModel": calc_gpt2_model_size,
            "GPT2LMHeadModelMoe": calc_gpt2moe_model_size,
            "LlamaForCausalLM": calc_llama_model_size,
            "SeedLLaMAForCausalLM": calc_llama_model_size,
        }
        total_weights_sz = model_size_calculator[self.model_type]()
        logging.info(f"XperfModelProphet get_model_size {total_weights_sz/1024**3}GB per card.")
        return total_weights_sz

    def profile_max_sample_io_buffer_size_per_token(self) -> int:
        # sampler io buffer
        logits_element_size = 2  # byte, default is torch.bfloat16
        topk_topp_index_element_size = 4  # byte
        logits_buf_size = self.vocab_size * logits_element_size
        topk_topp_buf_sz = self.vocab_size * (logits_element_size + topk_topp_index_element_size)
        sample_buf_sz = logits_buf_size + topk_topp_buf_sz
        logging.info(f"XperfModelProphet profile_max_sample_io_buffer_size_per_token got {sample_buf_sz/1024**3}GB.")
        return sample_buf_sz

    def profile_max_forward_io_buffer_size_per_token(self) -> int:
        # roughly estimate io buffer size.
        io_buffer_element_size = 2  # byte, default is torch.bfloat16

        def get_attn_io_buffer():
            # qkv projection
            num_kv_heads_per_card = self.num_kv_heads // self.mp_size if self.num_kv_heads > self.mp_size else 1
            num_q_heads_per_card = self.num_heads // self.mp_size
            qkv_proj_buffer_size = ((num_kv_heads_per_card * 2 + num_q_heads_per_card) * self.head_dim *
                                    io_buffer_element_size)
            # attn
            softmax_buffer_size = self.max_sequence_length * io_buffer_element_size
            attn_buffer_size = self.num_heads * self.head_dim * io_buffer_element_size
            # attn out projection
            attn_out_buffer_size = self.hidden_size * io_buffer_element_size

            return qkv_proj_buffer_size + softmax_buffer_size + attn_buffer_size + attn_out_buffer_size

        def get_activate_io_buffer():
            # activate
            activate_buffer_size = self.hidden_size * io_buffer_element_size

            return activate_buffer_size

        def get_layernorm_io_buffer():
            # layernorm
            layer_norm_buffer_size = self.hidden_size * io_buffer_element_size

            return layer_norm_buffer_size

        def calc_gpt2_io_buffer():
            # ffn
            ffn0_buffer_size = self.intermediate_size * io_buffer_element_size
            ffn1_buffer_size = self.hidden_size * io_buffer_element_size

            max_io_buffer_size = (get_activate_io_buffer() + get_layernorm_io_buffer() + get_attn_io_buffer() +
                                  ffn0_buffer_size + ffn1_buffer_size)

            return max_io_buffer_size

        def calc_gpt2moe_io_buffer():
            if self.is_exp_moe:
                # exp moe ffn
                exp_distribution = [2, 4, 8, 16]
                exp_moe_ffn0_buffer_size = (
                    sum([eg * self.moe_ffn_internal_dim // 2**i for i, eg in enumerate(exp_distribution)]) *
                    self.moe_topk / sum(exp_distribution) * io_buffer_element_size)
                exp_moe_ffn1_buffer_size = self.hidden_size * io_buffer_element_size * self.moe_topk
                # Because the activation will be released in time during xperf forward, max_io buffer can take the maximum value in activation
                max_io_buffer_size = max(
                    get_activate_io_buffer(),
                    get_layernorm_io_buffer(),
                    get_attn_io_buffer(),
                    exp_moe_ffn0_buffer_size,
                    exp_moe_ffn1_buffer_size,
                )
            else:
                # normal moe ffn
                normal_moe_ffn0_buffer_size = self.moe_topk * self.moe_ffn_internal_dim * io_buffer_element_size
                normal_meo_ffn1_buffer_size = self.hidden_size * io_buffer_element_size * self.moe_topk
                # Because the activation will be released in time during xperf forward, max_io buffer can take the maximum value in activation
                max_io_buffer_size = max(
                    get_activate_io_buffer(),
                    get_layernorm_io_buffer(),
                    get_attn_io_buffer(),
                    normal_moe_ffn0_buffer_size,
                    normal_meo_ffn1_buffer_size,
                )

            return max_io_buffer_size

        io_buf_calculator = {
            "GPT2LMHeadModel": calc_gpt2_io_buffer,
            "GPT2LMHeadModelMoe": calc_gpt2moe_io_buffer,
            "SeedLLaMAForCausalLM": calc_gpt2_io_buffer,
        }

        max_io_buf_size = io_buf_calculator[self.model_type]()
        # other io buffer are omited.
        logging.info(f"XperfModelProphet profile_max_forward_io_buffer_size_per_token got {max_io_buf_size/1024**3}GB.")
        return max_io_buf_size

    def profile_kv_cache_size_per_token(self) -> int:
        kv_cache_element_size_map = {
            QuantOption.NoQuant: 2,
            QuantOption.W8_PerChannelSymm: 2,
            QuantOption.W8C8_PerChannelSymm: 1,
            QuantOption.W4_ChannelGroupAsymm: 2,
            QuantOption.W4C8_ChannelGroupAsymm: 1,
            QuantOption.W8A8_PerChannelSymm: 1,
            QuantOption.FP8_W8_PerTensor: 2,
        }
        kv_cache_element_sz = kv_cache_element_size_map[self.quant_type]
        num_kv_heads_per_card = self.num_kv_heads // self.mp_size if self.num_kv_heads > self.mp_size else 1
        kv_cache_size_per_token = self.kv_num_layers * num_kv_heads_per_card * 2 * kv_cache_element_sz * self.head_dim
        logging.info(
            f"XperfModelProphet profile_kv_cache_size_per_token got {kv_cache_size_per_token/1024**3}GB per card.")
        return kv_cache_size_per_token

    def profile_available_vllm_cfg(self, gpu_memory_utilization: float) -> Dict:
        tot_gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        logging.info(f"XperfModelProphet profile_available_vllm_cfg got total {tot_gpu_mem_bytes/1024**3}GB per card.")
        model_sz = self.get_model_size()
        kv_cache_sz_per_token = self.profile_kv_cache_size_per_token()
        fwd_io_buf_sz_per_token = self.profile_max_forward_io_buffer_size_per_token()
        sample_io_buf_sz_per_token = self.profile_max_sample_io_buffer_size_per_token()
        avaliable_buf_sz = tot_gpu_mem_bytes * gpu_memory_utilization - model_sz
        if avaliable_buf_sz < 0:
            raise ValueError(
                f"XperfModelProphet profile_available_vllm_cfg got model_sz={model_sz/1024**3}GB, but available gpu buffer={tot_gpu_mem_bytes * gpu_memory_utilization/1024**3}GB."
            )
        io_buffer_reserve_ratio = 0.2
        max_io_buffer = avaliable_buf_sz * io_buffer_reserve_ratio
        orca_max_batch_size = max(1, int(max_io_buffer / max(fwd_io_buf_sz_per_token, sample_io_buf_sz_per_token)))
        orca_max_batch_size = min(orca_max_batch_size, self.orca_max_bs_clamp)

        orca_max_context_batch_size = min(
            orca_max_batch_size,
            int(max_io_buffer / max(self.ctx_seg_len * fwd_io_buf_sz_per_token, sample_io_buf_sz_per_token)),
        )
        orca_max_context_batch_size = min(orca_max_context_batch_size, self.orca_max_context_bs_clamp)
        vllm_max_slots = int((avaliable_buf_sz - max_io_buffer) / kv_cache_sz_per_token)
        if vllm_max_slots <= 0:
            raise ValueError(
                f"XperfModelProphet profile_available_vllm_cfg got vllm_max_slots={vllm_max_slots}, something wrong.")
        vllm_max_slots = min(vllm_max_slots, self.page_attn_slots_clamp)
        vllm_max_slots = vllm_max_slots // self.sched_cfg["vllm_block_size"]

        # summary memory usage
        kv_cache_mem_use = vllm_max_slots * self.sched_cfg["vllm_block_size"] * kv_cache_sz_per_token
        peak_io_buffer_mem_use = max(
            (orca_max_context_batch_size * self.ctx_seg_len + orca_max_batch_size - orca_max_context_batch_size) *
            fwd_io_buf_sz_per_token,
            orca_max_batch_size * sample_io_buf_sz_per_token,
        )
        predict_tot_use = model_sz + kv_cache_mem_use + peak_io_buffer_mem_use
        logging.info(
            f"XperfModelProphet profile_available_vllm_cfg got[orca_max_batch_size={orca_max_batch_size}, orca_max_context_batch_sz={orca_max_context_batch_size}, vllm_max_slots={vllm_max_slots}]"
        )
        logging.info(
            f"Total Memory {tot_gpu_mem_bytes/1024**3}GB, Gpu Memory Utilization={gpu_memory_utilization}, Expected Used Memory={tot_gpu_mem_bytes*gpu_memory_utilization/1024**3}GB."
        )
        logging.info(
            f"Predict Model Weight Memory Use={model_sz/1024**3}GB, KV Cache Memory Use={kv_cache_mem_use/1024**3}GB, Peak IO Buffer Use={peak_io_buffer_mem_use/1024**3}GB, Sum Of The Above={predict_tot_use/1024**3}GB."
        )

        return {
            "orca_max_batch_size": orca_max_batch_size,
            "orca_max_context_batch_size": orca_max_context_batch_size,
            "vllm_num_slots": vllm_max_slots,
        }

    def profile_available_orca_cfg(self, gpu_memory_utilization: float) -> Dict:
        tot_gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        logging.info(f"XperfModelProphet profile_available_orca_cfg got total {tot_gpu_mem_bytes/1024**3}GB per card.")
        model_sz = self.get_model_size()
        kv_cache_sz_per_token = self.profile_kv_cache_size_per_token()
        fwd_io_buf_sz_per_token = self.profile_max_forward_io_buffer_size_per_token()
        sample_io_buf_sz_per_token = self.profile_max_sample_io_buffer_size_per_token()

        # we should satisfy: tot_gpu_mem * gpu_memory_utilization - model_size > kv_cache_size + max_io_buf_size}
        avaliable_buf_sz = tot_gpu_mem_bytes * gpu_memory_utilization - model_sz
        if avaliable_buf_sz < 0:
            raise ValueError(
                f"XperfModelProphet profile_available_orca_cfg got model_sz={model_sz/1024**3}GB, but available gpu buffer={tot_gpu_mem_bytes * gpu_memory_utilization/1024**3}GB."
            )

        kv_cache_len = (sum(self.window_size) / len(self.window_size) if
                        (self.window_size is not None and len(self.window_size) > 0 and
                         not self.force_use_full_cache) else self.max_sequence_length)

        # First, let's assume that all decode_batch_size is 0, and estimate the maximum context_batch_size.
        orca_max_context_batch_sz = avaliable_buf_sz / (
            max(self.ctx_seg_len * fwd_io_buf_sz_per_token, sample_io_buf_sz_per_token) +
            kv_cache_len * kv_cache_sz_per_token)
        # Secondly, we scale context_batch_size to free up the space of decode batch
        max_context_batch_ratio = 0.5
        if orca_max_context_batch_sz < 1:
            raise ValueError(
                f"XperfModelProphet profile_available_orca_cfg got orca_max_context_batch_sz={orca_max_context_batch_sz}, which is invalid, maybe gpu memory space is not enough."
            )
        orca_max_context_batch_sz = max(int(orca_max_context_batch_sz * max_context_batch_ratio), 1)

        orca_max_context_batch_sz = min(orca_max_context_batch_sz, self.orca_max_context_bs_clamp)

        # Calculate how large decode_batch_size can be according to the free space.
        orca_decode_batch_buf_sz = (avaliable_buf_sz -
                                    orca_max_context_batch_sz * self.ctx_seg_len * fwd_io_buf_sz_per_token -
                                    orca_max_context_batch_sz * kv_cache_len * kv_cache_sz_per_token)
        if orca_decode_batch_buf_sz < 0:
            raise ValueError(
                f"XperfModelProphet profile_available_orca_cfg orca_decode_batch_buf_sz got invalid value: {orca_decode_batch_buf_sz/1024**3}GB, something wrong."
            )
        orca_max_decode_batch_sz = int(
            orca_decode_batch_buf_sz /
            (kv_cache_sz_per_token * kv_cache_len + max(fwd_io_buf_sz_per_token, sample_io_buf_sz_per_token)))

        orca_max_batch_sz = orca_max_decode_batch_sz + orca_max_context_batch_sz
        orca_max_batch_sz = min(orca_max_batch_sz, self.orca_max_bs_clamp)

        # summary memory usage
        kv_cache_mem_use = orca_max_batch_sz * kv_cache_len * kv_cache_sz_per_token
        peak_io_buffer_mem_use = max(
            (orca_max_context_batch_sz * self.ctx_seg_len + orca_max_decode_batch_sz) * fwd_io_buf_sz_per_token,
            orca_max_batch_sz * sample_io_buf_sz_per_token,
        )
        predict_tot_use = model_sz + kv_cache_mem_use + peak_io_buffer_mem_use
        logging.info(
            f"XperfModelProphet profile_available_orca_cfg got[orca_max_batch_size={orca_max_batch_sz}, orca_max_context_batch_sz={orca_max_context_batch_sz}]"
        )
        logging.info(
            f"Total Memory {tot_gpu_mem_bytes/1024**3}GB, Gpu Memory Utilization={gpu_memory_utilization}, Expected Used Memory={tot_gpu_mem_bytes*gpu_memory_utilization/1024**3}GB."
        )
        logging.info(
            f"Predict Model Weight Memory Use={model_sz/1024**3}GB, KV Cache Memory Use={kv_cache_mem_use/1024**3}GB, Peak IO Buffer Use={peak_io_buffer_mem_use/1024**3}GB, Sum Of The Above={predict_tot_use/1024**3}GB."
        )

        return {
            "orca_max_batch_size": orca_max_batch_sz,
            "orca_max_context_batch_size": orca_max_context_batch_sz,
        }
