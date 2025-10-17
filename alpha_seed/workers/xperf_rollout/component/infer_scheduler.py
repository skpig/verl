import torch
import time
import bisect
import re
import logging
from typing import *
from xperf_gpt.utils import (logging_rank, logging_rank_only)


class InferScheduler():

    def __init__(self,
                 cache_manager,
                 engine,
                 sampler,
                 return_full_hidden_states,
                 return_padding_tensor,
                 last_token_only,
                 context_only,
                 enable_cuda_graph=False,
                 enable_metrics=False,
                 max_ngram_size=0,
                 num_pred_tokens=0,
                 enable_mtp_decoding=False,
                 return_selected_experts=False):
        self.cache_manager = cache_manager
        self.engine = engine
        self.sampler = sampler
        self.return_full_hidden_states = return_full_hidden_states
        self.return_padding_tensor = return_padding_tensor
        self.last_token_only = last_token_only
        self.context_only = context_only
        self.enable_metrics = enable_metrics
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens
        self.enable_cuda_graph = enable_cuda_graph
        self.enable_mtp_decoding = enable_mtp_decoding
        self.return_selected_experts = return_selected_experts
        self.init_cuda_graph()
        self.init_metrics()

    def get_roofline(self):
        device_intensity_map = {
            "NO_QUANT": {
                "A100": 156,
                "H20": 40,
                "L20": 149,
                "H800": 295,
            },
            "WFP8": {
                "A100": 312,
                "H20": 79,
                "L20": 299,
                "H800": 590,
            },
            "W4A8": {
                "A100": 312,
                "H20": 79,
                "L20": 299,
                "H800": 590,
            }
        }
        gemm_intensity_map = {
            "NO_QUANT": 1,
            "WFP8": 2,
            "W4A8": 4,
        }
        quant_mode = self.engine.module.quant_mode
        if quant_mode not in gemm_intensity_map:
            logging_rank(logging.info, f"gemm intensity is not found for {quant_mode} quant mode")
        gemm_intensity = gemm_intensity_map.get(quant_mode, 1)

        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_name = re.split("[ |-]", device_name)[1]
        if device_name not in device_intensity_map[quant_mode]:
            logging_rank(logging.info, f"device intensity is not found for {device_name} device")
        device_intensity = device_intensity_map[quant_mode].get(device_name, 1)

        roofline = device_intensity / gemm_intensity
        if hasattr(self.engine.module.config, "moe_expert_num"):
            roofline *= self.engine.module.config.moe_expert_num / self.engine.module.config.moe_topk
        return roofline

    def record(self, key, value):
        if not self.enable_metrics:
            return
        if key not in self.metrics.keys():
            self.metrics[key] = []
        self.metrics[key].extend(value)

    def incr(self, key, value=1):
        if not self.enable_metrics:
            return
        if key not in self.metrics.keys():
            self.metrics[key] = 0
        self.metrics[key] += value

    def switch_mode(self, mode):
        if mode == "rollout":
            if self.enable_mtp_decoding:
                self.return_full_hidden_states = True
                self.last_token_only = False
            else:
                self.return_full_hidden_states = False
                self.last_token_only = True
            self.return_full_hidden_states_after_layernorm = False
            self.return_padding_tensor = False
            self.context_only = False
        elif mode == "log_probs":
            self.context_only = True
            self.return_full_hidden_states = False
            self.return_padding_tensor = True
            self.last_token_only = False
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def init_metrics(self):
        if not self.enable_metrics:
            return
        self.metrics = {}
        self.metrics['ctx_bs'] = []
        self.metrics['ctx_prompt_length'] = []
        self.metrics['dec_bs'] = []
        self.metrics['kv_cache_utils'] = []
        self.metrics['tokens_num'] = []
        self.metrics['sample_token_num'] = 0
        self.metrics['prob_mean'] = 0
        self.metrics['prob_lt_0.0001'] = 0
        self.metrics['prob_lt_1e-5'] = 0
        self.metrics['prob_lt_1e-6'] = 0
        self.metrics['page_swap_out_bs'] = 0
        self.metrics['page_swap_out_token'] = 0
        self.metrics['per_token_latency'] = []
        self.metrics['prefill_latency'] = []
        self.metrics['decode_latency'] = []
        self.metrics['prefill_token_num'] = []
        self.metrics['prefix_cache_hit_length'] = []
        self.metrics['evict_count'] = 0
        self.metrics['accept_len'] = []
        self.metrics['roofline'] = self.get_roofline()
        self.sampler.metrics = self.metrics
        self.step_time = None

    def empty_cache(self):
        self.init_metrics()

    def next_step(self, has_prefill):
        if self.enable_metrics:
            if self.step_time is not None:
                duration = (time.time() - self.step_time) * 1000
                self.metrics['per_token_latency'].append(duration)
                if has_prefill:
                    self.metrics['prefill_latency'].append(duration)
                else:
                    self.metrics['decode_latency'].append(duration)
            self.step_time = time.time()

    def init_cuda_graph(self):
        self.release_cuda_graph()
        if not self.enable_cuda_graph:
            return
        assert not self.engine.module.wte_weight.is_meta
        memory_pool = None
        self.graph_capture_bs_list = [1, 2, 4] + list(range(8, 128 + 1, 8))
        self.graph_capture_max_bs = max(self.graph_capture_bs_list)
        need_to_warmup = True
        for bs in self.graph_capture_bs_list:
            self.graph_decode_input_ids_placeholder[bs] = torch.zeros(bs, 1).cuda().int()
            self.graph_total_length_placeholder[bs] = torch.Tensor([1024] * bs).cuda().int()
            if not self.cache_manager.use_vllm:
                self.graph_kv_cache_index_placeholder[bs] = torch.Tensor([i for i in range(bs)]).cuda().int()
            else:
                self.graph_kv_cache_index_placeholder[bs] = torch.zeros(bs, self.cache_manager.slot_num).cuda().int()
            self.output_placeholder[bs] = torch.zeros(bs, self.engine.module.config.vocab_size).cuda().bfloat16()

            self.bs_graph_map[bs] = torch.cuda.CUDAGraph()

            def run_inference():
                output = self.engine.forward_orca(context_input_ids=None,
                                                  decode_input_ids=self.graph_decode_input_ids_placeholder[bs],
                                                  total_length=self.graph_total_length_placeholder[bs],
                                                  kv_cache_index=self.graph_kv_cache_index_placeholder[bs],
                                                  orca_updated=True,
                                                  context_shifts=None,
                                                  return_full_hidden_states=False,
                                                  return_padding_tensor=False,
                                                  last_token_only=True,
                                                  return_selected_experts=self.return_selected_experts)
                self.output_placeholder[bs].copy_(output)

            if need_to_warmup:
                run_inference()
                need_to_warmup = False
            with torch.cuda.graph(self.bs_graph_map[bs], pool=memory_pool):
                run_inference()
            if memory_pool is None:
                memory_pool = self.bs_graph_map[bs].pool()

    def release_cuda_graph(self):
        self.bs_graph_map = {}
        self.graph_decode_input_ids_placeholder = {}
        self.graph_total_length_placeholder = {}
        self.graph_kv_cache_index_placeholder = {}
        self.output_placeholder = {}
        self.graph_capture_bs_list = []
        self.graph_capture_max_bs = 0

    def internal_inference_orca(self, context_input: torch.Tensor, decode_input: torch.Tensor,
                                total_length: torch.Tensor, kv_index: torch.Tensor, orca_updated: bool,
                                context_shifts: torch.Tensor):
        if decode_input is not None and decode_input.shape[0] <= self.graph_capture_max_bs and context_input is None:
            bs = decode_input.shape[0]
            pos = bisect.bisect_left(self.graph_capture_bs_list, bs)
            cuda_graph_bs = self.graph_capture_bs_list[pos]
            padding_bs = cuda_graph_bs - bs
            if padding_bs > 0:
                # add padding
                decode_input = torch.cat(
                    [decode_input,
                     torch.zeros([padding_bs, 1], dtype=decode_input.dtype, device=decode_input.device)])
                total_length = torch.cat(
                    [total_length,
                     torch.ones([padding_bs], dtype=total_length.dtype, device=total_length.device)])
                kv_slot_for_padding = self.cache_manager.slot_num
                if self.cache_manager.use_vllm:
                    kv_index = torch.cat([
                        kv_index,
                        torch.full([padding_bs, kv_index.shape[1]],
                                   kv_slot_for_padding,
                                   dtype=kv_index.dtype,
                                   device=kv_index.device)
                    ],
                                         dim=0)
                else:
                    kv_index = torch.cat([
                        kv_index,
                        torch.full([padding_bs], kv_slot_for_padding, dtype=kv_index.dtype, device=kv_index.device)
                    ])

            self.graph_decode_input_ids_placeholder[cuda_graph_bs].copy_(decode_input)
            self.graph_total_length_placeholder[cuda_graph_bs].copy_(total_length)
            if self.cache_manager.use_vllm:
                self.graph_kv_cache_index_placeholder[cuda_graph_bs][:, :kv_index.shape[1]].copy_(kv_index)
            else:
                self.graph_kv_cache_index_placeholder[cuda_graph_bs].copy_(kv_index)
            self.bs_graph_map[cuda_graph_bs].replay()
            output = self.output_placeholder[cuda_graph_bs]
            return output[:bs]
        else:
            context_max_kv_len = context_total_kv_len = -1
            decode_max_kv_len = decode_total_kv_len = -1
            if context_input is not None:
                context_kv_len = total_length[:context_input.shape[0]] + context_shifts
                context_max_kv_len = context_kv_len.max().item()
                context_total_kv_len = context_kv_len.sum().item()
            if decode_input is not None:
                context_bs = context_input.shape[0] if context_input is not None else 0
                decode_kv_len = total_length[context_bs:]
                decode_max_kv_len = decode_kv_len.max().item()
                decode_total_kv_len = decode_kv_len.sum().item()

            return self.engine.forward_orca(
                context_input_ids=None,
                context_input_embeds=context_input,
                decode_input_ids=decode_input,
                total_length=total_length,
                kv_cache_index=kv_index,
                orca_updated=orca_updated,
                context_shifts=context_shifts,
                return_full_hidden_states=self.return_full_hidden_states,
                return_padding_tensor=self.return_padding_tensor,
                return_full_hidden_states_after_layernorm=self.return_full_hidden_states_after_layernorm,
                last_token_only=self.last_token_only,
                context_max_kv_len=context_max_kv_len,
                context_total_kv_len=context_total_kv_len,
                decode_max_kv_len=decode_max_kv_len,
                decode_total_kv_len=decode_total_kv_len,
                return_selected_experts=self.return_selected_experts)

    def forward_and_sample(self,
                           context_input: torch.Tensor,
                           decode_input: torch.Tensor,
                           total_length: torch.Tensor,
                           kv_index: torch.Tensor,
                           orca_updated: bool,
                           context_shifts: torch.Tensor,
                           context_labels_ids: torch.Tensor,
                           history_ids: List[List[int]],
                           keys: torch.Tensor = None,
                           sample_kwargs: dict = None,
                           draft_input: torch.Tensor = None,
                           draft_total_length: torch.Tensor = None,
                           target_hidden_states: torch.Tensor = None):
        if self.enable_metrics:
            tokens_num = 0
            if context_input is not None:
                tokens_num += context_input.shape[0] * context_input.shape[1]
                self.metrics["ctx_bs"].append(context_input.shape[0])
                self.metrics["ctx_prompt_length"].append(context_input.shape[1])
            else:
                pass

            if decode_input is not None:
                decode_tokens = decode_input.shape[0]
                if self.enable_mtp_decoding:
                    decode_tokens *= (self.num_pred_tokens + 1)
                tokens_num += decode_tokens
                self.metrics["dec_bs"].append(decode_tokens)
            else:
                self.metrics["dec_bs"].append(0)

            self.metrics["kv_cache_utils"].append(self.cache_manager.get_kv_cache_utils())
            self.metrics["tokens_num"].append(tokens_num)
            self.metrics["page_swap_out_bs"] += self.cache_manager.page_swap_out_bs
            self.metrics["page_swap_out_token"] += self.cache_manager.page_swap_out_token

        accepted_len = None
        forward_spec = keys is not None or (self.enable_mtp_decoding and context_input is None)
        if not forward_spec:
            # 1. forward
            output = self.internal_inference_orca(context_input, decode_input, total_length, kv_index, orca_updated,
                                                  context_shifts)
            selected_experts = None
            if self.return_full_hidden_states:
                logits = output[0]
                target_hidden_states = list(output[1].split(total_length.tolist(), dim=0))
            else:
                logits = output
                target_hidden_states = None
            if self.return_selected_experts:
                logits = output[0]
                context_bs = context_input.shape[0] if context_input is not None else 0
                total_len = total_length.tolist()
                prefill_decode_length = total_len[:context_bs] + [1] * (len(total_len) - context_bs)
                selected_experts = output[-1].cpu().split(prefill_decode_length, dim=0)

            # 2. sample
            if self.context_only:
                assert (logits.shape[0] == 1)
                import torch.nn.functional as F
                log_probs = F.log_softmax(logits, dim=-1)
                token_ids = context_labels_ids.unsqueeze(0).unsqueeze(-1).cuda()
                log_probs = torch.gather(log_probs, dim=-1, index=token_ids).squeeze(-1)
                next_tokens = None
            else:
                next_tokens, log_probs, = self.sampler.sample(logits,
                                                              need_torch_tensor=True,
                                                              history_ids=history_ids,
                                                              sample_kwargs=sample_kwargs)
            return next_tokens, accepted_len, target_hidden_states, log_probs, selected_experts
        else:
            # forward mtp spec
            context_bs = context_input.shape[0] if context_input is not None else 0
            decode_kv_len = total_length[context_bs:]
            decode_max_kv_len = decode_kv_len.max().item()
            decode_total_kv_len = decode_kv_len.sum().item()
            draft_context_shifts = total_length - draft_total_length - 1
            draft_kv_len = draft_total_length + draft_context_shifts
            draft_max_kv_len = draft_kv_len.max().item()
            draft_total_kv_len = draft_kv_len.sum().item()
            accpeted_tokens, accepted_len, target_hidden_states = self.engine.forward_spec(
                draft_model=self.engine,
                draft_input_ids=draft_input,
                eagle_hidden_states=target_hidden_states,
                draft_total_length=draft_total_length,
                draft_context_shifts=draft_context_shifts,
                kv_cache_index=kv_index,
                k=self.num_pred_tokens,
                sampler=self.sampler,
                draft_max_kv_len=draft_max_kv_len,
                draft_total_kv_len=draft_total_kv_len,
                decode_max_kv_len=decode_max_kv_len,
                decode_total_kv_len=decode_total_kv_len,
            )
            target_hidden_states = target_hidden_states.split(self.num_pred_tokens + 1)
            target_hidden_states = [
                hidden_states[:accepted_len[i] + 1] for i, hidden_states in enumerate(target_hidden_states)
            ]
            return accpeted_tokens.contiguous(), accepted_len, target_hidden_states, None, None
