import torch
import time
from typing import *


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
                 enable_mtp_decoding=False):
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
        self.init_cuda_graph()
        self.init_metrics()

    def record(self, key, value):
        if not self.enable_metrics:
            return
        if key not in self.metrics.keys():
            self.metrics[key] = []
        self.metrics[key].extend(value)

    def switch_mode(self, mode):
        if mode == "rollout":
            self.return_full_hidden_states = False
            self.return_padding_tensor = False
            self.last_token_only = True
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
        self.sampler.metrics = self.metrics
        self.step_time = None

    def empty_cache(self):
        self.init_metrics()

    def next_step(self):
        if self.enable_metrics:
            if self.step_time is not None:
                duration = (time.time() - self.step_time) * 1000
                self.metrics['per_token_latency'].append(duration)
            self.step_time = time.time()

    def init_cuda_graph(self):
        self.bs_graph_map = {}
        self.graph_decode_input_ids_placeholder = {}
        self.graph_total_length_placeholder = {}
        self.graph_kv_cache_index_placeholder = {}
        self.output_placeholder = {}
        if not self.enable_cuda_graph or self.engine.module.wte_weight.is_meta or self.engine.is_xperf_triton:
            return
        for bs in [1, 2, 4]:
            self.graph_decode_input_ids_placeholder[bs] = torch.zeros(bs, 1).cuda().int()
            self.graph_total_length_placeholder[bs] = torch.Tensor([1024] * bs).cuda().int()
            if not self.cache_manager.use_vllm:
                self.graph_kv_cache_index_placeholder[bs] = torch.Tensor([i for i in range(bs)]).cuda().int()
            else:
                self.graph_kv_cache_index_placeholder[bs] = torch.zeros(bs, self.cache_manager.slot_num).cuda().int()
            self.output_placeholder[bs] = torch.zeros(bs, self.engine.module.config.vocab_size).cuda().bfloat16()

            self.bs_graph_map[bs] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.bs_graph_map[bs]):
                output = self.engine.forward_orca(context_labels_ids=None,
                                                  decode_input_ids=self.graph_decode_input_ids_placeholder[bs],
                                                  total_length=self.graph_total_length_placeholder[bs],
                                                  kv_cache_index=self.graph_kv_cache_index_placeholder[bs],
                                                  orca_updated=True,
                                                  context_shifts=None,
                                                  return_full_hidden_states=False,
                                                  return_padding_tensor=False,
                                                  last_token_only=True)
                self.output_placeholder[bs].copy_(output)

    def internal_inference_orca(self, context_input: torch.Tensor, decode_input: torch.Tensor,
                                total_length: torch.Tensor, kv_index: torch.Tensor, orca_updated: bool,
                                context_shifts: torch.Tensor):

        if self.enable_metrics:
            tokens_num = 0
            if context_input is not None:
                tokens_num += context_input.shape[0] * context_input.shape[1]
                self.metrics["ctx_bs"].append(context_input.shape[0])
                self.metrics["ctx_prompt_length"].append(context_input.shape[1])
            else:
                pass

            if decode_input is not None:
                tokens_num += decode_input.shape[0]
                self.metrics["dec_bs"].append(decode_input.shape[0])
            else:
                self.metrics["dec_bs"].append(0)

            self.metrics["kv_cache_utils"].append(self.cache_manager.get_kv_cache_utils())
            self.metrics["tokens_num"].append(tokens_num)
            self.metrics["page_swap_out_bs"] += self.cache_manager.page_swap_out_bs
            self.metrics["page_swap_out_token"] += self.cache_manager.page_swap_out_token

        if decode_input is not None and decode_input.shape[0] in self.bs_graph_map.keys() and context_input is None:
            bs = decode_input.shape[0]
            self.graph_decode_input_ids_placeholder[bs].copy_(decode_input)
            self.graph_total_length_placeholder[bs].copy_(total_length)
            if self.cache_manager.use_vllm:
                self.graph_kv_cache_index_placeholder[bs][:, :kv_index.shape[1]].copy_(kv_index)
            else:
                self.graph_kv_cache_index_placeholder[bs].copy_(kv_index)
            self.bs_graph_map[bs].replay()
            return self.output_placeholder[bs]
        else:
            return self.engine.forward_orca(context_input_ids=None,
                                            context_input_embeds=context_input,
                                            decode_input_ids=decode_input,
                                            total_length=total_length,
                                            kv_cache_index=kv_index,
                                            orca_updated=orca_updated,
                                            context_shifts=context_shifts,
                                            return_full_hidden_states=self.return_full_hidden_states,
                                            return_padding_tensor=self.return_padding_tensor,
                                            last_token_only=self.last_token_only)

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
        accepted_len = None
        forward_spec = keys is not None or (self.enable_mtp_decoding and context_input is None)
        if not forward_spec:
            # 1. forward
            output = self.internal_inference_orca(context_input, decode_input, total_length, kv_index, orca_updated,
                                                  context_shifts)
            if self.return_full_hidden_states:
                logits = output[0]
                target_hidden_states = list(output[1].split(total_length.tolist(), dim=0))
            else:
                logits = output
                target_hidden_states = None

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
            return next_tokens, accepted_len, target_hidden_states, log_probs
        else:
            # forward mtp spec
            accpeted_tokens, accepted_len, target_hidden_states = self.engine.forward_spec(
                draft_model=self.engine,
                draft_input_ids=draft_input,
                eagle_hidden_states=target_hidden_states,
                draft_total_length=draft_total_length,
                draft_context_shifts=total_length - draft_total_length - 1,
                kv_cache_index=kv_index,
                k=self.num_pred_tokens,
                sampler=self.sampler,
            )
            target_hidden_states = target_hidden_states.split(self.num_pred_tokens + 1)
            target_hidden_states = [
                hidden_states[:accepted_len[i] + 1] for i, hidden_states in enumerate(target_hidden_states)
            ]
            return accpeted_tokens, accepted_len, target_hidden_states, None
