"""
Large Language Model Inference Session Manager

Handles end-to-end inference process including:
- Prompt preprocessing
- KV cache management
- Context window optimization
- Token generation with various decoding strategies
- Multi-GPU distributed inference
"""
from dataclasses import dataclass
from queue import Queue
from typing import Optional

from torch.distributed import get_rank
from xperf_gpt.inference import init_inference
from alpha_seed.workers.xperf_rollout.component.cache_manager import CacheManager
from alpha_seed.workers.xperf_rollout.component.infer_scheduler import InferScheduler
from alpha_seed.workers.xperf_rollout.component.sampling_manager import Sampler
from alpha_seed.workers.xperf_rollout.component.prefix_cache import PrefixCache
from alpha_seed.workers.xperf_rollout.utils.custom_xperf_convert_helper import XCustomInferenceModuleAdapter
from xperf_gpt.multi_models.visual.inferencer import VITInferencer
from xperf_gpt.multi_models.visual.eva_vit import EVA_VIT_CONFIGS
from alpha_seed.workers.xperf_rollout.component.query import Query, AsyncQuery, InflightQueue, batch_sync_tp_queries
from alpha_seed.utils.observility import get_profiler_context_wrapped
from xperf_gpt.utils import (logging_rank, logging_rank_only)
from typing import List, Dict
import logging
import os
import time
import torch
import copy
import logging
import pickle
import dill
import base64
import time
from threading import Lock
from transformers import AutoTokenizer
from alpha_seed.workers.xperf_rollout.utils.vit_inferencer import TorchVitInferencer, gen_vit_cfg
import numpy as np
import torch.nn.functional as F
from alpha_seed.utils.dataset.utils import convert_numpy_to_tensor

# Constants
BLOCK_SIZE_ALIGNMENT = 256
DEFAULT_LOGGING_LEVEL = logging.INFO


@dataclass
class LoadMetric:
    ts: float  # 采样时间，单位s
    num_pending: int  # 还从来没开始跑
    num_waiting: int  # 跑过但因为内存不够被换出
    num_prefilling: int  # 正在跑prefill
    num_decoding: int  # 正在跑decode
    kv_cache_util: float  # kv cache util range 0~1
    pending_ids: List[str]  # 放在pending列表里的query ids
    waiting_ids: List[str]  # 放在waiting列表里的query ids
    is_weights_updating: bool  # 是否正在更新weights期间


class StepProfiler:

    def __init__(self, config):
        self.profile_at_steps = set([i for i in config.profile_at_steps])
        self.profile_first_n_execs = config.profile_first_n_execs
        self.global_rank = torch.distributed.get_rank()

        def _create_context(step: int, ctx_tokens: int, dec_tokens: int):
            return get_profiler_context_wrapped(
                filename=f"{config.filename}_rank_{self.global_rank}_step_{step}_ctx_{ctx_tokens}_dec{dec_tokens}",
                profile_on_ranks=config.profile_on_ranks,
                upload_to_mlx=config.upload_to_mlx,
                enable=config.enable,
                wait=0,
                warmup=0,
                active=1,
                repeat=1)

        self.create_context = _create_context

        self.step = 0
        self.exec_count = 0
        self.profiler = None

    def reset_exec(self):
        self.step = 0
        self.exec_count += 1
        self.profiler = None

    def record_step(self, ctx_tokens: int, dec_tokens: int):
        if self.exec_count >= self.profile_first_n_execs:
            return
        if self.step in self.profile_at_steps:
            if self.profiler is not None:
                self.profiler.__exit__(None, None, None)
            self.profiler = self.create_context(self.step, ctx_tokens, dec_tokens)
            self.profiler.__enter__()
        elif self.profiler is not None and hasattr(self.profiler, 'step'):
            self.profiler.step()
        self.step += 1


def _remove_query_list_inplace(lst: List[Query], to_remove: Dict[str, float]) -> List[Query]:
    # to_remove: query_id -> ts (abort the query if before this ts)
    original_len = len(lst)
    write_index = 0
    will_remove = []
    for read_index in range(original_len):
        q = lst[read_index]
        not_after = to_remove.get(q.id)
        if not_after is None or q.dispatch_time >= not_after:
            # keep this query
            if write_index != read_index:
                lst[write_index] = lst[read_index]
            write_index += 1
        else:
            will_remove.append(lst[read_index])
    del lst[write_index:]
    # 返回删除的元素
    return will_remove


class GetMaxSet:
    """A sorted multiset data structure that can get the maximum value in O(logn)"""

    def __init__(self):
        from sortedcontainers import SortedDict
        self.data = SortedDict()

    def add_one(self, val: int):
        key = -val
        if key not in self.data:
            self.data[key] = 0
        self.data[key] += 1

    def remove_one(self, val: int):
        key = -val
        self.data[key] -= 1
        if self.data[key] == 0:
            self.data.pop(key)

    def is_empty(self):
        return len(self.data) == 0

    def get_max(self) -> int:
        key = next(iter(self.data.keys()))
        return -key


class InferenceSession:
    """A session for managing inference process of a large language model.
    
    Handles prompt processing, context management, token generation and decoding.
    Supports features like paged attention, multi-stream processing, and n-gram decoding.
    
    Args:
        num_slots (int): Number of concurrent processing slots
        max_batch_size (int): Maximum batch size for inference
        max_length (int): Maximum total sequence length (prompt + generated)
        enable_paged_attn (bool): Use paged attention optimization
        multi_stream (bool): Use separate streams for context and decoding
    """

    def __init__(
            self,
            num_slots,
            max_batch_size,
            max_length=4096,
            enable_paged_attn=False,
            context_split_len=4 * 1024,
            vit_use_xperf_gpt=True,
            vit_use_dp=False,
            dp_vit_batching_step=0,
            max_prompt_length=None,
            context_limit_bs=1,
            max_context_shift=0,
            slot_block_size=1,
            vocab_tp=False,
            enable_truncation=True,
            ctx_shift_intra_micro_batch=True,
            is_prefill_decode_split=False,
            enable_ngrams_decoding=False,
            enable_ngrams_when_bs_below=32,
            max_ngram_size=3,
            num_pred_tokens=6,
            enable_cuda_graph=False,
            standalone=False,
            schedule_strategy="default",  # ['default','fifo']
            step_profiler: StepProfiler = None,
            enable_mtp_decoding=False,
            prefix_cache_slot_num=-1,
            prefix_cache_max_length=-1):
        """Initialize inference session with hardware/performance parameters"""
        # Memory management
        self.enable_paged_attn = enable_paged_attn
        self.num_slots = num_slots
        self.slot_block_size = slot_block_size
        self.standalone = standalone
        # Context processing
        self.context_split_len = context_split_len
        self.vit_use_xperf_gpt = vit_use_xperf_gpt
        self.vit_use_dp = vit_use_dp
        self.context_limit_bs = context_limit_bs

        self.max_prompt_length = max_prompt_length
        self.max_context_shift = max_context_shift
        self.max_length = max_length
        self.vocab_tp = vocab_tp
        self.enable_truncation = enable_truncation
        self.ctx_shift_intra_micro_batch = ctx_shift_intra_micro_batch
        self.enable_cuda_graph = enable_cuda_graph
        self.is_prefill_decode_split = is_prefill_decode_split
        self.enable_ngrams_decoding = enable_ngrams_decoding
        self.schedule_strategy = schedule_strategy
        self.step_profiler = step_profiler
        self.status = "idle"
        self.mode = "rollout"
        self.enable_mtp_decoding = enable_mtp_decoding
        self.record_input_prompt = True
        self.tokenizer = None
        self.max_off_policy_steps = 5
        self.enable_metrics = False
        self.current_steps = 0

        if enable_paged_attn:
            self._validate_paged_attention_config()
            self.max_batch_size = max_batch_size
            self.max_total_tokens = num_slots * slot_block_size
        else:
            self.max_batch_size = min(num_slots, max_batch_size)
            self.max_total_tokens = 0

        if self.enable_ngrams_decoding:
            assert self.is_prefill_decode_split, "N-grams decoding depends on prefill decode split."
            self.max_ngram_size = max_ngram_size
            self.num_pred_tokens = num_pred_tokens
            self.enable_ngrams_when_bs_below = enable_ngrams_when_bs_below
        elif self.enable_mtp_decoding:
            self.max_ngram_size = 0
            self.num_pred_tokens = 1
        else:
            self.max_ngram_size = 0
            self.num_pred_tokens = 0

        # components
        '''
        1. Pending Queue(For Rollout Server Only)
        Acts as the entry point for incoming requests
        Temporarily stores newly arrived queries before they're scheduled for execution
        Maintains requests in FIFO (First-In-First-Out) order by default
        2. Waiting Queue
        Serves as a "pause buffer" for interrupted queries
        Holds requests that were:
        Preempted during execution (e.g., due to priority weight updates)
        Evicted from memory (via paged eviction mechanisms)
        Preserves partial execution states for later resumption
        3. Running Queue
        Contains currently executing queries
        Represents active workloads consuming system resources
        Maintains real-time status of in-process operations
        4. Paused Queue
        Contains queries that are paused, similar to waiting queue, but these queries waits for an
        external event (e.g. tool calling) before they can be resumed.
        
        Workflow:
            Requests flow from Pending → Running → (Waiting/Paused if interrupted) → Finished. 
            The waiting queue enables stateful handling of mid-execution interruptions through check-pointing mechanisms.
        '''
        self.pending = InflightQueue()
        self.waiting: List[Query] = []
        self.running: List[Query] = []
        self.paused: List[Query] = []
        self.paused_ready: List[Query] = []
        self.paused_ready_step = 0
        self.paused_batching_step = dp_vit_batching_step if vit_use_dp else 0

        self.all_accepted_queries: Dict[str, Query] = {}
        self.queries_to_abort: Queue = Queue()  # 要abort掉的query将query_id放进queue里
        self.unfinished_off_policy_steps_set = GetMaxSet()
        self.stop_sequence_tokens: List[List[int]] = []
        self.common_prefix = ""
        self.common_prefix_tensor_len = 0
        self.return_full_hidden_states = False
        self.return_padding_tensor = False
        self.last_token_only = True
        self.eos_callback_fn = None
        self.stop_signal_tensor = torch.tensor([0.0]).float().cuda()
        self.update_weights_lock = Lock()
        self._accepted_queries_mutex = Lock()
        self.tp_group = None
        self.prefix_cache = None
        self.prefix_cache_slot_num = prefix_cache_slot_num
        self.prefix_cache_max_length = prefix_cache_max_length

    def switch_inference_mode(self, mode):
        self.mode = mode
        if mode == "log_probs":
            self.cache_manager.context_batchsize_limit = 1
        self.infer_scheduler.switch_mode(mode)

    def _validate_paged_attention_config(self):
        """Validate paged attention configuration constraints"""
        assert self.slot_block_size % BLOCK_SIZE_ALIGNMENT == 0, \
            f"Block size must be multiple of {BLOCK_SIZE_ALIGNMENT}"
        assert self.max_length % self.slot_block_size == 0, \
            "Max length must align with block size"
        assert self.num_slots * self.slot_block_size > self.max_length, \
            "Insufficient slot capacity, expecting at least {} tokens".format(self.max_length)

    def set_callback_function(self, eos_callback_fn):
        self.eos_callback_fn = eos_callback_fn

    def reset_logging_level(self):
        logging_level = int(os.getenv("XGPT_LOGGING_LEVEL",
                                      "20"))  # see https://docs.python.org/zh-cn/3/library/logging.html#logging-levels
        default_logger = logging.getLogger()
        default_logger.setLevel(logging_level)

    def init_inference_engine(
            self,
            session_config_path: str,
            generation_config: dict,
            # Model loading parameters
            tokenizer_path: str = None,
            checkpoint_path: str = None,
            vanilla_checkpoint_path: str = None,
            reshard_checkpoint_path: str = None,
            preshard_checkpoint_path: str = None,
            # XPerf custom parameters
            use_xperf_custom: bool = False,
            xperf_custom_backbone: str = None,
            xperf_custom_preset: str = None,
            # XPerf triton parameters
            use_xperf_triton: bool = False,
            xperf_triton_cfg: dict = None,
            # Output and monitoring settings
            enable_metrics: bool = False,
            return_full_hidden_states: bool = False,
            return_padding_tensor: bool = False,
            last_token_only: bool = True,
            # Additional parameters from kwargs
            save_mp_checkpoint_path: str = None,
            stop_sequence_tokens: list = None,
            record_input_prompt: bool = None,
            constraint_decoding: bool = None,
            decode_output: bool = True,
            # Vision model settings
            vit_config: dict = None,
            vit_model_cfg_path=None,
            # Additional parameters
            mp_size: int = None,
            use_ep: bool = None,
            multi_host_tp: bool = None,
            **kwargs):
        """Initialize model engine and associated components
        
        Args:
            session_config_path: Path to model config JSON
            generation_config: Generation parameters
            use_xperf_custom: Whether use xperf_gpt_custom
            vit_config: vision config dict
            kwargs: Overrides for model loading
            TODO: add more docstrings
        """

        if os.getenv("XPERF_SESSION_SET_TORCH_DEVICE", "1") == "1":
            torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

        # model loading
        self.tokenizer_path = tokenizer_path
        self.checkpoint_path = checkpoint_path
        self.vanilla_checkpoint_path = vanilla_checkpoint_path
        self.reshard_checkpoint_path = reshard_checkpoint_path
        self.preshard_checkpoint_path = preshard_checkpoint_path
        self.save_mp_checkpoint_path = save_mp_checkpoint_path

        # output and monitoring
        self.enable_metrics = enable_metrics
        self.return_full_hidden_states = return_full_hidden_states
        self.return_padding_tensor = return_padding_tensor
        self.last_token_only = last_token_only
        self.decode_output = decode_output

        # generation control
        self.stop_sequence_tokens = stop_sequence_tokens
        self.record_input_prompt = record_input_prompt
        self.constraint_decoding = constraint_decoding

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.padding_side = "left"
        if "eos_token_id" in generation_config:
            self.eos_token_id = generation_config["eos_token_id"]
            generation_config.pop("eos_token_id", None)
        else:
            self.eos_token_id = [self.tokenizer.eos_token_id]

        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_new_tokens = generation_config["max_new_tokens"]
        self.max_prompt_length = self.max_prompt_length or self.max_length - self.max_new_tokens

        init_inference_kwargs = dict(
            # Model configuration
            model_config_path=session_config_path,
            dtype=torch.bfloat16,
            # Model checkpoint paths
            checkpoint_path=self.checkpoint_path,
            vanilla_checkpoint_path=self.vanilla_checkpoint_path if not self.checkpoint_path else None,
            reshard_checkpoint_path=self.reshard_checkpoint_path,
            save_mp_checkpoint_path=self.save_mp_checkpoint_path,
            # Parallelism settings
            vocab_tp=self.vocab_tp,
            rank0_split=not self.checkpoint_path,  # if preshard, turn off rank0_split
            # Memory and batching settings
            use_vllm=self.enable_paged_attn,
            slot_block_size=self.slot_block_size,
            max_batch_size=self.max_batch_size,
            max_total_tokens=self.max_total_tokens,
            # Sequence length settings
            max_length=self.max_length,
            max_context_shift=self.max_context_shift,
            # Engine features
            use_xperf_gpt=True,
            use_orca=True,
            use_mtp=self.enable_mtp_decoding,
            multi_stream=1,
            # Token IDs (placeholders, not actually used)
            eos_token_id=0,  # won't use
            pad_token_id=0,  # won't use
            **generation_config)
        # kwargs override
        print("init_inference_engine kwargs", kwargs)
        init_inference_kwargs.update(kwargs)

        # Preserve legacy behavior: only set mp_size if explicitly provided
        if mp_size is not None:
            init_inference_kwargs['mp_size'] = mp_size
        if use_ep is not None:
            init_inference_kwargs['use_ep'] = use_ep
        if multi_host_tp is not None:
            init_inference_kwargs['multi_host_tp'] = multi_host_tp

        self.is_xperf_custom = use_xperf_custom
        self.is_xperf_triton = use_xperf_triton
        if use_xperf_custom:
            assert not self.enable_paged_attn, f"xperf custom for paged attention not supported yet."
            import xperf_gpt_custom
            engine = xperf_gpt_custom.create_network(backbone=xperf_custom_backbone,
                                                     preset=xperf_custom_preset,
                                                     **init_inference_kwargs)
            module = XCustomInferenceModuleAdapter(engine)
            setattr(engine, "module", module)
            setattr(engine.config, "model_config", {"hidden_size": engine.config.hidden_size})
            self.engine = engine
        elif use_xperf_triton:
            from alpha_seed.workers.xperf_rollout.utils.xperf_gpt_triton_helper import init_inference_triton
            init_inference_kwargs["eos_token_id"] = self.eos_token_id
            init_inference_kwargs['enable_cuda_graph'] = self.enable_cuda_graph
            init_inference_kwargs['max_ctx_batch_size'] = self.context_limit_bs
            init_inference_kwargs['xperf_triton_cfg'] = xperf_triton_cfg
            init_inference_kwargs.pop('reshard_checkpoint_path')
            init_inference_kwargs['preshard_checkpoint_path'] = self.preshard_checkpoint_path
            self.engine = init_inference_triton(**init_inference_kwargs)
            self.enable_cuda_graph = False
        else:
            self.engine = init_inference(None, **init_inference_kwargs)
            print("init_inference_kwargs: ", init_inference_kwargs)
        self.sampler = Sampler(generation_config=generation_config)
        self.oe_max_stride = max(getattr(self.engine.module.config, "over_enc_vocab_stride", [1]))

        self.reset_logging_level()

        cache_manager_num_slots = self.num_slots
        if self.enable_cuda_graph and not use_xperf_triton:
            # reserve one as cuda graph padding
            print(f"reserve slot {cache_manager_num_slots} as cuda_graph padding")
            cache_manager_num_slots -= 1
        self.cache_manager = CacheManager(cache_manager_num_slots,
                                          max_batch_size=self.max_batch_size,
                                          use_vllm=self.enable_paged_attn,
                                          slot_block_size=self.slot_block_size,
                                          context_batchsize_limit=self.context_limit_bs,
                                          enable_ngrams_decoding=self.enable_ngrams_decoding,
                                          num_pred_tokens=self.num_pred_tokens,
                                          moving_avg_length=self.max_length,
                                          schedule_strategy=self.schedule_strategy)
        if self.prefix_cache_slot_num > 0:
            num_kv_heads = self.engine.module.config.tp_kv_heads
            if self.engine.module.quant_mode in ["NO_QUANT", "WFP8"]:
                kv_cache_dtype = torch.bfloat16
            elif "C8" in self.engine.module.quant_mode or self.engine.module.quant_mode == "W8A8":
                kv_cache_dtype = torch.int8
            else:
                raise RuntimeError(f"Unsupported quant mode {self.engine.module.quant_mode} for prefix cache")
            kv_mirror_layers = 0
            if hasattr(self.engine.module.config, "kv_mirror_layers"):
                kv_mirror_layers = len(getattr(self.engine.module.config, "kv_mirror_layers"))
            valid_num_layers = self.engine.module.num_layers - kv_mirror_layers
            self.prefix_cache = PrefixCache(num_kv_heads, self.engine.module.head_dim, valid_num_layers, kv_cache_dtype,
                                            self.prefix_cache_slot_num, self.prefix_cache_max_length,
                                            self.enable_paged_attn, self.slot_block_size)
        self.infer_scheduler = InferScheduler(
            cache_manager=self.cache_manager,
            engine=self.engine,
            sampler=self.sampler,
            return_full_hidden_states=self.return_full_hidden_states,
            return_padding_tensor=self.return_padding_tensor,
            last_token_only=self.last_token_only,
            context_only=False,
            enable_cuda_graph=self.enable_cuda_graph,
            enable_metrics=self.enable_metrics,
            max_ngram_size=self.max_ngram_size,
            num_pred_tokens=self.num_pred_tokens,
            enable_mtp_decoding=self.enable_mtp_decoding,
        )

        if vit_config is not None:
            if self.vit_use_xperf_gpt:
                if vit_config['vit_model'] not in EVA_VIT_CONFIGS.keys():
                    vit_config.update(vit_config.get('transformer_config'))
                else:
                    detail_config = EVA_VIT_CONFIGS[vit_config['vit_model']]
                    vit_config.update(detail_config)
                if vit_config.get('use_navit', False):
                    vit_config['navit_anyres'] = True
                if self.vit_use_dp:
                    vit_config['dp_vit'] = True
                self.vit_engine = VITInferencer(vit_config_dict=vit_config,
                                                tokenization_path=self.tokenizer_path).cuda().to(torch.bfloat16)
            else:
                # use torch vit based on seed_models
                vit_cfg = gen_vit_cfg(vit_model_cfg_path)
                self.vit_engine = TorchVitInferencer(vit_cfg)

    def set_tp_group(self, tp_group):
        self.tp_group = tp_group

    def set_generator_strategy(self, **kwargs):
        self.sampler.set_generator_strategy(**kwargs)
        for key, value in kwargs.items():
            if key == "stop_sequence_tokens":
                self.__dict__[key] = value
            elif value is None:
                continue
            elif key not in self.__dict__.keys():
                continue
            elif self.__dict__[key] is not None and type(self.__dict__[key])(value) != value:
                raise ValueError(
                    "the argument '{}' should be '{}' but got '{}' and cannot be converted implicitly".format(
                        key,
                        type(self.__dict__[key]).__name__,
                        type(value).__name__))
            else:
                self.__dict__[key] = type(self.__dict__[key])(value) if self.__dict__[key] is not None else value

    def truncate_prompts(self, prompts, calculate_loss=False):
        truncate_input_ids = list()
        if isinstance(prompts, list) and all(isinstance(prompt, str) for prompt in prompts):
            for idx, prompt in enumerate(prompts):
                input_ids = self.tokenizer(prompt, padding=True, return_tensors="pt").input_ids.tolist()[0]
                if (len(input_ids) == 0):
                    logging_rank(logging.info, "[prompts tokenization] detect invalid empty input_ids")
                    input_ids = [self.pad_token_id]
                prompt_len = len(input_ids)
                if prompt_len > self.max_prompt_length and self.enable_truncation:
                    logging_rank(
                        logging.debug, "[truncate_prompts] truncate query #{} prompt length from {} to {}".format(
                            idx, prompt_len, self.max_prompt_length))
                    input_ids = input_ids[-self.max_prompt_length:]
                truncate_input_ids.append(input_ids)
        # special case for calculate ppl
        elif isinstance(prompts, list) and all(isinstance(prompt, list) for prompt in prompts):
            for idx, input_ids in enumerate(prompts):
                if calculate_loss:
                    assert len(input_ids) <= self.max_prompt_length, "invalid input_ids length"
                else:
                    if (len(input_ids) == 0):
                        logging_rank(logging.info, "[prompts tokenization] detect invalid empty input_ids")
                        input_ids = [self.pad_token_id]
                    prompt_len = len(input_ids)
                    if prompt_len > self.max_prompt_length and self.enable_truncation:
                        logging_rank(
                            logging.debug, "[truncate_prompts] truncate query #{} prompt length from {} to {}".format(
                                idx, prompt_len, self.max_prompt_length))
                        input_ids = input_ids[-self.max_prompt_length:]
                truncate_input_ids.append(input_ids)
        return truncate_input_ids

    def find_longest_common_prefix(self, input_ids_list):
        size = len(input_ids_list)
        assert (size > 0)
        if size == 1 or not int(os.getenv("USE_SESSION_CACHE", 1)):
            self.common_prefix = []
            return

        if self.enable_paged_attn:
            self.common_prefix = []
            logging_rank(logging.error, "VLLM not support find_longest_common_prefix yet")
            return

        end = min([len(input_ids) for input_ids in input_ids_list])

        i = 0
        while (i < end) and \
            all(input_ids_list[0][i] == input_ids_list[j][i] for j in range(1, size)):
            i += 1

        # in case we skip the truncation
        i = min(i, self.max_length)
        self.common_prefix = input_ids_list[0][0:i]
        return

    def _prepare_codebooks(self, input_ids, code_book=None):
        if code_book is None and self.enable_ngrams_decoding:
            logging.warning("code_books is None while enable n-grams decoding, prompts will be used as code_books!")
            return input_ids
        elif isinstance(code_book, str):
            return self.tokenizer(code_book, padding=True, return_tensors="pt").input_ids.tolist()[0]
        elif isinstance(code_book, list) and isinstance(code_book[0], int):
            return code_book

    def prepare_context_inputs(self, input_ids_list, prompt_meta_info: List[Dict]):
        code_books = [None for _ in range(len(input_ids_list))]
        off_policy_steps = [0 for _ in range(len(input_ids_list))]

        if prompt_meta_info is not None:
            assert (len(input_ids_list) == len(prompt_meta_info)
                   ), "input_ids_list and prompt_meta_info should have the same length, but got {} and {}".format(
                       len(input_ids_list), len(prompt_meta_info))
            code_books = [meta_info.get("code_book", None) for meta_info in prompt_meta_info]
            off_policy_steps = [meta_info.get("off_policy_steps", 0) for meta_info in prompt_meta_info]
            top_k = [meta_info.get("top_k", None) for meta_info in prompt_meta_info]
            top_p = [meta_info.get("top_p", None) for meta_info in prompt_meta_info]
            temperature = [meta_info.get("temperature", None) for meta_info in prompt_meta_info]
            max_new_tokens = [meta_info.get("max_new_tokens", self.max_new_tokens) for meta_info in prompt_meta_info]
            max_length = [meta_info.get("max_length", self.max_length) for meta_info in prompt_meta_info]
        for idx, input_ids in enumerate(input_ids_list):
            code_book = self._prepare_codebooks(input_ids, code_book=code_books[idx])
            prompt = ""
            if self.record_input_prompt:
                prompt = self.tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)
            prefix_already_computed_len = self.common_prefix_tensor_len
            if len(input_ids) <= prefix_already_computed_len:
                prefix_already_computed_len = len(input_ids) - 1
            query = Query(copy.deepcopy(input_ids),
                          code_book=code_book,
                          input_prompt=prompt,
                          idx=idx,
                          prefix_already_computed_len=prefix_already_computed_len)
            query.off_policy_steps = off_policy_steps[idx]
            query.meta_info = prompt_meta_info[idx] if prompt_meta_info is not None else {}
            query.top_k = top_k[idx] if prompt_meta_info is not None else None
            query.top_p = top_p[idx] if prompt_meta_info is not None else None
            query.temperature = temperature[idx] if prompt_meta_info is not None else None
            query.max_new_tokens = max_new_tokens[idx] if prompt_meta_info is not None else self.max_new_tokens
            query.max_length = max_length[idx] if prompt_meta_info is not None else self.max_length
            query.image_data = query.meta_info.pop("image_data", None)
            query.prefill_only = self.mode != "rollout"
            query.attach_session(session=self)
            with self._accepted_queries_mutex:
                self.all_accepted_queries[query.id] = query
            self.unfinished_off_policy_steps_set.add_one(query.off_policy_steps)
            # Add query to waiting queue if input length is within max_length limit
            if len(input_ids) <= self.max_length:
                self.waiting.append(query)
        logging_rank(logging.info, "[prepare_context_inputs] total queries: {}".format(idx + 1))

    def build_prefix_kv_cache(self):
        if len(self.common_prefix) == 0:
            self.common_prefix_tensor_len = 0
            return
        input_ids = torch.tensor([self.common_prefix]).cuda().long()
        self.common_prefix_tensor_len = context_len = input_ids.shape[1]
        input_ids = input_ids.cuda().int()
        kv_index = torch.tensor([0]).cuda().int()
        for context_shift in range(0, input_ids.shape[1], self.context_split_len):
            query_input_ids = input_ids[:, context_shift:context_shift + self.context_split_len]
            logging_rank_only(
                logging.info, 0,
                "prefix trigger context split {} -> {}:{}".format(input_ids.shape[1], context_shift,
                                                                  context_shift + query_input_ids.shape[1]))
            query_total_length = torch.tensor([query_input_ids.shape[1]]).cuda().int()
            context_shifts = torch.tensor([context_shift]).cuda().int()
            self.engine.forward_orca(query_input_ids,
                                     None,
                                     query_total_length,
                                     kv_index,
                                     True,
                                     context_shifts=context_shifts)

        for layer_idx in range(self.engine.module.num_layers):
            kv_cache = self.engine.module._get_kv_cache(layer_idx, self.engine.module.quant_mode != "NO_QUANT")

            # The naive implementation which requires tons of copy
            # dst_kv_cache = torch.zeros_like(kv_cache)
            # valid_slice = kv_cache[:, :, 0, :context_len, :]
            # for i in range(self.max_batch_size):
            #     dst_kv_cache[:, :, i, :context_len, :].copy_(valid_slice, non_blocking=True)

            cu_seqlens = torch.Tensor([0 for _ in range(self.max_batch_size + 1)]).int().cuda()
            src_idx = torch.Tensor([0 for _ in range(self.max_batch_size)]).long().cuda()
            torch.classes.XGPT.ReorderKVCache().inference(kv_cache, cu_seqlens, src_idx, context_len + 1)
        return

    def empty_cache(self):
        self.cache_manager.empty_cache()
        self.infer_scheduler.empty_cache()
        self.stop_signal_tensor = torch.tensor([0.0]).float().cuda()
        self.waiting = []
        self.running = []
        self.paused_ready = []
        self.paused_ready_step = 0
        self.all_accepted_queries = {}

    def get_inorder_responses(self):
        ordered_query = list({
            k: v for k, v in sorted(self.all_accepted_queries.items(), key=lambda item: item[1].idx)
        }.values())
        return ordered_query

    def get_valid_history_ids(self) -> List[str]:
        history_ids = []
        if self.prefix_cache is not None:
            self.prefix_cache: PrefixCache
            history_ids.extend(self.prefix_cache.req_id_to_slot_id.keys())
        return history_ids

    def get_all_queries(self, query_type: str, retain_finished: bool = True) -> List[Query]:
        """
        在async streaming模式下，读取所有query的状态和生成结果(包括中间结果)，并把完成的剔除掉
        注意：跟get_inorder_responses互斥，两者不可同时调用，用这个方法后，其他地方都不能再调用get_inorder_responses
        :param query_type: 区分一下validation/hybrid_rollout/standalone_rollout等，避免一个engine实例同时gen多个来源的时候，
                           来源侧不知道怎么取回之前add过来的。
        :param retain_finished: 取走后，如果已经finished，就不会再留在engine里
        """
        ret = []
        with self._accepted_queries_mutex:
            for q in self.all_accepted_queries.values():
                qt = q.meta_info.get('query_type')
                if qt == query_type:
                    ret_q = q.clone()
                    ret_q.detach()
                    ret.append(ret_q)

            if not retain_finished:
                for q in ret:
                    if q.is_finished:
                        self.all_accepted_queries.pop(q.id)
        return ret

    def abort(self, query_ids: List[str], not_after: float):
        for query_id in query_ids:
            self.queries_to_abort.put((query_id, not_after))

    def get_load_metrics(self) -> LoadMetric:
        num_prefill = 0
        num_decode = 0
        for q in self.running:
            if q.is_finished:
                continue
            if q.is_context_computing:
                num_prefill += 1
            else:
                num_decode += 1
        return LoadMetric(
            ts=time.time(),
            num_pending=len(self.pending),
            num_waiting=len(self.waiting),
            num_prefilling=num_prefill,
            num_decoding=num_decode,
            kv_cache_util=self.cache_manager.get_kv_cache_utils(),
            pending_ids=[q.id for q in self.pending.queue],
            waiting_ids=[q.id for q in self.waiting],
            is_weights_updating=self.status != "running",
        )

    def _finish_query(self, query):
        if self.eos_callback_fn:
            self.eos_callback_fn(query)
        query.set_finished()
        if self.prefix_cache is not None:
            self.prefix_cache: PrefixCache
            full_input_ids = torch.tensor(query.input_ids + query.new_token_ids).cuda()
            is_evict = self.prefix_cache.save_to_cache(query.id, full_input_ids,
                                                       torch.tensor(query.kv_slot_ids).cuda(), self.engine.module)
            if is_evict:
                self.infer_scheduler.incr("evict_count")
        self.unfinished_off_policy_steps_set.remove_one(query.off_policy_steps)
        self.finished_num += 1
        self.cache_manager.release_query(query)
        self.infer_scheduler.record("finished_tokens_by_step", [self.current_steps])
        # only set image_data to None when validation
        if 'image_data' in query.meta_info and query.meta_info.get('validate', False):
            image_data = query.meta_info.pop('image_data')
            del image_data
        if self.enable_metrics and self.num_pred_tokens > 0:
            accepted_steps = len(query.accepted_len)
            if accepted_steps > 0:
                accepted_len = sum(query.accepted_len) / (accepted_steps * self.num_pred_tokens)
                self.infer_scheduler.metrics['accept_len'].append(accepted_len)

    def _try_resume_paused_queries(self):
        if self.paused:
            batch_sync_tp_queries(self.paused, tp_group=self.tp_group)
            new_paused = []
            for query in self.paused:
                query.try_resume_from_paused()
                if query.meet_pause_condition():
                    new_paused.append(query)
                else:
                    threshold = self.num_pred_tokens + 1 if self.enable_ngrams_decoding else 0
                    if self._exceed_length_condition(query, tokens_threshold=threshold) or not query.action:
                        query.output_prompt = self.tokenizer.batch_decode([query.output_tokens
                                                                          ]) if self.decode_output else ""
                        self._finish_query(query)
                    else:
                        self.paused_ready.append(query)
            self.paused = new_paused
        if self.paused_ready_step >= self.paused_batching_step or len(self.paused_ready) >= self.engine.module.tp_size:
            for _ in range(self.engine.module.tp_size):
                if len(self.paused_ready) == 0:
                    break
                query = self.paused_ready.pop()
                if query.is_kv_cache_slot_allocated():
                    self.running.append(query)
                else:
                    self.waiting.append(query)
            self.paused_ready_step = 0
        else:
            self.paused_ready_step += 1

    # Preparing queries for next forward
    def _select_running_queries(self):
        for query in self.waiting:
            query.lazy_init_from_prompt_once(self.tokenizer)
        return self.cache_manager.update_queries(self.running, self.waiting, self.paused)

    def _prepare_image_embeds(self, input_ids, query, is_oe):
        # image embeddings has already been computed
        if query.input_embedding is not None:
            return query.input_embedding

        input_ids = input_ids[:, query.context_shift:]
        if query.image_shift < query.image_data['image_grid_hw'].shape[0]:
            # get increasemental image embeddings
            image_grid_hw = query.image_data['image_grid_hw'][query.image_shift:]
            pixel_values = query.image_data['pixel_values'][-(image_grid_hw[:, 0] * image_grid_hw[:, 1]).sum():]
            query.image_shift += image_grid_hw.shape[0]

            assert pixel_values is not None
            if isinstance(pixel_values, np.ndarray):
                pixel_values = convert_numpy_to_tensor(pixel_values, float)
            pixel_values = pixel_values.to(torch.bfloat16).cuda(non_blocking=True)
            if isinstance(image_grid_hw, np.ndarray):
                image_grid_hw = convert_numpy_to_tensor(image_grid_hw, int)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if self.vit_use_xperf_gpt:
                    img_emb = self.vit_engine.visual_encoder(pixel_values, grid_hw=image_grid_hw)
                    if self.vit_engine.ln_vision is not None:
                        img_emb = self.vit_engine.ln_vision(img_emb)
                    if self.vit_engine.seed_proj is not None:
                        img_emb = self.vit_engine.seed_proj(img_emb)
                else:
                    img_emb = self.vit_engine.get_image_features(pixel_values, image_grid_hw)

            image_token_id = -100
            image_mask = input_ids == image_token_id
            # fill image tokens to padding tokens, to avoid negative token_ids for text embedding
            input_ids[image_mask] = 1
            text_embeds = self._get_input_embeddings(input_ids, query, is_oe)
            image_mask = image_mask.unsqueeze(-1).expand_as(text_embeds).to(text_embeds.device)
            img_emb = img_emb.to(text_embeds.device)
            text_embeds = text_embeds.masked_scatter(image_mask, img_emb)
        else:
            text_embeds = self._get_input_embeddings(input_ids, query, is_oe)
        query.input_embedding = text_embeds
        return query.input_embedding

    def _prepare_image_embeds_dp(self, running, is_oe):
        image_queries = [
            query for query in running if query.is_context_computing and query.image_data is not None and
            query.input_embedding is None and query.image_shift < query.image_data['image_grid_hw'].shape[0]
        ]
        if image_queries:
            tp_size = self.engine.module.tp_size
            tp_rank = torch.distributed.get_rank(group=self.tp_group)
            hidden_size = self.engine.config.model_config['hidden_size']
            assert tp_size >= len(image_queries), "context_limit_bs must smaller than tp_size when use dp vit"
            img_token_len = [0] * tp_size
            for i, query in enumerate(image_queries):
                input_ids = torch.tensor(query.input_ids, device="cuda")[query.context_shift:]
                img_token_len[i] = (input_ids == -100).sum().item()
            if tp_rank < len(image_queries):
                query = image_queries[tp_rank]
                image_grid_hw = query.image_data['image_grid_hw'][query.image_shift:]
                pixel_values = query.image_data['pixel_values'][-(image_grid_hw[:, 0] * image_grid_hw[:, 1]).sum():]
                if isinstance(image_grid_hw, np.ndarray):
                    image_grid_hw = torch.from_numpy(image_grid_hw.astype(int))
                if isinstance(pixel_values, np.ndarray):
                    pixel_values = torch.from_numpy(pixel_values.astype(float))
                pixel_values = pixel_values.to(torch.bfloat16).cuda(non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if self.vit_use_xperf_gpt:
                        img_emb = self.vit_engine.visual_encoder(pixel_values, grid_hw=image_grid_hw)
                        if self.vit_engine.ln_vision is not None:
                            img_emb = self.vit_engine.ln_vision(img_emb)
                        if self.vit_engine.seed_proj is not None:
                            img_emb = self.vit_engine.seed_proj(img_emb)
                    else:
                        img_emb = self.vit_engine.get_image_features(pixel_values, image_grid_hw)
            else:
                img_emb = torch.empty((0, hidden_size), dtype=torch.bfloat16, device="cuda")
            all_emb_list = [
                torch.empty((img_token_len[i], hidden_size), dtype=torch.bfloat16, device="cuda")
                for i in range(tp_size)
            ]
            torch.distributed.all_gather(all_emb_list, img_emb, group=self.tp_group)

            for i, query in enumerate(image_queries):
                input_ids = torch.tensor(query.input_ids, device="cuda").unsqueeze(0)
                input_ids = input_ids[:, query.context_shift:]
                image_token_id = -100
                image_mask = input_ids == image_token_id
                input_ids[image_mask] = 1
                text_embeds = self._get_input_embeddings(input_ids, query, is_oe)
                image_mask = image_mask.unsqueeze(-1).expand_as(text_embeds).to(text_embeds.device)
                text_embeds = text_embeds.masked_scatter(image_mask, all_emb_list[i].to(text_embeds.device))
                query.input_embedding = text_embeds
                query.image_shift = query.image_data['image_grid_hw'].shape[0]

    def _get_input_embeddings(self,
                              input_ids,
                              query,
                              is_oe: Optional[bool] = False,
                              start: int = None,
                              end: int = None):
        if start is not None and end is not None:  # text
            input_ids = input_ids[:, start:end]
            context_shift = start
        else:  # vlm
            context_shift = query.context_shift

        if is_oe:
            raw_inputs_id = torch.tensor(query.input_ids, device="cuda").unsqueeze(0)
            oe_history = raw_inputs_id[:, max(context_shift - self.oe_max_stride + 1, 0):context_shift]
            pad_len = (self.oe_max_stride - 1) - oe_history.shape[1]
            oe_history = F.pad(oe_history, (pad_len, 0), value=self.pad_token_id)
            step_seq_length = torch.tensor((input_ids.shape[1]), device=input_ids.device, dtype=torch.int).unsqueeze(0)
            total_seq_length = torch.tensor((input_ids.shape[1] + context_shift),
                                            device=input_ids.device,
                                            dtype=torch.int).unsqueeze(0)
            input_ids = torch.concat([oe_history, input_ids], dim=1)
            return self.engine.get_input_oe_embeddings(input_ids, step_seq_length, total_seq_length)
        return self.engine.get_input_embeddings(input_ids=input_ids)

    def _prepare_forward_inputs(self, running: List[Query]):
        max_context_len = -1
        max_kv_index_len = -1
        phase0_index = []  # index in running
        phase0_list = []
        phase0_labels_list = []  # for log probs only
        phase1_index = []
        phase1_list = []
        phase0_total_length = []
        phase1_total_length = []
        phase1_oe_history = []
        phase0_kv_index = []
        phase1_kv_index = []
        context_shift = []
        history_ids = []
        # for ngrams
        keys_list = []
        code_books_list = []
        max_code_book_len = -1
        sample_kwargs = {"top_k": [], "top_p": [], "temperature": []}
        # for MTP
        max_draft_len = -1
        draft_list = []
        draft_oe_histroy = []
        draft_total_length = []
        target_hidden_states = []
        # MTP is PD separate
        prefill_only = self.enable_mtp_decoding and any([query.is_context_computing for query in running])

        if self.vit_use_dp:
            self._prepare_image_embeds_dp(running, self.oe_max_stride > 1)

        for index, query in enumerate(running):
            assert len(query.input_ids) > 0 and len(query.input_ids) <= self.max_length, \
                f"input_ids length {len(query.input_ids)} must be greater than 0 and less than or equal to max_length {self.max_length}"
            context_len = len(query.input_ids) - query.prefix_already_computed_len
            max_kv_index_len = max(max_kv_index_len, len(query.kv_slot_ids))
            if query.is_context_computing:

                def _get_inp_embs_and_labels(input_ids, start: int, end: int):
                    is_vlm = query.image_data is not None
                    labels_ids = input_ids + [self.pad_token_id]
                    labels_ids = labels_ids[start + 1:end + 1]
                    input_ids = torch.tensor(input_ids).cuda().unsqueeze(0)
                    is_oe = self.oe_max_stride > 1
                    if is_vlm:
                        input_embs = self._prepare_image_embeds(input_ids, query, is_oe)
                        cur_shift = end - start
                        input_embs = input_embs[:, query.image_context_shift:query.image_context_shift + cur_shift, :]
                        query.image_context_shift += cur_shift
                    else:
                        input_embs = self._get_input_embeddings(input_ids, query, is_oe, start, end)
                    return input_embs, torch.tensor(labels_ids)

                self.infer_scheduler.record("prefill_token_num", [len(query.input_ids)])
                if self.prefix_cache is not None:
                    self.prefix_cache: PrefixCache
                    # note: only query prefix cache once (context_shift == 0)
                    if query.context_shift == 0:
                        input_ids_cuda = torch.tensor(query.input_ids).cuda()
                        prefix_hit_length = self.prefix_cache.calc_prefix_length(query.id, input_ids_cuda)
                        self.infer_scheduler.record("prefix_cache_hit_length", [prefix_hit_length])
                        if prefix_hit_length > query.prefix_already_computed_len:
                            self.prefix_cache.load_from_cache(query.id, input_ids_cuda,
                                                              torch.tensor(query.kv_slot_ids).cuda(),
                                                              self.engine.module, prefix_hit_length)
                            query.prefix_already_computed_len = prefix_hit_length
                            context_len = len(query.input_ids) - query.prefix_already_computed_len

                current_context_shift = query.context_shift + query.prefix_already_computed_len
                # start from context_shift pos
                if context_len > self.context_split_len or query.context_shift > 0:
                    query_input_emb, labels_ids = _get_inp_embs_and_labels(
                        query.input_ids, current_context_shift, current_context_shift + self.context_split_len)
                    logging_rank_only(
                        logging.debug, 0,
                        "trigger context split {} -> {}:{}".format(context_len, current_context_shift,
                                                                   current_context_shift + context_len))
                    context_len = query_input_emb.shape[1]
                else:
                    query_input_emb, labels_ids = _get_inp_embs_and_labels(query.input_ids, current_context_shift,
                                                                           len(query.input_ids))

                query.context_shift += query_input_emb.shape[1]
                max_context_len = max(max_context_len, context_len)
                phase0_index.append(index)
                phase0_list.append(query_input_emb)
                phase0_labels_list.append(labels_ids)
                phase0_total_length.append(context_len)
                phase0_kv_index.append(query.kv_slot_ids) if self.enable_paged_attn else phase0_kv_index.extend(
                    query.kv_slot_ids)
                context_shift.append(current_context_shift)

            elif not prefill_only:
                phase1_index.append(index)
                phase1_list.append(query.new_token_ids[-1])
                phase1_total_length.append(context_len + len(query.new_token_ids) + query.prefix_already_computed_len)
                phase1_kv_index.append(query.kv_slot_ids) if self.enable_paged_attn else phase1_kv_index.extend(
                    query.kv_slot_ids)
                if len(query.new_token_ids) >= self.oe_max_stride:
                    oe_history = query.new_token_ids[-self.oe_max_stride:-1]
                else:
                    oe_history = query.input_ids[-(self.oe_max_stride - len(query.new_token_ids) + 1) +
                                                 1:] + query.new_token_ids[:-1]
                phase1_oe_history.append(oe_history)
                if self.enable_ngrams_decoding:
                    max_code_book_len = max(max_code_book_len, len(query.code_book))
                    key = query.new_token_ids[-self.max_ngram_size:]
                    if len(key) < self.max_ngram_size:
                        key = [self.pad_token_id] * (self.max_ngram_size - len(key)) + key
                    keys_list.append(key)
                    code_books_list.append(query.code_book)
                if self.enable_mtp_decoding:
                    if len(query.new_token_ids) <= 1:
                        # prepare mtp layer kv cache
                        draft_list.append(query.input_ids[1:] + query.new_token_ids)
                        draft_oe_histroy.append(query.input_ids[:1])
                    else:
                        draft_list.append(query.new_token_ids[-query.accepted_len[-1] - 1:])
                        if len(query.new_token_ids) - query.accepted_len[-1] >= self.oe_max_stride:
                            draft_oe_histroy.append(
                                query.new_token_ids[-self.oe_max_stride -
                                                    query.accepted_len[-1]:-query.accepted_len[-1] - 1])
                        else:
                            draft_oe_histroy.append(query.input_ids[-(self.oe_max_stride - len(query.new_token_ids) +
                                                                      query.accepted_len[-1]):] +
                                                    query.new_token_ids[:-query.accepted_len[-1] - 1])
                    max_draft_len = max(len(draft_list[-1]), max_draft_len)
                    draft_total_length.append(len(draft_list[-1]))
                    target_hidden_states.append(query.hidden_states)
            # set sample args for each query
            for k, v in sample_kwargs.items():
                v.append(getattr(query, k, None))

        results = dict()

        # left pad context_input
        if len(phase0_list) > 0:
            context_emb = None
            context_input_ids = None
            for i, query in enumerate(phase0_list):
                pad_tokens_len = max_context_len - query.shape[1]
                if pad_tokens_len > 0:
                    phase0_list[i] = F.pad(query, (0, 0, pad_tokens_len, 0))
                    running[phase0_index[i]].cur_batch_pad_token = pad_tokens_len
                if context_emb is None:
                    context_emb = phase0_list[i]
                    context_labels_ids = phase0_labels_list[i]
                else:
                    context_emb = torch.concat([context_emb, phase0_list[i]], dim=0)
                    context_labels_ids = torch.concat([context_labels_ids, phase0_labels_list[i]], dim=0)
            context_input = context_emb
            results['context_input'] = context_input
            results['context_labels_ids'] = context_labels_ids
        else:
            results['context_input'] = None
            results['context_labels_ids'] = None

        if len(phase1_list) > 0:
            if self.oe_max_stride > 1:
                for i, (query, oe_history) in enumerate(zip(phase1_list, phase1_oe_history)):
                    phase1_list[i] = [self.pad_token_id] * (self.oe_max_stride - 1 - len(oe_history)) + oe_history + [
                        query
                    ]
                decode_input = torch.tensor(phase1_list, dtype=torch.int64, device="cuda")
            else:
                decode_input = torch.tensor(phase1_list, dtype=torch.int64, device="cuda").unsqueeze(1)
            for i, code_book in enumerate(code_books_list):
                code_books_list[i] = code_book + [self.pad_token_id] * (max_code_book_len - len(code_book))
            results['decode_input'] = decode_input
        else:
            results['decode_input'] = None

        if len(draft_list) > 0:
            for i, (query, oe_histroy) in enumerate(zip(draft_list, draft_oe_histroy)):
                if self.oe_max_stride > 1:
                    draft_list[i] = [self.pad_token_id] * (max_draft_len + self.oe_max_stride - 1 - len(query) -
                                                           len(oe_histroy)) + oe_histroy + query
                else:
                    draft_list[i] = [self.pad_token_id] * (max_draft_len - len(query)) + query
            results['draft_input'] = torch.tensor(draft_list, dtype=torch.int64, device="cuda")
            results['draft_total_length'] = torch.tensor(draft_total_length, dtype=torch.int, device="cuda")
            results['target_hidden_states'] = torch.cat(target_hidden_states, dim=0)
        else:
            results['draft_input'] = None
            results['draft_total_length'] = None
            results['target_hidden_states'] = None

        results['forward_index'] = phase0_index + phase1_index

        # right pad the page_table (required by flash2)
        if self.enable_paged_attn:
            for i, kv_index in enumerate(phase0_kv_index):
                kv_list = kv_index + [-1] * (max_kv_index_len - len(kv_index))
                phase0_kv_index[i] = kv_list
            for i, kv_index in enumerate(phase1_kv_index):
                kv_list = kv_index + [-1] * (max_kv_index_len - len(kv_index))
                phase1_kv_index[i] = kv_list

        bs_total_length = len(phase0_total_length + phase1_total_length)
        bs_context_shift = len(context_shift)
        total_input_list = []

        if self.enable_paged_attn:
            packed_kv_index = phase0_kv_index + phase1_kv_index
            packed_kv_rows = len(packed_kv_index)
            packed_kv_cols = len(packed_kv_index[0]) if packed_kv_rows > 0 else 0
            packed_kv_index_1d = [item for sublist in packed_kv_index for item in sublist]
            total_input_list.extend(packed_kv_index_1d)
            bs_kv = packed_kv_rows * packed_kv_cols
        else:
            packed_kv_index = phase0_kv_index + phase1_kv_index
            total_input_list.extend(packed_kv_index)
            bs_kv = len(packed_kv_index)

        total_input_list.extend(phase0_total_length + phase1_total_length)
        total_input_list.extend(context_shift)

        total_input_tensor = torch.tensor(total_input_list, dtype=torch.int32, device="cuda")
        packed_kv_index = total_input_tensor[:bs_kv].view(
            -1, packed_kv_cols) if self.enable_paged_attn else total_input_tensor[:bs_kv]
        total_length = total_input_tensor[bs_kv:bs_kv + bs_total_length]
        context_shift_tensor = total_input_tensor[bs_kv + bs_total_length:bs_kv + bs_total_length + bs_context_shift]
        if logging.root.level <= logging.DEBUG:
            logging_rank_only(
                logging.debug, 0,
                f"context_input: {context_input}\ndecode_input: {decode_input}\ntotal_length: {total_length.tolist()}\npacked_kv_index: {packed_kv_index.tolist()}\ncontext_shift: {context_shift_tensor}"
            )
        results['kv_index'] = packed_kv_index

        results['total_length'] = total_length
        results['context_shifts'] = context_shift_tensor
        results['history_ids'] = history_ids
        results['sample_kwargs'] = sample_kwargs
        return results

    def _should_terminate(self, prompts, complete_ratio, stop_event):
        # complete ratio break, only works when stop event not set
        if stop_event is None and self.finished_num >= int(complete_ratio * len(prompts)):
            return True
        if stop_event is not None:
            stop_event_is_set = stop_event.is_set()  # noqa: py-spy
            # wait for the max_off_policy rollout
            if (not self.unfinished_off_policy_steps_set.is_empty()) and (
                    self.unfinished_off_policy_steps_set.get_max() > self.max_off_policy_steps):
                return False

            # stop event break
            if (stop_event.is_set()):
                self.stop_signal_tensor.fill_(1.0)
            else:
                self.stop_signal_tensor.zero_()
            # reuse first nccl layer to comm signal
            if self.engine.module.tp_size > 1:
                assert self.tp_group is not None, "tp_group not set!"
                torch.distributed.all_reduce(self.stop_signal_tensor, group=self.tp_group)

            if self.stop_signal_tensor.item() == self.engine.module.tp_size:
                self.stop_signal_tensor.zero_()
                return True
        return False

    # abort掉abort queue 里的 query
    def _remove_aborted_queries(self):
        num_abort = self.queries_to_abort.qsize()
        num_abort_local = num_abort
        num_abort_query_local_tensor = torch.tensor([num_abort], dtype=torch.int32, device="cuda")
        if self.engine.module.tp_size > 1:
            assert self.tp_group is not None, "tp_group not set!"
            reduce_op = torch.distributed.ReduceOp.MIN
            torch.distributed.all_reduce(num_abort_query_local_tensor, group=self.tp_group, op=reduce_op)
            num_abort = num_abort_query_local_tensor.int().item()

        if num_abort == 0:
            return num_abort, num_abort_local

        # get minimum synchronized aborts
        to_abort: Dict[str, float] = {}  # query_id -> abort ts before
        for i in range(num_abort):
            query_id, not_after = self.queries_to_abort.get()
            to_abort[query_id] = not_after  # 如果query_id有重复，那queue后面的时间戳肯定大于前面的时间戳

        # remove from local list
        with self._accepted_queries_mutex:
            for query_id, not_after in to_abort.items():
                q = self.all_accepted_queries.get(query_id)
                if q is None or q.dispatch_time >= not_after:
                    continue
                self.all_accepted_queries.pop(query_id)
        self.pending.remove(to_abort)
        removed = []
        removed.extend(_remove_query_list_inplace(self.paused, to_abort))
        removed.extend(_remove_query_list_inplace(self.waiting, to_abort))
        removed.extend(_remove_query_list_inplace(self.running, to_abort))

        # release kv cache
        for q in removed:
            self.cache_manager.release_query(q)

        return num_abort, num_abort_local

    # consume more queries from pending to waiting list
    def _fetch_from_pending_queries(self) -> List[Query]:
        num_ready_query_local_tensor = torch.tensor([len(self.pending)], dtype=torch.float32, device="cuda")
        num_ready = len(self.pending)
        if self.engine.module.tp_size > 1:
            assert self.tp_group is not None, "tp_group not set!"
            reduce_op = torch.distributed.ReduceOp.MIN
            torch.distributed.all_reduce(num_ready_query_local_tensor, group=self.tp_group, op=reduce_op)
            num_ready = num_ready_query_local_tensor.int().item()
        if num_ready == 0:
            return self.waiting
        new_joins: List[AsyncQuery] = self.pending.get_earliest(num_ready)
        new_queries = [aq.query for aq in new_joins]
        with self._accepted_queries_mutex:
            for q in new_queries:
                q.attach_session(session=self)
                self.all_accepted_queries[q.id] = q
                self.unfinished_off_policy_steps_set.add_one(q.off_policy_steps)
        self.pending.truncate(num_ready)
        return self.waiting + new_queries

    def _wait_all_paused_resume(self):
        # wait all paused queries to resume. this is necessary for plugin call:
        # paused queries may have pending plugin calls, which makes it impossible to be serialized
        # and get_resume_state could fail.
        while len(self.paused) > 0:
            self._try_resume_paused_queries()
            time.sleep(0.1)

    def execute(self, prompts, complete_ratio=1, stop_event=None, prompt_meta_info: List[Dict] = None):
        """Main inference execution loop
        
        Args:
            prompts: List of text prompts or tokenized IDs
            complete_ratio: Stop when this fraction of queries complete
            stop_event: External termination signal
            prompt_meta_info: Additional per-prompt metadata
            
        Returns:
            List of completed Query objects with results by method get_inorder_responses()
        """
        if "XPERF_RANDOM_SEED" in os.environ:
            torch.manual_seed(int(os.getenv('XPERF_RANDOM_SEED', '0')))
        input_ids_list = self.truncate_prompts(prompts, False)
        self.find_longest_common_prefix(input_ids_list)
        if not self.is_xperf_custom and not self.is_xperf_triton:
            self.build_prefix_kv_cache()
            logging_rank(
                logging.info, "find common prefix which contains {} tokens, reuse this kv cache!".format(
                    self.common_prefix_tensor_len))
        self.prepare_context_inputs(input_ids_list, prompt_meta_info)
        self.current_steps = 0
        self.finished_num = 0
        if self.step_profiler is not None:
            self.step_profiler.reset_exec()
        while not self._should_terminate(prompts, complete_ratio, stop_event):
            self._try_resume_paused_queries()
            self.running, self.waiting = self._select_running_queries()
            if len(self.running) == 0:
                assert (len(self.waiting) == 0)
                if len(self.paused) > 0:
                    # all queries are paused, wait for a while
                    time.sleep(0.1)
                    continue
                elif len(self.paused_ready) > 0:
                    self.paused_ready_step = self.paused_batching_step
                    continue
                else:
                    break
            self.current_steps += 1
            forward_inputs = self._prepare_forward_inputs(self.running)
            context_input = forward_inputs['context_input']
            decode_input = forward_inputs['decode_input']
            next_tokens, accepted_len, hidden_states, log_probs = self.infer_scheduler.forward_and_sample(
                context_input=context_input,
                decode_input=decode_input,
                total_length=forward_inputs['total_length'],
                kv_index=forward_inputs['kv_index'],
                orca_updated=True,
                context_shifts=forward_inputs['context_shifts'],
                context_labels_ids=forward_inputs['context_labels_ids'],
                history_ids=forward_inputs['history_ids'],
                sample_kwargs=forward_inputs['sample_kwargs'],
                draft_input=forward_inputs['draft_input'],
                draft_total_length=forward_inputs['draft_total_length'],
                target_hidden_states=forward_inputs['target_hidden_states'])

            if self.engine.module.tp_size > 1 and next_tokens is not None:
                assert self.tp_group is not None, "tp_group not set!"
                tp_src_rank = torch.distributed.get_global_rank(self.tp_group, group_rank=0)
                torch.distributed.broadcast(next_tokens, src=tp_src_rank, group=self.tp_group)

            self._update_running_batch(next_tokens=next_tokens,
                                       accepted_len=accepted_len,
                                       index_in_running_batch=forward_inputs['forward_index'],
                                       log_probs=log_probs,
                                       hidden_states=hidden_states)
            has_prefill_input = context_input is not None
            self.infer_scheduler.next_step(has_prefill_input)
            if self.step_profiler is not None:
                ctx_tokens = context_input.shape[0] if context_input is not None else 0
                dec_tokens = decode_input.shape[0] if decode_input is not None else 0
                self.step_profiler.record_step(ctx_tokens=ctx_tokens, dec_tokens=dec_tokens)

        self._wait_all_paused_resume()
        if self.prefix_cache is not None:
            self.prefix_cache.clear_cache()
        torch.cuda.synchronize()
        self.infer_scheduler.record("cur_steps", [self.current_steps])

    def async_execute(self, update_weight_event):
        import time
        if "XPERF_RANDOM_SEED" in os.environ:
            torch.manual_seed(int(os.getenv('XPERF_RANDOM_SEED', '0')))
        self.current_steps = 0
        self.finished_num = 0
        last_time = 0

        def _check_stop_event():
            while (self._should_terminate(None, 1.0, stop_event=update_weight_event)):
                # recompute prefill
                for query in self.running:
                    self.cache_manager.release_query(query)
                    query.reset_compute()
                    self.waiting.append(query)
                self.running = []
                if self.prefix_cache is not None:
                    self.prefix_cache.clear_cache()
                update_weight_event_is_set = update_weight_event.is_set()  # noqa: for py-spy
                time.sleep(0.01)
                return True
            return False

        def _idle():
            return (len(self.waiting) == 0 and len(self.running) == 0)

        while True:
            try:
                self.status = "running"
                if (_check_stop_event()):
                    break
                # each rank should have the same running and waiting
                if (_idle()) or (self.current_steps % 20 == 0):
                    self._remove_aborted_queries()
                    self.waiting = self._fetch_from_pending_queries()
                self._try_resume_paused_queries()
                self.running, self.waiting = self._select_running_queries()
                if _idle():
                    if len(self.paused) > 0:
                        time.sleep(0.1)
                    continue
                self.current_steps += 1

                forward_inputs = self._prepare_forward_inputs(self.running)
                context_input = forward_inputs['context_input']
                decode_input = forward_inputs['decode_input']
                # if self.current_steps % 100 == 0:
                #     ctx_tokens = context_input.shape[0] if context_input is not None else 0
                #     dec_tokens = decode_input.shape[0] if decode_input is not None else 0
                #     print(
                #         f"{self.current_steps}: ctx_tokens: {ctx_tokens}, dec_tokens: {dec_tokens}, swap tokens: {self.cache_manager.page_swap_out_token}, per step: {(time.time() - last_time) / 100 * 1000} ms"
                #     )
                #     last_time = time.time()
                next_tokens, accepted_len, hidden_states, log_probs = self.infer_scheduler.forward_and_sample(
                    context_input=context_input,
                    decode_input=decode_input,
                    total_length=forward_inputs['total_length'],
                    kv_index=forward_inputs['kv_index'],
                    orca_updated=True,
                    context_labels_ids=None,
                    context_shifts=forward_inputs['context_shifts'],
                    history_ids=forward_inputs['history_ids'],
                    sample_kwargs=forward_inputs['sample_kwargs'],
                    draft_input=forward_inputs['draft_input'],
                    draft_total_length=forward_inputs['draft_total_length'],
                    target_hidden_states=forward_inputs['target_hidden_states'])

                if self.engine.module.tp_size > 1 and next_tokens is not None:
                    assert self.tp_group is not None, "tp_group not set!"
                    tp_src_rank = torch.distributed.get_global_rank(self.tp_group, group_rank=0)
                    torch.distributed.broadcast(next_tokens, src=tp_src_rank, group=self.tp_group)

                self._update_running_batch(next_tokens=next_tokens,
                                           accepted_len=accepted_len,
                                           index_in_running_batch=forward_inputs['forward_index'],
                                           log_probs=log_probs,
                                           hidden_states=hidden_states)
                has_prefill_input = context_input is not None
                self.infer_scheduler.next_step(has_prefill_input)
                if self.step_profiler is not None:
                    ctx_tokens = context_input.shape[0] if context_input is not None else 0
                    dec_tokens = decode_input.shape[0] if decode_input is not None else 0
                    self.step_profiler.record_step(ctx_tokens=ctx_tokens, dec_tokens=dec_tokens)
            except Exception as e:
                for query in self.running:
                    query.set_finished(exception=e)
                logging.error(f"Error handling request: {str(e)}")
                raise (e)
        self._wait_all_paused_resume()
        self.status = "idle"

    def _exceed_length_condition(self, query, tokens_threshold):
        return query.new_token_len + tokens_threshold >= query.max_new_tokens or len(query.new_token_ids) + len(
            query.input_ids) + tokens_threshold >= query.max_length

    def _meet_eos_condition(self, query, next_token):
        finished_sequences = False
        if next_token in self.eos_token_id:
            query.output_prompt = self.tokenizer.batch_decode([query.output_tokens[:-1]]) \
                if self.decode_output else ""
            finished_sequences = True
        elif self._exceed_length_condition(query,
                                           tokens_threshold=self.num_pred_tokens +
                                           1 if self.enable_ngrams_decoding else 0):
            query.output_prompt = self.tokenizer.batch_decode([query.output_tokens]) \
                if self.decode_output else ""
            finished_sequences = True
        elif self.stop_sequence_tokens and next_token in [tokens[-1] for tokens in self.stop_sequence_tokens]:
            for stop_sequences in self.stop_sequence_tokens:
                seq_len = len(stop_sequences)
                if query.new_token_len < seq_len:
                    continue
                finished_sequences |= (query.new_token_ids[-seq_len:] == stop_sequences)
            if finished_sequences:
                query.output_prompt = self.tokenizer.batch_decode([query.output_tokens]) \
                    if self.decode_output else ""

        return finished_sequences

    def _update_running_batch(self,
                              next_tokens,
                              index_in_running_batch,
                              accepted_len=None,
                              log_probs=None,
                              hidden_states=None):
        next_running = [[], []]
        new_paused = []
        next_tokens = next_tokens.cpu().tolist() if next_tokens is not None else None
        if accepted_len is not None:
            assert (accepted_len.shape[0] == len(index_in_running_batch))
            accepted_len = accepted_len.cpu().tolist()
        if log_probs is not None:
            assert (log_probs.shape[0] == len(index_in_running_batch)), (
                f"log_probs shape mismatch, {log_probs.shape[0]} vs {len(index_in_running_batch)}")
            log_probs = log_probs.cpu().tolist()
        running_index_to_i = {idx: i for i, idx in enumerate(index_in_running_batch)}
        for idx, query in enumerate(self.running):
            # prefill only step
            if idx not in running_index_to_i:
                next_running[1].append(query)
                continue
            i = running_index_to_i[idx]
            # decoding
            if query.is_to_decoding_compute():
                if query.prefill_only:
                    query.add_token(token_id=self.pad_token_id,
                                    accepted_len=accepted_len[i] if accepted_len is not None else 0,
                                    log_prob=log_probs[i] if log_probs is not None else 0)
                    finished_sequences = True
                else:
                    # Each query might have multile next tokens when spec/ngrams is enabled
                    query_next_tokens = next_tokens[i]
                    if isinstance(query_next_tokens, int):
                        query_next_tokens = [query_next_tokens]
                    query_next_tokens_len = 1 if accepted_len is None else accepted_len[i] + 1
                    finished_sequences = False
                    paused_triggered = False
                    for token_idx in range(query_next_tokens_len):
                        next_token = query_next_tokens[token_idx]
                        if len(query.new_token_ids) == 0:
                            query.recent_first_token_time = time.time() * 1000
                            if not query.first_token_time:
                                query.first_token_time = query.recent_first_token_time
                        query.add_token(token_id=next_token,
                                        accepted_len=accepted_len[i] if accepted_len is not None else 0,
                                        log_prob=log_probs[i] if log_probs is not None else 0)
                        if query.meet_pause_condition():
                            query.pause()
                            new_paused.append(query)
                            paused_triggered = True
                            break
                        elif self._meet_eos_condition(query, next_token):
                            finished_sequences = True
                            break

                if not finished_sequences:
                    if not paused_triggered:
                        if self.enable_mtp_decoding:
                            query.hidden_states = hidden_states[i] if query.hidden_states is None or len(
                                query.new_token_ids) > 1 else torch.cat([query.hidden_states, hidden_states[i]], dim=0)

                        next_running[1].append(query)
                else:
                    self._finish_query(query)
            # still prefill
            else:
                if self.enable_mtp_decoding:
                    query.hidden_states = hidden_states[i] if query.hidden_states is None else torch.cat(
                        [query.hidden_states, hidden_states[i]], dim=0)
                if query.prefill_only:
                    query.log_probs.extend(log_probs[i])
                next_running[0].append(query)

        self.running = next_running[0] + next_running[1]
        self.paused.extend(new_paused)
