"""
Large Language Model Inference Session Manager

Handles end-to-end inference process including:
- Prompt preprocessing
- KV cache management
- Context window optimization
- Token generation with various decoding strategies
- Multi-GPU distributed inference
"""

from xperf_gpt.inference import init_inference
from alpha_seed.workers.xperf_rollout.component.cache_manager import CacheManager
from alpha_seed.workers.xperf_rollout.component.infer_scheduler import InferScheduler
from alpha_seed.workers.xperf_rollout.component.sampling_manager import Sampler
from alpha_seed.workers.xperf_rollout.utils.custom_xperf_convert_helper import XCustomInferenceModuleAdapter
from xperf_gpt.multi_models.visual.eva_vit import EVA_VIT_CONFIGS
from alpha_seed.workers.xperf_rollout.component.query import Query, InflightQueue
from alpha_seed.utils.observility import get_profiler_context_wrapped
from xperf_gpt.utils import (logging_rank, logging_rank_only)
from typing import List, Dict
import logging
import os
import torch
import copy
import logging
from threading import Lock
from transformers import AutoTokenizer
from xperf_gpt.multi_models.visual.inferencer import VITInferencer
import numpy as np

# Constants
BLOCK_SIZE_ALIGNMENT = 256
DEFAULT_LOGGING_LEVEL = logging.INFO


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
        schedule_strategy="default", # ['default','fifo']
        step_profiler: StepProfiler = None,
    ):
        """Initialize inference session with hardware/performance parameters"""
        # Memory management
        self.enable_paged_attn = enable_paged_attn
        self.num_slots = num_slots
        self.slot_block_size = slot_block_size
        self.standalone = standalone
        # Context processing
        self.context_split_len = context_split_len
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
        
        Workflow:
            Requests flow from Pending → Running → (Waiting if interrupted) → Finished. 
            The waiting queue enables stateful handling of mid-execution interruptions through check-pointing mechanisms.
        '''
        self.pending = InflightQueue()
        self.waiting = []
        self.running = []

        self.finished = {}
        self.stop_sequence_tokens: List[List[int]] = []
        self.common_prefix = ""
        self.common_prefix_tensor_len = 0
        self.return_full_hidden_states = False
        self.return_padding_tensor = False
        self.last_token_only = True
        self.eos_callback_fn = None
        self.stop_signal_tensor = torch.tensor([0.0]).float().cuda()
        self.update_weights_lock = Lock()

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

    def _kwargs_wrapper(self, kwargs):
        params_with_defaults = {
            "vanilla_checkpoint_path": None,
            "checkpoint_path": None,
            "reshard_checkpoint_path": None,
            "tokenizer_path": None,
            "save_mp_checkpoint_path": None,
            "return_full_hidden_states": False,
            "return_padding_tensor": False,
            "last_token_only": False,
            "stop_sequence_tokens": None,
            "record_input_prompt": None,
            "enable_metrics": False,
            "constraint_decoding": None,
            "decode_output": True,
        }

        for param, default in params_with_defaults.items():
            if param in kwargs:
                value = kwargs.pop(param)
                setattr(self, param, value)
            else:
                setattr(self, param, default)

    def init_inference_engine(self,
                              session_config_path,
                              generation_config,
                              use_xperf_custom=False,
                              vit_config=None,
                              **kwargs):
        """Initialize model engine and associated components
        
        Args:
            session_config_path: Path to model config JSON
            generation_config: Generation parameters
            use_xperf_custom: Whether use xperf_gpt_custom
            vit_config: vision config dict
            kwargs: Overrides for model loading
        """

        if os.getenv("XPERF_SESSION_SET_TORCH_DEVICE", "1") == "1":
            torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
        self._kwargs_wrapper(kwargs)
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
            model_config_path=session_config_path,
            vanilla_checkpoint_path=self.vanilla_checkpoint_path if not self.checkpoint_path else None,
            checkpoint_path=self.checkpoint_path,
            reshard_checkpoint_path=self.reshard_checkpoint_path,
            save_mp_checkpoint_path=self.save_mp_checkpoint_path,
            eos_token_id=0,  # won't use
            pad_token_id=0,  # won't use
            use_xperf_gpt=True,
            use_orca=True,
            rank0_split=not self.checkpoint_path,  # if preshard, turn off rank0_split
            use_vllm=self.enable_paged_attn,
            slot_block_size=self.slot_block_size,
            max_batch_size=self.max_batch_size,
            max_total_tokens=self.max_total_tokens,
            max_length=self.max_length,
            max_context_shift=self.max_context_shift,
            dtype=torch.bfloat16,
            vocab_tp=self.vocab_tp,
            multi_stream=1,
            **generation_config)
        init_inference_kwargs.update(kwargs)  # overridable by kwargs
        if use_xperf_custom:
            assert not self.enable_paged_attn, f"xperf custom for paged attention not supported yet."
            import xperf_gpt_custom
            backbone = kwargs.pop('xperf_custom_backbone', None)
            preset = kwargs.pop('xperf_custom_preset', None)
            engine = xperf_gpt_custom.create_network(backbone=backbone, preset=preset, **init_inference_kwargs)
            module = XCustomInferenceModuleAdapter(engine)
            setattr(engine, "module", module)
            setattr(engine.config, "model_config", {"hidden_size": engine.config.hidden_size})
            self.engine = engine
            self.is_xperf_custom = True
        else:
            self.engine = init_inference(None, **init_inference_kwargs)
            self.is_xperf_custom = False
        self.sampler = Sampler(generation_config=generation_config)
        self.num_return_sequences = self.engine.module.num_return_sequences

        self.reset_logging_level()

        self.cache_manager = CacheManager(slot_num=self.num_slots,
                                          max_batch_size=self.max_batch_size,
                                          pp_size=1,
                                          use_vllm=self.enable_paged_attn,
                                          slot_block_size=self.slot_block_size,
                                          context_batchsize_limit=self.context_limit_bs,
                                          enable_ngrams_decoding=self.enable_ngrams_decoding,
                                          num_pred_tokens=self.num_pred_tokens,
                                          moving_avg_length=self.max_length,
                                          schedule_strategy=self.schedule_strategy)
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
        )
        if vit_config is not None:
            if vit_config['vit_model'] not in EVA_VIT_CONFIGS.keys():
                vit_config.update(vit_config.get('transformer_config'))
            else:
                detail_config = EVA_VIT_CONFIGS[vit_config['vit_model']]
                vit_config.update(detail_config)
            if vit_config.get('use_navit', False):
                vit_config['navit_anyres'] = True
            self.vit_engine = VITInferencer(vit_config_dict=vit_config,
                                            tokenization_path=self.tokenizer_path).cuda().to(torch.bfloat16)

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

    def find_longest_common_prefix(self, input_ids_list, logits_masks):
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
        if logits_masks is not None:
            max_pre_len = min([mask.index(1) for mask in logits_masks])
            end = min(end, max_pre_len)

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

    def prepare_context_inputs(self, input_ids_list, logits_masks, prompt_meta_info: List[Dict] = None):
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
            for _ in range(self.num_return_sequences):
                prefix_already_computed_len = self.common_prefix_tensor_len
                if len(input_ids) <= prefix_already_computed_len:
                    prefix_already_computed_len = len(input_ids) - 1
                query = Query(copy.deepcopy(input_ids),
                              code_book=code_book,
                              input_prompt=prompt,
                              idx=idx,
                              prefix_already_computed_len=prefix_already_computed_len)
                if logits_masks is not None:
                    self.return_padding_tensor = True
                    assert len(logits_masks[idx]) == len(query.input_ids), "logits_mask length not match input_ids"
                    query.logits_mask = torch.tensor(logits_masks[idx]).long()
                    query.shift_label = torch.tensor(input_ids[1:] + [-1]).cuda()
                if len(input_ids) <= self.max_length:
                    self.waiting.append(query)
                query.off_policy_steps = off_policy_steps[idx]
                query.meta_info = prompt_meta_info[idx] if prompt_meta_info is not None else {}
                query.top_k = top_k[idx] if prompt_meta_info is not None else None
                query.top_p = top_p[idx] if prompt_meta_info is not None else None
                query.temperature = temperature[idx] if prompt_meta_info is not None else None
                query.max_new_tokens = max_new_tokens[idx] if prompt_meta_info is not None else self.max_new_tokens
                query.max_length = max_length[idx] if prompt_meta_info is not None else self.max_length
                self.finished[query.id] = query
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
        self.pending = InflightQueue()
        self.waiting = []
        self.running = []
        self.finished = {}

    def get_inorder_responses(self):
        ordered_query = list({k: v for k, v in sorted(self.finished.items(), key=lambda item: item[1].idx)}.values())
        if self.num_return_sequences > 1:
            responses = []
            for i, query in enumerate(ordered_query):
                if i > 0 and query.idx == responses[-1].idx:
                    responses[-1].output_prompt.append(query.output_prompt[0])
                else:
                    responses.append(query)
            return responses
        else:
            return ordered_query

    # Preparing queries for next forward
    def _select_running_queries(self):
        return self.cache_manager.update_queries(self.running, self.waiting)

    def _prepare_image_embeds(self, input_ids, pixel_values, image_grid_hw):
        assert pixel_values is not None
        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values.astype(float))
        pixel_values = pixel_values.to(torch.bfloat16).cuda(non_blocking=True)
        if isinstance(image_grid_hw, np.ndarray):
            image_grid_hw = torch.from_numpy(image_grid_hw.astype(int))

        # compute image embedding
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            img_emb = self.vit_engine.visual_encoder(pixel_values, grid_hw=image_grid_hw)
            if self.vit_engine.ln_vision is not None:
                img_emb = self.vit_engine.ln_vision(img_emb)
            if self.vit_engine.seed_proj is not None:
                img_emb = self.vit_engine.seed_proj(img_emb)
        image_token_id = -100
        image_mask = input_ids == image_token_id
        # fill image tokens to padding tokens, to avoid negative token_ids for text embedding
        input_ids[image_mask] = 1
        text_embeds = self.engine.get_input_embeddings(input_ids=input_ids)
        image_mask = image_mask.unsqueeze(-1).expand_as(text_embeds).to(text_embeds.device)
        img_emb = img_emb.to(text_embeds.device)
        text_embeds = text_embeds.masked_scatter(image_mask, img_emb)
        return text_embeds

    def _prepare_forward_inputs(self, running: List[Query]):
        max_context_len = -1
        max_kv_index_len = -1
        phase0_index = []  # index in running
        phase0_list = []
        phase1_index = []
        phase1_list = []
        phase0_total_length = []
        phase1_total_length = []
        phase0_kv_index = []
        phase1_kv_index = []
        context_shift = []
        history_ids = []
        # for ngrams
        keys_list = []
        code_books_list = []
        max_code_book_len = -1
        sample_kwargs = {"top_k": [], "top_p": [], "temperature": []}

        for index, query in enumerate(running):
            context_len = len(query.input_ids) - query.prefix_already_computed_len
            max_kv_index_len = max(max_kv_index_len, len(query.kv_slot_ids))
            if query.is_context_computing:
                input_ids = torch.tensor(query.input_ids).cuda()
                if query.input_embedding is None:
                    if query.meta_info.get('pixel_values') is not None:
                        input_embs = self._prepare_image_embeds(input_ids, query.meta_info['pixel_values'],
                                                                query.meta_info['image_grid_hw'])
                    else:
                        input_embs = self.engine.get_input_embeddings(input_ids=input_ids)
                    if input_embs.ndim == 2:
                        input_embs = input_embs.unsqueeze(0)
                    query.input_embedding = input_embs
                else:
                    assert query.input_embedding.shape[1] == len(input_ids)

                current_context_shift = query.context_shift + query.prefix_already_computed_len
                # start from context_shift pos
                query_input_emb = query.input_embedding[:, current_context_shift:, :]
                if context_len > self.context_split_len or query.context_shift > 0:
                    query_input_emb = query.input_embedding[:, current_context_shift:current_context_shift +
                                                            self.context_split_len, :]
                    logging_rank_only(
                        logging.debug, 0,
                        "trigger context split {} -> {}:{}".format(context_len, current_context_shift,
                                                                   current_context_shift + context_len))
                    query.context_shift += query_input_emb.shape[1]
                    context_len = query_input_emb.shape[1]

                max_context_len = max(max_context_len, context_len)
                phase0_index.append(index)
                phase0_list.append(query_input_emb)
                phase0_total_length.append(context_len)
                phase0_kv_index.append(query.kv_slot_ids) if self.enable_paged_attn else phase0_kv_index.extend(
                    query.kv_slot_ids)
                context_shift.append(current_context_shift)

            else:
                phase1_index.append(index)
                phase1_list.append(query.new_token_ids[-1])
                phase1_total_length.append(context_len + len(query.new_token_ids) + query.prefix_already_computed_len)
                phase1_kv_index.append(query.kv_slot_ids) if self.enable_paged_attn else phase1_kv_index.extend(
                    query.kv_slot_ids)
                if self.enable_ngrams_decoding:
                    max_code_book_len = max(max_code_book_len, len(query.code_book))
                    key = query.new_token_ids[-self.max_ngram_size:]
                    if len(key) < self.max_ngram_size:
                        key = [self.pad_token_id] * (self.max_ngram_size - len(key)) + key
                    keys_list.append(key)
                    code_books_list.append(query.code_book)
            # set sample args for each query
            for k, v in sample_kwargs.items():
                v.append(getattr(query, k, None))

        results = dict()

        # left pad context_input
        if len(phase0_list) > 0:
            context_emb = None
            for i, query in enumerate(phase0_list):
                pad_tokens = None if max_context_len - query.shape[1] == 0 else [self.pad_token_id] * (max_context_len -
                                                                                                       query.shape[1])
                if pad_tokens is not None:
                    pad_tokens = torch.tensor(pad_tokens, dtype=torch.int64)
                    pad_emb = self.engine.get_input_embeddings(pad_tokens.cuda()).unsqueeze(0)
                    phase0_list[i] = torch.concat([pad_emb, query], dim=1)
                    running[phase0_index[i]].cur_batch_pad_token = max_context_len - query.shape[1]
                if context_emb is None:
                    context_emb = phase0_list[i]
                else:
                    context_emb = torch.concat([context_emb, phase0_list[i]], dim=0)
            context_input = context_emb
            results['context_input'] = context_input
        else:
            results['context_input'] = None

        if len(phase1_list) > 0:
            decode_input = torch.tensor(phase1_list, dtype=torch.int64).unsqueeze(1).cuda()
            for i, code_book in enumerate(code_books_list):
                code_books_list[i] = code_book + [self.pad_token_id] * (max_code_book_len - len(code_book))
            results['decode_input'] = decode_input
        else:
            results['decode_input'] = None

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
            # stop event break
            if (stop_event.is_set()):
                self.stop_signal_tensor.fill_(1.0)
            # reuse first nccl layer to comm signal
            if self.engine.module.tp_size > 1:
                assert self.tp_group is not None, "tp_group not set!"
                torch.distributed.all_reduce(self.stop_signal_tensor, group=self.tp_group)

            # wait for the max_off_policy rollout
            skip_break = False
            for query in self.finished.values():
                if (not query.is_finished and query.off_policy_steps >= self.max_off_policy_steps):
                    skip_break = True
            if not skip_break and self.stop_signal_tensor.item() == self.engine.module.tp_size:
                self.stop_signal_tensor.fill_(0.0)
                return True
        return False

    def _fetch_from_pending_queries(self):
        num_ready_query_local_tensor = torch.tensor([len(self.pending)], dtype=torch.float32, device="cuda")
        num_ready = len(self.pending)
        if self.engine.module.tp_size > 1:
            assert self.tp_group is not None, "tp_group not set!"
            reduce_op = torch.distributed.ReduceOp.MIN
            torch.distributed.all_reduce(num_ready_query_local_tensor, group=self.tp_group, op=reduce_op)
            num_ready = num_ready_query_local_tensor.int().item()
        if num_ready == 0:
            return self.waiting
        new_joins = self.pending.get_earliest(num_ready)
        for query in new_joins:
            self.finished[query.id] = query
        self.pending.truncate(num_ready)
        return self.waiting + new_joins

    def execute(self,
                prompts,
                logits_masks: List[List[int]] = None,
                complete_ratio=1,
                stop_event=None,
                prompt_meta_info: List[Dict] = None):
        """Main inference execution loop
        
        Args:
            prompts: List of text prompts or tokenized IDs
            logits_masks: Per-token logits masks for constrained decoding
            complete_ratio: Stop when this fraction of queries complete
            stop_event: External termination signal
            prompt_meta_info: Additional per-prompt metadata
            
        Returns:
            List of completed Query objects with results by method get_inorder_responses()
        """

        torch.manual_seed(int(os.getenv('XPERF_RANDOM_SEED', '0')))
        input_ids_list = self.truncate_prompts(prompts, logits_masks is not None)
        self.find_longest_common_prefix(input_ids_list, logits_masks)
        if not self.is_xperf_custom:
            self.build_prefix_kv_cache()
            logging_rank(
                logging.info, "find common prefix which contains {} tokens, reuse this kv cache!".format(
                    self.common_prefix_tensor_len))
        self.prepare_context_inputs(input_ids_list, logits_masks, prompt_meta_info)
        self.current_steps = 0
        self.finished_num = 0
        tokens_len = None
        accepted_len = None
        if self.step_profiler is not None:
            self.step_profiler.reset_exec()
        while (not self._should_terminate(prompts, complete_ratio, stop_event)):
            self.current_steps += 1
            self.running, self.waiting = self._select_running_queries()
            if len(self.running) == 0:
                assert (len(self.waiting) == 0)
                break
            forward_inputs = self._prepare_forward_inputs(self.running)
            context_input = forward_inputs['context_input']
            decode_input = forward_inputs['decode_input']
            next_tokens, _, _, log_probs, probs_gt_threshold_num, probs_lt_threshold_sum = self.infer_scheduler.forward_and_sample(
                context_input=context_input,
                decode_input=decode_input,
                total_length=forward_inputs['total_length'],
                kv_index=forward_inputs['kv_index'],
                orca_updated=True,
                context_shifts=forward_inputs['context_shifts'],
                history_ids=forward_inputs['history_ids'],
                sample_kwargs=forward_inputs['sample_kwargs'])

            if self.engine.module.tp_size > 1 and next_tokens is not None:
                assert self.tp_group is not None, "tp_group not set!"
                tp_src_rank = torch.distributed.get_global_rank(self.tp_group, group_rank=0)
                torch.distributed.broadcast(next_tokens, src=tp_src_rank, group=self.tp_group)

            self._update_running_batch(next_tokens=next_tokens,
                                       tokens_len=tokens_len,
                                       accepted_len=accepted_len,
                                       index_in_running_batch=forward_inputs['forward_index'],
                                       log_probs=log_probs,
                                       probs_gt_threshold_num=probs_gt_threshold_num,
                                       probs_lt_threshold_sum=probs_lt_threshold_sum)
            self.infer_scheduler.next_step()
            if self.step_profiler is not None:
                ctx_tokens = context_input.shape[0] if context_input is not None else 0
                dec_tokens = decode_input.shape[0] if decode_input is not None else 0
                self.step_profiler.record_step(ctx_tokens=ctx_tokens, dec_tokens=dec_tokens)
        torch.cuda.synchronize()
        self.infer_scheduler.record("cur_steps", [self.current_steps])

    def async_execute(self, update_weight_event):
        import time
        torch.manual_seed(int(os.getenv('XPERF_RANDOM_SEED', '0')))
        self.current_steps = 0
        self.finished_num = 0
        tokens_len = None
        accepted_len = None
        last_time = 0

        def _check_stop_event():
            while (self._should_terminate(None, 1.0, stop_event=update_weight_event)):
                # recompute prefill
                for query in self.running:
                    self.cache_manager.release_query(query)
                    query.reset_compute()
                    self.waiting.append(query)
                self.running = []
                time.sleep(0.01)

        def _idle():
            return (len(self.waiting) == 0 and len(self.running) == 0)

        while (True):
            try:
                _check_stop_event()
                # each rank should have the same running and waiting
                if (_idle()) or (self.step % 20 == 0):
                    self.waiting = self._fetch_from_pending_queries()
                if _idle():
                    continue
                self.current_steps += 1
                self.running, self.waiting = self._select_running_queries()
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
                next_tokens, _, _, log_probs, probs_gt_threshold_num, probs_lt_threshold_sum = self.infer_scheduler.forward_and_sample(
                    context_input=context_input,
                    decode_input=decode_input,
                    total_length=forward_inputs['total_length'],
                    kv_index=forward_inputs['kv_index'],
                    orca_updated=True,
                    context_shifts=forward_inputs['context_shifts'],
                    history_ids=forward_inputs['history_ids'],
                    sample_kwargs=forward_inputs['sample_kwargs'])

                if self.engine.module.tp_size > 1 and next_tokens is not None:
                    assert self.tp_group is not None, "tp_group not set!"
                    tp_src_rank = torch.distributed.get_global_rank(self.tp_group, group_rank=0)
                    torch.distributed.broadcast(next_tokens, src=tp_src_rank, group=self.tp_group)

                self._update_running_batch(next_tokens=next_tokens,
                                           tokens_len=tokens_len,
                                           accepted_len=accepted_len,
                                           index_in_running_batch=forward_inputs['forward_index'],
                                           log_probs=log_probs,
                                           probs_gt_threshold_num=probs_gt_threshold_num,
                                           probs_lt_threshold_sum=probs_lt_threshold_sum)
                self.infer_scheduler.next_step()
                if self.step_profiler is not None:
                    ctx_tokens = context_input.shape[0] if context_input is not None else 0
                    dec_tokens = decode_input.shape[0] if decode_input is not None else 0
                    self.step_profiler.record_step(ctx_tokens=ctx_tokens, dec_tokens=dec_tokens)
            except Exception as e:
                for query in self.running:
                    query.set_finished(exception=e)
                logging.error(f"Error handling request: {str(e)}")
                raise (e)

    def _exceed_length_condition(self, query, tokens_threshold):
        return query.new_token_len + tokens_threshold >= query.max_new_tokens or len(query.new_token_ids) + len(
            query.input_ids) + tokens_threshold >= query.max_length

    def _meet_eos_condition(self, query, next_token):
        finished_sequences = False
        if next_token in self.eos_token_id:
            self.finished[query.id].output_prompt = self.tokenizer.batch_decode([query.new_token_ids[:-1]
                                                                                ]) if self.decode_output else ""
            finished_sequences = True
        elif self._exceed_length_condition(query,
                                           tokens_threshold=self.num_pred_tokens +
                                           1 if self.enable_ngrams_decoding else 0):
            self.finished[query.id].output_prompt = self.tokenizer.batch_decode([query.new_token_ids
                                                                                ]) if self.decode_output else ""
            finished_sequences = True
        elif self.stop_sequence_tokens and next_token in [tokens[-1] for tokens in self.stop_sequence_tokens]:
            for stop_sequences in self.stop_sequence_tokens:
                seq_len = len(stop_sequences)
                if query.new_token_len < seq_len:
                    continue
                finished_sequences |= (query.new_token_ids[-seq_len:] == stop_sequences)
            if finished_sequences:
                self.finished[query.id].output_prompt = self.tokenizer.batch_decode([query.new_token_ids
                                                                                    ]) if self.decode_output else ""

        return finished_sequences

    def _update_running_batch(self,
                              next_tokens,
                              tokens_len,
                              index_in_running_batch,
                              accepted_len=None,
                              log_probs=None,
                              probs_gt_threshold_num=None,
                              probs_lt_threshold_sum=None):
        next_running = [[], []]
        next_tokens = next_tokens.cpu().tolist()
        if accepted_len is not None:
            assert (accepted_len.shape[0] == len(self.running))
        if log_probs is not None:
            assert (log_probs.shape[0] == len(self.running)), "log_probs shape mismatch, {} vs {}".format(
                log_probs.shape[0], len(self.running))
            log_probs = log_probs.cpu().tolist()
        if probs_gt_threshold_num is not None:
            assert (probs_gt_threshold_num.shape[0] == len(self.running))
            probs_gt_threshold_num = probs_gt_threshold_num.cpu().tolist()
        if probs_lt_threshold_sum is not None:
            assert (probs_lt_threshold_sum.shape[0] == len(self.running))
            probs_lt_threshold_sum = probs_lt_threshold_sum.cpu().tolist()
        running_index_to_i = {idx: i for i, idx in enumerate(index_in_running_batch)}
        for idx, query in enumerate(self.running):
            i = running_index_to_i[idx]
            # decoding
            if (query._is_to_decoding_compute()):
                # Each query might have multile next tokens when spec/ngrams is enabled
                query_next_tokens = next_tokens[i]
                if isinstance(query_next_tokens, int):
                    query_next_tokens = [query_next_tokens]
                query_next_tokens_len = 1 if tokens_len is None else tokens_len[i]
                # Ngrams need more kv-cache per forward
                finished_sequences = False
                query.accepted_len.append(accepted_len[i] if accepted_len is not None else -1)
                query.new_token_log_probs.append(log_probs[i] if log_probs is not None else 0)
                query.probs_gt_threshold_num.append(
                    probs_gt_threshold_num[i] if probs_gt_threshold_num is not None else 0)
                query.probs_lt_threshold_sum.append(
                    probs_lt_threshold_sum[i] if probs_lt_threshold_sum is not None else 0)
                for token_idx in range(query_next_tokens_len):
                    next_token = query_next_tokens[token_idx]
                    query.new_token_ids.append(next_token)
                    query.is_context_computing = False
                    query.new_token_len += 1
                    finished_sequences = self._meet_eos_condition(query, query.new_token_ids[-1])
                    if finished_sequences:
                        break
                if not finished_sequences:
                    next_running[1].append(query)
                else:
                    if self.eos_callback_fn:
                        self.eos_callback_fn(query)
                    query.set_finished()
                    self.finished_num += 1
                    self.cache_manager.release_query(query)
                    self.infer_scheduler.record("finished_tokens_by_step", [self.current_steps])
            # still prefill
            else:
                next_running[0].append(query)

        self.running = next_running[0] + next_running[1]
