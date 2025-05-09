from typing import *
from dataclasses import dataclass
import uuid
import torch
from threading import Lock
import asyncio
import copy
from .query_plugin import QueryPlugin, batch_sync_tp_plugin_queries


@dataclass
class Query:
    id: Optional[uuid.UUID]
    idx: int
    original_input_ids: Optional[List[int]]
    input_ids: Optional[List[int]]
    input_embedding: Optional[torch.Tensor]
    code_book: Optional[List[int]]
    accepted_len: Optional[List[int]]
    input_prompt: Union[str, List[str]]
    input_len: Optional[int]
    new_token_ids: Optional[List[int]]
    new_token_log_probs: Optional[List[int]]
    probs_gt_threshold_num: Optional[List[int]]
    probs_lt_threshold_sum: Optional[List[int]]
    kv_slot_ids: Optional[List[int]]
    is_context_computing: bool
    new_token_len: int
    context_shift: int
    output_prompt: Union[str, List[str]]
    prefix_already_computed_len: int
    multiround_id: int
    multiround_len: int
    system_ids_len: int
    first_scheduled_time: int
    first_token_time: int
    finished_time: int
    hidden_states: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    logits_mask: Optional[torch.Tensor]
    shift_label: Optional[torch.Tensor]
    cur_batch_pad_token: int
    nll_loss: Optional[torch.Tensor]
    is_finished: bool
    off_policy_steps: int
    meta_info: Optional[Dict]
    plugin_query: QueryPlugin

    def __init__(self,
                 input_ids,
                 input_prompt,
                 idx,
                 prefix_already_computed_len=0,
                 system_ids_len=0,
                 code_book=None,
                 constraint_decoding_predictor=None):
        self.id = uuid.uuid4()
        self.idx = idx
        self.original_input_ids = copy.copy(input_ids)
        self.input_ids = input_ids
        self.code_book = code_book
        self.accepted_len = []
        self.input_prompt = input_prompt
        self.prefix_already_computed_len = prefix_already_computed_len
        self.input_len = len(input_ids)
        self.is_context_computing = True
        self.new_token_ids = []
        self.global_new_token_ids = []
        self.new_token_log_probs = []
        self.probs_gt_threshold_num = []
        self.probs_lt_threshold_sum = []
        self.kv_slot_ids = []
        self.new_token_len = 0
        self.output_prompt = ""
        self.context_shift = 0
        self.multiround_id = 0
        self.multiround_len = len(input_prompt) if isinstance(input_prompt, list) else 0
        self.multiround_input_len = 0
        self.multiround_new_token_len = 0
        self.system_ids_len = system_ids_len
        self.hidden_states = None
        self.logits = None
        self.logits_mask = None
        self.shift_label = None
        self.cur_batch_pad_token = 0
        self.nll_loss = None
        self.is_finished = False
        self.meta_info = {}

        self.first_scheduled_time = 0
        self.first_token_time = 0
        self.finished_time = 0
        self.is_jumping = False
        self.jump_tokens = 0
        self.off_policy_steps = 0
        self.top_k = None
        self.top_p = None
        self.temperature = None
        self.max_new_tokens = None
        self.max_length = None
        self.input_embedding = None

        self.plugin_query = None

    # Check whether current query is going to enter the decoding stage
    def _is_to_decoding_compute(self):
        # already in decode stage
        if not self.is_context_computing:
            return True
        # called after context stage, no context_shift means it doesn't need context-split
        if self.is_context_computing and self.context_shift == 0:
            return True
        # for query which needs context-split, all input_ids finished context computing
        if self.is_context_computing and (self.context_shift + self.prefix_already_computed_len) == len(self.input_ids):
            return True
        return False

    def is_kv_cache_slot_allocated(self):
        return len(self.kv_slot_ids) > 0

    def set_finished(self, is_partial=False):
        self.is_finished = not is_partial
        self.global_new_token_ids.extend(self.new_token_ids)

    def reset_compute(self):
        self.kv_slot_ids = []
        self.is_context_computing = True
        self.input_ids.extend(self.new_token_ids)
        self.new_token_ids = []
        self.context_shift = 0
        self.prefix_already_computed_len = 0
        self.hidden_states = None
        return

    @property
    def original_input_len(self):
        return len(self.original_input_ids)

    @property
    def output_tokens(self) -> List[int]:
        return (self.input_ids + self.new_token_ids)[self.original_input_len:]

    def add_token(self, token_id, accepted_len=-1, log_prob=0.0, probs_gt_threshold_num=0, probs_lt_threshold_sum=0.0):
        self.accepted_len.append(accepted_len)
        self.new_token_log_probs.append(log_prob)
        self.probs_gt_threshold_num.append(probs_gt_threshold_num)
        self.probs_lt_threshold_sum.append(probs_lt_threshold_sum)

        self.new_token_ids.append(token_id)
        self.is_context_computing = False
        self.new_token_len += 1
        if self.plugin_query:
            self.plugin_query.record_model_token(token_id)

    def set_plugin_query(self, plugin_config, tokenizer, env_strs, tp_group):
        self.plugin_query = QueryPlugin(query=self,
                                        config=plugin_config,
                                        tokenizer=tokenizer,
                                        env_strs=env_strs,
                                        tp_group=tp_group)

    def meet_pause_condition(self) -> bool:
        if self.plugin_query:
            return self.plugin_query.meet_pause_condition()
        return False

    def try_resume_from_paused(self):
        if self.plugin_query:
            self.plugin_query.try_resume_from_paused()

    def get_resume_state(self) -> Dict:
        """States that should be consistent between off-policy and on-policy steps"""
        if self.plugin_query:
            state = dict()
            state['plugin_query'] = self.plugin_query.get_resume_state()
            return state
        return None

    def set_resume_state(self, state: Dict):
        if state is None:
            return
        if self.plugin_query:
            self.plugin_query.set_resume_state(state['plugin_query'])

    @property
    def env_state_bytes(self) -> bytes:
        if self.plugin_query:
            return self.plugin_query.env_state_bytes
        return None

    @property
    def model_output_mask(self) -> List[bool]:
        if self.plugin_query:
            return self.plugin_query.model_output_mask
        return [True] * (len(self.input_ids) + len(self.new_token_ids) - self.original_input_len)

    @property
    def metrics(self) -> Dict:
        ret = dict()
        if self.plugin_query:
            plugin_metrics = self.plugin_query.metrics
            for key, val in plugin_metrics.items():
                ret[f"plugin/{key}"] = val
        return ret


@dataclass
class AsyncQuery(Query):

    def __init__(self,
                 input_ids,
                 input_prompt,
                 idx,
                 prefix_already_computed_len=0,
                 system_ids_len=0,
                 code_book=None,
                 constraint_decoding_predictor=None):
        super().__init__(input_ids, input_prompt, idx, prefix_already_computed_len, system_ids_len, code_book,
                         constraint_decoding_predictor)
        self._event = None
        self._loop = None
        self._exception = None

    def set_finished(self, is_partial=False, exception=None):
        super().set_finished(is_partial)
        self._loop.call_soon_threadsafe(self._event.set)
        self._exception = exception

    async def wait_until_done(self):
        await self._event.wait()

    @classmethod
    def from_request(cls, input_ids, request_id, sampling_kwargs):
        query = AsyncQuery(input_ids, code_book=None, input_prompt='', idx=request_id, prefix_already_computed_len=0)
        query.id = request_id
        query.top_k = sampling_kwargs.get("top_k", 0)
        query.top_p = sampling_kwargs.get("top_p", 1.0)
        query.temperature = sampling_kwargs.get("temperature", 1.0)
        query.max_new_tokens = sampling_kwargs.get("max_new_tokens", 32)
        query.max_length = sampling_kwargs.get("max_length", 1024)
        query.meta_info = {}
        return query


class InflightQueue:

    def __init__(self):
        self.query_pool = {}
        self.queue = []
        self.lock = Lock()

    def append(self, item):
        with self.lock:
            item._event = asyncio.Event()
            item._loop = asyncio.get_running_loop()
            self.query_pool[item.id] = item
            self.queue.append(item)

    def truncate(self, length):
        with self.lock:
            self.queue = self.queue[length:]

    def get_earliest(self, length):
        with self.lock:
            return self.queue[:length]

    def __len__(self):
        with self.lock:
            return len(self.queue)

    def __iter__(self):
        with self.lock:
            return iter(self.queue.copy())


def batch_sync_tp_queries(queries: List[Query], tp_group):
    plugin_queries = []
    for query in queries:
        if query.plugin_query is not None:
            plugin_queries.append(query.plugin_query)
    batch_sync_tp_plugin_queries(plugin_queries, tp_group=tp_group)
