import copy
import time
import warnings
from typing import *
from dataclasses import dataclass
import uuid
import torch
from threading import Lock
import asyncio
import copy
import base64
import dill
from .query_plugin import QueryPlugin, batch_sync_tp_plugin_queries


@dataclass
class Query:
    id: str
    idx: int
    original_input_ids: Optional[List[int]]
    input_ids: Optional[List[int]]
    input_embedding: Optional[torch.Tensor]
    code_book: Optional[List[int]]
    constraint_decoding_predictor: Optional[Any]
    accepted_len: Optional[List[int]]
    input_prompt: Union[str, List[str]]
    input_len: Optional[int]
    new_token_ids: Optional[List[int]]
    new_token_log_probs: Optional[List[int]]
    kv_slot_ids: Optional[List[int]]
    is_context_computing: bool
    new_token_len: int
    context_shift: int
    output_prompt: Union[str, List[str]]
    prefix_already_computed_len: int
    multiround_id: int
    multiround_len: int
    system_ids_len: int
    created_time: float  # 此对象在request pool创建时间
    received_time: float  # 在engine侧第一次收到进入队列的时间
    first_scheduled_time: float  # 开始prefill的时间
    first_token_time: float  # prefill完的时间
    finished_time: float  # decode完的时间
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
        self.id = uuid.uuid4().hex
        self.idx = idx
        self.original_input_ids = copy.copy(input_ids)
        self.input_ids = input_ids
        self.code_book = code_book
        self.constraint_decoding_predictor = constraint_decoding_predictor
        self.accepted_len = []
        self.input_prompt = input_prompt
        self.prefix_already_computed_len = prefix_already_computed_len
        self.input_len = len(input_ids)
        self.is_context_computing = True
        self.new_token_ids = []
        self.new_token_log_probs = []
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

        # timestamp units are all milliseconds
        self.created_time = time.time() * 1000
        self.reset_timestamp()
        self.is_jumping = False
        self.jump_tokens = 0
        self.off_policy_steps = 0
        self.top_k = None
        self.top_p = None
        self.temperature = None
        self.max_new_tokens = None
        self.max_length = None
        self._exception = None

        self.plugin_query = None

    # Check whether current query is going to enter the decoding stage
    def is_to_decoding_compute(self):
        # already in decode stage
        if not self.is_context_computing:
            return True
        assert (self.context_shift + self.prefix_already_computed_len) <= len(self.input_ids)
        # all input_ids finished context computing
        if self.is_context_computing and (self.context_shift + self.prefix_already_computed_len) == len(self.input_ids):
            return True
        return False

    def is_kv_cache_slot_allocated(self):
        return len(self.kv_slot_ids) > 0

    def to_context_phase(self):
        self.input_ids.extend(self.new_token_ids)
        self.new_token_ids = []
        if not self.is_context_computing:
            self.is_context_computing = True
            # all context except the last decoded token has finished context compute
            self.context_shift = max(0, len(self.input_ids) - 1)

    def set_finished(self, is_partial=False, exception=None):
        self.is_finished = not is_partial
        self.finished_time = time.time() * 1000
        self._exception = exception
        self.detach()

    def reset_compute(self):
        self.kv_slot_ids = []
        self.to_context_phase()
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

    def add_token(self, token_id, accepted_len=-1, log_prob=0.0):
        self.accepted_len.append(accepted_len)
        self.new_token_log_probs.append(log_prob)

        self.new_token_ids.append(token_id)
        self.is_context_computing = False
        self.new_token_len += 1
        if self.plugin_query:
            self.plugin_query.record_model_token(token_id)

    def meet_pause_condition(self) -> bool:
        if self.plugin_query:
            return self.plugin_query.meet_pause_condition()
        return False

    def pause(self):
        if self.plugin_query:
            self.plugin_query.trigger_plugin_call()

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

    def reset_timestamp(self):
        self.received_time = 0
        self.first_scheduled_time = 0
        self.first_token_time = 0
        self.finished_time = 0

    def clone(self) -> 'Query':
        ret = copy.copy(self)
        # skip any gpu tensors, as they might be mutated shortly
        if ret.hidden_states is not None and ret.hidden_states.device != torch.device('cpu'):
            ret.hidden_states = None
        return ret

    @classmethod
    def from_request(cls, input_ids, request_id, sampling_kwargs, meta_info=None) -> 'Query':
        query = Query(input_ids, input_prompt='', code_book=None, idx=request_id, prefix_already_computed_len=0)
        query.id = request_id
        query.top_k = sampling_kwargs.get("top_k", 0)
        query.top_p = sampling_kwargs.get("top_p", 1.0)
        query.temperature = sampling_kwargs.get("temperature", 1.0)
        query.max_new_tokens = sampling_kwargs.get("max_new_tokens", 32)
        query.max_length = sampling_kwargs.get("max_length", 1024)
        query.meta_info = meta_info or {}
        return query

    @property
    def extra_data(self) -> Dict[str, str]:
        extra_data = copy.copy(self.meta_info.get('extra_data', {}))
        extra_data.pop('env_states', None)
        extra_data.pop('resume_state', None)
        if self.is_finished:
            if self.plugin_query is not None:
                # get env_states only when finished
                extra_data['env_states'] = self.plugin_query.env_state_b64
        else:
            # get resume_state only when unfinished
            resume_state = self.get_resume_state()
            if resume_state is not None:
                extra_data['resume_state'] = base64.b64encode(dill.dumps(resume_state)).decode('utf-8')
        return extra_data

    def attach_session(self, session):
        """Attach session, initialize session-dependant fields"""
        generation_kwargs = self.meta_info['generation_kwargs']
        plugin_config = generation_kwargs.get('plugin_config', None)
        plugin_enabled = plugin_config and plugin_config.get('enable', False)

        if plugin_enabled:
            self.plugin_query = QueryPlugin(plugin_config)
            self.plugin_query.attach_session(session=session, query=self)

        extra_data = self.meta_info.get('extra_data', {})
        resume_state = extra_data.get('resume_state', None)
        if resume_state is not None:
            assert isinstance(resume_state, str), f"resume_state should be a b64 string, got {type(resume_state)}"
            resume_state_bytes = base64.b64decode(resume_state)
            self.set_resume_state(dill.loads(resume_state_bytes))

    def detach(self):
        """Detach session-dependant fields"""
        if self.plugin_query:
            self.plugin_query = copy.copy(self.plugin_query)
            self.plugin_query.detach()


class AsyncQuery:
    """
    the Query class wrapper with python async coroutines
    """

    def __init__(self, query: Query):
        self._query = query
        self._event = asyncio.Event()

    def set_finished(self, is_partial=False, exception=None):
        self._query.set_finished(is_partial, exception)
        self._event.set()

    async def wait_until_done(self):
        await self._event.wait()

    @property
    def exception(self):
        return self._query._exception

    @property
    def id(self):
        return self._query.id

    @property
    def query(self):
        return self._query


class InflightQueue:

    def __init__(self):
        self.query_pool: Dict[str, AsyncQuery] = {}
        self.queue: List[AsyncQuery] = []
        self.lock = Lock()

    def append(self, item: AsyncQuery):
        with self.lock:
            item.query.received_time = time.time() * 1000
            self.query_pool[item.id] = item
            self.queue.append(item)

    def truncate(self, length):
        with self.lock:
            self.queue = self.queue[length:]

    def get_earliest(self, length) -> List[AsyncQuery]:
        with self.lock:
            return self.queue[:length]

    def __len__(self):
        with self.lock:
            return len(self.queue)

    def __iter__(self) -> Iterator[AsyncQuery]:
        with self.lock:
            return iter(self.queue.copy())


def batch_sync_tp_queries(queries: List[Query], tp_group):
    plugin_queries = []
    for query in queries:
        if query.plugin_query is not None:
            plugin_queries.append(query.plugin_query)
    batch_sync_tp_plugin_queries(plugin_queries, tp_group=tp_group)
