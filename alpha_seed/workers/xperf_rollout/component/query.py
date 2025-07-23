import copy
import time
import logging
from typing import *
from dataclasses import dataclass
import uuid
import torch
from threading import Lock
import asyncio
import copy
import base64
import dill
from xperf_gpt.utils import (logging_rank, logging_rank_only)
from alpha_seed.workers.xperf_rollout.component.query_plugin import QueryPlugin, batch_sync_tp_plugin_queries


def call_once_method(method):

    def wrapper(self, *args, **kwargs):
        flag_name = f"_has_run_{method.__name__}"
        if getattr(self, flag_name, False):
            return
        setattr(self, flag_name, True)
        return method(self, *args, **kwargs)

    return wrapper


@dataclass
class Query:
    id: str
    idx: int
    original_input_ids: Optional[List[int]]
    input_ids: Optional[List[int]]
    code_book: Optional[List[int]]
    accepted_len: Optional[List[int]]
    input_prompt: Union[str, List[str]]
    new_token_ids: Optional[List[int]]
    log_probs: Optional[List[float]]
    kv_slot_ids: Optional[List[int]]
    is_context_computing: bool
    new_token_len: int
    context_shift: int
    output_prompt: Union[str, List[str]]
    prefix_already_computed_len: int
    system_ids_len: int
    # 下面几个time的单位都是ms
    created_time: float  # 此对象在client侧创建时间
    enqueue_time: float  # 对象放入request pool的时间
    dispatch_time: float  # 从request pool取出来分配给某个engine的时刻
    received_time: float  # 在engine侧第一次收到进入队列的时间
    first_scheduled_time: float  # 开始prefill的时间
    first_token_time: float  # prefill完的时间
    finished_time: float  # decode完的时间
    hidden_states: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    cur_batch_pad_token: int
    nll_loss: Optional[torch.Tensor]
    is_finished: bool
    prefill_only: bool
    off_policy_steps: int
    meta_info: Optional[Dict]
    plugin_query: QueryPlugin
    image_data: Optional[Dict]
    image_data_ref: Optional[str]
    images_bytes_ref: Optional[str]

    def __init__(self,
                 input_ids,
                 input_prompt,
                 idx,
                 prefix_already_computed_len=0,
                 system_ids_len=0,
                 code_book=None,
                 image_data=None,
                 image_data_ref=None,
                 images_bytes_ref=None):
        self.id = uuid.uuid4().hex
        self.idx = idx
        self.original_input_ids = copy.copy(input_ids)
        self.input_ids = input_ids
        self.code_book = code_book
        self.accepted_len = []
        self.input_prompt = input_prompt
        self.prefix_already_computed_len = prefix_already_computed_len
        self.is_context_computing = True
        self.new_token_ids = []
        self.log_probs: List[float] = []
        self.kv_slot_ids = []
        self.new_token_len = 0
        self.output_prompt = ""
        self.context_shift = 0
        self.system_ids_len = system_ids_len
        self.hidden_states = None
        self.logits = None
        self.cur_batch_pad_token = 0
        self.nll_loss = None
        self.is_finished = False
        self.prefill_only = False
        self.meta_info = {}

        # timestamp units are all milliseconds
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
        self.image_data = image_data
        self.image_data_ref = image_data_ref
        self.images_bytes_ref = images_bytes_ref
        self.action = True

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

    @call_once_method
    def lazy_init_from_prompt_once(self, tokenizer):
        if len(self.input_ids) > 0:
            return
        self.input_ids = tokenizer.encode(self.input_prompt)
        self.original_input_ids = copy.copy(self.input_ids)

    @property
    def original_input_len(self):
        return len(self.original_input_ids)

    @property
    def output_tokens(self) -> List[int]:
        return (self.input_ids + self.new_token_ids)[self.original_input_len:]

    def add_token(self, token_id, accepted_len=-1, log_prob=0.0):
        self.accepted_len.append(accepted_len)
        self.is_context_computing = False
        if isinstance(log_prob, List):
            self.log_probs.extend(log_prob)
        else:
            self.log_probs.append(log_prob)
        if not self.prefill_only:
            self.new_token_ids.append(token_id)
            self.new_token_len += 1
        if self.plugin_query:
            self.plugin_query.record_model_token(token_id)

    def meet_pause_condition(self) -> bool:
        if self.plugin_query and self.action:
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
        """
        重置跟engine相关的时间戳，query生命周期时间戳不变
        """
        self.received_time = 0
        self.first_scheduled_time = 0
        self.first_token_time = 0
        self.finished_time = 0
        # not yet dispatch and not yet enqueued
        self.dispatch_time = -1
        self.enqueue_time = -1
        self.created_time = time.time() * 1000

    def clone(self) -> 'Query':
        ret = copy.copy(self)

        # copy的过程中，可能另外的线程正在调用add_token追加新的token，
        # 为避免这里出现脏读，始终以new_token_len的值表示已经commit的token
        # 所以这里复制已提交部分实现clone的读事务隔离
        ret.accepted_len = ret.accepted_len[:ret.new_token_len]
        ret.log_probs = ret.log_probs[:ret.new_token_len]
        # new_token_ids比较特殊，每次reset_compute会把new_token_ids追加到input_ids里面
        # 但new_token_len持续累加，所以这里算出来真正需要truncate的量
        total_committed_tokens = ret.original_input_len + ret.new_token_len
        new_token_ids_len = total_committed_tokens - len(ret.input_ids)
        ret.new_token_ids = ret.new_token_ids[:new_token_ids_len]
        # Note: 其他要保证事务隔离的列表对象在这里处理好再返回

        return ret

    @classmethod
    def from_request(cls,
                     input_ids,
                     input_prompt,
                     request_id,
                     sampling_kwargs,
                     meta_info=None,
                     image_kwargs=None) -> 'Query':
        query = Query(input_ids,
                      input_prompt=input_prompt,
                      code_book=None,
                      idx=request_id,
                      prefix_already_computed_len=0)
        query.id = request_id
        query.top_k = sampling_kwargs.get("top_k", 0)
        query.top_p = sampling_kwargs.get("top_p", 1.0)
        query.temperature = sampling_kwargs.get("temperature", 1.0)
        query.max_new_tokens = sampling_kwargs.get("max_new_tokens", 32)
        query.max_length = sampling_kwargs.get("max_length", 1024)
        query.meta_info = meta_info or {}
        if image_kwargs is not None and len(image_kwargs) > 0:
            query.image_data = image_kwargs.get('image_data')
            query.image_data_ref = image_kwargs.get('image_data_ref')
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
        generation_kwargs = self.meta_info.get('generation_kwargs', {})
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

        # skip any gpu tensors, as they might be mutated shortly
        if self.hidden_states is not None and self.hidden_states.device != torch.device('cpu'):
            self.hidden_states = None

        if self.image_data_ref is not None:
            self.image_data = None


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
        self.queue: List[AsyncQuery] = []
        self.lock = Lock()

    def append(self, item: AsyncQuery):
        with self.lock:
            item.query.received_time = time.time() * 1000
            self.queue.append(item)

    def truncate(self, length):
        with self.lock:
            self.queue = self.queue[length:]

    def remove(self, to_remove: Dict[str, float]):
        # to_remove: query_id -> ts (abort the query if before this ts)
        with self.lock:
            original_len = len(self.queue)

            # in-place remove and compact the list
            write_index = 0
            for read_index in range(original_len):
                q = self.queue[read_index]
                not_after = to_remove.get(q.id)
                if not_after is None or q.query.dispatch_time >= not_after:
                    # keep this query
                    if write_index != read_index:
                        self.queue[write_index] = self.queue[read_index]
                    write_index += 1
            del self.queue[write_index:]

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
