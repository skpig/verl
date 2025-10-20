import copy
import time
from enum import Enum, auto
from typing import *
from dataclasses import dataclass, field, fields
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


# 在engine内的生命周期event，在engine外的由RequestManager管理
# 为了方便json序列化，这里直接用字符串表示enum
class ProcessEventType:
    RECEIVED = "RECEIVED"  # (pool) -> waiting
    PREFILL_START = "PREFILL_START"  # waiting -> prefill
    PREFILL_DONE = "PREFILL_DONE"  # prefill -> decode
    FINISHED = "FINISHED"  # decode -> done
    EVICTED = "EVICTED"  # prefill/decode -> waiting, kv 满了被evict


@dataclass
class QueryProcessEvent:
    event: str | ProcessEventType
    ts_ms: float
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryCheckpoint:
    saved_length: int = 0  # request pool里已经保存了多少长度的new token_ids
    updated_at: float = 0  # 上次update时间戳


@dataclass
class QueryUpdate:
    # 用于全量更新的一些字段，非None的字段需要覆盖到request pool里
    query_primitive: 'Query'

    # 下面都是增量部分
    accepted_len: List[int]  # speculative decoding 对应的长度，跟new_token_ids一一对应 (单调递增)
    log_probs: List[float]  # 跟new_token_ids一一对应 (单调递增)
    new_token_ids: List[int]  # 增量decode出来的token部分，无论engine内是否reset过
    prefill_len: int  # 保存原Query的len(input_ids)，算指标用到
    saved_length: int  # 加上此增量后，保存到了第多少个token

    @property
    def id(self) -> str:
        return self.query_primitive.id

    @property
    def is_finished(self) -> bool:
        return False

    def __getattr__(self, item):
        qp = object.__getattribute__(self, "query_primitive")
        return getattr(qp, item)


@dataclass
class Query:
    id: str
    idx: int
    original_input_ids: Optional[List[int]]  # 最开始输入进来的prompt部分 (不会变)
    input_ids: Optional[List[int]]  # decode一半中断再继续时，需要prefill的所有token id (单调递增)
    code_book: Optional[List[int]]
    accepted_len: Optional[List[int]]  # speculative decoding 对应的长度，跟new_token_ids一一对应 (单调递增)
    input_prompt: Optional[Union[str, List[str]]]  # 同original_input_ids (不变)
    new_token_ids: Optional[List[int]]  # (会reset，不一定单调)
    log_probs: Optional[List[float]]  # 跟new_token_ids一一对应 (单调递增)
    kv_slot_ids: Optional[List[int]]
    is_context_computing: bool
    new_token_len: int
    context_shift: int
    image_shift: int
    image_context_shift: int
    output_prompt: Union[str, List[str]]
    prefix_already_computed_len: int
    system_ids_len: int

    # trace event相关
    process_events: List[QueryProcessEvent]  # 记录query在此engine上的事件
    # 下面几个time的单位都是ms
    created_time: float  # 此对象在client侧创建时间
    enqueue_time: float  # 对象放入request pool的时间
    dispatch_time: float  # 最近一次从request pool取出来分配给某个engine的时刻
    received_time: float  # 在engine侧第一次收到进入队列的时间
    first_scheduled_time: float  # 开始prefill的时间
    first_token_time: float  # prefill完的时间
    recent_scheduled_time: float  # 最近一次重新prefill时间
    recent_first_token_time: float  # 最近一次重新prefill完的时间
    finished_time: float  # decode完的时间
    release_count: int  # 在decode或者prefill过程中因kv cache满了或者weights变了需要重新prefill的次数

    hidden_states: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    return_selected_experts: bool
    selected_experts: Optional[torch.Tensor]
    selected_experts_offset: int
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

    update_checkpoint: QueryCheckpoint

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
        self.log_probs = []
        self.kv_slot_ids = []
        self.new_token_len = 0
        self.output_prompt = ""
        self.context_shift = 0
        self.image_shift = 0
        self.image_context_shift = 0
        self.system_ids_len = system_ids_len
        self.hidden_states = None
        self.logits = None
        self.return_selected_experts = False
        self.selected_experts = None
        self.selected_experts_offset = 0
        self.cur_batch_pad_token = 0
        self.nll_loss = None
        self.is_finished = False
        self.prefill_only = False
        self.meta_info = {}

        # timestamp units are all milliseconds
        self.init_timestamp()
        self.reset_timestamp()
        self.release_count = 0
        self.is_jumping = False
        self.jump_tokens = 0
        self.off_policy_steps = 0
        self.top_k = None
        self.top_p = None
        self.temperature = None
        self.max_new_tokens = None
        self.max_length = None
        self.input_embedding = None
        self._exception = None

        self.plugin_query = None
        self.image_data = image_data
        self.image_data_ref = image_data_ref
        self.images_bytes_ref = images_bytes_ref
        self.action = True
        self.update_checkpoint = QueryCheckpoint()

    @property
    def cache_id(self) -> str:
        # 用语匹配prefix cache的id，如果一个trajectory不分裂，可能没设定cache_id，直接用uid即可
        return self.meta_info.get('cache_id') or self.meta_info.get('uid')

    def add_event(self, event: str | ProcessEventType, info: Optional[Dict] = None):
        info = info or {}
        if event in [ProcessEventType.FINISHED, ProcessEventType.EVICTED]:
            info["length_generated"] = self.new_token_len - (len(self.input_ids) - self.original_input_len)
            info["total_output_len"] = self.new_token_len
            info["release_count"] = self.release_count
        self.process_events.append(QueryProcessEvent(event, ts_ms=time.time() * 1000, info=info))

    def get_event_time(self, event: ProcessEventType) -> Optional[float]:
        for e in self.process_events:
            if e.event == event:
                return e.ts_ms
        return None

    def init_timestamp(self):
        """
        query 生命周期时间戳，初始化时只调用一次，中途不改变
        """
        self.dispatch_time = 0  # 最近一次的调度时间戳，即使reset_compute也不改变这个值
        self.enqueue_time = 0
        self.created_time = time.time() * 1000

    def reset_timestamp(self):
        self.process_events = []  #
        self.received_time = 0
        self.first_scheduled_time = 0
        self.first_token_time = 0
        self.recent_scheduled_time = 0
        self.recent_first_token_time = 0
        self.finished_time = 0

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
        self.add_event(ProcessEventType.FINISHED)

    def reset_compute(self, info: Optional[Dict] = None):
        self.add_event(ProcessEventType.EVICTED, info)
        self.kv_slot_ids = []
        self.to_context_phase()
        self.input_embedding = None
        self.context_shift = 0
        self.image_shift = 0
        self.image_context_shift = 0
        self.prefix_already_computed_len = 0
        self.hidden_states = None
        self.release_count += 1
        self.selected_experts = None
        self.selected_experts_offset = 0

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
    def prefill_len(self):
        return len(self.input_ids)

    @property
    def output_tokens(self) -> List[int]:
        return (self.input_ids + self.new_token_ids)[self.original_input_len:]

    def add_token(self, token_id, accepted_len=-1, log_prob=0.0):
        self.accepted_len.append(accepted_len)
        self.is_context_computing = False
        self.input_embedding = None
        self.image_context_shift = 0
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

        ret.input_embedding = None
        ret.image_context_shift = 0
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
        query.return_selected_experts = meta_info.get('return_selected_experts', False)
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

    def clear_volatile(self):
        # 更新到request pool之前需要忽略掉的跟engine local相关的易变变量
        # 不用存这些用不上的值
        if self.plugin_query:
            self.plugin_query = copy.copy(self.plugin_query)
            self.plugin_query.detach()
        self.input_embedding = None
        self.hidden_states = None
        self.kv_slot_ids = None
        self.logits = None
        self.nll_loss = None
        self.image_data = None

    def clear_transient(self):
        # 持久化之前需要忽略掉的字段，即不持久化，但需要返回到client侧
        # experts信息不能增量存储，中途切换engine可能需要重新prefill
        self.selected_experts = None

    def to_incremental(self) -> QueryUpdate:
        # 0. 去掉易变部分 (包括增量部分非determined结果)
        # 1. 计算增量部分
        # 2. 去掉不变的字段

        q = self.clone()
        q.clear_volatile()
        q.clear_transient()

        saved_length = q.update_checkpoint.saved_length
        accepted_len = q.accepted_len[saved_length:]
        log_probs = q.log_probs[saved_length:]
        incremental_new_token_len = saved_length - (len(q.input_ids) - q.original_input_len)

        # 这部分取值有点复杂
        # [ original input ][ decoded ids ][ new token ids ]
        #                           |            |
        #                       分这两种情况计算增量的部分
        if incremental_new_token_len >= 0:
            new_token_ids = q.new_token_ids[incremental_new_token_len:]
        else:
            new_token_ids = q.input_ids[incremental_new_token_len:] + q.new_token_ids
        prefill_len = len(q.input_ids)

        # 不变量
        q.original_input_ids = None
        q.input_prompt = None
        q.meta_info = None

        # 增量部分已经包含了
        q.accepted_len = None
        q.log_probs = None
        q.new_token_ids = None
        q.input_ids = None

        return QueryUpdate(
            query_primitive=q,
            accepted_len=accepted_len,
            log_probs=log_probs,
            new_token_ids=new_token_ids,
            prefill_len=prefill_len,
            saved_length=q.new_token_len,
        )

    def apply_update(self, update: QueryUpdate):
        self.accepted_len.extend(update.accepted_len)
        self.log_probs.extend(update.log_probs)
        self.new_token_ids.extend(update.new_token_ids)

        # 覆盖更新其他非None的字段
        for f in fields(update.query_primitive):
            v = getattr(update.query_primitive, f.name)
            if v is not None:
                setattr(self, f.name, v)
        self.update_checkpoint.saved_length = self.new_token_len


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
            item.query.add_event(ProcessEventType.RECEIVED)
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
