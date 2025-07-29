import asyncio
import copy
import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Container
from dataclasses import dataclass, field
from collections import defaultdict

import ray
from omegaconf import DictConfig

from alpha_seed.utils.server_client import is_local_ray_instance
from alpha_seed.workers.streaming_service.rollout_request_manager_diagnosis import FinishedEventStats, RequestDigest, \
    FiniteDict
from alpha_seed.workers.xperf_rollout.component.query import Query
from alpha_seed.utils.profile.timeline import CoherentCompleteEvent, Tracer, TracingEvent, CompleteEvent, \
    FlowEvent, \
    CombinedEvents, WaterfallSlotTracer, CounterEvent


@dataclass
class StaleHistory:
    assigned_engine_id: str
    assigned_engine_name: str
    start_step: int  # 最早开始的step
    # 下面时间戳均为stale之前在engine侧的时间戳 (unit: ms)
    received_time: float  # 进入engine waiting队列的时间
    first_scheduled_time: float  # 开始prefill时间
    first_token_time: float  # 最早开始decode时间(prefill完成时间)
    end_ts: float  # 发现stale的时刻 (unit: ms)
    stale_action: str  # 是什么直接导致的stale，如release/engine-died/...
    stale_reason: str  # 基于什么原因要做这个stale的操作，如内存不够/rebalance/...
    length_generated: int  # 在这一次stale之前，decode了多少(不含prefill部分)
    update_count: int  # stale之前update过多少次
    release_count: int  # 这次stale区间内kv cache 被release的次数

    @classmethod
    def from_request(cls, req: 'Request', stale_action, stale_reason) -> 'StaleHistory':
        delta_generated = req.query.new_token_len - (len(req.query.input_ids) - req.query.original_input_len)
        # 时间单位都是ms
        received_time = req.query.received_time or req.last_pending_reschedule_ts
        first_scheduled_time = req.query.first_scheduled_time or (received_time + 1e-3)
        first_token_time = req.query.first_token_time or (first_scheduled_time + 1e-3)
        history = StaleHistory(
            assigned_engine_id=req.assigned_engine_id,  # noqa
            assigned_engine_name=req.assigned_engine_name,
            start_step=req.global_step,
            received_time=received_time,
            first_scheduled_time=first_scheduled_time,
            first_token_time=first_token_time,
            end_ts=time.time() * 1e3,
            stale_action=stale_action,
            stale_reason=stale_reason,
            length_generated=delta_generated,
            update_count=req.update_count,
            release_count=req.query.release_count,
        )
        return history


@dataclass
class AbortHistory:
    engine_id: str  # 从哪个engine abort
    timestamp: float  # 何时abort


@dataclass
class Request:
    request_id: str  # 同Query.id
    query: Query
    finished: bool = False  # 是否已经结束生成
    assigned: bool = False  # 是否已经分出去了（分出去了，但不确定具体分配的engine_id，取决于passive/active模式
    assigned_engine_id: Optional[str] = None  # 被分配到的engine
    assigned_engine_name: Optional[str] = None  # 比engine_id更可读的name，作为key优先使用engine_id
    updated_at: float = None  # 标记最后更新时间，并发更新时可以判断数据是否过期 (unit: s)

    # fields need to assign back
    global_step: int = 0  # 当前这个query来自哪个global step的sample
    last_assigned_at: float = 0  # 最近一次调度到此engine的时间 (unit: s)
    last_pending_reschedule_ts: float = 0  # 最近一次各种原因被重新放回池里等待调度的时间戳 (unit: ms)
    update_count: int = 0  # 统计更新了多少次
    stale_histories: List[StaleHistory] = field(default_factory=list)  # 记录所有更换过的engine
    # 只记录从proxy主动abort的记录（新的在前），在重新分发时，会跳过从最近abort的engine，避免抖动
    abort_histories: List[AbortHistory] = field(default_factory=list)

    def is_recent_aborted_from(self, engine_id: str, cool_down_seconds) -> bool:
        now = time.time()
        for his in reversed(self.abort_histories):
            elapsed = now - his.timestamp
            if elapsed > cool_down_seconds:
                # 由于abort_histories是新的在后面，那前面的超过CD的都不管了
                break
            if his.engine_id == engine_id:
                # 遇到任意一个engine则true
                return True
        return False


@dataclass
class ProgressStat:
    step: int
    total: int
    finished: int
    token_throughput: float
    running_queries: int
    pending_queries: int
    active_engines: int
    oldest_updated_time: float  # 最老的更新时间
    oldest_query_time: float  # 目前最老的query开始跑的时间戳
    latest_query_time: float  # 目前最新的query开始的时间戳


class RequestPool:

    def __init__(self):
        self.requests: Dict[str, Request] = {}  # {query_id -> } 中间结果会被update进来
        self.finished_requests: Dict[str, Request] = {}  # finished部分会被移到这里
        self.historical_finished_requests = FiniteDict(204800)  # 记录所有完成的query，FIFO，便于query_tool查询诊断
        self.finished_counter = defaultdict(int)  # {step -> count} 统计每个step完成的数量(因为多轮每个step数量是会变化的)
        self._finished_events: Dict[str, asyncio.Event] = {}  # 标记请求完成的async event
        self._mutex = threading.Lock()

        # metrics for observability
        self._metrics_max_retain_seconds = 1200
        # ts_bucket(?s) -> engine_id -> step -> count
        self._accumulated_token_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def __len__(self) -> int:
        return len(self.requests)

    def put_new_requests(self, reqs: List[Request]):
        assert len(reqs) > 0
        with self._mutex:
            for r in reqs:
                self.requests[r.request_id] = r
                self._finished_events[r.request_id] = asyncio.Event()

    def get_pending_size(self):
        # 返回还未分发出去的请求的数量
        count = 0
        with self._mutex:
            for _, request in self.requests.items():
                if request.assigned or request.assigned_engine_id is not None:
                    continue
                count += 1
            return count

    def get_next_pending_requests_with_cache(self,
                                             batch_size,
                                             engine_id: str,
                                             wg_name: str,
                                             cache_ids: List[str] = None) -> Dict[str, Request]:
        # 每个engine实例来这里pull空闲的请求
        # 简单处理，暂不允许并发获取请求
        cool_down_seconds = 10
        with self._mutex:
            batch_size = min(batch_size, len(self.requests))
            ret = {}
            for request_id, request in self.requests.items():
                # 跳过已分发
                if request.assigned or request.assigned_engine_id is not None:
                    continue
                # 跳过最近abort
                if request.is_recent_aborted_from(engine_id, cool_down_seconds):
                    continue
                # 如果有cache_ids，且当前request不在cache_ids中，则跳过
                if cache_ids is not None and request.query.id not in cache_ids:
                    continue
                # 标记请求已被认领了再分发出去
                now = time.time()
                request.assigned = True
                request.assigned_engine_id = engine_id
                request.assigned_engine_name = wg_name
                request.last_assigned_at = now
                request.updated_at = now
                request.query.dispatch_time = now
                ret[request_id] = request
                if len(ret) == batch_size:
                    break
            return ret

    def get_pool_size(self) -> Tuple[int, int]:
        return len(self.requests), len(self.finished_requests)

    # 等待某个request生成完成
    async def wait(self, request_id: str) -> Request:
        """
        一直等待某个request 生成完成才返回，返回后，这个request不再保存在pool里
        """
        await self._finished_events[request_id].wait()
        self._finished_events.pop(request_id, None)
        return self.finished_requests.pop(request_id)

    # 按request全量更新到request pool里
    def update(self, reqs: List[Request]):
        # assign global_step back
        for r in reqs:
            cur_req = self.requests.get(r.request_id)
            if cur_req is None:
                # already finished by another engine
                continue
            r.global_step = cur_req.global_step
            r.stale_histories = cur_req.stale_histories
            r.abort_histories = cur_req.abort_histories
            r.last_pending_reschedule_ts = cur_req.last_pending_reschedule_ts
            r.last_assigned_at = cur_req.last_assigned_at
            r.update_count = cur_req.update_count + 1

        # 从engine中取出的结果，update到这里
        ts_bucket_2s = int(time.time()) // 2 * 2
        for r in reqs:
            this_req = self.requests.get(r.request_id)
            if this_req is None:
                # (speculation) already finished by another replica
                continue
            if this_req.assigned_engine_id != r.assigned_engine_id:
                # (rebalanced) request maybe staled in this engine, skip
                continue
            if this_req.updated_at > r.updated_at and not r.finished:
                # (out of order) in coming request is out of date, skip
                continue

            # compute decoding throughput
            new_decoded_len = r.query.new_token_len - this_req.query.new_token_len
            self._accumulated_token_counts[ts_bucket_2s][r.assigned_engine_id][r.global_step] += new_decoded_len

            if r.finished:
                self.finished_requests[r.request_id] = r
                self.historical_finished_requests.add(r.request_id, r)
                self.finished_counter[r.global_step] += 1
                self.requests.pop(r.request_id)
                # 注意event不要pop，可能调用方还没开始wait
                evt = self._finished_events.get(r.request_id)
                if evt is not None:
                    evt.set()
            else:
                # 匹配的request，暂时全量更新
                self.requests[r.request_id] = r

    # mark stale queries as pending in request pool
    def update_stale(self, ready_engine_ids: Container[str]):
        # engine 死了立刻把请求释放，等待另外的engine处理
        with self._mutex:
            for _, req in self.requests.items():
                if req.assigned_engine_id is not None and req.assigned_engine_id not in ready_engine_ids:
                    # skip this if never been scheduled
                    history = StaleHistory.from_request(req, "engine-died", "engine-died")
                    req.stale_histories.append(history)
                    req.query.reset_timestamp()
                    req.query.reset_compute()
                    req.assigned_engine_id = None
                    req.assigned_engine_name = None
                    req.assigned = False
                    req.last_pending_reschedule_ts = time.time() * 1e3

    # mark the queries as pending from busy engines
    def release(self, waiting_query_ids: List[str], engine_id: str, stale_reason: str) -> List[str]:
        ret = []
        with self._mutex:
            for query_id in waiting_query_ids:
                req = self.requests.get(query_id)
                if req is None or req.assigned_engine_id is None:
                    continue
                if req.assigned_engine_id is not None and req.assigned_engine_id != engine_id:
                    # already assigned to another engine, ignore
                    continue
                history = StaleHistory.from_request(req, "release", stale_reason)
                abort_history = AbortHistory(
                    engine_id=engine_id,
                    timestamp=time.time(),
                )
                req.stale_histories.append(history)
                req.abort_histories.append(abort_history)
                req.query.reset_timestamp()
                req.query.reset_compute()
                req.assigned_engine_id = None
                req.assigned_engine_name = None
                req.assigned = False
                req.last_pending_reschedule_ts = time.time() * 1e3
                ret.append(query_id)
        return ret

    # 返回某engine正在跑的前n个生成的最短的request
    def get_shortest_n(self, n: int, engine_id: str) -> List[Request]:
        reqs_for_engine = []
        with self._mutex:
            for req in self.requests.values():
                if req.assigned_engine_id is not None and req.assigned_engine_id != engine_id:
                    continue
                reqs_for_engine.append(req)

        reqs_for_engine.sort(key=lambda r: r.query.new_token_len)
        return reqs_for_engine[:n]

    # 返回长时间assigned但未被更新的query，返回query_id和engine_id
    def get_possible_hang_query_ids(self, threshold: float) -> List[Tuple[str, str]]:
        now = time.time()
        ret = []
        with self._mutex:
            for req_id, req in self.requests.items():
                if req.assigned_engine_id is None:
                    continue
                if now - req.updated_at > threshold:
                    ret.append((req_id, req.assigned_engine_id))
        return ret

    # 返回每个engine_id在最近给定的interval里的throughput
    def get_throughput(self, /, step: Optional[int] = None, interval=30) -> Dict[str, float]:
        # step: 查属于第几个step的query的吞吐，None则不区分step算总吞吐
        # interval: 从前interval seconds到现在区间的统计的吞吐
        now = time.time()
        since = now - interval
        ts_buckets = sorted(self._accumulated_token_counts.keys())
        token_count = defaultdict(int)
        for bucket in reversed(ts_buckets):
            if bucket > since:
                for engine_id in list(self._accumulated_token_counts[bucket].keys()):
                    if step is None:
                        count_all_steps = list(self._accumulated_token_counts[bucket][engine_id].values())
                        for count in count_all_steps:
                            token_count[engine_id] += count
                    else:
                        count = self._accumulated_token_counts[bucket][engine_id][step]
                        token_count[engine_id] += count

        ret = {}
        for engine_id, count in token_count.items():
            ret[engine_id] = count / interval

        # clean up out dated ts buckets
        out_dated_ts = now - self._metrics_max_retain_seconds
        for bucket in ts_buckets:
            if bucket < out_dated_ts:
                self._accumulated_token_counts.pop(bucket, None)

        return ret

    def get_concurrency(self) -> Dict[str, int]:
        """
        返回每个engine当前的concurrency
        """
        ret = defaultdict(int)
        with self._mutex:
            for _, req in self.requests.items():
                ret[req.assigned_engine_id] += 1
        return ret


@ray.remote
class RequestManagerRegisterCenter:

    def __init__(self, config: DictConfig):
        # config: the root config
        self.config = config
        self.names = set()
        self.registry = []

    def ready(self):
        return True

    def register(self, name: str):
        assert name not in self.names, f"duplicate request manager name: {name}"
        self.names.add(name)

    def get_all_names(self) -> List[str]:
        return list(self.names)

    @classmethod
    def init(cls, config: DictConfig):
        resources = {}
        if not is_local_ray_instance():
            resources = {"worker": 1}
        rmrc = RequestManagerRegisterCenter.options(name='RequestManagerRegisterCenter',
                                                    resources=resources).remote(config)
        ray.get(rmrc.ready.remote())
        return rmrc

    def create(self, instance_name: str):
        resources = {}
        if not is_local_ray_instance():
            # 非local模式下，让RequestManager只跑在stable resources上
            resources = {"worker": 1}
        # note(lixiang): concurrency必须超过global batch size才行，不然会卡住更新不了请求，导致死锁
        store_decoding_output = self.config.streaming_rollout.query_trace.store_decoding_output
        request_manager = RequestManager.options(name=f'RequestManager/{instance_name}',
                                                 resources=resources,
                                                 max_concurrency=102400).remote(store_decoding_output)
        ray.wait([request_manager.ready.remote()])
        self.registry.append(request_manager)
        self.names.add(instance_name)
        return request_manager

    @staticmethod
    def get(name: str) -> 'RequestManager':
        return ray.get_actor(f'RequestManager/{name}')  # noqa


class ProgressBar:

    def __init__(self, name, log_interval=0.1):
        self.name = name
        self.total_finished = 0
        self.recent_finished = 0
        self.recent_update = 0
        self.log_interval = log_interval
        self._next_log_at = log_interval

    def update(self, size: int, remaining_size: int, finished_size: int, engine_id: str):
        self.total_finished += finished_size
        self.recent_finished += finished_size
        self.recent_update += size
        total = remaining_size + self.total_finished
        if total <= 0 or finished_size <= 0:
            return

        progress = self.total_finished / total
        if progress > self._next_log_at:
            print(
                f"will update {self.recent_update}x queries(dup) from engine({engine_id}) "
                f"to {self.name}, finished={self.recent_finished}/remain={remaining_size}, progress={progress*100:.2f}%"
            )
            self.recent_finished = 0
            self.recent_update = 0
            self._next_log_at += self.log_interval
            if self._next_log_at > 1:
                self._next_log_at = 1

    def reset(self):
        self.total_finished = 0
        self.recent_finished = 0
        self._next_log_at = self.log_interval


@ray.remote
class RequestManager:

    def __init__(self, tracer_store_decoding_output: bool):
        self.req_pool = RequestPool()
        self._step = 0
        self._tracer_store_decoding_output = tracer_store_decoding_output
        self.tracer = Tracer.get_instance()
        self.waterfall_tracer = WaterfallSlotTracer(self.tracer)
        self._pending_events_to_flows = []
        self.actor_name = ray.get_runtime_context().get_actor_name()
        self._rm_name = self.actor_name.removeprefix('RequestManager/')
        self._progress_bar = ProgressBar(self.actor_name)
        self._query_id_log = defaultdict(set)  # step -> set(query.id)

    def ready(self):
        print(f'RequestManager ready, {self.actor_name=}')
        return True

    async def put_new_query(self, query: Query) -> str:
        query.enqueue_time = time.time() * 1e3
        step = query.meta_info.get('step', self._step)
        self.req_pool.put_new_requests([
            Request(
                request_id=query.id,
                query=query,
                global_step=step,
                last_pending_reschedule_ts=query.created_time,
                updated_at=time.time(),
            )
        ])
        self._query_id_log[step].add(query.id)
        return query.id

    async def wait_until_finished(self, query_id: str) -> Query:
        req = await self.req_pool.wait(query_id)
        events = self._make_trace_event(req)
        for evt in events:
            self.waterfall_tracer.trace(evt)
        self._pending_events_to_flows.append(events)
        return req.query

    def update_intermediate_queries(self, queries: List[Query], engine_id: str, wg_name: str, ts: float):
        finished = len(list(None for q in queries if q.is_finished))
        self._progress_bar.update(len(queries), len(self.req_pool), finished, engine_id)
        # 从engine取出的结果，更新到request pool里
        reqs = [
            Request(
                request_id=q.id,
                query=q,
                finished=q.is_finished,
                assigned=True,
                assigned_engine_id=engine_id,
                assigned_engine_name=wg_name,
                updated_at=ts,
            ) for q in queries
        ]
        self.req_pool.update(reqs)
        update_metrics = CounterEvent(name='sync metrics:',
                                      pid=f'{self._rm_name} {wg_name}',
                                      ts=ts * 1e6,
                                      data={
                                          "#update": len(queries),
                                          "#finished": finished,
                                      })
        self.tracer.trace(update_metrics)

    def _debug(self):
        return self.req_pool.requests, self.req_pool.finished_requests

    def get_next_pending_requests(self, batch_size: int, engine_id: str, wg_name: str) -> List[Query]:
        next_reqs = self.req_pool.get_next_pending_requests_with_cache(batch_size, engine_id, wg_name)
        return [r.query for r in next_reqs.values()]

    def get_next_pending_requests_with_cache(self, batch_size: int, engine_id: str, wg_name: str,
                                             cache_ids: List[str]) -> List[Query]:
        next_reqs = self.req_pool.get_next_pending_requests_with_cache(batch_size, engine_id, wg_name, cache_ids)
        return [r.query for r in next_reqs.values()]

    # 释放掉给定的query_ids，返回确定释放的query_id
    def release_by_ids(self, query_ids: List[str], engine_id: str, reason: str) -> List[str]:
        return self.req_pool.release(query_ids, engine_id, reason)

    # 释放掉最短的前n个，返回确定释放的query_id
    def release_shortest_n(self, n: int, engine_id: str, reason: str) -> List[str]:
        queries = self.req_pool.get_shortest_n(n, engine_id)
        return self.req_pool.release([q.request_id for q in queries], engine_id, reason)

    def handle_stale_requests(self, ready_engine_ids: Container[str]):
        # 根据还存活的engine id，将其他死掉的engine在跑的request标记为待认
        self.req_pool.update_stale(ready_engine_ids)

    def get_possible_hang_query_ids(self, threshold: float) -> List[Tuple[str, str]]:
        # return [(query_id, engine_id), ...]
        return self.req_pool.get_possible_hang_query_ids(threshold)

    def set_global_step(self, global_step: int):
        self._step = global_step
        self._progress_bar.reset()
        self.waterfall_tracer.flush()
        # reorder后，重新确定了tid，这时再计算flows
        evts_to_flows = self._pending_events_to_flows
        self._pending_events_to_flows = []
        for events in evts_to_flows:
            flows = self._make_stale_flow_trace_event(events)
            self.tracer.trace(CombinedEvents(flows))

    def get_pending_size(self):
        return self.req_pool.get_pending_size()

    def get_size(self) -> Tuple[int, int]:
        total = len(self.req_pool)
        pending = self.req_pool.get_pending_size()
        return total, pending

    def get_estimated_throughput(self) -> Dict[str, float]:
        # 返回{engine_id -> 最近interval内的平均throughput}
        return self.req_pool.get_throughput()

    def get_concurrency(self) -> Dict[str, int]:
        return self.req_pool.get_concurrency()

    def _normalize_output(self, output: str) -> str:
        # 有些输出的字符会导致perfetto UI显示不了，因此可以通过开关关掉输出，要看rollout具体输出内容可以去tracking里
        if self._tracer_store_decoding_output:
            return output.replace("\"", ".").replace("\'", ".").replace("$", ".").replace("{", ".").replace("}", ".")
        else:
            return ""

    def _make_trace_event(self, req: Request) -> List[Union[CompleteEvent, CoherentCompleteEvent]]:
        # [C][W][P][D](stale) --> [W][P][D]
        # [C]Query对象被创建出来的时刻，也就是从dataloader里取出来的时刻，每个batch应该几乎统一开始
        # [W]始终以re/scheduled ts开始，
        # [P]尽量以engine里的first_scheduled_time作为开始
        # [D]尽量以first_token_time作为开始
        # 如果任意一个时间为0，则往前回退一个已标记的时间

        query = req.query
        received_time = query.received_time
        first_scheduled_time = query.first_scheduled_time or query.created_time
        first_token_time = query.first_token_time or first_scheduled_time
        meta_info = copy.deepcopy(query.meta_info)
        meta_info.get("extra_data", {}).pop("config", None)  # pop掉没用又冗余的字段

        # 暂时不记录C，因为C属于driver/cpu上的时间分配，不属于engine那边的分配，从W开始才算时engine的
        new_created = CompleteEvent(
            name='C',
            cat='rollout-new',
            pid=f'{self._rm_name} {req.assigned_engine_name}',
            tid=0,
            ts=req.last_pending_reschedule_ts * 1000,
            dur=(received_time - req.last_pending_reschedule_ts) * 1000,
            args={
                'query_id': query.id,
                'input_len': query.original_input_len,
                'step': req.global_step,
                'stale_count': len(req.stale_histories),
            },
        )
        wait = CompleteEvent(
            name='W',
            cat='rollout-wait',
            pid=f'{self._rm_name} {req.assigned_engine_name}',
            tid=0,
            ts=received_time * 1000 + 1,
            dur=(first_scheduled_time - received_time) * 1000 - 1,
            args={
                'query_id': query.id,
                'original_input_len': query.original_input_len,
                # 如果engine更新过参数，这个值也会显示为更新参数前已经decode的长度
                'previous_generated_len': len(query.input_ids) - query.original_input_len,
                'step': req.global_step,
                'stale_count': len(req.stale_histories),
                'abort_count': len(req.abort_histories),
                'age': (received_time - query.created_time) / 1e3,
                'enqueue_delay': query.enqueue_time - query.created_time,
            },
        )
        prefill = CompleteEvent(
            name='P',
            cat='rollout-prefill',
            pid=f'{self._rm_name} {req.assigned_engine_name}',
            tid=0,
            ts=first_scheduled_time * 1000 + 1,  # 处理渲染上对齐的误差，偏移1us
            dur=(first_token_time - first_scheduled_time) * 1000 - 1,
            args={
                'query_id': query.id,
                'prefill_len': len(query.input_ids),  # 用这个表示prefill里真正输入的token数，可能包含中途decode中断重新prefill的token
                'step': req.global_step,
                'stale_count': len(req.stale_histories),
                'age': (first_scheduled_time - query.created_time) / 1e3,
            },
        )
        decode = CompleteEvent(
            name='D',
            cat='rollout-decode',
            pid=f'{self._rm_name} {req.assigned_engine_name}',
            tid=0,
            ts=first_token_time * 1000 + 1,
            dur=(query.finished_time - first_token_time) * 1000 - 1,
            args={
                'query_id': query.id,
                'input': query.input_prompt,
                'output': self._normalize_output(query.output_prompt[0]) if query.output_prompt else "",
                'original_input_len': query.original_input_len,  # 原始输入给定的prefill token数，对齐openai usage的指标
                'length_generated': query.new_token_len -
                                    (len(query.input_ids) - query.original_input_len),  # 本次decode的token数
                'total_output_len': query.new_token_len,  # 总共decode的token数
                'update_count': req.update_count,
                'release_count': req.query.release_count,
                'step': req.global_step,
                'meta_info': meta_info,
                'sample_kwargs': {
                    'top_k': query.top_k,
                    'top_p': query.top_p,
                    'temperature': query.temperature,
                }
            },
        )
        histories = []
        for idx, his in enumerate(req.stale_histories):
            stale_wait = CompleteEvent(
                name='W',
                cat='rollout-wait',
                pid=f'{self._rm_name} {his.assigned_engine_name}',
                tid=0,
                ts=his.received_time * 1000 + 1,
                dur=(his.first_scheduled_time - his.received_time) * 1000 - 1,
                args={
                    'stale_count': idx,
                    'query_id': query.id,
                    'step': his.start_step,
                },
            )
            stale_prefill = CompleteEvent(
                name='P',
                cat='rollout-prefill',
                pid=f'{self._rm_name} {his.assigned_engine_name}',
                tid=0,
                ts=his.first_scheduled_time * 1000 + 1,
                dur=(his.first_token_time - his.first_scheduled_time) * 1000 - 1,
                args={
                    'stale_count': idx,
                    'query_id': query.id,
                    'step': his.start_step,
                },
            )
            stale_decode = CompleteEvent(
                name=his.stale_reason,
                cat='rollout-stale',
                pid=f'{self._rm_name} {his.assigned_engine_name}',
                tid=0,
                ts=his.first_token_time * 1000 + 1,
                dur=(his.end_ts - his.first_token_time) * 1000 - 1,
                args={
                    'stale_count': idx,
                    'query_id': query.id,
                    'step': his.start_step,
                    'stale_action': his.stale_action,
                    'stale_reason': his.stale_reason,
                    'length_generated': his.length_generated,
                    'update_count': his.update_count,
                    'release_count': his.release_count,
                },
            )
            histories.append(CoherentCompleteEvent([stale_wait, stale_prefill, stale_decode], 1))

        return [CoherentCompleteEvent([wait, prefill, decode], 1)] + histories

    def _make_stale_flow_trace_event(self, events: List[TracingEvent]) -> List[TracingEvent]:
        # 这个函数跟上面那个_make_trace_event配合用，先通过waterfall tracer分配了tid之后，再调用这个构造flow
        working, *histories = events
        working: CoherentCompleteEvent
        history_flows = []
        for i in range(len(histories) - 1):
            his0 = histories[i]
            his1 = histories[i + 1]
            from_ = (his0.pid, his0.tid, his0.ts + his0.dur - 2)
            to = (his1.pid, his1.tid, his1.ts + 2)
            flow = FlowEvent(name=f'stale-flow-{his0.args["query_id"]}-{i}', cat='stale-flow', flows=[from_, to])
            history_flows.append(flow)
        # from last stale to final prefill
        if len(histories) > 0:
            last_one = histories[-1]
            from_ = (last_one.pid, last_one.tid, last_one.ts + last_one.dur - 2)
            to = (working.pid, working.tid, working.ts + 2)
            flow = FlowEvent(name=f'stale-flow-last-{last_one.args["query_id"]}', cat='stale-flow', flows=[from_, to])
            history_flows.append(flow)
        return history_flows

    def dump_request_trace(self) -> List[dict]:
        return Tracer.merge_all()

    ## query_tool util function ##

    def get_inflight_query_digest(self) -> List[RequestDigest]:
        ret = []
        query_ids = list(self.req_pool.requests.keys())
        for query_id in query_ids:
            req = self.req_pool.requests.get(query_id)
            reg_digest = RequestDigest(
                query_id=query_id,
                assigned_engine_id=req.assigned_engine_id,
                assigned_engine_name=req.assigned_engine_name,
                assigned_at=req.last_assigned_at,
                updated_at=req.updated_at,
                input_length=req.query.original_input_len,
                output_length=req.query.new_token_len,
                aborted_count=len(req.abort_histories),
                stale_count=len(req.stale_histories),
            )
            ret.append(reg_digest)
        return ret

    def get_inflight_query_ids(self) -> List[str]:
        return list(self.req_pool.requests.keys())

    def get_finished_query_ids(self) -> List[str]:
        return list(self.req_pool.finished_requests.keys())

    def get_finished_stats(self, step=None) -> FinishedEventStats:
        finished_events = list(self.req_pool._finished_events.items())
        done = 0
        waiting = 0
        done_keys = []
        for req_id, evt in finished_events:
            if evt.is_set():
                done += 1
                done_keys.append(req_id)
            else:
                waiting += 1
        finished_staging_size = len(self.req_pool.finished_requests)
        finished_accumulated_size = len(self.req_pool.historical_finished_requests)
        if step is None:
            finished_size = sum(self.req_pool.finished_counter.values())
        else:
            finished_size = self.req_pool.finished_counter[step]
        return FinishedEventStats(waiting, done, finished_size, finished_staging_size, finished_accumulated_size,
                                  done_keys)

    def get_by_id(self, query_id: str) -> Optional[Request]:
        return self.req_pool.requests.get(query_id) or self.req_pool.historical_finished_requests.get(query_id)

    def get_progress(self) -> List[ProgressStat]:

        # 获取当前正在跑的
        current_inflight = set(self.get_inflight_query_ids())

        # 计算所有旧步骤的统计信息
        ret = []
        for step in list(self._query_id_log.keys()):
            step_total = len(self._query_id_log[step])
            step_finished = step_total - len(current_inflight.intersection(self._query_id_log[step]))

            throughput = self.req_pool.get_throughput(step=step)
            total_throughput = sum(throughput.values())

            # 为当前step计算指标
            step_running = 0
            step_pending = 0
            step_active_engines = set()
            step_oldest_query_time = time.time()
            step_oldest_updated_time = time.time()
            step_latest_query_time = 0

            # 遍历该step的所有queries
            step_query_ids = self._query_id_log[step]
            for query_id in step_query_ids:
                req = self.req_pool.requests.get(query_id)
                if req is None:
                    continue
                if req.assigned:
                    step_running += 1
                    step_active_engines.add(req.assigned_engine_id)
                    if req.updated_at:
                        step_oldest_updated_time = min(step_oldest_updated_time, req.updated_at)
                    if req.last_assigned_at:
                        step_oldest_query_time = min(step_oldest_query_time, req.last_assigned_at)
                        step_latest_query_time = max(step_latest_query_time, req.last_assigned_at)
                else:
                    step_pending += 1

            # 如果没有任何assigned的请求，重置时间戳
            if step_running == 0:
                step_oldest_query_time = 0
                step_oldest_updated_time = 0
                step_latest_query_time = 0

            if step_finished < step_total:
                ret.append(
                    ProgressStat(
                        step=step,
                        total=step_total,
                        finished=step_finished,
                        token_throughput=total_throughput,
                        running_queries=step_running,
                        pending_queries=step_pending,
                        active_engines=len(step_active_engines),
                        oldest_updated_time=step_oldest_updated_time,
                        oldest_query_time=step_oldest_query_time,
                        latest_query_time=step_latest_query_time,
                    ))

        ret = sorted(ret, key=lambda p: p.step)
        return ret


def get_all_request_manager_actors() -> List[RequestManager]:
    rmrc: RequestManagerRegisterCenter = ray.get_actor('RequestManagerRegisterCenter')  # noqa
    names = ray.get(rmrc.get_all_names.remote())
    rms = []
    for n in names:
        rms.append(RequestManagerRegisterCenter.get(n))
    return rms
