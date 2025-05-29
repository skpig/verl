import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict

import ray

from alpha_seed.workers.xperf_rollout.component.query import Query
from alpha_seed.utils.profile.timeline import CoherentCompleteEvent, Tracer, TracingEvent, CompleteEvent, CounterEvent, \
    FlowEvent, \
    CombinedEvents, WaterfallSlotTracer, OrderedTracer


@dataclass
class StaleHistory:
    assigned_engine_id: str
    start_step: int  # 最早开始的step
    start_ts: float  # 最早开始的时刻 (unit: ms)
    end_ts: float  # 发现stale的时刻 (unit: ms)
    stale_reason: str
    length_generated: int


@dataclass
class Request:
    request_id: str
    query: Query
    finished: bool = False  # 是否已经结束生成
    assigned: bool = False  # 是否已经分出去了（分出去了，但不确定具体分配的engine_id，取决于passive/active模式
    assigned_engine_id: Optional[str] = None  # 被分配到的engine

    # fields need to assign back
    global_step: int = 0  # 当前这个query来自哪个global step的sample
    last_pending_reschedule_ts: float = 0  # 最近一次stale被重新调度的时间戳 (unit: ms)
    stale_histories: List[StaleHistory] = field(default_factory=list)


@dataclass
class StepRequestSpan:
    global_step: int = 0
    start_ts: float = 0  # step第一个request开始时间
    end_ts: float = 0  # step结束时间，也是下一个step的开始时间
    step_clear_ts: float = 0  # step内最后一个request完成的时间

    def get_busy_ratio(self) -> float:
        return (self.step_clear_ts - self.start_ts) / (self.end_ts - self.start_ts)

    @property
    def finished(self) -> bool:
        return self.end_ts > self.start_ts


class RequestPool:

    def __init__(self):
        self.requests: Dict[str, Request] = {}  # 中间结果会被update进来
        self.finished_requests: Dict[str, Request] = {}  # finished部分会被移到这里
        self._finished_events: Dict[str, asyncio.Event] = {}  # 标记请求完成的async event
        self._mutex = threading.Lock()
        self._step_span_marks: Dict[int, StepRequestSpan] = {}  # global_step -> StepRequestSpan

    def __len__(self) -> int:
        return len(self.requests)

    def put_new_requests(self, reqs: List[Request]):
        assert len(reqs) > 0
        with self._mutex:
            for r in reqs:
                self.requests[r.request_id] = r
                self._finished_events[r.request_id] = asyncio.Event()

        # 纪录这个step开始时间
        global_step = reqs[0].global_step
        if global_step not in self._step_span_marks:
            self._step_span_marks[global_step] = StepRequestSpan(global_step=global_step, start_ts=time.time())

        span = self._step_span_marks.get(global_step - 1)
        if span is not None:
            span.end_ts = time.time()

    def get_pending_size(self):
        # 返回还未分发出去的请求的数量
        count = 0
        with self._mutex:
            for _, request in self.requests.items():
                if request.assigned or request.assigned_engine_id is not None:
                    continue
                count += 1
            return count

    def get_next_pending_requests(self, batch_size, engine_id: Optional[str] = None) -> Dict[str, Request]:
        # 每个engine实例来这里pull空闲的请求
        # 简单处理，暂不允许并发获取请求
        with self._mutex:
            batch_size = min(batch_size, len(self.requests))
            ret = {}
            for request_id, request in self.requests.items():
                if request.assigned or request.assigned_engine_id is not None:
                    continue
                # 标记请求已被认领了再分发出去
                request.assigned = True
                request.assigned_engine_id = engine_id
                ret[request_id] = request
                if len(ret) == batch_size:
                    break
            return ret

    def get_pool_size(self) -> Tuple[int, int]:
        return len(self.requests), len(self.finished_requests)

    async def wait(self, request_id: str) -> Request:
        """
        一直等待某个request 生成完成才返回，返回后，这个request不再保存在pool里
        """
        await self._finished_events[request_id].wait()
        return self.finished_requests.pop(request_id)

    def update(self, reqs: List[Request]):
        # assign global_step back
        for r in reqs:
            cur_req = self.requests[r.request_id]
            r.global_step = cur_req.global_step
            r.stale_histories = cur_req.stale_histories
            r.last_pending_reschedule_ts = cur_req.last_pending_reschedule_ts

        # 从engine中取出的结果，update到这里
        for req in reqs:
            if req.finished:
                self.finished_requests[req.request_id] = req
                self.requests.pop(req.request_id)
                evt = self._finished_events.pop(req.request_id)
                evt.set()

                # 纪录这个request的结束时间，始终更新对应step的clear ts
                span = self._step_span_marks.get(req.global_step)
                if span is not None:
                    span.step_clear_ts = time.time()
            else:
                self.requests[req.request_id] = req

    def update_stale(self, alive_engine_ids: List[str]):
        # engine 死了立刻把请求释放，等待另外的engine处理
        for _, req in self.requests.items():
            if req.assigned_engine_id is not None and req.assigned_engine_id not in alive_engine_ids:
                # skip this if never been scheduled
                history = StaleHistory(
                    assigned_engine_id=req.assigned_engine_id,
                    start_step=req.global_step,
                    # if never be scheduled in engine, use the rescheduled time in requst manager
                    start_ts=req.query.first_scheduled_time or req.last_pending_reschedule_ts,
                    end_ts=time.time() * 1e3,
                    stale_reason='engine-died',
                    length_generated=req.query.new_token_len,
                )
                req.stale_histories.append(history)
                req.query.reset_timestamp()
                req.query.reset_compute()
                req.assigned_engine_id = None
                req.assigned = False
                req.last_pending_reschedule_ts = time.time() * 1e3

    def get_recent_step_busy_ratio(self, recent_n_steps: int) -> List[Tuple[int, float]]:
        # 获取最近观测到的每个step的busy ratio
        # busy ratio定义为，step时间内，poll中存在未完成的request的时间占比
        # 从最新往旧的返回busy ratio，范围0～1
        if len(self._step_span_marks) == 0:
            return []

        ret = []
        current_step = max(self._step_span_marks.keys())
        # 取多1个，因为可能最后一个sample没有完成
        for step in range(current_step, current_step - recent_n_steps - 1, -1):
            span = self._step_span_marks.get(step)
            # 取到None就不再往前取，因为肯定没了，假设step不会跳
            if span is None:
                break
            if not span.finished:
                continue
            ret.append((span.global_step, span.get_busy_ratio()))
        return ret


class RequestManagerRegisterCenter:

    def __init__(self):
        self.names = set()

    def ready(self):
        return True

    def register(self, name: str):
        assert name not in self.names, f"duplicate request manager name: {name}"
        self.names.add(name)

    def get_all_names(self) -> List[str]:
        return list(self.names)


class ProgressBar:

    def __init__(self, name, log_interval=0.1):
        self.name = name
        self.total_finished = 0
        self.log_interval = log_interval
        self._next_log_at = log_interval

    def update(self, size: int, remaining_size: int, finished_size: int, engine_id: str):
        self.total_finished += finished_size
        total = remaining_size + self.total_finished
        if total <= 0 or finished_size <= 0:
            return

        progress = self.total_finished / total
        if progress > self._next_log_at:
            print(f"will update {size}/{remaining_size} queries from engine({engine_id}) "
                  f"to {self.name}, finished={finished_size}, progress={progress*100:.2f}%")
            self._next_log_at += self.log_interval
            if self._next_log_at > 1:
                self._next_log_at = 1

    def reset(self):
        self.total_finished = 0
        self._next_log_at = self.log_interval


# this will be run on ray remote
class RequestManager:

    def __init__(self):
        self.req_pool = RequestPool()
        self._step = 0
        self.tracer = Tracer.get_instance()
        self.ordered_tracer = OrderedTracer(self.tracer)
        self._pending_events_to_flows = []
        self.actor_name = ray.get_runtime_context().get_actor_name()
        self._rm_name = self.actor_name.removeprefix('RequestManager/')
        self._progress_bar = ProgressBar(self.actor_name)
        try:
            rmrc = ray.get_actor('RequestManagerRegisterCenter')
            ray.get(rmrc.register.remote(self.actor_name))
        except ValueError:
            raise RuntimeError("please initialize RequestManagerRegisterCenter first before creating RequestManager")

    def ready(self):
        print(f'RequestManager ready, {self.actor_name=}')
        return True

    def put_new_query(self, query: Query) -> str:
        self.req_pool.put_new_requests([
            Request(
                request_id=query.id,
                query=query,
                global_step=self._step,
                last_pending_reschedule_ts=query.created_time,
            )
        ])
        return query.id

    async def wait_until_finished(self, query_id: str) -> Query:
        req = await self.req_pool.wait(query_id)
        events = self._make_trace_event(req)
        for evt in events:
            self.ordered_tracer.trace(evt)
        self._pending_events_to_flows.append(events)
        return req.query

    def update_intermediate_queries(self, queries: List[Query], engine_id: str):
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
            ) for q in queries
        ]
        self.req_pool.update(reqs)

    def _debug(self):
        return self.req_pool.requests, self.req_pool.finished_requests

    def get_next_pending_requests(self, batch_size: int, engine_id: str) -> List[Query]:
        next_reqs = self.req_pool.get_next_pending_requests(batch_size, engine_id)
        return [r.query for r in next_reqs.values()]

    def handle_stale_requests(self, alive_engine_ids: List[str]):
        # 根据还存活的engine id，将其他死掉的engine在跑的request标记为待认
        self.req_pool.update_stale(alive_engine_ids)

    def set_global_step(self, global_step: int):
        self._step = global_step
        self._progress_bar.reset()
        self.ordered_tracer.reorder_flush()
        # reorder后，重新确定了tid，这时再计算flows
        for events in self._pending_events_to_flows:
            flows = self._make_stale_flow_trace_event(events)
            self.tracer.trace(CombinedEvents(flows))

    def get_pending_size(self):
        return self.req_pool.get_pending_size()

    def get_size(self) -> Tuple[int, int]:
        total = len(self.req_pool)
        pending = self.req_pool.get_pending_size()
        return total, pending

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

        new_created = CompleteEvent(
            name='C',
            cat='rollout-new',
            pid=f'{self._rm_name} {req.assigned_engine_id}',
            tid=0,
            ts=req.last_pending_reschedule_ts * 1000,
            dur=(received_time - req.last_pending_reschedule_ts) * 1000,
            args={
                'query_id': query.id,
                'input_len': query.input_len,
                'step': req.global_step,
                'stale_count': len(req.stale_histories),
            },
        )
        wait = CompleteEvent(
            name='W',
            cat='rollout-wait',
            pid=f'{self._rm_name} {req.assigned_engine_id}',
            tid=0,
            ts=received_time * 1000 + 1,
            dur=(first_scheduled_time - received_time) * 1000 - 1,
            args={
                'query_id': query.id,
                'input_len': query.input_len,
                'step': req.global_step,
                'stale_count': len(req.stale_histories),
            },
        )
        prefill = CompleteEvent(
            name='P',
            cat='rollout-prefill',
            pid=f'{self._rm_name} {req.assigned_engine_id}',
            tid=0,
            ts=first_scheduled_time * 1000 + 1,  # 处理渲染上对齐的误差，偏移1us
            dur=(first_token_time - first_scheduled_time) * 1000 - 1,
            args={
                'query_id': query.id,
                'input_len': query.input_len,
                'step': req.global_step,
                'stale_count': len(req.stale_histories),
            },
        )
        decode = CompleteEvent(
            name='D',
            cat='rollout-decode',
            pid=f'{self._rm_name} {req.assigned_engine_id}',
            tid=0,
            ts=first_token_time * 1000 + 1,
            dur=(query.finished_time - first_token_time) * 1000 - 1,
            args={
                'query_id': query.id,
                'input': query.input_prompt,
                'output': query.output_prompt,
                'input_len': query.input_len,
                'output_len': query.new_token_len,
                'step': req.global_step,
                'meta_info': query.meta_info,
                'sample_kwargs': {
                    'top_k': query.top_k,
                    'top_p': query.top_p,
                    'temperature': query.temperature,
                }
            },
        )
        histories = []
        for his in req.stale_histories:
            stale = CompleteEvent(
                name='stale',
                cat='rollout-stale',
                pid=f'{self._rm_name} {his.assigned_engine_id}',
                tid=0,
                ts=his.start_ts * 1000,
                dur=(his.end_ts - his.start_ts) * 1000,
                args={
                    'query_id': query.id,
                    'step': his.start_step,
                    'stale_reason': his.stale_reason,
                    'length_generated': his.length_generated,
                },
            )
            histories.append(stale)

        return [CoherentCompleteEvent([new_created, wait, prefill, decode], 2)] + histories

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

    def get_recent_step_busy_ratio(self, recent_n: int) -> List[Tuple[int, float]]:
        ratio = self.req_pool.get_recent_step_busy_ratio(recent_n)
        if ratio:
            gs, last_busy_ratio = ratio[0]
            now = time.time() * 1e6
            self.tracer.trace(CounterEvent('metric', 'RequestManager', now, {"global step": gs}))
            self.tracer.trace(CounterEvent('metric', 'RequestManager', now, {"busy%": last_busy_ratio}))
        return ratio


def get_all_request_manager_actors() -> List[RequestManager]:
    rmrc = ray.get_actor('RequestManagerRegisterCenter')
    names = ray.get(rmrc.get_all_names.remote())
    rms = []
    for n in names:
        rms.append(ray.get_actor(n))
    return rms
