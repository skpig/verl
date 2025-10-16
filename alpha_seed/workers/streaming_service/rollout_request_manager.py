import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple, Container, Set, Any
from collections import defaultdict

import ray
from omegaconf import DictConfig

from alpha_seed.workers.streaming_service.rollout_query_trace import QueryTracer
from alpha_seed.workers.streaming_service.rollout_request import StaleHistory, AbortHistory, Request
from alpha_seed.workers.streaming_service.rollout_request_manager_diagnosis import FinishedEventStats, RequestDigest, \
    FiniteDict, ProgressStat, RequestStatCollector, RequestPoolInternalDiagnosis
from alpha_seed.workers.xperf_rollout.component.query import Query, QueryUpdate
from alpha_seed.utils.profile.timeline import CounterEvent
from alpha_seed.utils.server_client import get_stable_res


class FIFOListIter:

    def __init__(self, lst: list, head_idx: int):
        self.lst = lst
        self.cur_idx = head_idx - 1

    def __next__(self):
        while True:
            self.cur_idx += 1
            if self.cur_idx >= len(self.lst):
                raise StopIteration
            val = self.lst[self.cur_idx]
            if val is None:
                continue
            else:
                return val


class FIFOList:
    """
    一个简单数据结构用来做快速fifo，元素值存request_id
    先进来的先被分发，直到finish了才从list里移除掉，
    """

    def __init__(self):
        self.lst = []
        self.index_map: Dict[Any, int] = {}  # value -> index反向映射
        self.head_idx = 0
        self.mutex = threading.Lock()

    def append(self, item):
        assert item is not None, "FIFOList item cannot be None"
        with self.mutex:
            idx = len(self.lst)
            self.lst.append(item)
            self.index_map[item] = idx

    def remove(self, val):
        with self.mutex:
            idx = self.index_map.pop(val, None)
            if idx is not None:
                self.lst[idx] = None  # 用None标记删除
                # 如果删掉的是head idx所在的元素，才有可能需要移动head
                if idx == self.head_idx:
                    # 尝试移动head idx到最近一个非None元素
                    while self.head_idx < len(self.lst) and self.lst[self.head_idx] is None:
                        self.head_idx += 1

                    # 头部空到一定程度再截断(超过1000个element或前1/4已经空了)
                    if self.head_idx > 1000 or self.head_idx > len(self.lst) // 4:
                        # 更新index_map中的索引：直接减去偏移量
                        for item in self.index_map:
                            self.index_map[item] -= self.head_idx
                        self.lst = self.lst[self.head_idx:]
                        self.head_idx = 0

    def empty(self):
        return self.head_idx >= len(self.lst)

    def __iter__(self):
        # 只作view返回
        # 只读： iter里不能修改内容
        # 脏读： iter过程中不保证lst内容不变，可能还没iter到某个item时，这个item被删掉了
        return FIFOListIter(self.lst, self.head_idx)


class StepPriorityList:
    """
    按step排序的类FIFO，iterate时优先返回step更小的
    """

    def __init__(self):
        self.steps = defaultdict(FIFOList)  # step -> FIFOList

    def append(self, step: int, item):
        self.steps[step].append(item)

    def remove(self, step: int, item):
        self.steps[step].remove(item)

    def __iter__(self):
        for step in sorted(self.steps.keys()):
            fifo = self.steps[step]
            if not fifo.empty():
                yield from fifo


class RequestPool:

    def __init__(self):
        self.requests: Dict[str, Request] = {}  # {query_id -> } 中间结果会被update进来
        # 按顺序记录每个(step, query id)，分发的时候优先从取更早的step，锁跟着self.requests的一起就好
        self.fifo = StepPriorityList()
        self.finished_requests: Dict[str, Request] = {}  # finished部分会被移到这里
        self.historical_finished_requests = FiniteDict(204800)  # 记录所有完成的query，FIFO，便于query_tool查询诊断
        self.finished_counter = defaultdict(int)  # {step -> count} 统计每个step完成的数量(因为多轮每个step数量是会变化的)
        self._finished_events: Dict[str, asyncio.Event] = {}  # 标记请求完成的async event
        self._mutex = threading.Lock()

        # metrics for observability
        self._metrics_max_retain_seconds = 600
        # ts_bucket(?s) -> engine_id -> step -> count
        self._accumulated_token_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self._accumulated_prefill_token_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self._internal_metrics = RequestPoolInternalDiagnosis()

    def __len__(self) -> int:
        return len(self.requests)

    def put_new_requests(self, reqs: List[Request]):
        assert len(reqs) > 0
        with self._mutex:
            for r in reqs:
                self.requests[r.request_id] = r
                self._finished_events[r.request_id] = asyncio.Event()
                self.fifo.append(r.global_step, r.request_id)

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
                                             cache_ids: Set[str] = None) -> Dict[str, Request]:
        # 每个engine实例来这里pull空闲的请求
        # 简单处理，暂不允许并发获取请求
        cool_down_seconds = 10
        with self._mutex:
            batch_size = min(batch_size, len(self.requests))
            ret = {}
            for request_id in self.fifo:
                request = self.requests.get(request_id)
                if request is None:
                    # 可能在迭代中已经完成了，忽略
                    continue
                # 跳过已分发
                if request.assigned or request.assigned_engine_id is not None:
                    continue
                # 跳过最近abort
                if request.is_recent_aborted_from(engine_id, cool_down_seconds):
                    continue
                # 如果有cache_ids，且当前request不在cache_ids中，则跳过
                if cache_ids is not None and request.query.cache_id not in cache_ids:
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

    def peak_all_pending_requests(self) -> List[Request]:
        # pull全部空闲的请求，只是先将assigned设置成True
        # 后续再进行实际的分发
        ret = []
        with self._mutex:
            for request_id in self.fifo:
                request = self.requests.get(request_id)
                # 可能在迭代中已经完成了，忽略
                if request is None:
                    continue
                # 跳过已分发
                if request.assigned or request.assigned_engine_id is not None:
                    continue
                request.assigned = True
                ret.append(request)
        return ret

    def set_requests_assigned(self, request_ids: List[str], engine_id: str, wg_name: str, ts: float):
        # 真实分发请求
        # 注意，这里更新的字段不会发到给engine，
        # 为了让engine侧收到的query也跟这里的值保持一致，发给engine前记得手动补充这些字段
        for request_id in request_ids:
            request = self.requests.get(request_id)
            if request is None:
                continue
            request.assigned_engine_id = engine_id
            request.assigned_engine_name = wg_name
            request.last_assigned_at = ts
            request.updated_at = ts
            request.query.dispatch_time = ts

    def clear_requests_assigned(self, request_ids: List[str]):
        with self._mutex:
            for request_id in request_ids:
                request = self.requests.get(request_id)
                if request is None:
                    continue
                request.assigned = False
                request.assigned_engine_id = None
                request.assigned_engine_name = None

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
                # 开了增量更新后，这个乱序很危险，数据会丢失，先监控
                self._internal_metrics.out_of_order_count += 1
                continue

            # compute prefilling/decoding throughput
            new_decoded_len = r.query.new_token_len - this_req.query.new_token_len
            self._accumulated_token_counts[ts_bucket_2s][r.assigned_engine_id][r.global_step] += new_decoded_len
            self.accumulate_prefilling_throughput(r, this_req)

            if r.finished:
                self.finished_requests[r.request_id] = r
                self.historical_finished_requests.add(r.request_id, r)
                self.finished_counter[r.global_step] += 1
                self.requests.pop(r.request_id)
                self.fifo.remove(r.global_step, r.request_id)
                # 注意event不要pop，可能调用方还没开始wait
                evt = self._finished_events.get(r.request_id)
                if evt is not None:
                    evt.set()
            else:
                if isinstance(r.query, QueryUpdate):
                    # 增量更新Query对象后，让r更新成最新状态覆盖到原来的request
                    this_req.query.apply_update(r.query)
                    r.query = this_req.query
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
                    req.query.reset_compute()
                    req.query.reset_timestamp()
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
                req.query.reset_compute()
                req.query.reset_timestamp()
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

    def get_running_requests(self) -> List[Request]:
        ret = []
        with self._mutex:
            for req_id, req in self.requests.items():
                if req.assigned:
                    ret.append(req)
        return ret

    def accumulate_prefilling_throughput(self, new_req: Request, prev_req: Request):
        if new_req.query.recent_first_token_time > prev_req.query.recent_first_token_time:
            prefill_dur = (new_req.query.recent_first_token_time - new_req.query.recent_scheduled_time) / 1e3  # unit: s
            input_token_len = new_req.query.prefill_len
            # 因为是实时数据，只算最近30s的interval，超过的部分不用算了，并按等比例折算
            if prefill_dur > 30:
                input_token_len *= 30 / prefill_dur
                prefill_dur = 30
            # 保持整2s一个bucket
            prefill_end_bucket = int(new_req.query.recent_first_token_time / 1e3) // 2 * 2
            prefill_start_bucket = int(prefill_end_bucket - prefill_dur) // 2 * 2
            for ts_bucket in range(prefill_start_bucket, prefill_end_bucket, 2):
                self._accumulated_prefill_token_counts[ts_bucket][new_req.assigned_engine_id][
                    new_req.global_step] += input_token_len

    # 返回每个engine_id在最近给定的interval里的throughput
    def get_decode_throughput(self, /, step: Optional[int] = None, interval=30) -> Dict[str, float]:
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
            else:
                break

        return ret

    def get_prefill_throughput(self, /, step: Optional[int] = None, interval=10) -> Dict[str, float]:
        # step: 查属于第几个step的query的吞吐，None则不区分step算总吞吐
        # interval: 从前interval seconds到现在区间的统计的吞吐
        now = time.time()
        since = now - interval
        ts_buckets = sorted(self._accumulated_prefill_token_counts.keys())
        token_count = defaultdict(int)
        for bucket in reversed(ts_buckets):
            if bucket > since:
                for engine_id in list(self._accumulated_prefill_token_counts[bucket].keys()):
                    if step is None:
                        count_all_steps = list(self._accumulated_prefill_token_counts[bucket][engine_id].values())
                        for count in count_all_steps:
                            token_count[engine_id] += count
                    else:
                        count = self._accumulated_prefill_token_counts[bucket][engine_id][step]
                        token_count[engine_id] += count

        ret = {}
        for engine_id, count in token_count.items():
            ret[engine_id] = count / interval

        # clean up out dated ts buckets
        out_dated_ts = now - self._metrics_max_retain_seconds
        for bucket in ts_buckets:
            if bucket < out_dated_ts:
                self._accumulated_prefill_token_counts.pop(bucket, None)
            else:
                break

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

    def get_internal_metrics(self) -> dict:
        return self._internal_metrics.to_dict()


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
        stable_res = get_stable_res()
        rmrc = RequestManagerRegisterCenter.options(name='RequestManagerRegisterCenter',
                                                    resources=stable_res).remote(config)
        ray.get(rmrc.ready.remote())
        return rmrc

    def create(self, instance_name: str):
        resources = get_stable_res()
        # note(lixiang): concurrency必须超过global batch size才行，不然会卡住更新不了请求，导致死锁
        query_trace_config = self.config.streaming_rollout.query_trace
        request_manager = RequestManager.options(name=f'RequestManager/{instance_name}',
                                                 resources=resources,
                                                 max_concurrency=102400).remote(query_trace_config)
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

    def __init__(self, query_trace_config: DictConfig):
        self.req_pool = RequestPool()
        self.req_stat = RequestStatCollector()
        self.query_trace_config = query_trace_config
        self._step = 0
        self.actor_name = ray.get_runtime_context().get_actor_name()
        self._rm_name = self.actor_name.removeprefix('RequestManager/')
        self.query_tracer = QueryTracer(self.query_trace_config, self._rm_name)
        self._progress_bar = ProgressBar(self.actor_name)
        self._query_id_log = defaultdict(set)  # step -> set(query.id)

    def ready(self):
        print(f'RequestManager ready, {self.actor_name=}')
        return True

    async def put_new_query(self, query: Query) -> str:
        now = time.time()
        query.enqueue_time = now * 1e3  # first enqueue time
        step = query.meta_info.get('step', self._step)
        self.req_pool.put_new_requests([
            Request(
                request_id=query.id,
                query=query,
                global_step=step,
                last_pending_reschedule_ts=now * 1e3,
                updated_at=now,
            )
        ])
        self._query_id_log[step].add(query.id)
        return query.id

    async def wait_until_finished(self, query_id: str) -> Query:
        req = await self.req_pool.wait(query_id)
        self.req_stat.finish(self._rm_name, req)
        self.query_tracer.trace(req)
        return req.query

    def update_intermediate_queries(self, queries: List[Query | QueryUpdate], engine_id: str, wg_name: str, ts: float):
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
        self.query_tracer.trace_event(update_metrics)

    def _debug(self):
        return self.req_pool.requests, self.req_pool.finished_requests

    def get_next_pending_requests(self, batch_size: int, engine_id: str, wg_name: str) -> List[Query]:
        next_reqs = self.req_pool.get_next_pending_requests_with_cache(batch_size, engine_id, wg_name)
        return [r.query for r in next_reqs.values()]

    def get_next_pending_requests_with_cache(self, batch_size: int, engine_id: str, wg_name: str,
                                             cache_ids: Set[str]) -> List[Query]:
        next_reqs = self.req_pool.get_next_pending_requests_with_cache(batch_size, engine_id, wg_name, cache_ids)
        return [r.query for r in next_reqs.values()]

    # 获取全部request
    # 二阶段分发请求，先获取所有pending，再去跟所有engine匹配，
    # 匹配完了之后再提交匹配结果，未匹配到的则会滚到分配前的状态
    def peak_all_pending_requests(self) -> List[Request]:
        return self.req_pool.peak_all_pending_requests()

    def set_requests_assigned(self, query_ids: List[str], engine_id: str, wg_name: str, ts: float):
        # commit
        self.req_pool.set_requests_assigned(query_ids, engine_id, wg_name, ts)

    def clear_requests_assigned(self, query_ids: List[str]):
        # rollback
        self.req_pool.clear_requests_assigned(query_ids)

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
        self.query_tracer.set_global_step(global_step)

    def get_pending_size(self):
        return self.req_pool.get_pending_size()

    def get_size(self) -> Tuple[int, int]:
        total = len(self.req_pool)
        pending = self.req_pool.get_pending_size()
        return total, pending

    def get_estimated_throughput(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        # 返回{engine_id -> 最近interval内的平均throughput}
        return self.req_pool.get_prefill_throughput(), self.req_pool.get_decode_throughput()

    def get_concurrency(self) -> Dict[str, int]:
        return self.req_pool.get_concurrency()

    def dump_request_trace(self, after_ts: float = 0.) -> List[dict]:
        running_reqs = self.req_pool.get_running_requests()
        extra_events = []
        for r in running_reqs:
            cce = self.query_tracer.trace(r, persist=False)
            extra_events.append(cce)
        return self.query_tracer.dump_request_trace(extra_events, after_ts)

    def get_step_metrics(self, step) -> Dict[str, float]:
        internal_metrics = self.req_pool.get_internal_metrics()
        step_metrics = self.req_stat.get_step_metrics(step)
        step_metrics.update(internal_metrics)
        return step_metrics

    ## query_tool util function ##

    def get_inflight_query_digest(self, step: Optional[int] = None) -> List[RequestDigest]:
        ret = []
        query_ids = list(self.req_pool.requests.keys())
        for query_id in query_ids:
            req = self.req_pool.requests.get(query_id)
            if step is not None and req.global_step != step:
                continue
            reg_digest = RequestDigest(
                query_id=query_id,
                pool_name=self._rm_name,
                assigned=req.assigned,
                assigned_engine_id=req.assigned_engine_id,
                assigned_engine_name=req.assigned_engine_name,
                global_step=req.global_step,
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

            decode_throughput = self.req_pool.get_decode_throughput(step=step)
            total_decode_tps = sum(decode_throughput.values())
            prefill_throughput = self.req_pool.get_prefill_throughput(step=step)
            total_prefill_tps = sum(prefill_throughput.values())

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
                if req.assigned and req.assigned_engine_id is not None:
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
                        pool_name=self._rm_name,
                        step=step,
                        total=step_total,
                        finished=step_finished,
                        token_throughput=total_decode_tps,
                        prefill_throughput=total_prefill_tps,
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


def get_all_request_manager_actors_with_names() -> List[RequestManager]:
    rmrc: RequestManagerRegisterCenter = ray.get_actor('RequestManagerRegisterCenter')  # noqa
    names = ray.get(rmrc.get_all_names.remote())
    rms = []
    for n in names:
        rms.append((n, RequestManagerRegisterCenter.get(n)))
    return rms
