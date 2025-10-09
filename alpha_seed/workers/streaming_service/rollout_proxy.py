import asyncio
import inspect
import os
import random
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Union, Tuple, Set, Deque

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from ray import ObjectRef
from ray.exceptions import ActorDiedError, GetTimeoutError, RayActorError, RayTaskError, ActorUnavailableError

from alpha_seed.utils.profile.timeline import Tracer, CompleteEvent, CounterEvent
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from alpha_seed.workers.streaming_service.rollout_request import Request
from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManager, RequestManagerRegisterCenter
from alpha_seed.workers.streaming_service.streaming_rollout import RemoteAsyncXPerfGPTRollout
from alpha_seed.workers.xperf_rollout.component.query import Query
from alpha_seed.workers.xperf_rollout.utils.base_weights_communicator import WeightsRankInfo
from mono_rl.single_controller.ray import RayWorkerGroup
from mono_rl.single_controller.ray.base import func_generator
from mono_rl.single_controller.ray.replicated_worker_group import ReplicatedRayWorkerGroup, ScalingRayWorkerGroup
from mono_rl import DataProto

from alpha_seed.workers.streaming_service.auto_scaling import MetricSource, TimeSeriesMetrics
from alpha_seed.workers.xperf_rollout.session import LoadMetric


class NoAvailableWorker(RuntimeError):
    pass


@dataclass
class LoadSkewness:
    mean: float  # kv mean
    p90: float  # kv p90
    p25: float  # kv p25
    cov: float  # kv Coefficient of Variation
    running_avg: float  # 平均每个engine运行数


class TraceMetrics:

    def get_title(self) -> str:
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, int | float]:
        raise NotImplementedError()


class EngineTraceMetrics(TraceMetrics):

    def get_wg_name(self) -> str:
        raise NotImplementedError()


@dataclass
class ProxyLoopCost(TraceMetrics):
    total: float = 0.  # 每个loop的耗时
    get_engine_info: float = 0.  # 读取一轮全部engine的load信息等 (从最开始)
    matching: float = 0.  # 读取所有pending queries并做engine matching (从最开始)
    staging: float = 0.  # 将matching结果拟分发到engine，但没有执行真的commit (从最开始)

    def get_title(self) -> str:
        return 'loop cost'

    def to_dict(self) -> Dict[str, int | float]:
        return {
            'total': self.total,
            'engine_info': self.get_engine_info,
            'matching': self.matching,
            'staging': self.staging,
        }


@dataclass
class InternalDiagnosisMetrics(TraceMetrics):
    num_target_replicas: int
    num_ready_replicas: int
    num_initialized_replicas: int
    num_alive_replicas: int
    gmem_insufficient_count: int
    gmem_high_water_level_count: int
    total_standby_wgs: int  # 还有多少wg本轮可以接收请求
    total_overload_num_slots: int  # 多少请求在wg上溢出要分发给别的wg
    total_available_num_slots: int  # 本轮可接收请求的wg总共能接收多少
    total_rebalanced: int  # 本轮发生重平衡的请求多少个
    max_concurrency: int  # 本轮计算得到每个wg最大接受多少并发
    dispatch_delay_acc: float  # query在request pool里到分发出去那一刻总共等待的时间，时间越长表示proxy分发能力越弱
    enqueue_delay_acc: float  # query从创建到进入pool里产生的delay的累积

    def get_title(self) -> str:
        return "internal"

    def to_dict(self):
        # $前缀表示requests/query数量
        # #前缀表示replica数量
        return {
            '#target': self.num_target_replicas,
            '#ready': self.num_ready_replicas,
            '#initialized': self.num_initialized_replicas,
            '#alive': self.num_alive_replicas,
            '#gmem full': self.gmem_insufficient_count,
            '#gmem high': self.gmem_high_water_level_count,
            '#standby': self.total_standby_wgs,
            '$overload': self.total_overload_num_slots,
            '$available': self.total_available_num_slots,
            '$rebalanced': self.total_rebalanced,
            '$max_concurrency': self.max_concurrency,
            'dispatch_delay': self.dispatch_delay_acc,
            'enqueue_delay': self.enqueue_delay_acc,
        }


@dataclass
class InternalShedMetrics(TraceMetrics):
    cache_match: int = 0  # 分配kv匹配的
    steal: int = 0  # 分配kv不匹配的(从别的engine偷过来的)
    standalone: int = 0  # 分配无任何kv cache的 (全新query或kv cache太久已被evict)
    history_count: int = 0  # pending中有多少个匹配了engine cache的
    standalone_count: int = 0  # pending中有多少个是不匹配任何cache的

    def get_title(self) -> str:
        return "shed"

    def to_dict(self):
        return {
            '$cache_match': self.cache_match,
            '$steal': self.steal,
            '$no_cache': self.standalone,
            '$history_pending': self.history_count,
            '$standalone_pending': self.standalone_count,
        }


@dataclass
class EngineDispatchMetrics(EngineTraceMetrics):
    # 本轮分发给某个engine的状况
    history_remain_pending: int = 0  # 属于这个engine但没有分过来的数量
    steal: int = 0  # 不属于这个engine但属于别的engine拿出来的数量

    def __init__(self, wg_name: str, history_remain_pending: int = 0, steal: int = 0):
        self.wg_name = wg_name
        self.history_remain_pending = history_remain_pending
        self.steal = steal

    def get_title(self) -> str:
        return "dispatch"

    def get_wg_name(self) -> str:
        return self.wg_name

    def to_dict(self):
        return {
            '$history_remain_pending': self.history_remain_pending,
            '$steal': self.steal,
        }


class _MetricSourceImpl(MetricSource):

    def __init__(self, request_manager: RequestManager):
        self.req_mgr = request_manager
        self.concurrency_ts = []  # [(ts, concurrency), ...]

    def get_recent_time_series_metrics(self, recent_seconds: float) -> TimeSeriesMetrics:
        now = time.time()
        concurrency = ray.get(self.req_mgr.get_concurrency.remote())
        concurrency_values = concurrency.values()
        if not concurrency_values:
            return TimeSeriesMetrics([], [])
        min_con = min(concurrency_values)
        max_con = max(concurrency_values)
        total_con = sum(concurrency_values)
        print(f'[{time.ctime()}] get recent concurrency min={min_con} max={max_con} total={total_con}')
        self.concurrency_ts.append((now, total_con))
        tss = []
        metrics = []
        oldest_idx = 0
        for idx, (ts, total_con) in enumerate(reversed(self.concurrency_ts)):
            if ts > now - recent_seconds:
                tss.append(ts)
                metrics.append(total_con)
            else:
                # out of date
                oldest_idx = idx + 1
                break
        # 取最后的N个，扔掉前面过期的指标
        self.concurrency_ts = self.concurrency_ts[-oldest_idx:]
        return TimeSeriesMetrics(
            timestamps=list(reversed(tss)),
            metrics=list(reversed(metrics)),
        )


def split_by_indices(big_worker_group: RayWorkerGroup, indices: List[List[int]],
                     original_class_name) -> List['RayWorkerGroup']:
    # indices: [[0, 1], [2, 3], ...] 外层是切的worker groups数，内层是每个worker_group取第几个index
    rollout_cls = big_worker_group.ray_cls_with_init.cls.raw_cls_dict[original_class_name]
    worker_groups = []
    for dp_rank, worker_indices in enumerate(indices):
        worker_names = [big_worker_group._worker_names[i] for i in worker_indices]
        name_prefix = f'{big_worker_group.name_prefix}_dp{dp_rank}'
        new_wg = RayWorkerGroup.from_detached(name_prefix=name_prefix,
                                              worker_names=worker_names,
                                              ray_cls_with_init=big_worker_group.ray_cls_with_init)
        # 参考RayWorkerGroup.spawn，重新给detached worker bind回Worker的方法
        new_wg._bind_worker_method(rollout_cls, func_generator)
        new_wg.sub_cls_name = big_worker_group.sub_cls_name
        worker_groups.append(new_wg)
    return worker_groups


# 将一个大的world拆成N个小world，每个world只含1个DP group
class FixedReplicatedRayWorkerGroupAdapter(ReplicatedRayWorkerGroup):

    def __init__(self, wg_with_dp: RayWorkerGroup, tp_size: int, original_class_name: str):
        # 不支持scale，initializer传None
        # original_class_name是fuse之前的class name, 见RayPPOTrainer.resource_pool_to_cls的定义
        super().__init__(None, wg_with_dp.resource_pool)  # noqa
        self.tp_size = tp_size
        assert wg_with_dp.world_size % tp_size == 0, \
            f"world_size({wg_with_dp.world_size}) must be divisible by tp_size({tp_size})"
        # assume only dp and tp
        dp_size = wg_with_dp.world_size // tp_size
        world = list(range(wg_with_dp.world_size))
        split_indices = []
        for dp_rank in range(dp_size):
            split_indices.append(world[:tp_size])
            world = world[tp_size:]
        split_wgs = split_by_indices(wg_with_dp, split_indices, original_class_name)
        self.wgs = {uuid.uuid4().hex: wg for wg in split_wgs}
        self.alive_worker_group_ids = set(self.wgs.keys())
        self.ready_worker_group_ids = self.alive_worker_group_ids
        self.initialized_worker_group_ids = self.alive_worker_group_ids

    @property
    def guaranteed(self):
        return self

    def set_dead_callback(self, fn):
        pass

    def get_alive_worker_groups(self):
        return self.wgs

    def get_initialized_worker_groups(self):
        return self.wgs

    def get_ready_worker_groups(self):
        return self.wgs

    @property
    def target_num_replicas(self) -> int:
        return len(self.wgs)


class CombinedRayWorkerGroupAdapter(ReplicatedRayWorkerGroup):

    ScalingOrReplicated = ReplicatedRayWorkerGroup | ScalingRayWorkerGroup

    # noqa: no initializing base class, use as interface only
    def __init__(self, intermittent: Dict[str, ScalingOrReplicated], persistent: Dict[str, ScalingOrReplicated]):
        # intermittent: 时间上时分复用，有时候可用有时候不可用，不可用期间不会访问到对应的方法，由_replica_active决定
        # persistent: 时间上持续存在，无论何时都可用
        assert all(
            [isinstance(val, (ReplicatedRayWorkerGroup, ScalingRayWorkerGroup)) for val in intermittent.values()])
        self._intermittent_replicas = intermittent  # {name -> }
        self._persistent_replicas = persistent  # {name -> }
        self._replica_active = {
            name: True for name in self._intermittent_replicas.keys() | self._persistent_replicas.keys()
        }

    @property
    def replicas(self):
        return {
            **self._intermittent_replicas,
            **self._persistent_replicas,
        }

    @property
    def guaranteed(self):
        persistent_keys = list(self._persistent_replicas.keys())
        assert len(persistent_keys) > 0, f"should register at least 1 persistent replica when calling {self}.guaranteed"
        any_persistent_key = persistent_keys[0]
        return self._persistent_replicas[any_persistent_key].guaranteed

    def set_dead_callback(self, fn):
        for replica in self.replicas.values():
            replica.set_dead_callback(fn)

    def get_alive_worker_groups(self):
        ret = {}
        for replica in self.replicas.values():
            ret.update(replica.get_alive_worker_groups())
        return ret

    def get_initialized_worker_groups(self):
        ret = {}
        for replica in self.replicas.values():
            ret.update(replica.get_initialized_worker_groups())
        return ret

    def get_ready_worker_groups(self):
        ret = {}
        for name, replica in self.replicas.items():
            if self._replica_active[name]:
                ret.update(replica.get_ready_worker_groups())
        return ret

    @property
    def target_num_replicas(self) -> int:
        return sum(
            [replica.target_num_replicas for name, replica in self.replicas.items() if self._replica_active[name]])

    def set_replica_ready_state(self, name: str, ready: bool):
        assert name in self._intermittent_replicas, \
            f"name({name}) should be in intermittent replicas({self._intermittent_replicas.keys()})"
        self._replica_active[name] = ready

    @property
    def alive_worker_group_ids(self) -> Set[str]:
        return set(self.get_alive_worker_groups().keys())

    @property
    def initialized_worker_group_ids(self) -> Set[str]:
        return set(self.get_initialized_worker_groups().keys())

    @property
    def ready_worker_group_ids(self) -> Set[str]:
        return set(self.get_ready_worker_groups().keys())


class StandaloneRolloutWGAdapter:
    """Provide the same api as standalone rollout worker group"""

    def __init__(self, replicas: ScalingRayWorkerGroup):
        self.replicas = replicas
        self.worker_group_unit_size = self.replicas.resource_unit.world_size
        self._tracer = Tracer.get_instance()
        self._update_worker_start_ts = 0  # 调用update worker开始的时刻
        self._stop_server_ts = 0
        self._update_worker_num = 0  # 调用update worker时的dp数
        self._update_worker_success_num = 0  # 调用update worker结束时成功的dp数(有些elastic在途中死掉)
        self._stop_server_num = 0
        self._stop_server_success_num = 0
        self._restart_server_num = 0
        self._restart_server_success_num = 0  # (同上)

    def get_master_addr(self) -> List[DataProto]:
        # 获取每个workergroup的每个rank的address
        ret = []
        key = "standalone_master_addr"
        for wg in self.replicas.get_alive_worker_groups().values():
            out = DataProto.from_dict(tensors={'mock': torch.tensor([[0]])}, meta_info={key: wg.master_address})
            ret.extend([out] * wg.world_size)  # 先保证外部调用接口一致，给每个rank都返回一个master_addr，即使他们应该是相同的
        return ret

    def get_master_free_port(self) -> int:
        for wg in self.replicas.get_alive_worker_groups().values():
            # 返回任意一个即可
            return wg.get_master_free_port()
        assert False, "should have at least 1 alive worker group"

    def update_standalone_worker(self, role) -> List[ObjectRef]:
        # update转发给所有initialized的worker，不用管其是否ready，一开始肯定不ready，需要update weights后才会ready
        initialized_wgs = self.replicas.get_initialized_worker_groups().values()
        self._update_worker_start_ts = time.time() * 1e6
        self._update_worker_num = len(initialized_wgs) * self.worker_group_unit_size
        futs = []
        for wg in initialized_wgs:
            wg: Union[RayWorkerGroup, RemoteAsyncXPerfGPTRollout, AsyncActorRolloutRefWorker]
            refs = wg.update_standalone_worker(role)
            futs.append((wg, refs))
        num_success = self.wait_ignore_actor_died(futs)
        self._update_worker_success_num = num_success
        # 返回一个占位符即可
        return [ray.put(None)]

    def update_standalone_worker_end(self):
        # 通知server侧结束参数拉取
        # 任意一个client通知即可，调用此方法时需要确保所有standalone rollout worker已经同步完参数
        no_available_worker_retry = 0
        max_no_available_worker_retry = 15
        while True:
            try:
                # 注意这里必须用guaranteed去广播，因为其他best effort可能在广播过程中死掉，导致caller task runner那边 hang
                all_initialized_workers = list(self.replicas.guaranteed.get_initialized_worker_groups().values())
                if len(all_initialized_workers) == 0:
                    raise NoAvailableWorker("no initialized workers during weights update or all actors died")
                # any worker group sending will be ok
                wg = random.choice(all_initialized_workers)
                wg.update_standalone_worker_end()
            except ActorDiedError:
                # do nothing when actor dies unfortunately, try next run
                time.sleep(0.1)
            except NoAvailableWorker:
                # underlying workers are still in liveness/readiness probe gap, wait for a bit more seconds
                no_available_worker_retry += 1
                if no_available_worker_retry > max_no_available_worker_retry:
                    raise
                time.sleep(1)
                continue
            else:
                break
        evt = CompleteEvent(
            pid='RolloutProxy',
            tid='update',
            cat='update weights',
            name='update weights',
            ts=self._update_worker_start_ts,
            dur=time.time() * 1e6 - self._update_worker_start_ts,
            args={
                "worker_num": self._update_worker_num,
                "worker_success_num": self._update_worker_success_num,
            },
        )
        self._tracer.trace(evt)

    def stop_server_before_weights_update(self):
        initialized_wgs = self.replicas.get_initialized_worker_groups().values()
        self._stop_server_ts = time.time() * 1e6
        self._stop_server_num = len(initialized_wgs) * self.worker_group_unit_size
        futs = []
        for wg in initialized_wgs:
            wg: RemoteAsyncXPerfGPTRollout
            ref = wg.stop_server_before_weights_update_non_blocking()
            futs.append((wg, ref))
        num_success = self.wait_ignore_actor_died(futs)
        self._stop_server_success_num = num_success

    def restart_server_after_weights_update(self):
        initialized_wgs = self.replicas.get_initialized_worker_groups().values()
        self._restart_server_num = len(initialized_wgs) * self.worker_group_unit_size
        futs = []
        for wg in initialized_wgs:
            wg: RemoteAsyncXPerfGPTRollout
            ref = wg.restart_server_after_weights_update_non_blocking()
            futs.append((wg, ref))
        num_success = self.wait_ignore_actor_died(futs)
        self._restart_server_success_num = num_success

        evt = CompleteEvent(
            pid='RolloutProxy',
            tid='update',
            cat='stop/start server',
            name='stop/start server',
            ts=self._stop_server_ts,
            dur=time.time() * 1e6 - self._stop_server_ts,
            args={
                "stop_server_num": self._stop_server_num,
                "stop_server_success_num": self._stop_server_success_num,
                "restart_server_num": self._restart_server_num,
                "restart_server_success_num": self._restart_server_success_num,
            },
        )
        self._tracer.trace(evt)

    def wait_ignore_actor_died(self, refs: List[Tuple[RayWorkerGroup, List[ray.ObjectRef]]]) -> int:
        obj_wg_map = {}
        remaining = set()
        ready = set()
        for (wg, ref_list) in refs:
            for obj in ref_list:
                obj_wg_map[obj] = wg
                remaining.add(obj)
        total_count = len(remaining)  # noqa: py-spy

        while remaining:
            done, not_done = ray.wait(list(remaining), num_returns=len(remaining), timeout=1.0)
            for obj in done:
                try:
                    ray.get(obj)
                    ready.add(obj)
                except ActorDiedError as e:
                    wg = obj_wg_map[obj]
                    caller = inspect.stack()[1].frame.f_code.co_name
                    print(f"[{time.ctime()}] actor({e.actor_id}) of wg({wg.group_name}) died at function({caller}). "
                          f"ignore this as this is expected.")
                except Exception as e:
                    # for other exceptions, carefully check whether it's caused by actor recycling by auto-scaling
                    # if the wg is scheduled to destroy, ignore all errors on it
                    wg = obj_wg_map[obj]
                    if not wg.is_destroying:
                        raise
                finally:
                    remaining.discard(obj)

            not_ready_wgs = set()
            for obj in remaining:
                not_ready_wgs.add(obj_wg_map[obj])
            not_ready_wg_names_list = [wg.worker_names for wg in not_ready_wgs]  # noqa: py-spy
            not_ready_count = len(not_ready_wg_names_list)  # noqa: py-spy

            # Optional: avoid tight loop
            time.sleep(0.1)
        return len(ready)


class DebounceAccumulatedLogger:

    def __init__(self, log_interval_seconds=15., accumulated_type=int):
        self.log_interval_seconds = log_interval_seconds
        self.wg_latest_log_ts = defaultdict(time.time)  # wg_name -> ts
        self.wg_accumulate_value = defaultdict(accumulated_type)  # wg_name -> val
        self.zero_val = accumulated_type()

    def log(self, wg_name: str, val: int | float, fmt: str):
        last_ts = self.wg_latest_log_ts[wg_name]
        self.wg_accumulate_value[wg_name] += val
        now = time.time()
        if now - last_ts >= self.log_interval_seconds:
            self.wg_latest_log_ts[wg_name] = now
            acc = self.wg_accumulate_value[wg_name]
            self.wg_accumulate_value[wg_name] = self.zero_val
            content = fmt.format(accumulated_value=acc)
            print(f"[{time.ctime()}] {content}")


@dataclass
class StatisticalMetric:
    minimum: float
    maximum: float
    mean: float
    sum: float


zero_stats = StatisticalMetric(0, 0, 0, 0)


class ProxyMetricsLogger:

    def __init__(self):
        self.metrics = defaultdict(list)  # name -> val
        self._last_step_metrics = {}
        self.global_step = 0

    def log(self, kv: dict):
        for k, v in kv.items():
            self.metrics[k].append(v)

    def step(self, global_step: int):
        # go to next step
        self.global_step = global_step
        self._last_step_metrics = self.metrics
        self.metrics = defaultdict(list)

    def get_last_step_metrics(self) -> Dict[str, StatisticalMetric]:
        ret = {}
        for k, vl in self._last_step_metrics.items():
            if len(vl) == 0:
                continue
            minimum = min(vl)
            sum_ = sum(vl)
            mean = sum_ / len(vl)
            maximum = max(vl)
            ret[k] = StatisticalMetric(minimum, maximum, mean, sum_)
        return ret


class RolloutWorkerGroupProxy(_MetricSourceImpl):
    """
    负责代理底下N个replicas的请求分发、负载平衡
    负责请求状态同步到request pool
    """

    def __init__(self, replicas: Union[ReplicatedRayWorkerGroup, ScalingRayWorkerGroup],
                 actor_info: List[WeightsRankInfo], request_manager_name: str, config: DictConfig):
        self.request_manager: RequestManager = RequestManagerRegisterCenter.get(request_manager_name)  # noqa
        super().__init__(self.request_manager)
        self.replicas = replicas
        self.actor_info = actor_info  # hybrid rollout actor info
        self.config = config  # .streaming_rollout
        self._tracer: Optional[Tracer] = None  # dispatch loop thread专用tracer
        self._stop_server_ts = 0
        self._update_worker_start_ts = 0
        self._request_manager_name = request_manager_name
        self.poll_interval = self.config.proxy.poll_internal_seconds
        self._progress_logger = DebounceAccumulatedLogger()
        self._metrics_logger = ProxyMetricsLogger()
        self._wg_stack_trace_logger = open("/tmp/wg_stack_trace.log", "a")

        self.replicas.set_dead_callback(self._worker_group_dead_callback)
        self._loop_should_stop = threading.Event()
        self._loop_should_continue = threading.Event()
        self._loop_should_continue.set()
        self.is_waiting = False
        self._dispatch_loop_thread = threading.Thread(target=self._run_dispatch_loop,
                                                      name=f'{request_manager_name}-rollout-wg-proxy-dispatch-loop',
                                                      daemon=True)
        self._dispatch_loop_thread.start()
        self._update_loop_thread = threading.Thread(target=self._update_loop,
                                                    name=f'{request_manager_name}-rollout-wg-proxy-update-loop',
                                                    daemon=True)
        self._update_loop_thread.start()

    def _run_dispatch_loop(self):
        self._tracer = Tracer.get_instance(retention_hours=self.config.query_trace.retention_hours)
        asyncio.run(self._dispatch_loop())

    def _update_loop(self):
        print(f'start background update loop for {self._request_manager_name}')
        sleep_interval = self.poll_interval * 2

        while True:
            if self._loop_should_stop.is_set():
                break
            self.update_is_waiting = True
            self._loop_should_continue.wait()
            self.update_is_waiting = False
            time.sleep(sleep_interval)

            t0 = time.time()
            ready_wg_items = list(self.replicas.get_ready_worker_groups().items())

            for engine_id, wg in ready_wg_items:
                wg: RayWorkerGroup | RemoteAsyncXPerfGPTRollout | AsyncActorRolloutRefWorker
                try:
                    wg.update_queries(self._request_manager_name, engine_id, wg.group_name)
                except (ActorDiedError, RayTaskError) as e:
                    self._finalize(wg, e)

            loop_cost = time.time() - t0  # noqa: py-spy

    @property
    def world_size(self):
        # 按照设定的replicas数计算world_size，而不管是否已完成初始化
        alive_worker_groups: Dict[str, RayWorkerGroup] = self.replicas.get_alive_worker_groups()
        return sum(w.world_size for w in alive_worker_groups.values())

    def _worker_group_dead_callback(self, worker_group_ids: List[str]):
        # worker group任意死了之后，通知request manager将运行中的请求释放掉
        ready_worker_group_ids = self.replicas.ready_worker_group_ids
        ray.get(self.request_manager.handle_stale_requests.remote(ready_worker_group_ids))

    async def _dispatch_loop(self):
        print(f'start background dispatch loop for {self._request_manager_name}')

        sleep_interval = self.poll_interval
        while True:
            if self._loop_should_stop.is_set():
                break
            self.is_waiting = True
            self._loop_should_continue.wait()
            self.is_waiting = False
            await asyncio.sleep(sleep_interval)

            # 按照总量平分给每个ready replica，均匀分发
            # 注意一开始可能还没有request进去request pool
            # 也可能replicas还没ready
            total, pending_size = ray.get(self.request_manager.get_size.remote())
            num_ready_replicas = len(self.replicas.ready_worker_group_ids)
            max_concurrency = total // max(1, num_ready_replicas)  # replicas可能还没ready
            max_concurrency = min(max(max_concurrency, 1), 512)  # 限制在1-512范围内

            # 纪录负载指标
            loads = {}  # (engine_id, wg_name) -> LoadMetric

            # 只将请求dispatch给ready worker group，每次循环都是最新的ready状态
            # 在dispatch过程中，worker group死了也没关系，这个request会之后被标记为stale
            t0 = time.time()
            ready_wg_items = list(self.replicas.get_ready_worker_groups().items())

            wg_history_map: Dict[str, Set[str]] = {}  # engine_id -> cache_ids
            wg_queries: Dict[str, List[Query]] = {}  # engine_id -> 记录分给engine的queries
            for engine_id, wg in ready_wg_items:
                try:
                    history_ids = wg.get_history_ids()
                    wg_history_map[engine_id] = set(history_ids) if history_ids else set()
                except (ActorDiedError, RayTaskError) as e:
                    self._finalize(wg, e)

            # 第一轮先按照kvcache亲和性分发
            for engine_id, wg in ready_wg_items:
                wg: Union[RayWorkerGroup, RemoteAsyncXPerfGPTRollout, AsyncActorRolloutRefWorker]  # type anno
                wg_name = wg.group_name
                try:
                    # send new request to worker group (engine)
                    load: LoadMetric = wg.get_load_metrics()
                    gmem_insufficient = load.kv_cache_util > self.config.proxy.gmem_insufficient_threshold

                    short = max_concurrency - load.num_prefilling - load.num_decoding - load.num_pending - load.num_waiting
                    # 如果gmem不够了就不发了
                    if short > 0 and not gmem_insufficient:
                        queries: List[Query] = ray.get(
                            self.request_manager.get_next_pending_requests_with_cache.remote(
                                short, engine_id, wg_name, wg_history_map[engine_id]))
                        if len(queries) > 0:
                            wg_queries[engine_id] = queries

                    loads[(engine_id, wg_name)] = load

                except (ActorDiedError, RayTaskError) as e:
                    self._finalize(wg, e)

            # 如果engine还有空闲则再分其他的一些
            for engine_id, wg in ready_wg_items:
                wg: Union[RayWorkerGroup, RemoteAsyncXPerfGPTRollout, AsyncActorRolloutRefWorker]  # type anno
                wg_name = wg.group_name
                try:
                    load = loads[(engine_id, wg_name)]
                    gmem_insufficient = load.kv_cache_util > 0.9

                    short = max_concurrency - load.num_prefilling - load.num_decoding - load.num_pending - load.num_waiting - len(
                        wg_queries.get(engine_id, []))
                    queries = wg_queries.get(engine_id, [])
                    # 如果gmem不够了就不发了
                    if short > 0 and not gmem_insufficient:
                        queries.extend(
                            ray.get(self.request_manager.get_next_pending_requests.remote(short, engine_id, wg_name)))
                    if len(queries) > 0:
                        for q in queries:
                            # query的分类，区分一下是hybrid_rollout/standalone_rollout/validation，避免共用engine时不知道怎么update回去对应的来源
                            q.meta_info['query_type'] = self._request_manager_name
                        wg.add_inflight_queries(queries)
                        pending_size -= len(queries)
                        fmt = (
                            "dispatch {accumulated_value} "
                            f"(remain={pending_size}) queries from({self._request_manager_name}) to wg({wg_name}, "
                            f"pending={load.num_pending}, W={load.num_waiting}/P={load.num_prefilling}/D={load.num_decoding}, "
                            f"kv={load.kv_cache_util:.2f})")
                        self._progress_logger.log(wg_name, len(queries), fmt)

                except (ActorDiedError, RayTaskError) as e:
                    self._finalize(wg, e)

            total, pending_size = ray.get(self.request_manager.get_size.remote())
            load_skewness = self._calc_load_skewness(loads)
            prefill_throughput, decode_throughput = ray.get(self.request_manager.get_estimated_throughput.remote())
            self._trace_load_metrics(loads, load_skewness, prefill_throughput, decode_throughput, total, pending_size)

            loop_cost = time.time() - t0  # noqa: for py-spy
            sleep_interval = max(0., self.poll_interval - loop_cost)

    def stop(self):
        self._loop_should_stop.set()
        self._dispatch_loop_thread.join()
        self._update_loop_thread.join()

    def pause_loop(self):
        self._loop_should_continue.clear()
        while not self.is_waiting or not self.update_is_waiting:
            time.sleep(0.5)

    def continue_loop(self):
        self._loop_should_continue.set()

    def step(self, global_step):
        self._metrics_logger.step(global_step)

    def get_step_metrics(self) -> dict:
        step = self._metrics_logger.global_step
        metrics = ray.get(self.request_manager.get_step_metrics.remote(step))
        return metrics

    def _finalize(self, wg: RayWorkerGroup, e: Exception):
        # proactively kill the actor who causes any RayTaskErrors, RayTaskErrors or so on..
        # let underlying replicated worker group to handle the ready/alive of worker group
        #   in elastic scenario, there won't be any impact to task runner's proxy thread
        #   in static server scenario, any actor deadness could cause the other rpc call failure from
        #   main thread
        if not isinstance(e, ActorDiedError):
            # ignore the stack of ActorDiedError, no useful information
            tb = traceback.format_exc()
            self._wg_stack_trace_logger.write(f"[{time.ctime()}] {tb}\n")
            print(f'[{time.ctime()}] show error of remote worker group only, {wg=}, {e}. '
                  f'for stack trace refers to {self._wg_stack_trace_logger.name}')
        try:
            wg.destroy()
        except Exception:
            pass
        if not self.config.elastic.enable:
            # teardown the task_runner process in static server mode
            print('due to some worker group failed to execute tasks in non-elastic mode, '
                  'will teardown to process to propagate errors in time')
            os._exit(11)  # noqa

    def _trace_load_metrics(self, loads: Dict[Tuple[str, str], LoadMetric], load_skewness: LoadSkewness,
                            prefill_throughput: Dict[str, float], decode_throughput: Dict[str, float], total: int,
                            global_pending: int):
        total_waiting_num = 0
        total_prefilling_num = 0
        total_decoding_num = 0
        kv_cache_util_min = 1
        kv_cache_util_max = 0
        for (wg_id, wg_name), metric in loads.items():
            total_waiting_num += metric.num_waiting
            total_prefilling_num += metric.num_prefilling
            total_decoding_num += metric.num_decoding
            prefill_tps = prefill_throughput.get(wg_id) or 0
            decode_tps = decode_throughput.get(wg_id) or 0
            kv_cache_util_min = min(kv_cache_util_min, metric.kv_cache_util)
            kv_cache_util_max = max(kv_cache_util_max, metric.kv_cache_util)
            evt = CounterEvent(
                name='load:',
                pid=f'{self._request_manager_name} {wg_name}',
                ts=metric.ts * 1e6,
                data={
                    'kv cache%': metric.kv_cache_util,
                    'kv swap%': metric.kv_prefix_cache_swap_util,
                    'prefilling': metric.num_prefilling,
                    'decoding': metric.num_decoding,
                    'pending': metric.num_pending,
                    'waiting': metric.num_waiting,
                    '$no_caching': len(metric.no_caching_query_ids),
                    'deviate': metric.num_running - load_skewness.running_avg,
                    'prefill TPS': prefill_tps,
                    'decode TPS': decode_tps,
                },
            )
            self._tracer.trace(evt)

        # global pending running event
        evt = CounterEvent(
            name='request:',
            pid=f'RequestManager/{self._request_manager_name}',  # 不区分hybrid/standalone
            ts=time.time() * 1e6,
            data={
                '$waiting': total_waiting_num,
                '$prefilling': total_prefilling_num,
                '$decoding': total_decoding_num,
                '$running(avg)': load_skewness.running_avg,
                '$processing': total,
                '$pending dispatch': global_pending,
                '$prefill TPS': sum(prefill_throughput.values()),
                '$decode TPS': sum(decode_throughput.values()),
                'kv cache%(avg)': load_skewness.mean,
                'kv cache%(min)': kv_cache_util_min,
                'kv cache%(p90)': load_skewness.p90,
                'kv cache%(p25)': load_skewness.p25,
                'kv cache%(CoV)': load_skewness.cov,
            },
        )
        self._tracer.trace(evt)

    def _trace_metrics(self, metrics: TraceMetrics):
        evt = CounterEvent(
            name=f'{metrics.get_title()}:',
            pid=f'RequestManager/{self._request_manager_name}',  # 不区分hybrid/standalone
            ts=time.time() * 1e6,
            data=metrics.to_dict(),
        )
        self._tracer.trace(evt)

    def _trace_engine_metrics(self, metrics: EngineTraceMetrics):
        evt = CounterEvent(
            name=f'{metrics.get_title()}:',
            pid=f'{self._request_manager_name} {metrics.get_wg_name()}',
            ts=time.time() * 1e6,
            data=metrics.to_dict(),
        )
        self._tracer.trace(evt)

    def _calc_load_skewness(self, loads: Dict[Tuple[str, str], LoadMetric]) -> LoadSkewness:
        kv_cache_util_lst = [v.kv_cache_util for _, v in loads.items()] or [0]
        mean = float(np.mean(kv_cache_util_lst))
        cov = float(np.std(kv_cache_util_lst)) / mean if mean > 0 else 0
        p90 = float(np.percentile(kv_cache_util_lst, 90))
        p25 = float(np.percentile(kv_cache_util_lst, 25))
        running_total = sum(m.num_running for m in loads.values())
        running_avg = 0
        if loads:
            running_avg = running_total / len(loads)
        return LoadSkewness(mean, p90, p25, cov, running_avg)


class BalancedRolloutWorkerGroupProxy(RolloutWorkerGroupProxy):

    def __init__(self, replicas: Union[ReplicatedRayWorkerGroup, ScalingRayWorkerGroup],
                 actor_info: List[WeightsRankInfo], request_manager_name: str, config: DictConfig):
        super().__init__(replicas, actor_info, request_manager_name, config)
        self._rebalance_threshold = self.config.proxy.rebalance_threshold
        self.abort_logger = DebounceAccumulatedLogger()

    async def _dispatch_loop(self):
        """
        每个loop内自动均衡每个worker group正在跑的query，
        如某些worker group跑得比较快，会自动从别的worker group匀过来一些，
        如有新的query进来，会自动按照worker当前数量均分
        """
        print(f'start background dispatch loop with balanced mode for {self._request_manager_name}')

        loop_start_ts_list = []  # 记录每个loop开始的时间
        sleep_interval = self.poll_interval
        # 记录上一个loop的ready wg，两次loop之间如果有ready engine变inactive，
        # 不会检测到有engine dead(inactive)，则可能导致query hang
        ready_wg_ids_prev = set()
        while True:
            if self._loop_should_stop.is_set():
                break
            loop_start_ts_list.append(time.time())
            self.is_waiting = True
            self._loop_should_continue.wait()
            self.is_waiting = False
            await asyncio.sleep(sleep_interval)

            # 按照总量平分给每个ready replica，均匀分发
            # 注意一开始可能还没有request进去request pool
            # 也可能replicas还没ready
            total, pending_size = ray.get(self.request_manager.get_size.remote())
            num_ready_replicas = len(self.replicas.ready_worker_group_ids)
            max_concurrency = total // max(1, num_ready_replicas)  # replicas可能还没ready
            max_concurrency = min(max(max_concurrency, 1), 512)  # 限制在1-512范围内

            # 记录负载指标
            loads = {}  # (engine_id, wg_name) -> LoadMetric
            engine_concurrency_cap = defaultdict(int)  # engine_id -> 最大可并发数

            # internal metrics
            total_overload_num_slots = 0
            total_available_num_slots = 0
            total_standby_wgs = 0
            gmem_insufficient_count = 0
            gmem_high_water_level_count = 0
            gmem_insufficient_relabenced_count = 0
            load_rebalanced_count = 0
            dispatch_delay = 0
            enqueue_delay = 0

            # 只将请求dispatch给ready worker group，每次循环都是最新的ready状态
            # 在dispatch过程中，worker group死了也没关系，这个request会之后被标记为stale
            t0 = time.time()
            ready_wg_items = self.replicas.get_ready_worker_groups().items()
            ready_wg_ids0 = set(engine_id for engine_id, _ in ready_wg_items)
            for engine_id, wg in ready_wg_items:
                wg: Union[RayWorkerGroup, RemoteAsyncXPerfGPTRollout, AsyncActorRolloutRefWorker]  # type anno
                wg_name = wg.group_name
                try:
                    # send new request to worker group (engine)
                    load: LoadMetric = wg.get_load_metrics()
                    gmem_insufficient = load.kv_cache_util > self.config.proxy.gmem_insufficient_threshold
                    gmem_high_water_level = load.kv_cache_util > self.config.proxy.gmem_high_water_level_threshold
                    gmem_abundant = load.kv_cache_util < self.config.proxy.gmem_abundant_threshold

                    short = max_concurrency - load.num_prefilling - load.num_decoding - load.num_pending - load.num_waiting
                    if gmem_high_water_level:
                        # 统计engine可以同时运行的最大query数
                        running_concurrency = load.num_prefilling + load.num_decoding
                        engine_concurrency_cap[engine_id] = max(engine_concurrency_cap[engine_id], running_concurrency)
                        # 限制最大分发不超过engine并发能力，额外补充固定10个余量，避免发过去过多然后又abort
                        short = min(short, engine_concurrency_cap[engine_id] + 2)

                    # metrics
                    total_overload_num_slots += short if short < 0 else 0
                    total_available_num_slots += short if short > 0 else 0
                    total_standby_wgs += 1 if short > 0 and not gmem_insufficient else 0
                    gmem_insufficient_count += 1 if gmem_insufficient else 0
                    gmem_high_water_level_count += 1 if gmem_high_water_level else 0

                    # 优先让各个wg都均匀得到相等的query，内存满了就不再放过去
                    if short > 0 and not gmem_high_water_level:
                        queries: List[Query] = ray.get(
                            self.request_manager.get_next_pending_requests.remote(short, engine_id, wg_name))
                        if len(queries) > 0:
                            for q in queries:
                                # query的分类，区分一下是hybrid_rollout/standalone_rollout/validation，避免共用engine时不知道怎么update回去对应的来源
                                q.meta_info['query_type'] = self._request_manager_name
                            wg.add_inflight_queries(queries)
                            pending_size -= len(queries)
                            fmt = (
                                "dispatch {accumulated_value} "
                                f"(remain={pending_size}) queries from({self._request_manager_name}) to wg({wg_name}, "
                                f"pending={load.num_pending}, W={load.num_waiting}/P={load.num_prefilling}/D={load.num_decoding}, "
                                f"kv={load.kv_cache_util:.2f})")
                            self._progress_logger.log(wg_name, len(queries), fmt)
                            now = time.time()
                            for q in queries:
                                dispatch_delay += now - q.enqueue_time / 1e3
                                # 还没开始生成过的才考虑算上enqueue delay
                                if q.new_token_len == 0:
                                    enqueue_delay += (q.enqueue_time - q.created_time) / 1e3

                    elif pending_size < 128 or pending_size < 2 * len(ready_wg_items):
                        # 仅当pending较少时才进行rebalance，否则优先将未分发出去的先分出去
                        # 只驱逐，等下一轮循环时再分配

                        # gmem不够导致的waiting，可以将其驱逐给别的wg
                        waiting_query_ids = []
                        release_due_to_gmem_insufficient_ref = None
                        if gmem_insufficient and not load.is_weights_updating:
                            waiting_query_ids = load.waiting_ids + load.pending_ids  # 在pending里的也释放掉
                            if len(waiting_query_ids) > 0:
                                release_due_to_gmem_insufficient_ref = self.request_manager.release_by_ids.remote(
                                    waiting_query_ids, engine_id, 'memory insufficient')

                        # 驱逐了waiting仍然超了，再去除掉一些 (short/still_short是一个negative number)
                        # rebalance_threshold的目的：不平衡只超出一点点就不管，让他继续跑，避免来回震荡调整
                        release_due_to_lb = None
                        still_short = short + len(waiting_query_ids)
                        if not gmem_abundant and still_short < -self._rebalance_threshold:
                            release_due_to_lb = self.request_manager.release_shortest_n.remote(
                                -still_short, engine_id, 'rebalance')

                        # 汇总要abort的所有query_ids
                        to_abort = []
                        if release_due_to_gmem_insufficient_ref is not None:
                            released_ids = ray.get(release_due_to_gmem_insufficient_ref)
                            to_abort.extend(released_ids)
                            gmem_insufficient_relabenced_count += len(released_ids)
                        if release_due_to_lb is not None:
                            released_ids = ray.get(release_due_to_lb)
                            to_abort.extend(released_ids)
                            load_rebalanced_count += len(released_ids)

                        # 从engine abort掉
                        if len(to_abort) > 0:
                            pending_size += len(to_abort)
                            wg.abort_queries(to_abort, time.time())
                            fmt = (
                                'aborting {accumulated_value}x queries from ' +
                                f'engine({wg_name}) W={load.num_waiting}/P={load.num_prefilling}/D={load.num_decoding}')
                            self.abort_logger.log(wg_name, len(to_abort), fmt)

                    loads[(engine_id, wg_name)] = load

                except (ActorDiedError, RayTaskError) as e:
                    self._finalize(wg, e)

            # handle dead engines during the loop to avoid request from staling for too long
            ready_wg_ids1 = self.replicas.ready_worker_group_ids
            dead_wg_ids_during_loop = ready_wg_ids0 - ready_wg_ids1
            dead_wg_ids_between_loop = ready_wg_ids_prev - ready_wg_ids1
            ready_wg_ids_prev = ready_wg_ids1  # save to previous
            if dead_wg_ids_during_loop or dead_wg_ids_between_loop:
                ray.get(self.request_manager.handle_stale_requests.remote(ready_wg_ids1))

            # observability
            total, pending_size = ray.get(self.request_manager.get_size.remote())
            num_ready_replicas = len(self.replicas.ready_worker_group_ids)
            load_skewness = self._calc_load_skewness(loads)
            prefill_throughput, decode_throughput = ray.get(self.request_manager.get_estimated_throughput.remote())
            finished_stats = ray.get(self.request_manager.get_finished_stats.remote(self._metrics_logger.global_step))
            self._trace_load_metrics(loads, load_skewness, prefill_throughput, decode_throughput, total, pending_size)

            loop_cost = time.time() - t0
            sleep_interval = max(0., self.poll_interval - loop_cost)
            num_target_replicas = self.replicas.target_num_replicas
            num_alive_replicas = len(self.replicas.alive_worker_group_ids)
            num_initialized_replicas = len(self.replicas.initialized_worker_group_ids)
            loop_cost_metrics = ProxyLoopCost(
                total=loop_cost,
            )
            internal_metrics = InternalDiagnosisMetrics(
                num_target_replicas=num_target_replicas,
                num_ready_replicas=num_ready_replicas,
                num_initialized_replicas=num_initialized_replicas,
                num_alive_replicas=num_alive_replicas,
                gmem_insufficient_count=gmem_insufficient_count,
                gmem_high_water_level_count=gmem_high_water_level_count,
                total_standby_wgs=total_standby_wgs,
                total_overload_num_slots=total_overload_num_slots,
                total_available_num_slots=total_available_num_slots,
                total_rebalanced=gmem_insufficient_relabenced_count,
                max_concurrency=max_concurrency,
                dispatch_delay_acc=dispatch_delay,
                enqueue_delay_acc=enqueue_delay,
            )
            self._trace_metrics(loop_cost_metrics)
            self._trace_metrics(internal_metrics)

            self._metrics_logger.log({
                'num_ready_replicas': num_ready_replicas,
                'loop_cost': loop_cost,
                'gmem_insufficient_rebalanced_count': gmem_insufficient_relabenced_count,
                'load_rebalanced_count': load_rebalanced_count,
                'total_token_TPS': sum(decode_throughput.values()),
                'total_processes_queries': finished_stats.finished_size,
            })

    def get_step_metrics(self) -> dict:
        super_metrics = super().get_step_metrics()
        metrics = self._metrics_logger.get_last_step_metrics()
        try:
            num_ready_replicas = metrics.get('num_ready_replicas', zero_stats)
            loop_cost = metrics.get('loop_cost', zero_stats)
            gmem_insufficient_rebalanced_count = metrics.get('gmem_insufficient_rebalanced_count', zero_stats)
            load_rebalanced_count = metrics.get('load_rebalanced_count', zero_stats)
            decode_tps = metrics.get('total_token_TPS', zero_stats)
            total_processes_queries = metrics.get('total_processes_queries', zero_stats)
            this_metrics = {
                'rollout/elastic/num_ready_replicas_mean': num_ready_replicas.mean,
                'rollout/elastic/num_ready_replicas_min': num_ready_replicas.minimum,
                'rollout/elastic/num_ready_replicas_max': num_ready_replicas.maximum,
                'rollout/proxy/loop_cost': loop_cost.mean,
                'rollout/proxy/gmem_insufficient_rebalanced_count_total': gmem_insufficient_rebalanced_count.sum,
                'rollout/proxy/load_rebalanced_count_total': load_rebalanced_count.sum,
                'rollout/proxy/total_token_TPS': decode_tps.mean,
                'rollout/proxy/total_processes_queries': total_processes_queries.maximum,
            }
            super_metrics.update(this_metrics)
        except KeyError as e:
            traceback.print_exc()
            print("got error on get_step_metrics, will be ignored")
        return super_metrics


class CacheAwareBalancedRolloutWorkerGroupProxy(RolloutWorkerGroupProxy):

    def __init__(self, replicas: Union[ReplicatedRayWorkerGroup, ScalingRayWorkerGroup],
                 actor_info: List[WeightsRankInfo], request_manager_name: str, config: DictConfig):
        super().__init__(replicas, actor_info, request_manager_name, config)
        self.abort_logger = DebounceAccumulatedLogger()

    async def _dispatch_loop(self):
        print(f'start background dispatch loop with cache_aware balanced mode for {self._request_manager_name}')

        sleep_interval = self.poll_interval
        ready_wg_ids_prev = set()
        engine_concurrency_cap = defaultdict(int)  # engine_id -> 最大可并发数
        while True:
            if self._loop_should_stop.is_set():
                break
            self.is_waiting = True
            self._loop_should_continue.wait()
            self.is_waiting = False
            await asyncio.sleep(sleep_interval)

            # handle dead engines during the loop to prevent request from staling(dangling)
            ready_wg_ids1 = self.replicas.ready_worker_group_ids
            dead_wg_ids_between_loop = ready_wg_ids_prev - ready_wg_ids1
            ready_wg_ids_prev = ready_wg_ids1  # save to previous
            if dead_wg_ids_between_loop:
                await self.request_manager.handle_stale_requests.remote(ready_wg_ids1)

            # 先考虑cache命中，再按照总量平分给每个ready replica，均匀分发
            # 注意一开始可能还没有request进去request pool
            # 也可能replicas还没ready
            total, pending_size = await self.request_manager.get_size.remote()

            # 纪录负载指标
            loads: Dict[Tuple[str, str], LoadMetric] = {}  # (engine_id, wg_name) -> LoadMetric
            gmem_high_water_level_count = 0
            gmem_insufficient_count = 0
            total_available_num_slots = 0
            total_overload_num_slots = 0
            load_rebalanced_count = 0
            enqueue_delay, dispatch_delay = 0.0, 0.0
            loop_cost_metrics = ProxyLoopCost()
            shed_metrics = InternalShedMetrics()
            dispatch_metrics: Dict[str, EngineDispatchMetrics] = {}  # engine_id -> EngineDispatchMetrics

            # 只将请求dispatch给ready worker group，每次循环都是最新的ready状态
            # 在dispatch过程中，worker group死了也没关系，这个request会之后被标记为stale
            t0 = time.time()
            ready_wg_items = list(self.replicas.get_ready_worker_groups().items())
            died_wgs = []

            query_history_map: Dict[str, Tuple[str, float]] = {}  # cache_id -> (engine_id, save_ts)

            async def get_wg_history(wg, engine_id):
                nonlocal gmem_high_water_level_count, gmem_insufficient_count
                try:
                    wg_name = wg.group_name
                    history_ids = await wg.get_history_ids_async()
                    for cache_id, save_ts in history_ids:
                        # Note(lixiang):
                        # 这里可能会出现一个cache_id有多个matched engine，优先选后跑过的那个，因为prefix length可能会更长
                        if cache_id not in query_history_map:
                            query_history_map[cache_id] = (engine_id, save_ts)
                        else:
                            # 存更新的
                            if save_ts > query_history_map[cache_id][1]:
                                query_history_map[cache_id] = (engine_id, save_ts)
                    load: LoadMetric = await wg.get_load_metrics_async()
                    loads[(engine_id, wg_name)] = load
                    running_concurrency = load.num_prefilling + load.num_decoding
                    if load.kv_cache_util > self.config.proxy.gmem_high_water_level_threshold:
                        engine_concurrency_cap[engine_id] = max(engine_concurrency_cap[engine_id], running_concurrency)
                        gmem_high_water_level_count += 1
                    if load.kv_cache_util > self.config.proxy.gmem_insufficient_threshold:
                        gmem_insufficient_count += 1

                except (ActorDiedError, RayTaskError) as e:
                    # ignore actor died error, underlying replicated worker group will handle
                    # worker group and actors lifecycle
                    self._finalize(wg, e)
                    died_wgs.append(engine_id)

            async with asyncio.TaskGroup() as tg:
                for engine_id, wg in ready_wg_items:
                    tg.create_task(get_wg_history(wg, engine_id))

            # 过滤死掉的wg
            ready_wg_items = [(engine_id, wg) for engine_id, wg in ready_wg_items if engine_id not in died_wgs]

            # collect perf metrics
            load_skewness = self._calc_load_skewness(loads)
            prefill_throughput, decode_throughput = await self.request_manager.get_estimated_throughput.remote()
            self._trace_load_metrics(loads, load_skewness, prefill_throughput, decode_throughput, total, pending_size)
            loop_cost_metrics.get_engine_info = time.time() - t0

            abort_cool_down_seconds = 10
            # 一次rebalance不要太多，调节要平缓
            # 这里从别的engine rebalance过来太多会导致目标engine因为跑了别人的query而把prefix cache evict掉了，最终造成命中率下降
            max_rebalance_count_per_wg = 1
            recent_aborted_ids: Dict[str, Set[str]] = {}  # engine_id -> [query_id]
            # rebalance1
            # 在engine waiting中一段时间但没有命中cache
            # (可能是刚step更新完，大家都没cache，重新abort掉平衡一下)
            enable_no_cache_rebalancing = self.config.proxy.enable_no_cache_rebalancing
            if enable_no_cache_rebalancing and load_skewness.p25 < self.config.proxy.gmem_high_water_level_threshold:
                no_cache_query_ids: Dict[str, Set[str]] = {}  # engine_id -> {query_id}
                for (engine_id, wg_name), load in loads.items():
                    gmem_insufficient = load.kv_cache_util > self.config.proxy.gmem_insufficient_threshold
                    if gmem_insufficient:
                        no_cache_query_ids[engine_id] = set(load.no_caching_query_ids[-max_rebalance_count_per_wg:])
                if len(no_cache_query_ids) > 0:
                    aborted_ids = await self._rebalance_by_ids(loads, no_cache_query_ids)
                    aborted_count = sum([len(v) for v in aborted_ids.values()])
                    pending_size += aborted_count
                    load_rebalanced_count += aborted_count
                    recent_aborted_ids.update(aborted_ids)

            # rebalance2
            # 负载倾斜，平衡的收益更大
            # 触发条件(需要苛刻一点，当另一个engine重新prefill的好处大于继续等cache的好处才rebalance)
            # 1. 有超过25%的engine kv低于高水位
            # 2. 负载倾斜达到阈值
            if (load_skewness.p25 < self.config.proxy.gmem_high_water_level_threshold and
                    load_skewness.cov > self.config.proxy.rebalance_skewness_coef):
                # 当有空闲engine才会触发rebalance
                aborted_ids = await self._rebalance(loads, load_skewness, max_rebalance_count_per_wg)
                aborted_count = sum([len(v) for v in aborted_ids.values()])
                pending_size += aborted_count
                load_rebalanced_count += aborted_count
                for engine_id, aborted in aborted_ids.items():
                    if engine_id not in recent_aborted_ids:
                        recent_aborted_ids[engine_id] = aborted
                    else:
                        recent_aborted_ids[engine_id].update(aborted)

            # matching
            # 找出哪些query命中cache，哪些没命中(standalone)，跳过刚从某个engine abort出来的
            # Note: all_requests 是有序的，queries_in_wg_history和standalone_queries也要满足偏序关系
            all_requests = await self.request_manager.peak_all_pending_requests.remote()
            queries_in_wg_history: Dict[str, List[Query]] = defaultdict(list)  # engine_id ->
            standalone_queries = []
            # engine_id -> {query_id}，记录是否最近从某个engine abort，在实际dispatching时要跳过这个
            engine_black_list: Dict[str, Set[str]] = defaultdict(set)
            engine_black_list.update(recent_aborted_ids)
            # 只去取cache_id->engine_id，忽略save ts
            query_cache_id_map = {cache_id: engine_id for cache_id, (engine_id, _) in query_history_map.items()}
            for req in all_requests:
                req: Request
                query = req.query
                history_engine_id = query_cache_id_map.get(query.cache_id)
                # 刚abort出来的query不要放回同一个engine里
                aborted_ids = recent_aborted_ids.get(history_engine_id) or set()
                if history_engine_id is not None and query.id not in aborted_ids:
                    # 暂时没有考虑rebalance，之后再算
                    queries_in_wg_history[history_engine_id].append(query)
                else:
                    standalone_queries.append(query)

            shed_metrics.history_count = len(all_requests) - len(standalone_queries)
            shed_metrics.standalone_count = len(standalone_queries)

            # 按照load水平从低到高分query
            # 先分没命中cache(standalone)的query，然后看load水平
            # 如果仍然比较低，就继续分命中的，否则命中的仍然以正常的方式分
            wg_loads: Dict[str, int] = {}  # engine_id -> 已经分的数量
            wg_id_load_from_low_to_high = []
            for engine_id, wg in ready_wg_items:
                wg: Union[RayWorkerGroup, RemoteAsyncXPerfGPTRollout, AsyncActorRolloutRefWorker]  # type anno
                wg_name = wg.group_name
                load = loads[(engine_id, wg_name)]
                wg_loads[engine_id] = load.num_prefilling + load.num_decoding + load.num_pending + load.num_waiting
                wg_id_load_from_low_to_high.append((engine_id, wg_name))
            wg_id_load_from_low_to_high.sort(key=lambda v: wg_loads[v[0]])

            loop_cost_metrics.matching = time.time() - t0

            # dispatching(staging)
            queries_assigned: Dict[str, List[Query]] = defaultdict(list)  # engine_id ->

            # 重新计算一次当前最新的总量，staging的数量 + 已经分出去的
            current_total = len(all_requests) + sum(wg_loads.values())
            num_ready_replicas = len(self.replicas.ready_worker_group_ids)
            max_concurrency = current_total // max(1, num_ready_replicas)  # replicas可能还没ready
            max_concurrency = min(max(max_concurrency, 1), 512)  # 限制在1-512范围内

            # 第一轮先匹配kv
            for engine_id, wg_name in wg_id_load_from_low_to_high:
                dispatch_metrics[engine_id] = EngineDispatchMetrics(wg_name)

                # 从低到高，先试图分配到max_concurrency
                # 有本来就命中cache且属于他的
                # pending量前x%必发，或不超过历史记录的concurrency+bleeding
                short = max_concurrency - wg_loads[engine_id]
                load = loads[(engine_id, wg_name)]
                gmem_high = load.kv_cache_util > self.config.proxy.gmem_high_water_level_threshold
                has_waiting = load.num_waiting + load.num_pending > 4

                # 如果engine本身还有waiting，则先跳过这一轮，engine内可能会kv满了然后evict掉一轮
                # 可能会出现从decoding swap到waiting的过程中，因为保存了新的kv，导致老的kv被LRU，
                # 然后原本可能match的query进来时kv没了，来回prefill降低了命中率
                if has_waiting and load.kv_prefix_cache_swap_util > 0.4:
                    continue

                if gmem_high:
                    short = min(short, engine_concurrency_cap[engine_id] + 1)
                    short_for_history = short
                else:
                    # 充分时可以多分一些history cache匹配的(提高命中率)
                    # 尽量优先分配cache，即使在waiting中排队也没关系，避免过多的P打断D
                    short_for_history = short + 6

                if short < 0:
                    # 已经跑满了就跳过分给这个engine
                    total_overload_num_slots += (-short)
                    continue
                total_available_num_slots += short

                staging = queries_in_wg_history[engine_id][:short_for_history]
                queries_in_wg_history[engine_id] = queries_in_wg_history[engine_id][short_for_history:]
                dispatch_metrics[engine_id].history_remain_pending += len(queries_in_wg_history[engine_id])
                queries_assigned[engine_id].extend(staging)
                short -= len(staging)
                shed_metrics.cache_match += len(staging)

                # 仍没有达到cap
                # 用没命中cache(standalone)的query往里分配，并跳过刚abort出来的
                if short > 0 and standalone_queries:
                    blacklist = engine_black_list[engine_id]
                    skipped: List[Query] = []
                    staging: List[Query] = []
                    for q in standalone_queries:
                        if q.id in blacklist:
                            skipped.append(q)
                        else:
                            staging.append(q)
                            if len(staging) == short:
                                break
                    total_iterated = len(skipped) + len(staging)
                    standalone_queries = skipped + standalone_queries[total_iterated:]  # skipped要还回去
                    queries_assigned[engine_id].extend(staging)
                    short -= len(staging)
                    shed_metrics.standalone += len(staging)

            # 第二轮当自己kv不满且别人还有分剩的才拿过来
            for engine_id, wg_name in wg_id_load_from_low_to_high:
                num_staging = len(queries_assigned[engine_id])
                num_assigned = wg_loads[engine_id]
                short = max_concurrency - num_assigned - num_staging
                gmem_abundant = loads[(engine_id, wg_name)].kv_cache_util < self.config.proxy.gmem_abundant_threshold

                # 发完standalone仍可分
                # 则从属于其他engine有cache的里面偷一点过来，要求如下
                # 1. gmem充足
                # 2. 这个engine负载只有另外的engine的一半，否则拿过来降低cache命中率不划算
                # 3. 当不是刚起来的新engine时，不超过engine自身cap
                if num_assigned > 0:
                    short = min(short, engine_concurrency_cap[engine_id])
                if gmem_abundant and short > 0:
                    # FIXME(lixiang): 这里为了满足fifo，需要merge sort，暂时简化成依次顺序访问，观察调度指标，延迟过大了再考虑优化
                    this_engine_total = num_assigned + num_staging
                    for another_engine_id, queries in queries_in_wg_history.items():
                        another_engine_total = wg_loads[another_engine_id] + len(queries_assigned[another_engine_id])
                        if this_engine_total * 2 > another_engine_total:
                            # 另一个engine负载大于此engine两倍才偷
                            continue
                        from_another_engine = queries[:short]
                        queries_in_wg_history[another_engine_id] = queries[short:]
                        queries_assigned[engine_id].extend(from_another_engine)
                        short -= len(from_another_engine)
                        shed_metrics.steal += len(from_another_engine)
                        dispatch_metrics[engine_id].steal += len(from_another_engine)
                        if short < 0:
                            break

            loop_cost_metrics.staging = time.time() - t0

            # dispatching(commit/rollback)
            async def dispatching(wg, engine_id):
                nonlocal dispatch_delay, enqueue_delay, pending_size
                wg: Union[RayWorkerGroup, RemoteAsyncXPerfGPTRollout, AsyncActorRolloutRefWorker]  # type anno
                wg_name = wg.group_name
                load = loads[(engine_id, wg_name)]

                commit_queries = queries_assigned.get(engine_id)
                if not commit_queries:
                    return

                now = time.time()
                commit_query_ids = [q.id for q in commit_queries]
                await self.request_manager.set_requests_assigned.remote(commit_query_ids, engine_id, wg_name, now)
                for q in commit_queries:
                    # 补充这个字段到query，保证一致性
                    q.dispatch_time = now

                    dispatch_delay += now - q.enqueue_time / 1e3
                    if q.new_token_len == 0:
                        # 还没开始生成过的才考虑算上enqueue delay
                        enqueue_delay += (q.enqueue_time - q.created_time) / 1e3

                    # query的分类，区分一下是hybrid_rollout/standalone_rollout/validation，避免共用engine时不知道怎么update回去对应的来源
                    q.meta_info['query_type'] = self._request_manager_name

                try:
                    await asyncio.gather(*wg.add_inflight_queries_non_blocking(commit_queries))
                    pending_size -= len(commit_queries)
                    fmt = (
                        "dispatch {accumulated_value} "
                        f"(remain={pending_size}) queries from({self._request_manager_name}) to wg({wg_name}, "
                        f"pending={load.num_pending}, W={load.num_waiting}/P={load.num_prefilling}/D={load.num_decoding}, "
                        f"kv={load.kv_cache_util:.2f})")
                    self._progress_logger.log(wg_name, len(commit_queries), fmt)
                except (ActorDiedError, RayTaskError) as e:
                    # ignore actor died error, underlying replicated worker group will handle
                    # worker group and actors lifecycle
                    # 分发失败的，需要把assigned flag给clear掉，不然会泄漏
                    await self.request_manager.clear_requests_assigned.remote(commit_query_ids)
                    self._finalize(wg, e)

            async with asyncio.TaskGroup() as tg:
                for engine_id, wg in ready_wg_items:
                    tg.create_task(dispatching(wg, engine_id))

            # 没有分出去的要rollback掉 (history+standalone)
            rollback_query_ids = []
            for queries in queries_in_wg_history.values():
                rollback_query_ids.extend([q.id for q in queries])
            rollback_query_ids.extend([q.id for q in standalone_queries])
            if len(rollback_query_ids) > 0:
                await self.request_manager.clear_requests_assigned.remote(rollback_query_ids)

            # metrics collect
            finished_stats = await self.request_manager.get_finished_stats.remote(self._metrics_logger.global_step)

            loop_cost_metrics.total = time.time() - t0
            sleep_interval = max(0., self.poll_interval - loop_cost_metrics.total)

            num_target_replicas = self.replicas.target_num_replicas
            num_alive_replicas = len(self.replicas.alive_worker_group_ids)
            num_initialized_replicas = len(self.replicas.initialized_worker_group_ids)
            internal_metrics = InternalDiagnosisMetrics(
                num_target_replicas=num_target_replicas,
                num_ready_replicas=num_ready_replicas,
                num_initialized_replicas=num_initialized_replicas,
                num_alive_replicas=num_alive_replicas,
                gmem_insufficient_count=gmem_insufficient_count,
                gmem_high_water_level_count=gmem_high_water_level_count,
                total_standby_wgs=0,
                total_overload_num_slots=total_overload_num_slots,
                total_available_num_slots=total_available_num_slots,
                total_rebalanced=load_rebalanced_count,
                max_concurrency=max_concurrency,
                dispatch_delay_acc=dispatch_delay,
                enqueue_delay_acc=enqueue_delay,
            )
            self._trace_metrics(internal_metrics)
            self._trace_metrics(shed_metrics)
            self._trace_metrics(loop_cost_metrics)
            [self._trace_engine_metrics(m) for m in dispatch_metrics.values()]

            self._metrics_logger.log({
                'num_ready_replicas': num_ready_replicas,
                'loop_cost': loop_cost_metrics.total,
                'loop_cost_get_engine_info': loop_cost_metrics.get_engine_info,
                'loop_cost_matching': loop_cost_metrics.matching,
                'total_token_TPS': sum(decode_throughput.values()),
                'total_processes_queries': finished_stats.finished_size,
            })

    async def _rebalance(self, loads: Dict[Tuple[str, str], LoadMetric], load_skewness: LoadSkewness,
                         max_rebalance_count: int) -> Dict[str, Set[str]]:
        aborted_ids = {}
        for engine_id, wg in self.replicas.get_ready_worker_groups().items():
            wg: Union[RayWorkerGroup, RemoteAsyncXPerfGPTRollout, AsyncActorRolloutRefWorker]
            load = loads.get((engine_id, wg.group_name))
            if load is None:
                # skip new joined
                continue
            if load.is_weights_updating:
                continue
            if load.kv_cache_util < self.config.proxy.gmem_abundant_threshold:
                # skip abundant engines
                continue

            # 计算要abort的量
            num_deviated = int(round(load.num_running - load_skewness.running_avg, 0))

            # 把gmem较满的切溢出到waiting中的query给abort了
            gmem_insufficient = load.kv_cache_util > self.config.proxy.gmem_insufficient_threshold
            to_abort = []
            if gmem_insufficient:
                waiting_query_ids = load.waiting_ids + load.pending_ids  # 在pending里的也释放掉
                waiting_query_ids = waiting_query_ids[-max_rebalance_count:]  # 从后面开始取
                if len(waiting_query_ids) > 0:
                    released_ids = await self.request_manager.release_by_ids.remote(waiting_query_ids, engine_id,
                                                                                    "memory insufficient")
                    to_abort.extend(released_ids)
            max_rebalance_count -= len(to_abort)

            # abort running中不平衡的部分
            if max_rebalance_count > 0 and num_deviated > self.config.proxy.rebalance_threshold:
                num_release = num_deviated - self.config.proxy.rebalance_threshold
                num_release = min(num_release, max_rebalance_count)
                released_ids = await self.request_manager.release_shortest_n.remote(num_release, engine_id, "rebalance")
                to_abort.extend(released_ids)

            # do abort
            if len(to_abort) > 0:
                try:
                    await asyncio.gather(*wg.abort_queries_non_blocking(to_abort, time.time()))
                    fmt = (
                        'aborting {accumulated_value}x queries from ' +
                        f'engine({wg.group_name}) W={load.num_waiting}/P={load.num_prefilling}/D={load.num_decoding}')
                    self.abort_logger.log(wg.group_name, len(to_abort), fmt)
                    aborted_ids[engine_id] = set(to_abort)
                except (ActorDiedError, RayTaskError) as e:
                    self._finalize(wg, e)

                # update load metrics (roughly)
                load.subtract_released(len(to_abort))

        return aborted_ids

    async def _rebalance_by_ids(self, loads: Dict[Tuple[str, str], LoadMetric],
                                release_ids: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        refs: List[Tuple[str, ObjectRef]] = []  # [engine_id, ref]
        wg_refs = []
        for engine_id, wg in self.replicas.get_ready_worker_groups().items():
            wg: Union[RayWorkerGroup, RemoteAsyncXPerfGPTRollout, AsyncActorRolloutRefWorker]
            to_release = release_ids.get(engine_id, set())
            if len(to_release) > 0:
                ref = self.request_manager.release_by_ids.remote(to_release, engine_id, "rebalance")
                refs.append((engine_id, ref))
                wg_ref = wg.abort_queries_non_blocking(list(to_release), time.time())
                wg_refs.extend(wg_ref)

                # update load counts
                load = loads.get((engine_id, wg.group_name))
                if load is not None:
                    fmt = (
                        'aborting {accumulated_value}x queries from ' +
                        f'engine({wg.group_name}) W={load.num_waiting}/P={load.num_prefilling}/D={load.num_decoding}')
                    self.abort_logger.log(wg.group_name, len(to_release), fmt)
                    load.subtract_released(len(to_release))

        aborted_ids: Dict[str, Set[str]] = {}
        for engine_id, ref in refs:
            aborted_ids[engine_id] = set(await ref)
        await asyncio.gather(*wg_refs)

        return aborted_ids

    def get_step_metrics(self) -> dict:
        super_metrics = super().get_step_metrics()
        metrics = self._metrics_logger.get_last_step_metrics()
        try:
            num_ready_replicas = metrics.get('num_ready_replicas', zero_stats)
            loop_cost = metrics.get('loop_cost', zero_stats)
            gmem_insufficient_rebalanced_count = metrics.get('gmem_insufficient_rebalanced_count', zero_stats)
            load_rebalanced_count = metrics.get('load_rebalanced_count', zero_stats)
            decode_tps = metrics.get('total_token_TPS', zero_stats)
            total_processes_queries = metrics.get('total_processes_queries', zero_stats)
            this_metrics = {
                'rollout/elastic/num_ready_replicas_mean': num_ready_replicas.mean,
                'rollout/elastic/num_ready_replicas_min': num_ready_replicas.minimum,
                'rollout/elastic/num_ready_replicas_max': num_ready_replicas.maximum,
                'rollout/proxy/loop_cost': loop_cost.mean,
                'rollout/proxy/gmem_insufficient_rebalanced_count_total': gmem_insufficient_rebalanced_count.sum,
                'rollout/proxy/load_rebalanced_count_total': load_rebalanced_count.sum,
                'rollout/proxy/total_token_TPS': decode_tps.mean,
                'rollout/proxy/total_processes_queries': total_processes_queries.maximum,
            }
            super_metrics.update(this_metrics)
        except KeyError as e:
            traceback.print_exc()
            print("got error on get_step_metrics, will be ignored")
        return super_metrics
