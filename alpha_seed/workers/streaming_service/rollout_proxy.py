import asyncio
import inspect
import os
import random
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Union, Tuple, Set

import ray
import torch
from omegaconf import DictConfig
from ray import ObjectRef
from ray.exceptions import ActorDiedError, GetTimeoutError, RayActorError, RayTaskError, ActorUnavailableError

from alpha_seed.utils.profile.timeline import Tracer, CompleteEvent, CounterEvent
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
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
class InternalDiagnosisMetrics:
    loop_cost: float
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

    def to_dict(self):
        # $前缀表示requests/query数量
        # #前缀表示replica数量
        return {
            'loop_cost': self.loop_cost,
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
        print(f'get recent concurrency min={min_con} max={max_con} total={total_con}')
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

    def __init__(self, replicas):
        self.replicas = replicas
        self._tracer = Tracer.get_instance()
        self._update_worker_start_ts = 0
        self._stop_server_ts = 0

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
        self._update_worker_start_ts = time.time() * 1e6
        futs = []
        for wg in self.replicas.get_initialized_worker_groups().values():
            wg: Union[RayWorkerGroup, RemoteAsyncXPerfGPTRollout, AsyncActorRolloutRefWorker]
            refs = wg.update_standalone_worker(role)
            futs.append((wg, refs))
        self.wait_ignore_actor_died(futs)
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
        )
        self._tracer.trace(evt)

    def stop_server_before_weights_update(self):
        self._stop_server_ts = time.time() * 1e6
        futs = []
        for wg in self.replicas.get_initialized_worker_groups().values():
            wg: RemoteAsyncXPerfGPTRollout
            ref = wg.stop_server_before_weights_update_non_blocking()
            futs.append((wg, ref))
        self.wait_ignore_actor_died(futs)

    def restart_server_after_weights_update(self):
        futs = []
        for wg in self.replicas.get_initialized_worker_groups().values():
            wg: RemoteAsyncXPerfGPTRollout
            ref = wg.restart_server_after_weights_update_non_blocking()
            futs.append((wg, ref))
        self.wait_ignore_actor_died(futs)

        evt = CompleteEvent(
            pid='RolloutProxy',
            tid='update',
            cat='stop/start server',
            name='stop/start server',
            ts=self._stop_server_ts,
            dur=time.time() * 1e6 - self._stop_server_ts,
        )
        self._tracer.trace(evt)

    def wait_ignore_actor_died(self, refs: List[Tuple[RayWorkerGroup, List[ray.ObjectRef]]]):
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
                    caller = inspect.stack()[1].frame.f_code.co_name
                    print(f"actor({e.actor_id}) died at function({caller}). ignore this as this is expected.")
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
            print(content)


@dataclass
class StatisticalMetric:
    minimum: float
    maximum: float
    mean: float
    sum: float


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
        self.config = config  # .streaming_rollout.proxy
        self._tracer = Tracer.get_instance()
        self._stop_server_ts = 0
        self._update_worker_start_ts = 0
        self._request_manager_name = request_manager_name
        self.poll_interval = self.config.proxy.poll_internal_seconds
        self._progress_logger = DebounceAccumulatedLogger()
        self._metrics_logger = ProxyMetricsLogger()

        self.replicas.set_dead_callback(self._worker_group_dead_callback)
        self._loop_should_stop = threading.Event()
        self._loop_should_continue = threading.Event()
        self._loop_should_continue.set()
        self.is_waiting = False
        self._loop_thread = threading.Thread(target=self._dispatch_loop,
                                             name=f'{request_manager_name}-rollout-wg-proxy-dispatch-loop',
                                             daemon=True)
        self._loop_thread.start()

    @property
    def world_size(self):
        # 按照设定的replicas数计算world_size，而不管是否已完成初始化
        alive_worker_groups: Dict[str, RayWorkerGroup] = self.replicas.get_alive_worker_groups()
        return sum(w.world_size for w in alive_worker_groups.values())

    def _worker_group_dead_callback(self, worker_group_ids: List[str]):
        # worker group任意死了之后，通知request manager将运行中的请求释放掉
        ready_worker_group_ids = self.replicas.ready_worker_group_ids
        ray.get(self.request_manager.handle_stale_requests.remote(ready_worker_group_ids))

    def _dispatch_loop(self):
        print(f'start background dispatch loop for {self._request_manager_name}')

        sleep_interval = self.poll_interval
        while True:
            if self._loop_should_stop.is_set():
                break
            self.is_waiting = True
            self._loop_should_continue.wait()
            self.is_waiting = False
            time.sleep(sleep_interval)

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

            wg_history_map: Dict[str, Set[str]] = {}
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
                    # 1. collect intermediate result
                    # get result(including partial) from engine, update to centralized request pool
                    queries: List[Query] = wg.get_all_queries(self._request_manager_name)
                    if len(queries) > 0:
                        self.request_manager.update_intermediate_queries.remote(queries, engine_id, wg_name,
                                                                                time.time())

                    # 2. send new request to worker group (engine)
                    load: LoadMetric = wg.get_load_metrics()[0]  # noqa, dp_size always =1
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
            throughput = ray.get(self.request_manager.get_estimated_throughput.remote())
            self._trace_load_metrics(loads, throughput, total, pending_size)

            loop_cost = time.time() - t0  # noqa: for py-spy
            sleep_interval = max(0., self.poll_interval - loop_cost)

    def stop(self):
        self._loop_should_stop.set()
        self._loop_thread.join()

    def pause_loop(self):
        self._loop_should_continue.clear()
        while not self.is_waiting:
            time.sleep(0.5)

    def continue_loop(self):
        self._loop_should_continue.set()

    def step(self, global_step):
        self._metrics_logger.step(global_step)

    def get_step_metrics(self):
        return {}

    def _finalize(self, wg: RayWorkerGroup, e: Exception):
        # proactively kill the actor who causes any RayTaskErrors, RayTaskErrors or so on..
        # let underlying replicated worker group to handle the ready/alive of worker group
        #   in elastic scenario, there won't be any impact to task runner's proxy thread
        #   in static server scenario, any actor deadness could cause the other rpc call failure from
        #   main thread
        if not isinstance(e, ActorDiedError):
            # ignore the stack of ActorDiedError, no useful information
            traceback.print_exc()
            print(f'show stack trace of task error of remote worker group only, {wg=}, {type(e)=}')
        try:
            wg.destroy()
        except Exception:
            pass
        if not self.config.elastic.enable:
            # teardown the task_runner process in static server mode
            print('due to some worker group failed to execute tasks in non-elastic mode, '
                  'will teardown to process to propagate errors in time')
            os._exit(11)  # noqa

    def _trace_load_metrics(self, loads: Dict[Tuple[str, str], LoadMetric], throughput: Dict[str, float], total: int,
                            global_pending: int):
        total_decoding_num = 0
        for (wg_id, wg_name), metric in loads.items():
            total_decoding_num += metric.num_decoding
            tp = throughput.get(wg_id) or 0
            evt = CounterEvent(
                name='load metrics:',
                pid=f'{self._request_manager_name} {wg_name}',
                ts=metric.ts * 1e6,
                data={
                    'kv cache%': metric.kv_cache_util,
                    'prefilling': metric.num_prefilling,
                    'decoding': metric.num_decoding,
                    'pending': metric.num_pending,
                    'waiting': metric.num_waiting,
                    'decode TPS': tp,
                },
            )
            self._tracer.trace(evt)

        # global pending running event
        evt = CounterEvent(
            name='request:',
            pid=f'RequestManager/{self._request_manager_name}',  # 不区分hybrid/standalone
            ts=time.time() * 1e6,
            data={
                '$decoding': total_decoding_num,
                '$processing': total,
                '$pending dispatch': global_pending,
                '$decode TPS': sum(throughput.values()),
            },
        )
        self._tracer.trace(evt)

    def _trace_internal_diagnosis(self, internal_metrics: InternalDiagnosisMetrics):
        evt = CounterEvent(
            name='internal:',
            pid=f'RequestManager/{self._request_manager_name}',  # 不区分hybrid/standalone
            ts=time.time() * 1e6,
            data=internal_metrics.to_dict(),
        )
        self._tracer.trace(evt)


class BalancedRolloutWorkerGroupProxy(RolloutWorkerGroupProxy):

    def __init__(self, replicas: Union[ReplicatedRayWorkerGroup, ScalingRayWorkerGroup],
                 actor_info: List[WeightsRankInfo], request_manager_name: str, config: DictConfig):
        super().__init__(replicas, actor_info, request_manager_name, config)
        self._rebalance_threshold = self.config.proxy.rebalance_threshold
        self.abort_logger = DebounceAccumulatedLogger()

    def _dispatch_loop(self):
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
            time.sleep(sleep_interval)

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
                    # 1. collect intermediate result
                    # get result(including partial) from engine, update to centralized request pool
                    queries: List[Query] = wg.get_all_queries(self._request_manager_name)
                    if len(queries) > 0:
                        self.request_manager.update_intermediate_queries.remote(queries, engine_id, wg_name,
                                                                                time.time())

                    # 2. send new request to worker group (engine)
                    load: LoadMetric = wg.get_load_metrics()[0]  # noqa, dp_size always =1
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
            throughput = ray.get(self.request_manager.get_estimated_throughput.remote())
            finished_stats = ray.get(self.request_manager.get_finished_stats.remote(self._metrics_logger.global_step))
            self._trace_load_metrics(loads, throughput, total, pending_size)

            loop_cost = time.time() - t0
            sleep_interval = max(0., self.poll_interval - loop_cost)
            num_target_replicas = self.replicas.target_num_replicas
            num_alive_replicas = len(self.replicas.alive_worker_group_ids)
            num_initialized_replicas = len(self.replicas.initialized_worker_group_ids)
            internal_metrics = InternalDiagnosisMetrics(
                loop_cost=loop_cost,
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
            self._trace_internal_diagnosis(internal_metrics)

            self._metrics_logger.log({
                'num_ready_replicas': num_ready_replicas,
                'loop_cost': loop_cost,
                'gmem_insufficient_rebalanced_count': gmem_insufficient_relabenced_count,
                'load_rebalanced_count': load_rebalanced_count,
                'total_token_TPS': sum(throughput.values()),
                'total_processes_queries': finished_stats.finished_size,
            })

    def _aggregate_throughput(self, throughput: Dict[str, float]):
        return

    def get_step_metrics(self) -> dict:
        metrics = self._metrics_logger.get_last_step_metrics()
        try:
            num_ready_replicas = metrics['num_ready_replicas']
            loop_cost = metrics['loop_cost']
            gmem_insufficient_rebalanced_count = metrics['gmem_insufficient_rebalanced_count']
            load_rebalanced_count = metrics['load_rebalanced_count']
            return {
                'rollout/elastic/num_ready_replicas_mean': num_ready_replicas.mean,
                'rollout/elastic/num_ready_replicas_min': num_ready_replicas.minimum,
                'rollout/elastic/num_ready_replicas_max': num_ready_replicas.maximum,
                'rollout/proxy/loop_cost': loop_cost.mean,
                'rollout/proxy/gmem_insufficient_rebalanced_count_total': gmem_insufficient_rebalanced_count.sum,
                'rollout/proxy/load_rebalanced_count_total': load_rebalanced_count.sum,
                'rollout/proxy/total_token_TPS': metrics['total_token_TPS'].mean,
                'rollout/proxy/total_processes_queries': metrics['total_processes_queries'].maximum,
            }
        except KeyError as e:
            return {}
