import asyncio
import inspect
import random
import threading
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Union

import numpy as np
import ray
import torch
from ray import ObjectRef
from ray.exceptions import ActorDiedError

from alpha_seed.utils.profile.timeline import Tracer, CompleteEvent, CounterEvent
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManager
from alpha_seed.workers.streaming_service.streaming_rollout import RemoteAsyncXPerfGPTRollout
from alpha_seed.workers.xperf_rollout.component.query import Query
from verl.single_controller.ray import RayWorkerGroup
from verl.single_controller.ray.base import func_generator
from verl.single_controller.ray.replicated_worker_group import ReplicatedRayWorkerGroup, ScalingRayWorkerGroup
from verl import DataProto

from alpha_seed.workers.streaming_service.auto_scaling import MetricSource, SeriesMetrics
from alpha_seed.workers.xperf_rollout.session import LoadMetric


def wait_ignore_actor_died(refs: List[ObjectRef]):
    for f in refs:
        try:
            ray.get(f)
        except ActorDiedError as e:
            caller = inspect.stack()[1].frame.f_code.co_name
            print(f"actor({e.actor_id}) died at function({caller}). ignore this as this is expected.")


class _MetricSourceImpl(MetricSource):

    def __init__(self, request_manager: RequestManager):
        self.req_mgr = request_manager

    def get_recent_series_metrics(self, recent_n: int) -> SeriesMetrics:
        assert recent_n >= 1, f"need to retrieve at least one sample, got({recent_n})"
        busy_ratio = ray.get(self.req_mgr.get_recent_step_busy_ratio.remote(recent_n))
        if busy_ratio:
            print(f"get recent busy% from request mgr got {[int(r * 100) for _, r in busy_ratio]}%")
        return SeriesMetrics(
            steps=[global_step for global_step, _ in busy_ratio],
            metrics=[ratio for _, ratio in busy_ratio],
        )


def split_by_indices(big_worker_group: RayWorkerGroup, indices: List[List[int]],
                     original_class_name) -> List['RayWorkerGroup']:
    # indices: [[0, 1], [2, 3], ...] 外层是切的worker groups数，内层是每个worker_group取第几个index
    rollout_cls = big_worker_group.ray_cls_with_init.cls.raw_cls_dict[original_class_name]
    worker_groups = []
    for worker_indices in indices:
        workers = [big_worker_group._worker_names[i] for i in worker_indices]
        new_wg = RayWorkerGroup.from_detached(workers, big_worker_group.ray_cls_with_init)
        # 参考RayWorkerGroup.spawn，重新给detached worker bind回Worker的方法
        new_wg._bind_worker_method(rollout_cls, func_generator)
        new_wg.sub_cls_name = big_worker_group.sub_cls_name
        worker_groups.append(new_wg)
    return worker_groups


# 将一个大的world拆成N个小world，每个world只含1个DP group
class FixedReplicatedRayWorkerGroupAdapter(ReplicatedRayWorkerGroup):

    def __init__(self, wg_with_dp: RayWorkerGroup, tp_size: int, original_class_name: str):
        # 不支持scale，initializer传None
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

    def set_dead_callback(self, fn):
        pass

    def get_alive_worker_groups(self):
        return self.wgs

    def get_ready_worker_groups(self):
        return self.wgs


class RolloutWorkerGroupProxy(_MetricSourceImpl):
    """
    负责代理底下N个replicas的请求分发、负载平衡
    负责请求状态同步到request pool
    """

    def __init__(self, replicas: Union[ReplicatedRayWorkerGroup, ScalingRayWorkerGroup], actor_addresses: List[str],
                 request_manager_name: str):
        self.request_manager: RequestManager = ray.get_actor(f'RequestManager/{request_manager_name}')  # noqa
        super().__init__(self.request_manager)
        self.replicas = replicas
        self.actor_addresses = actor_addresses
        self._tracer = Tracer.get_instance()
        self._update_worker_start_ts = 0
        self._update_worker_finished_ts = 0
        self._request_manager_name = request_manager_name

        self.replicas.set_dead_callback(self._worker_group_dead_callback)
        self._loop_should_stop = threading.Event()
        self._loop_thread = threading.Thread(target=self._dispatch_loop,
                                             name=f'{request_manager_name}-rollout-wg-proxy-dispatch-loop',
                                             daemon=True)
        self._loop_thread.start()

    @property
    def world_size(self):
        # 按照设定的replicas数计算world_size，而不管是否已完成初始化
        alive_worker_groups: Dict[str, RayWorkerGroup] = self.replicas.get_alive_worker_groups()
        return sum(w.world_size for w in alive_worker_groups.values())

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

    def _worker_group_dead_callback(self, worker_group_ids: List[str]):
        # worker group任意死了之后，通知request manager将运行中的请求释放掉
        alive_worker_group_ids = self.replicas.ready_worker_group_ids
        self.request_manager.handle_stale_requests.remote(alive_worker_group_ids)

    def _dispatch_loop(self):
        print('start background dispatch loop')

        # 初始concurrency 按照总量平分给每个ready replica
        pending_size = ray.get(self.request_manager.get_pending_size.remote())
        num_ready_replicas = len(self.replicas.ready_worker_group_ids)
        max_concurrency = 512

        poll_interval = 1
        while True:
            if self._loop_should_stop.is_set():
                break
            time.sleep(poll_interval)

            # 纪录负载指标
            loads = {}

            # 只将请求dispatch给ready worker group，每次循环都是最新的ready状态
            # 在dispatch过程中，worker group死了也没关系，这个request会之后被标记为stale
            t0 = time.time()
            for engine_id, wg in self.replicas.get_ready_worker_groups().items():
                wg: Union[RemoteAsyncXPerfGPTRollout, AsyncActorRolloutRefWorker]  # type anno
                try:
                    # 1. collect intermediate result
                    # get result(including partial) from engine, update to centralized request pool
                    queries: List[Query] = wg.get_all_queries(self._request_manager_name)
                    if len(queries) > 0:
                        self.request_manager.update_intermediate_queries.remote(queries, engine_id)

                    # 2. send new request to worker group (engine)
                    load: LoadMetric = wg.get_load_metrics()[0]  # noqa, dp_size always =1
                    gmem_insufficient = load.num_waiting > 0
                    # 如果engine目前并发已经达到这里自适应的max_concurrency，但不等于engine负载已经满了，
                    # 根据kv cache util自适应所以这里额外补发extra个请求，下一轮如果有扩容，会重新自动平衡
                    extra = self._get_adaptive_extra_num(load.kv_cache_util)

                    # note(hongbin): 始终让engine处于一个固定满并发的状态即可，减少动态插入新的具体进行prefill打断decode的case
                    short = max_concurrency - load.num_prefilling - load.num_decoding
                    # 如果gmem不够了就不发了
                    if short > 0 and not gmem_insufficient:
                        queries: List[Query] = ray.get(
                            self.request_manager.get_next_pending_requests.remote(short, engine_id))
                        if len(queries) > 0:
                            for q in queries:
                                # query的分类，区分一下是hybrid_rollout/standalone_rollout/validation，避免共用engine时不知道怎么update回去对应的来源
                                q.meta_info['query_type'] = self._request_manager_name
                            wg.add_inflight_queries(queries)
                            print(
                                f"dispatch {len(queries)}/{pending_size} queries from({self._request_manager_name}) "
                                f"to wg({engine_id}, pending={load.num_pending}, P={load.num_prefilling}/D={load.num_decoding}, kv={load.kv_cache_util:.2f})"
                            )

                    loads[engine_id] = load

                except ray.exceptions.ActorDiedError as e:
                    # ignore actor died error, underlying replicated worker group will handle
                    # worker group and actors lifecycle
                    pass

            # TODO: 识别卡住太久的query object，从engine中主动释放掉
            # 1. request manager内自动管理staleness和每个query的gen速度，卡住的或者太慢的自动释放掉
            # proxy拉取stale清单，向engine发送release请求，减少浪费算力
            # 从engine将query update回request manager时，判断是否属于当前所assigned engine id

            pending_size = ray.get(self.request_manager.get_pending_size.remote())  # noqa: for py-spy
            num_ready_replicas = len(self.replicas.ready_worker_group_ids)  # noqa: for py-spy

            loop_cost = time.time() - t0  # noqa: for py-spy
            self._trace_load_metrics(loads)

    def update_standalone_worker(self, role) -> List[ObjectRef]:
        # update转发给所有alive的worker，不用管其是否ready，一开始肯定不ready，需要update weights后才会ready
        self._update_worker_start_ts = time.time() * 1e6
        futs = []
        for wg in self.replicas.get_alive_worker_groups().values():
            fut = wg.update_standalone_worker(role)
            futs.extend(fut)
        # 这里直成同步等update完成，因为caller也是要等这个update完了才会进行下一步，这样就可以在这里把ActorDiedError也一起处理了
        for f in futs:
            try:
                ray.get(f)
            except ActorDiedError:
                # ignore dead actors during the update. This worker group will not be included in the next dispatching loop
                pass
        # 返回一个占位符即可
        return [ray.put(None)]

    def update_standalone_worker_end(self):
        # 通知server侧结束参数拉取
        # 任意一个client通知即可，调用此方法时需要确保所有standalone rollout worker已经同步完参数
        while True:
            try:
                all_alive_workers = list(self.replicas.get_alive_worker_groups().values())
                if len(all_alive_workers) == 0:
                    raise RuntimeError("no alive workers during weights update or all actors died")
                # any worker group sending will be ok
                wg = random.choice(all_alive_workers)
                wg.update_standalone_worker_end()
            except ActorDiedError:
                # do nothing when actor dies unfortunately, try next run
                time.sleep(0.1)
            else:
                break
        self._update_worker_finished_ts = time.time() * 1e6
        evt = CompleteEvent(
            pid='RolloutProxy',
            tid='update',
            cat='update weights',
            name='update weights',
            ts=self._update_worker_start_ts,
            dur=self._update_worker_finished_ts - self._update_worker_start_ts,
        )
        self._tracer.trace(evt)

    def stop_server_before_weights_update(self):
        futs = []
        for wg in self.replicas.get_alive_worker_groups().values():
            fut = wg.stop_server_before_weights_update_non_blocking()
            futs.extend(fut)
        wait_ignore_actor_died(futs)

    def restart_server_after_weights_update(self):
        futs = []
        for wg in self.replicas.get_alive_worker_groups().values():
            fut = wg.restart_server_after_weights_update_non_blocking()
            futs.extend(fut)
        wait_ignore_actor_died(futs)

    def stop(self):
        self._loop_should_stop.set()
        self._loop_thread.join()

    def _get_adaptive_extra_num(self, kv_cache_util: float) -> int:
        # 简单根据engine的kv cache决定着一轮发多少个请求
        if kv_cache_util < 0.5:
            return 32
        elif kv_cache_util < 0.7:
            return 16
        elif kv_cache_util < 0.8:
            return 8
        elif kv_cache_util < 0.9:
            return 4
        elif kv_cache_util < 0.95:
            return 2
        else:
            return 0

    def _trace_load_metrics(self, loads: Dict[str, LoadMetric]):
        for wg_id, metric in loads.items():
            evt = CounterEvent(
                name='load metrics',
                pid=f'{self._request_manager_name} {wg_id}',
                ts=metric.ts * 1e6,
                data={
                    'kv cache%': metric.kv_cache_util,
                    'prefilling': metric.num_prefilling,
                    'decoding': metric.num_decoding,
                    'pending': metric.num_pending,
                    'waiting': metric.num_waiting,
                },
            )
            self._tracer.trace(evt)
