import os
import queue
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import ray
from ray import ObjectRef
from mono_rl.single_controller.ray import RayWorkerGroup

from mono_rl.single_controller.ray.replicated_worker_group import ReplicatedRayWorkerGroup, ScalingRayWorkerGroup

from alpha_seed.utils.profile.timeline import Tracer, CompleteEvent, CounterEvent
from alpha_seed.utils.server_client import is_local_ray_instance


@dataclass
class ScalePolicyConfig:
    metrics_sampling_seconds: float
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_wait: float
    scale_down_wait: float
    min_replicas: int
    max_replicas: int


@dataclass
class TimeSeriesMetrics:
    timestamps: List[float]
    metrics: List[float]


class MetricSource:
    """
    负责提供周期性观测指标用于计算扩缩容动作
    """

    def get_recent_time_series_metrics(self, recent_seconds: float) -> TimeSeriesMetrics:
        raise NotImplementedError


class HorizontalAutoScaling:
    """
    管理replicated worker group的副本数，通过给定metrics检测方法确定scale数量。
    暂时一起接管cluster资源，观察集群资源
    """

    def __init__(self, replicas: ReplicatedRayWorkerGroup, worker_pool_name: str, config: ScalePolicyConfig,
                 metric_source: MetricSource):
        self.replicas = replicas
        self.group_name = self.replicas.resource_unit.name_prefix  # 要伸缩的group的名字
        self.group_unit_size = self.replicas.resource_unit.world_size  # 要伸缩的group的一个unit的大小(i.e. 一个dp的大小)
        self.worker_pool_name = worker_pool_name
        self.config = config
        self.metric_source = metric_source
        self._scale_up_down_refs_queue: queue.Queue[List[ObjectRef]] = queue.Queue()
        self._is_local_ray_cluster = is_local_ray_instance()
        self._tracer: Optional[Tracer] = None  # _scaling_loop 专用tracer

        threading.Thread(target=self._scaling_loop,
                         daemon=True,
                         name=f'scaling-loop/{worker_pool_name}/{self.group_name}').start()
        threading.Thread(target=self._scaling_materializing,
                         daemon=True,
                         name=f'scaling-materialize/{worker_pool_name}/{self.group_name}').start()

    def _scaling_loop(self):
        self._tracer = Tracer.get_instance()
        check_interval = self.config.metrics_sampling_seconds
        loop_count = 0
        while True:
            time.sleep(check_interval)
            loop_count += 1

            try:
                # 检查是否需要扩容
                should_scale_up, num_scale_up = self.should_scale_up()
                if should_scale_up and num_scale_up > 0:
                    num_able_scale_up = min(num_scale_up, self.max_scale_up_available())
                    if num_able_scale_up > 0:
                        with self._tracing('scale_up', num_able_scale_up):
                            futs = self.replicas.scale_up(num_able_scale_up)
                            self._scale_up_down_refs_queue.put(futs)
                            print(f"[{time.ctime()}] scale up {num_able_scale_up} more replica "
                                  f"on pool({self.worker_pool_name})")
                            # 忽略这里返回的futs，因为等也没用，就让他们后台自己跑
                            alive_num_replicas = len(self.replicas.alive_worker_group_ids)
                            print(f"scale up done on pool({self.worker_pool_name}), current {alive_num_replicas=}")
                    else:
                        # do nothing, wait for next turn
                        print(f"[{time.ctime()}] try to scale up {num_scale_up} on pool({self.worker_pool_name}) "
                              f"more replica, but underlying resource is not enough. wait for cluster HPA ready")

                    # 如果可扩，则不再判断是否要缩容
                    continue

                should_scale_down, num_scale_down = self.should_scale_down()
                if should_scale_down and num_scale_down > 0:
                    with self._tracing('scale_down', num_scale_down):
                        futs = self.replicas.scale_down(num_scale_down)
                        self._scale_up_down_refs_queue.put(futs)
                        print(f"[{time.ctime()}] scale down {num_scale_down} replica on pool({self.worker_pool_name})")
                        alive_num_replicas = len(self.replicas.alive_worker_group_ids)
                        print(f"scale down done on pool({self.worker_pool_name}), current {alive_num_replicas=}")
            except Exception as e:
                print(f'got exception during scaling resource loop, ignore this run')
                traceback.print_exc()

    def _scaling_materializing(self):
        # 不断把所有refs消费掉，既不让他们变成unhandled actor task，也不会无限积压
        # 暂时假设refs一定会结束，不做cancel处理
        # 如有用可以顺便用来做tracing
        remaining = set()
        while True:
            remaining_in_queue = self._scale_up_down_refs_queue.qsize()  # noqa: py-spy
            try:
                new_from_queue = self._scale_up_down_refs_queue.get(timeout=2.)
            except queue.Empty:
                new_from_queue = []
            for ref in new_from_queue:
                remaining.add(ref)

            if not remaining:
                time.sleep(1)
                continue

            remaining_size = len(remaining)
            done, not_done = ray.wait(list(remaining), num_returns=remaining_size, timeout=2.)
            for obj in done:
                try:
                    ray.get(obj)
                except Exception as e:
                    # ignore everything
                    pass
                finally:
                    remaining.discard(obj)

    @contextmanager
    def _tracing(self, event_name, num_replicas):
        t0 = time.time() * 1e6
        yield
        t1 = time.time() * 1e6
        original_dur = t1 - t0
        dur = max(1e6, original_dur)  # 最小显示1s的方块，避免找不到
        evt = CompleteEvent(pid=self._tracer_pid,
                            tid=0,
                            name=event_name,
                            cat=event_name,
                            ts=t0,
                            dur=dur,
                            args={
                                'num_replicas': num_replicas,
                                'original_dur_us': original_dur,
                            })
        self._tracer.trace(evt)

    @contextmanager
    def _tracing_each(self, object_name):
        t0 = time.time() * 1e6
        yield
        t1 = time.time() * 1e6
        evt = CompleteEvent(pid=self._tracer_pid, tid=0, name=object_name, cat=object_name, ts=t0, dur=t1 - t0)
        self._tracer.trace(evt)

    def _trace_scaling_metrics(self, direction: str, metrics: TimeSeriesMetrics):
        ts = metrics.timestamps[-1]
        val = metrics.metrics[-1]
        evt = CounterEvent(name=f'scale {direction}', pid=self._tracer_pid, ts=ts * 1e6, data={
            'current': val,
        })
        self._tracer.trace(evt)

    def should_scale_up(self):
        # 先暂时把策略都实现在这个类里面

        # hard limit
        target_replicas = self.replicas.target_num_replicas
        if target_replicas >= self.config.max_replicas:
            return False, 0

        # 持续一段时间超过threshold，则scale up
        recent_metrics = self.metric_source.get_recent_time_series_metrics(int(self.config.scale_up_wait))

        if not recent_metrics.metrics:
            return False, 0

        self._trace_scaling_metrics('up', recent_metrics)

        # 针对stable rollout丢失的情形，缺少就立刻scale_up
        if self.config.scale_up_threshold == -1:
            return True, self.config.max_replicas - target_replicas

        # 过去观测的窗口每个值都超过阈值则scale up
        # 计算理论应该承载并发度
        current_target_concurrency = self.config.scale_up_threshold * target_replicas
        # 实际观测总并发度
        latest_real_concurrency = recent_metrics.metrics[-1]
        should = all(v > current_target_concurrency for v in recent_metrics.metrics)
        # 按线性计算将实际并发度均摊到每个replica应有的并发度时需要scale up的replicas数
        num_scale_up = latest_real_concurrency // self.config.scale_up_threshold - target_replicas  # 线性scale
        return should, min(num_scale_up, self.config.max_replicas - target_replicas)

    def should_scale_down(self):
        # hard limit
        target_replicas = self.replicas.target_num_replicas
        if target_replicas <= self.config.min_replicas:
            return False, 0

        # 持续一段时间超过threshold，则scale up
        recent_metrics = self.metric_source.get_recent_time_series_metrics(int(self.config.scale_down_wait))
        if not recent_metrics.metrics:
            return False, 0

        self._trace_scaling_metrics('down', recent_metrics)

        # 针对stable不需要metric_source的情形
        if self.config.scale_down_threshold == -1:
            return False, 0

        # 过去观测的窗口每个值都超过阈值则scale up
        # 计算理论应该承载并发度
        current_target_concurrency = self.config.scale_down_threshold * target_replicas
        # 实际观测总并发度
        latest_real_concurrency = recent_metrics.metrics[-1]
        # 过去观测的窗口每个值都超过阈值
        should = all(v < current_target_concurrency for v in recent_metrics.metrics)
        num_scale_down = latest_real_concurrency // self.config.scale_down_threshold - target_replicas  # 线性scale
        return should, min(abs(num_scale_down), abs(self.config.min_replicas - target_replicas))

    def resource_available(self) -> bool:
        # 只检查资源数量，不检查资源拓扑，最终的拓扑检查交给实例创建时的PG调度
        unit = self.replicas.resource_unit
        num_gpus_required = unit.world_size
        total_available = ray.available_resources()
        num_gpus_available = total_available.get('GPU', 0)
        if not self._is_local_ray_cluster:
            num_role_placeholder = total_available.get(self.worker_pool_name, 0)
        else:
            # 在local ray cluster上， role placeholder 不一定有，所以这里直接给固定1表示跳过role placeholder检测
            num_role_placeholder = 1
        # note(lixiang): ray现在的版本暂不支持每个node可用资源检查，所以这里只检查数量，不做拓扑检查
        return num_gpus_available >= num_gpus_required and num_role_placeholder >= 1

    def max_scale_up_available(self) -> int:
        unit = self.replicas.resource_unit
        num_gpus_required = unit.world_size
        total_available = ray.available_resources()
        num_gpus_available = total_available.get('GPU', 0)

        # note(lixiang): ray现在的版本暂不支持每个node可用资源检查，所以这里只检查数量，不做拓扑检查
        return int(num_gpus_available // num_gpus_required)

    @property
    def _tracer_pid(self) -> str:
        return f"{self.group_name} HAS.{self.worker_pool_name}"
