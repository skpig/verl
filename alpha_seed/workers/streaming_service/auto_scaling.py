import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import ray
from mono_rl.single_controller.ray import RayWorkerGroup

from mono_rl.single_controller.ray.replicated_worker_group import ReplicatedRayWorkerGroup, ScalingRayWorkerGroup

from alpha_seed.utils.profile.timeline import Tracer, CompleteEvent
from alpha_seed.utils.server_client import is_local_ray_instance


@dataclass
class ScalePolicyConfig:
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_wait: float
    scale_down_wait: float
    min_replicas: int
    max_replicas: int


@dataclass
class SeriesMetrics:
    steps: List[float]
    metrics: List[float]


class MetricSource:
    """
    负责提供周期性观测指标用于计算扩缩容动作
    """

    def get_recent_series_metrics(self, recent_n: int) -> SeriesMetrics:
        raise NotImplementedError


class HorizontalAutoScaling:
    """
    管理replicated worker group的副本数，通过给定metrics检测方法确定scale数量。
    暂时一起接管cluster资源，观察集群资源
    """

    def __init__(self, replicas: ScalingRayWorkerGroup, worker_pool_name: str, config: ScalePolicyConfig,
                 metric_source: MetricSource):
        self.replicas = replicas
        self.worker_pool_name = worker_pool_name
        self.config = config
        self.metric_source = metric_source
        self._recent_action_step = -1
        self._is_local_ray_cluster = is_local_ray_instance()
        self._tracer = Tracer.get_instance()

        threading.Thread(target=self._resource_loop, daemon=True, name=f'scaling-loop/{worker_pool_name}').start()

    def _resource_loop(self):
        check_interval = 5
        while True:
            time.sleep(check_interval)

            # 检查是否需要扩容
            if self.should_scale_up():
                if self.resource_available():
                    with self._tracing('scale_up'):
                        fut = self.replicas.scale_up(1)
                        print(f"scale up 1 more replica on pool({self.worker_pool_name})")
                        ray.get(fut)
                        alive_num_replicas = len(self.replicas.alive_worker_group_ids)
                        print(f"scale up done on pool({self.worker_pool_name}), current {alive_num_replicas=}")
                else:
                    # do nothing, wait for next turn
                    print("try to scale up 1 more replica, but underlying resource is not enough. "
                          "wait for cluster HPA ready")
            elif self.should_scale_down():
                with self._tracing('scale_down'):
                    fut = self.replicas.scale_up(-1)
                    print(f"scale down 1 replica on pool({self.worker_pool_name})")
                    ray.get(fut)
                    alive_num_replicas = len(self.replicas.alive_worker_group_ids)
                    print(f"scale down done on pool({self.worker_pool_name}), current {alive_num_replicas=}")

    @contextmanager
    def _tracing(self, event_name):
        t0 = time.time() * 1e6
        yield
        t1 = time.time() * 1e6
        evt = CompleteEvent(
            pid='HorizontalAutoScaling',
            tid=0,
            name=event_name,
            cat=event_name,
            ts=t0,
            dur=t1 - t0,
        )
        self._tracer.trace(evt)

    def should_scale_up(self):
        # 先暂时把策略都实现在这个类里面

        # hard limit
        if len(self.replicas) >= self.config.max_replicas:
            return False

        # 持续一段时间超过threshold，则scale up
        recent_metrics = self.metric_source.get_recent_series_metrics(int(self.config.scale_up_wait))
        if not recent_metrics.metrics:
            return False
        latest_step = recent_metrics.steps[-1]
        # 最近一个step scale过则跳过
        if self._recent_action_step >= latest_step:
            return False
        # 过去观测的窗口每个值都超过阈值则scale up
        ret = all(v > self.config.scale_up_threshold for v in recent_metrics.metrics)
        if ret is True:
            self._recent_action_step = latest_step
        return ret

    def should_scale_down(self):
        # hard limit
        if len(self.replicas) <= self.config.min_replicas:
            return False

        # 持续一段时间超过threshold，则scale up
        recent_metrics = self.metric_source.get_recent_series_metrics(int(self.config.scale_down_wait))
        if not recent_metrics.metrics:
            return False
        latest_step = recent_metrics.steps[-1]
        # 最近一个step scale过则跳过
        if self._recent_action_step >= latest_step:
            return False
        # 过去观测的窗口每个值都超过阈值则scale up
        ret = all(v < self.config.scale_down_threshold for v in recent_metrics.metrics)
        if ret is True:
            self._recent_action_step = latest_step
        return ret

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
