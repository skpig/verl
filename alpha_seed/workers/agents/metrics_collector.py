import time
import traceback
from collections import defaultdict
from typing import Dict, List, Any, Optional
from dataclasses import asdict

import ray

from alpha_seed.utils.server_client import is_local_ray_instance
from alpha_seed.workers.agents.monitoring import AgentWorkerStats


@ray.remote
class AgentMetricsCollector:
    """
    简单的Ray Actor，收集所有agent worker的统计数据并合并
    """

    def __init__(self):
        self.worker_stats: Dict[str, AgentWorkerStats] = {}  # worker_id -> stats
        self.last_update_time: Dict[str, float] = defaultdict(float)  # worker_id -> timestamp
        self.start_time = time.time()

    def collect(self, worker_id: str, stats: AgentWorkerStats, submission_time: float):
        """接收单个worker的统计数据"""
        if submission_time < self.last_update_time[worker_id]:
            # out-dated due to out-of-order
            return
        self.worker_stats[worker_id] = stats
        self.last_update_time[worker_id] = submission_time

    def get_collector_info(self) -> Dict[str, Any]:
        """获取collector自身的信息"""
        current_time = time.time()
        return {
            "uptime_seconds": current_time - self.start_time,
            "total_workers": len(self.worker_stats),
            "active_workers": len([w for w, t in self.last_update_time.items() if current_time - t <= 30])
        }

    def get_basic_stats(self) -> Dict[str, Any]:
        """获取基础统计信息，用于实时监控"""
        try:
            total_workers = len(self.worker_stats)
            full_stats = AgentWorkerStats.merge(list(self.worker_stats.values()))

            return {
                "workers": f"{total_workers}",
                "tasks": f"{full_stats.task_stats.active_tasks}/{full_stats.task_stats.pending_tasks}",
                "completed_tasks": full_stats.task_stats.completed_tasks,
                "failed_tasks": full_stats.task_stats.failed_tasks,
                "avg_throughput": full_stats.perf_metrics.throughput_per_second,
                "agent_type_complete": dict(full_stats.task_stats.agent_type_complete),
                "agent_type_active": dict(full_stats.task_stats.agent_type_active),
                "agent_type_pending": dict(full_stats.task_stats.agent_type_pending),
                "agent_type_perf": {
                    k: asdict(v) for k, v in full_stats.task_type_perf_metrics.items()
                },
                "tool_use_perf": {
                    k: asdict(v) for k, v in full_stats.tool_use_perf_metrics.items()
                },
            }
        except Exception as e:
            return {"error": str(e), 'traceback': traceback.format_exc()}


def init_agent_metrics_collector(stable_pool_name):
    resources = {}
    if stable_pool_name and not is_local_ray_instance():
        resources = {stable_pool_name: 1}
    return AgentMetricsCollector.options(name="AgentMetricsCollector", get_if_exists=True,
                                         resources=resources).remote()  # noqa


def get_agent_metrics_collector() -> AgentMetricsCollector:
    return ray.get_actor("AgentMetricsCollector")  # noqa
