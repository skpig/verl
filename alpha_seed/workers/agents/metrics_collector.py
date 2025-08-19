import time
import traceback
from collections import defaultdict
from typing import Dict, List, Any, Optional
from dataclasses import asdict

import ray
from omegaconf import DictConfig

from alpha_seed.utils.server_client import is_local_ray_instance
from alpha_seed.workers.agents.monitoring import AgentWorkerStats


@ray.remote
class AgentMetricsCollector:
    """
    简单的Ray Actor，收集所有agent worker的统计数据并合并
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.worker_stats: Dict[str, AgentWorkerStats] = {}  # worker_id -> stats
        self.last_update_time: Dict[str, float] = defaultdict(float)  # worker_id -> timestamp
        self.worker_submission_delay = defaultdict(float)  # worker_id -> duration(s), worker调用到collector处理的时差
        self.start_time = time.time()

    def collect(self, worker_id: str, stats: AgentWorkerStats, submission_time: float):
        """接收单个worker的统计数据"""
        if submission_time < self.last_update_time[worker_id]:
            # out-dated due to out-of-order
            return
        now = time.time()
        self.worker_stats[worker_id] = stats
        self.last_update_time[worker_id] = submission_time
        self.worker_submission_delay[worker_id] = now - submission_time

    def get_collector_info(self) -> Dict[str, Any]:
        """获取collector自身的信息"""
        current_time = time.time()
        return {
            "uptime_seconds": current_time - self.start_time,
            "total_workers": len(self.worker_stats),
            "active_workers": len([w for w, t in self.last_update_time.items() if current_time - t <= 30]),
            "last_update_time": dict(self.last_update_time),
            "worker_submission_delay": dict(self.worker_submission_delay),
        }

    def get_basic_stats(self) -> Dict[str, Any]:
        """获取基础统计信息，用于实时监控"""
        try:
            total_workers = len(self.worker_stats)
            if total_workers == 0:
                return {}
            full_stats = AgentWorkerStats.merge(list(self.worker_stats.values()))

            return {
                "max_workers": self.config.rollout_server.agent.max_workers,
                "worker_max_concurrency": self.config.rollout_server.agent.worker_max_concurrency,
                "executor_class": self.config.rollout_server.agent.executor_class,
                "active_tasks": full_stats.task_stats.active_tasks,
                "pending_tasks": full_stats.task_stats.pending_tasks,
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

    def get_current_metrics(self, step: int) -> Dict[str, float | int]:
        ret = {}
        try:
            full_stats = AgentWorkerStats.merge(list(self.worker_stats.values()))

            # worker stats
            ret.update({
                "agent/worker/active_tasks": full_stats.task_stats.active_tasks,
                "agent/worker/pending_tasks": full_stats.task_stats.pending_tasks,
                "agent/worker/completed_tasks": full_stats.task_stats.completed_tasks,
                "agent/worker/failed_tasks": full_stats.task_stats.failed_tasks,
            })

            # agent task stats
            for task_type, count in dict(full_stats.task_stats.agent_type_complete).items():
                ret.update({f"agent/task/{task_type}/complete_num": count})
            for task_type, perf in dict(full_stats.task_type_perf_metrics).items():
                ret.update({
                    f"agent/task/{task_type}/wait_time_avg": perf.avg_waiting_time,
                    f"agent/task/{task_type}/wait_time_max": perf.max_waiting_time,
                    f"agent/task/{task_type}/wait_time_min": perf.min_waiting_time,
                    f"agent/task/{task_type}/exec_time_avg": perf.avg_execution_time,
                    f"agent/task/{task_type}/exec_time_max": perf.max_execution_time,
                    f"agent/task/{task_type}/exec_time_min": perf.min_execution_time,
                    f"agent/task/{task_type}/tool_call_time_avg": perf.avg_tool_call_time,
                    f"agent/task/{task_type}/tool_call_time_max": perf.max_tool_call_time,
                    f"agent/task/{task_type}/tool_call_time_min": perf.min_tool_call_time,
                    f"agent/task/{task_type}/tool_call/success": perf.tool_success,
                    f"agent/task/{task_type}/tool_call/error": perf.tool_error,
                    f"agent/task/{task_type}/tool_call/retries": perf.tool_retries,
                    f"agent/task/{task_type}/tool_call/max_attempts_exceeds": perf.tool_max_attempts_exceeds,
                    f"agent/task/{task_type}/llm_call/success": perf.llm_success,
                })

            # agent tool stats
            for tool_name, stats in dict(full_stats.tool_use_perf_metrics).items():
                ret.update({
                    f"agent/tool/{tool_name}/success_count": stats.success_count,
                    f"agent/tool/{tool_name}/error_count": stats.error_count,
                    f"agent/tool/{tool_name}/retried_count": stats.retried_count,
                    f"agent/tool/{tool_name}/max_attempts_exceeds_count": stats.max_attempts_exceeds,
                    f"agent/tool/{tool_name}/time_avg": stats.avg_time,
                })

        except Exception as e:
            traceback.print_exc()
        return ret


def init_agent_metrics_collector(config, stable_pool_name):
    resources = {}
    if stable_pool_name and not is_local_ray_instance():
        resources = {stable_pool_name: 1}
    max_concurrency = config.rollout_server.agent.max_workers * 2 + 5  # 额外5个监控用
    return AgentMetricsCollector.options(name="AgentMetricsCollector",
                                         get_if_exists=True,
                                         max_concurrency=max_concurrency,
                                         resources=resources).remote(config)  # noqa


def get_agent_metrics_collector() -> AgentMetricsCollector:
    return ray.get_actor("AgentMetricsCollector")  # noqa
