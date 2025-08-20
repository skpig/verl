import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, Any, List, Type, Iterable, Container, Optional
import threading

import torch

from alpha_seed.utils.profile.timeline import CompleteEvent
from alpha_seed.workers.agents.monitor_ctx import current_agent_tracker, set_current_agent_tracker
from alpha_seed.workers.agents.tool import ToolResult
from mono_rl import DataProto


def avg(l: list) -> float:
    if not l:
        return 0
    return sum(l) / len(l)


@dataclass
class AgentWorkerTaskStats:
    worker_id: str

    # 基础统计一个AsyncAgent/ThreadedAgent实例算一个task
    total_tasks: int = 0
    pending_tasks: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    # 按agent类型统计 - 使用动态名称
    # {name -> count}
    agent_type_active: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    agent_type_pending: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    agent_type_complete: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # 时间戳
    last_updated: float = field(default_factory=time.time)


@dataclass
class AgentWorkerQueueStats:
    # 队列状态
    semaphore_available: int = 0
    semaphore_total: int = 0
    thread_pool_active: int = 0
    thread_pool_total: int = 0


@dataclass
class TaskTypePerfMetrics:
    """单个agent类型的性能指标"""
    agent_name: str
    # create
    avg_creation_time: float = 0.0
    min_creation_time: float = 0.0
    max_creation_time: float = 0.0
    # wait
    avg_waiting_time: float = 0.0
    min_waiting_time: float = 0.0
    max_waiting_time: float = 0.0
    # exec
    avg_execution_time: float = 0.0
    min_execution_time: float = 0.0
    max_execution_time: float = 0.0

    # tool call time(不区分tool)
    avg_tool_call_time: float = 0.0
    min_tool_call_time: float = 0.0
    max_tool_call_time: float = 0.0

    total_executions: int = 0  # 上面统计的原始样本的长度，不是task执行的个数
    throughput_per_second: float = 0.0

    # tool/env/llm
    tool_success: int = 0
    tool_error: int = 0
    tool_retries: int = 0
    tool_max_attempts_exceeds: int = 0
    llm_success: int = 0

    @staticmethod
    def merge(task_type: str, metrics_list: List['TaskTypePerfMetrics']) -> 'TaskTypePerfMetrics':
        """合并同一task_type的多个性能指标"""
        if not metrics_list:
            return TaskTypePerfMetrics(agent_name=task_type)

        # 收集所有时间数据
        creation_time_accumulate = 0.0
        waiting_time_accumulate = 0.0
        exec_time_accumulate = 0.0
        total_executions = sum(m.total_executions for m in metrics_list)
        tool_call_time_accumulate = 0.0
        total_tool_calls = sum(m.tool_success for m in metrics_list)

        for metrics in metrics_list:
            # 根据执行次数加权收集时间数据
            if metrics.total_executions > 0:
                creation_time_accumulate += metrics.avg_creation_time * metrics.total_executions
                waiting_time_accumulate += metrics.avg_waiting_time * metrics.total_executions
                exec_time_accumulate += metrics.avg_execution_time * metrics.total_executions
            if metrics.tool_success:
                tool_call_time_accumulate += metrics.avg_tool_call_time * metrics.tool_success

        return TaskTypePerfMetrics(
            agent_name=task_type,
            avg_creation_time=creation_time_accumulate / total_executions if total_executions else 0,
            min_creation_time=min(m.min_creation_time for m in metrics_list),
            max_creation_time=max(m.max_creation_time for m in metrics_list),
            avg_waiting_time=waiting_time_accumulate / total_executions if total_executions else 0,
            min_waiting_time=min(m.min_waiting_time for m in metrics_list),
            max_waiting_time=max(m.max_waiting_time for m in metrics_list),
            avg_execution_time=exec_time_accumulate / total_executions if total_executions else 0,
            min_execution_time=min(m.min_execution_time for m in metrics_list),
            max_execution_time=max(m.max_execution_time for m in metrics_list),
            avg_tool_call_time=tool_call_time_accumulate / total_tool_calls if total_tool_calls else 0,
            min_tool_call_time=min(m.min_tool_call_time for m in metrics_list),
            max_tool_call_time=max(m.max_tool_call_time for m in metrics_list),
            total_executions=total_executions,
            throughput_per_second=sum(m.throughput_per_second for m in metrics_list),
            tool_success=total_tool_calls,
            tool_error=sum(m.tool_error for m in metrics_list),
            tool_retries=sum(m.tool_retries for m in metrics_list),
            tool_max_attempts_exceeds=sum(m.tool_max_attempts_exceeds for m in metrics_list),
            llm_success=sum(m.llm_success for m in metrics_list),
        )


@dataclass
class ToolUsePerfMetrics:
    tool_name: str  # 跟class name对应
    success_count: int = 0  # 成功的次数(最终成功了都算1)
    retried_count: int = 0  # 环境失败重试的次数(除了第一次，每重试1次+1，无论最后重试多少次以及是否失败)
    error_count: int = 0  # 环境导致的失败的次数，因为llm给错输入导致的错误不算
    max_attempts_exceeds: int = 0  # 重试达到最大次数的call次数(通常用来表示这个环境彻底坏掉)
    avg_time: float = 0.0  # 平均时间

    @staticmethod
    def merge(tool_name: str, metrics_list: List['ToolUsePerfMetrics']) -> 'ToolUsePerfMetrics':
        if not metrics_list:
            return ToolUsePerfMetrics(tool_name=tool_name)

        total_success = sum(m.success_count for m in metrics_list)
        return ToolUsePerfMetrics(
            tool_name=tool_name,
            success_count=total_success,
            retried_count=sum(m.retried_count for m in metrics_list),
            error_count=sum(m.error_count for m in metrics_list),
            max_attempts_exceeds=sum(m.max_attempts_exceeds for m in metrics_list),
            # 按成功次数加权
            avg_time=sum(m.avg_time * m.success_count / total_success for m in metrics_list) if total_success else 0,
        )


@dataclass
class WorkerPerfMetrics:
    """Worker级别的性能指标"""
    avg_execution_time: float = 0.0
    avg_creation_time: float = 0.0
    throughput_per_second: float = 0.0  # task完成数吞吐(不是tokens)

    @staticmethod
    def merge(metrics_list: List['WorkerPerfMetrics']) -> 'WorkerPerfMetrics':
        """合并多个WorkerPerfMetrics，计算平均值"""
        if not metrics_list:
            return WorkerPerfMetrics()

        return WorkerPerfMetrics(avg_execution_time=sum(m.avg_execution_time for m in metrics_list) / len(metrics_list),
                                 avg_creation_time=sum(m.avg_creation_time for m in metrics_list) / len(metrics_list),
                                 throughput_per_second=sum(m.throughput_per_second for m in metrics_list))


@dataclass
class AgentWorkerStats:
    task_stats: AgentWorkerTaskStats
    queue_stats: AgentWorkerQueueStats
    perf_metrics: WorkerPerfMetrics = field(default_factory=WorkerPerfMetrics)
    task_type_perf_metrics: Dict[str, TaskTypePerfMetrics] = field(default_factory=TaskTypePerfMetrics)  # task_type ->
    tool_use_perf_metrics: Dict[str, ToolUsePerfMetrics] = field(default_factory=ToolUsePerfMetrics)  # tool_name ->

    @staticmethod
    def merge(stats: List['AgentWorkerStats']) -> 'AgentWorkerStats':
        """合并多个AgentWorkerStats"""
        if not stats:
            raise ValueError('stats need to be non-empty')

        # 合并task_stats
        merged_task_stats = AgentWorkerTaskStats(
            worker_id='',  # 合并后的统计没有特定worker_id
            total_tasks=sum(s.task_stats.total_tasks for s in stats),
            pending_tasks=sum(s.task_stats.pending_tasks for s in stats),
            active_tasks=sum(s.task_stats.active_tasks for s in stats),
            completed_tasks=sum(s.task_stats.completed_tasks for s in stats),
            failed_tasks=sum(s.task_stats.failed_tasks for s in stats),
            last_updated=max(s.task_stats.last_updated for s in stats) if stats else time.time())

        # 合并agent_type_stats
        for stat in stats:
            for agent_type, count in stat.task_stats.agent_type_complete.items():
                merged_task_stats.agent_type_complete[agent_type] += count
            for agent_type, count in stat.task_stats.agent_type_active.items():
                merged_task_stats.agent_type_active[agent_type] += count
            for agent_type, count in stat.task_stats.agent_type_pending.items():
                merged_task_stats.agent_type_pending[agent_type] += count

        # 合并queue_stats
        merged_queue_stats = AgentWorkerQueueStats(
            semaphore_available=sum(s.queue_stats.semaphore_available for s in stats),
            semaphore_total=sum(s.queue_stats.semaphore_total for s in stats),
            thread_pool_active=sum(s.queue_stats.thread_pool_active for s in stats),
            thread_pool_total=sum(s.queue_stats.thread_pool_total for s in stats))

        # 合并perf_metrics
        merged_perf_metrics = WorkerPerfMetrics.merge([s.perf_metrics for s in stats])

        # 合并task_type_perf_metrics
        task_type_data = defaultdict(list)
        for stat in stats:
            for task_type, metrics in stat.task_type_perf_metrics.items():
                task_type_data[task_type].append(metrics)
        merged_task_type_perf = {
            task_type: TaskTypePerfMetrics.merge(task_type, metrics_list)
            for task_type, metrics_list in task_type_data.items()
        }

        # merge tool_use
        tool_use_data = defaultdict(list)
        for stat in stats:
            for tool_name, metrics in stat.tool_use_perf_metrics.items():
                tool_use_data[tool_name].append(metrics)
        merged_tool_use_perf_metrics = {
            tool_name: ToolUsePerfMetrics.merge(tool_name, metrics_list)
            for tool_name, metrics_list in tool_use_data.items()
        }

        return AgentWorkerStats(task_stats=merged_task_stats,
                                queue_stats=merged_queue_stats,
                                perf_metrics=merged_perf_metrics,
                                task_type_perf_metrics=merged_task_type_perf,
                                tool_use_perf_metrics=merged_tool_use_perf_metrics)


class NullLLMCapturer:

    def capture_input(self, item):
        pass

    def capture(self, item, completion):
        pass


class LLMResultCapturer:

    def __init__(self):
        self.input_ids: List[int] = []
        self.output_ids: List[int] = []

    def capture_input(self, prompt: DataProto):
        if isinstance(prompt, DataProto):
            input_ids = prompt.batch['input_ids']
            attention_mask = prompt.batch['attention_mask']
            valid_input_len = torch.sum(attention_mask)
            prompt_ids = input_ids[0, -valid_input_len:].tolist()
            self.input_ids = prompt_ids

    def capture_output(self, completion: dict):
        # ignore if not dict
        if isinstance(completion, dict):
            if completion and 'choices' in completion:
                choices = completion['choices']
                if choices:
                    self.output_ids = choices[0].get('message', {}).get('raw_output_ids', [])


class NullToolCapturer:

    def capture_input(self, instance_id: str, parameter: dict):
        pass

    def capture_output(self, instance_id: str, parameter: dict, result):
        pass

    def capture_exception(self, e: Exception, tb: str):
        pass


class ToolResultCapturer:

    def __init__(self):
        from alpha_seed.workers.agents.tool import ToolResult
        self.instance_id: str = ''
        self.parameter: dict = {}
        self.result: Optional[ToolResult] = None

    def capture_input(self, instance_id: str, parameter: dict):
        self.instance_id = instance_id
        self.parameter = parameter

    def capture_output(self, result):
        self.result = result

    def capture_exception(self, e: Exception, tb: str):
        self.result = ToolResult(f"exception during tool call, e={e}\n{tb}", success=False)


class AgentTaskTracker:
    """单次agent执行的跟踪器，分别统计创建和执行时间"""

    def __init__(self, monitor: 'AgentWorkerMonitor'):
        from alpha_seed.workers.agents.handlers.base import AsyncAgent, ThreadedAgent
        self.monitor = monitor
        self.creation_start_time = None
        self.creation_finish_time = None
        self.execution_start_time = None
        self.execution_finish_time = None
        self.agent_cls = None
        self.agent: Optional[AsyncAgent | ThreadedAgent] = None

    @contextmanager
    def creation(self, agent_cls: Type):
        """统计agent创建时间"""
        if not self.monitor.enabled:
            yield
            return

        current_agent_tracker.set(self)
        set_current_agent_tracker(self)
        self.agent_cls = agent_cls
        self.creation_start_time = time.time()
        self.monitor.increment_pending_tasks(agent_cls.__name__)
        try:
            yield
        finally:
            if self.creation_start_time is not None:
                self.creation_finish_time = time.time()

    @contextmanager
    def execution(self, agent):
        """统计agent执行时间"""
        if not self.monitor.enabled:
            yield
            return

        self.agent = agent
        self.execution_start_time = time.time()

        self.monitor.increment_active_tasks(self.agent_cls.__name__)

        try:
            yield
            self.monitor.increment_completed_tasks(self.agent_cls.__name__)
        except Exception:
            self.monitor.increment_failed_tasks()
            raise
        finally:
            if self.execution_start_time is not None:
                self.execution_finish_time = time.time()
                self.monitor.record_task_completion(task_type=self.agent_cls.__name__,
                                                    creation_time=self.creation_finish_time - self.creation_start_time,
                                                    waiting_time=self.execution_start_time - self.creation_finish_time,
                                                    execution_time=self.execution_finish_time -
                                                    self.execution_start_time,
                                                    total_time=self.execution_finish_time - self.creation_start_time)
                self._make_trace_event()

    @contextmanager
    def llm_call(self, prompt: DataProto):
        if not self.monitor.enabled:
            yield NullLLMCapturer()
            return

        from alpha_seed.workers.agents.trajectory import make_llm_seg_pair

        start = time.time()
        capturer = LLMResultCapturer()
        capturer.capture_input(prompt)
        llm_start, llm_end = make_llm_seg_pair(start, capturer.input_ids)
        traj = self.agent.trajectory_factory.get()
        traj.append(llm_start)
        try:
            self.incr_llm_call()
            yield capturer
        finally:
            finish = time.time()
            llm_end.end_ts = finish
            llm_end.output_ids = capturer.output_ids
            traj.append(llm_end)

    @contextmanager
    def tool_call(self, tool_class_name: str, instance_id: str, parameter: dict):
        # 这个函数内务必特别小心不要搞出错误来，任何这里的异常都可能作为工具的输入到llm下一轮里
        if not self.monitor.enabled:
            yield NullToolCapturer()
            return

        from alpha_seed.workers.agents.trajectory import make_tool_seg_pair

        start = time.time()
        capturer = ToolResultCapturer()
        capturer.capture_input(instance_id, parameter)
        tool_start, tool_end = make_tool_seg_pair(start, instance_id, tool_class_name, parameter)
        traj = self.agent.trajectory_factory.get()
        traj.append(tool_start)
        try:
            yield capturer
        finally:
            finish = time.time()
            self.monitor.record_tool_call(task_type=self.agent_cls.__name__,
                                          tool_type=tool_class_name,
                                          call_time=finish - start)
            self.incr_tool_call_counter(
                tool_class_name,
                capturer.result.retries,
                capturer.result.retries == capturer.result.max_attempts - 1,
                capturer.result.success,
            )
            tool_end.end_ts = finish
            tool_end.set_result(capturer.result)
            traj.append(tool_end)

    def incr_llm_call(self):
        self.monitor.incr_llm_call(self.agent_cls.__name__)

    def incr_tool_call_counter(self, tool_class_name: str, retries: int, exceeded_max_attempts: bool, success: bool):
        self.monitor.incr_tool_call_counter(self.agent_cls.__name__, tool_class_name, retries, exceeded_max_attempts,
                                            success)

    def _make_trace_event(self):
        pass


class AgentWorkerMonitor:

    def __init__(self, worker_id: str, enabled: bool = True, window_size: int = 2048):
        self.worker_id = worker_id
        self.enabled = enabled

        # task 粒度的统计
        self.task_stats = AgentWorkerTaskStats(worker_id=worker_id)
        self.execution_times: deque = deque(maxlen=window_size)
        self.creation_times: deque = deque(maxlen=window_size)

        # {agent_cls -> [dur, ...]
        self.agent_type_execution_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.agent_type_waiting_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.agent_type_creation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.agent_type_total_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

        # task内的调用的统计(tool/env/llm/...)
        self.tool_call_success = defaultdict(lambda: defaultdict(int))  # {agent_cls -> tool_cls -> count}
        self.tool_call_error = defaultdict(lambda: defaultdict(int))  # {agent_cls -> tool_cls -> count}
        self.tool_call_retries = defaultdict(lambda: defaultdict(int))  # {agent_cls -> tool_cls -> count}
        # 统计超过最大尝试次数的tool call的数量
        self.tool_call_max_attempts_exceeds = defaultdict(lambda: defaultdict(int))  # {agent_cls -> tool_cls -> count}
        self.llm_call_success = defaultdict(int)  # {agent_cls -> count} 暂时没有区分模型的必要性
        # {agent_cls -> tool_cls -> [dur, ...]
        self.tool_call_times = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window_size)))

        self._lock = threading.Lock()
        self._tool_call_mutex = threading.Lock()

    def task_tracker(self) -> AgentTaskTracker:
        """
        返回一个执行tracker，支持分别统计创建和执行时间
        """
        return AgentTaskTracker(self)

    def get_task_stats(self) -> AgentWorkerTaskStats:
        return self.task_stats

    def get_performance_metrics(self) -> WorkerPerfMetrics:
        # 计算总体性能指标
        with self._lock:
            if self.execution_times:
                total_times = list(self.execution_times)
                avg_execution_time = sum(total_times) / len(total_times)
                throughput_per_second = len(total_times) / (sum(total_times) + 1e-6)
            else:
                avg_execution_time = 0.0
                throughput_per_second = 0.0

            if self.creation_times:
                creation_times = list(self.creation_times)
                avg_creation_time = sum(creation_times) / len(creation_times)
            else:
                avg_creation_time = 0.0

            return WorkerPerfMetrics(avg_execution_time=avg_execution_time,
                                     avg_creation_time=avg_creation_time,
                                     throughput_per_second=throughput_per_second)

    def get_agent_type_performance_metrics(self) -> Dict[str, TaskTypePerfMetrics]:
        """动态计算每种agent类型的性能指标"""
        with self._lock, self._tool_call_mutex:
            performance = {}

            for agent_name in self.agent_type_execution_times.keys():
                exec_times = list(self.agent_type_execution_times[agent_name])
                creation_times = list(self.agent_type_creation_times[agent_name])
                waiting_times = list(self.agent_type_waiting_times[agent_name])
                tool_calls_times = reduce(lambda a, b: a + b,
                                          [list(v) for v in self.tool_call_times[agent_name].values()], [])
                tool_calls_success = sum(self.tool_call_success[agent_name].values())
                tool_calls_error = sum(self.tool_call_error[agent_name].values())
                tool_calls_retries = sum(self.tool_call_retries[agent_name].values())
                tool_calls_max_attempts_exceeds = sum(self.tool_call_max_attempts_exceeds[agent_name].values())
                llm_calls_success = self.llm_call_success[agent_name]

                if not exec_times:
                    continue

                if not tool_calls_times:
                    tool_calls_times = [0]  # 给个默认值避免除0错误

                metrics = TaskTypePerfMetrics(
                    agent_name=agent_name,
                    avg_creation_time=avg(creation_times),
                    min_creation_time=min(creation_times),
                    max_creation_time=max(creation_times),
                    avg_waiting_time=avg(waiting_times),
                    min_waiting_time=min(waiting_times),
                    max_waiting_time=max(waiting_times),
                    avg_execution_time=avg(exec_times),
                    min_execution_time=min(exec_times),
                    max_execution_time=max(exec_times),
                    avg_tool_call_time=avg(tool_calls_times),
                    min_tool_call_time=min(tool_calls_times),
                    max_tool_call_time=max(tool_calls_times),
                    total_executions=len(exec_times),
                    throughput_per_second=len(exec_times) / (sum(exec_times) + 1e-6),
                    tool_success=tool_calls_success,
                    tool_error=tool_calls_error,
                    tool_retries=tool_calls_retries,
                    tool_max_attempts_exceeds=tool_calls_max_attempts_exceeds,
                    llm_success=llm_calls_success,
                )

                performance[agent_name] = metrics

            return performance

    def get_tool_use_performance_metrics(self) -> Dict[str, ToolUsePerfMetrics]:
        perf: Dict[str, ToolUsePerfMetrics] = {}  # tool_name ->
        call_time = defaultdict(list)  # tool_name ->
        with self._lock, self._tool_call_mutex:
            # succ
            for task_name, tool_calls in self.tool_call_success.items():
                for tool_name, count in tool_calls.items():
                    if tool_name not in perf:
                        perf[tool_name] = ToolUsePerfMetrics(tool_name=tool_name)
                    perf[tool_name].success_count += count
            # error
            for task_name, tool_calls in self.tool_call_error.items():
                for tool_name, count in tool_calls.items():
                    if tool_name not in perf:
                        perf[tool_name] = ToolUsePerfMetrics(tool_name=tool_name)
                    perf[tool_name].error_count += count
            # retries
            for task_name, tool_calls in self.tool_call_retries.items():
                for tool_name, count in tool_calls.items():
                    if tool_name not in perf:
                        perf[tool_name] = ToolUsePerfMetrics(tool_name=tool_name)
                    perf[tool_name].retried_count += count
            # max attempts exceeds
            for task_name, tool_calls in self.tool_call_max_attempts_exceeds.items():
                for tool_name, count in tool_calls.items():
                    if tool_name not in perf:
                        perf[tool_name] = ToolUsePerfMetrics(tool_name=tool_name)
                    perf[tool_name].max_attempts_exceeds += count

            # call time
            for task_name, tool_calls in self.tool_call_times.items():
                for tool_name, dur in tool_calls.items():
                    call_time[tool_name].extend(dur)

        for tool_name, metric in perf.items():
            metric.avg_time = avg(call_time.get(tool_name, []))

        return perf

    def increment_pending_tasks(self, agent_name: str):
        with self._lock:
            self.task_stats.pending_tasks += 1
            self.task_stats.agent_type_pending[agent_name] += 1

    def increment_active_tasks(self, agent_name: str):
        """增加活跃任务计数"""
        with self._lock:
            self.task_stats.active_tasks += 1
            self.task_stats.pending_tasks -= 1
            self.task_stats.agent_type_active[agent_name] += 1
            self.task_stats.agent_type_pending[agent_name] -= 1

    def increment_completed_tasks(self, agent_name: str):
        """增加完成任务计数和agent类型统计"""
        with self._lock:
            self.task_stats.completed_tasks += 1
            self.task_stats.agent_type_complete[agent_name] += 1
            self.task_stats.agent_type_active[agent_name] -= 1

    def increment_failed_tasks(self):
        """增加失败任务计数"""
        with self._lock:
            self.task_stats.failed_tasks += 1

    def incr_tool_call_counter(self, agent_cls: str, tool_cls: str, retries: int, exceeded_max_attempts: bool,
                               success: bool):
        with self._tool_call_mutex:
            if success:
                self.tool_call_success[agent_cls][tool_cls] += 1
            else:
                self.tool_call_error[agent_cls][tool_cls] += 1
            self.tool_call_retries[agent_cls][tool_cls] += retries
            if exceeded_max_attempts:
                self.tool_call_max_attempts_exceeds[agent_cls][tool_cls] += 1

    def incr_llm_call(self, agent_cls: str):
        with self._tool_call_mutex:
            self.llm_call_success[agent_cls] += 1

    def record_task_completion(self, task_type: str, creation_time: float, waiting_time, execution_time: float,
                               total_time: float):
        """记录任务完成的时间统计"""
        with self._lock:
            self.task_stats.active_tasks -= 1
            self.task_stats.total_tasks += 1

            # 记录时间统计
            self.execution_times.append(total_time)
            self.creation_times.append(creation_time)

            # 按agent类型记录
            self.agent_type_execution_times[task_type].append(execution_time)
            self.agent_type_waiting_times[task_type].append(waiting_time)
            self.agent_type_creation_times[task_type].append(creation_time)
            self.agent_type_total_times[task_type].append(total_time)

            # 更新时间戳
            self.task_stats.last_updated = time.time()

    def record_tool_call(self, task_type, tool_type, call_time: float):
        with self._tool_call_mutex:
            self.tool_call_times[task_type][tool_type].append(call_time)
