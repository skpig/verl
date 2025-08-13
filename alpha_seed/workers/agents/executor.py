import asyncio
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import cycle
from typing import Type, List, Dict, Any, Optional

import ray
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, AutoProcessor

from alpha_seed.utils.server_client import is_local_ray_instance
from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import TaskContext, GlobalState
from alpha_seed.workers.agents.handlers.base import AsyncAgent, functional_agent, ThreadedAgent
from alpha_seed.workers.agents.llm import OpenAIAsyncClient, OpenAIClient, DirectAsyncClient, DirectClient
from alpha_seed.workers.agents.metrics_collector import get_agent_metrics_collector, init_agent_metrics_collector
from alpha_seed.workers.agents.monitoring import (AgentWorkerMonitor, AgentWorkerTaskStats, AgentWorkerStats,
                                                  AgentWorkerQueueStats)
from mono_rl import DataProto


class AgentWorker:

    def __init__(self, worker_id: int, config: DictConfig, tokenizer: PreTrainedTokenizer, processor: AutoProcessor,
                 host, port, request_manager_name, loop: Optional[asyncio.AbstractEventLoop]):
        """
        worker_id: worker id to identify different workers
        config: root config
        host: llm server host
        port: llm server port
        request_manager_name: the corresponding request manager of rollout
        loop: asyncio event loop, available only in LocalExecutor mode
        """
        self.worker_max_concurrency = config.rollout_server.agent.worker_max_concurrency
        self.llm_request_concurrency = config.rollout_server.agent.llm_request_concurrency
        self.llm_timeout = config.rollout_server.timeout
        self._thread_executor = ThreadPoolExecutor(max_workers=self.worker_max_concurrency,
                                                   thread_name_prefix=f"agent-worker-{worker_id}")
        self.tokenizer = tokenizer
        self.async_tokenizer = AsyncTokenizer(tokenizer)
        self.processor = processor
        self.loop = loop

        # 根据配置选择使用Direct/OpenAI client
        if config.rollout_server.agent.direct_submit_query:
            self.llm = DirectAsyncClient(request_manager_name)
            self.sync_llm = DirectClient(request_manager_name)
        else:
            self.llm = OpenAIAsyncClient(host, port, self.llm_request_concurrency, self.llm_timeout)
            self.sync_llm = OpenAIClient(host, port, self.llm_timeout)

        self.concurrency_limit = asyncio.Semaphore(self.worker_max_concurrency)
        self.config = config
        self.global_state = GlobalState()

        # 添加监控
        self.monitor = AgentWorkerMonitor(
            worker_id=f'{request_manager_name}.{worker_id}',
            enabled=config.rollout_server.agent.enable_monitoring,
        )
        stable_pool_names = self.config.elastic.resource_pools.stable_pool_names
        stable_pool_name = stable_pool_names[0] if stable_pool_names else ''
        self.collector = init_agent_metrics_collector(stable_pool_name)

        self._metrics_enabled = config.rollout_server.agent.enable_monitoring
        self._last_metrics_emit = 0
        self._metrics_emit_interval = 2.0  # 至少间隔N秒提交一次

        # metrics collector loop
        self._metrics_task = self.get_event_loop().create_task(self._metrics_collection_loop())

    async def execute(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent], /, *args, **kwargs):
        # 兼容旧的functional handler，保持task context中有tokenizer赋值
        for a in args:
            if isinstance(a, TaskContext):
                a.tokenizer = self.tokenizer
        agent_init_kwargs = self._make_essential_init_kwargs()

        if issubclass(agent_cls, AsyncAgent):
            tracker = self.monitor.task_tracker()
            with tracker.creation(agent_cls):
                agent = agent_cls(self.async_tokenizer, self.llm, **agent_init_kwargs)
            async with self.concurrency_limit:
                with tracker.execution(agent_cls):
                    return await agent.run_task(*args, **kwargs)

        elif issubclass(agent_cls, ThreadedAgent):
            tracker = self.monitor.task_tracker()
            with tracker.creation(agent_cls):
                agent = agent_cls(self.tokenizer, self.sync_llm, **agent_init_kwargs)
            async with self.concurrency_limit:
                # 不要漏了threaded agent也算在总并发度里
                with tracker.execution(agent_cls):
                    agent_task = partial(agent.run_task, **kwargs) if kwargs else agent.run_task
                    loop = self.get_event_loop()
                    return await loop.run_in_executor(self._thread_executor, agent_task, *args)
        else:
            raise TypeError(f"agent_cls must be a subclass of AsyncAgent or ThreadedAgent. got {type(agent_cls)}")

    def _make_essential_init_kwargs(self):
        return {
            'config': self.config,
            'executor': self._thread_executor,
            'global_state': self.global_state,
            'monitor': self.monitor,
            'processor': self.processor
        }

    def set_global_step(self, global_step: int):
        self.global_state.set_global_step(global_step)

    async def _metrics_collection_loop(self):
        """检查是否需要提交统计数据到collector"""
        if not self._metrics_enabled:
            return

        sleep_interval = self._metrics_emit_interval
        while True:
            await asyncio.sleep(sleep_interval)
            t0 = time.time()
            try:
                # 获取当前统计数据
                stats = self.get_stats()
                await self.collector.collect.remote(self.monitor.worker_id, stats, t0)
                self._last_metrics_emit = t0
            except Exception as e:
                print(f"Failed to submit metrics: {e}, ignoring")
            loop_cost = time.time() - t0
            sleep_interval = max(0.01, self._metrics_emit_interval - loop_cost)

    def get_stats(self) -> AgentWorkerStats:
        """获取worker统计信息"""
        task_stats = self.monitor.get_task_stats()
        queue_stats = AgentWorkerQueueStats(self.concurrency_limit._value, self.worker_max_concurrency,
                                            len(self._thread_executor._threads), self._thread_executor._max_workers)
        perf_metrics = self.monitor.get_performance_metrics()
        task_type_perf_metrics = self.monitor.get_agent_type_performance_metrics()
        tool_use_perf_metrics = self.monitor.get_tool_use_performance_metrics()
        return AgentWorkerStats(
            task_stats=task_stats,
            queue_stats=queue_stats,
            perf_metrics=perf_metrics,
            task_type_perf_metrics=task_type_perf_metrics,
            tool_use_perf_metrics=tool_use_perf_metrics,
        )

    def get_event_loop(self):
        return self.loop or asyncio.get_running_loop()


class ExecutorBase:

    async def submit(self, cls: Type[callable], /, *args, **kwargs) -> DataProto | List[DataProto]:
        """
        提交一个prompt到AgentWorker里运行(rollout)，
        返回rollout的结果，DataProto或者List[DataProto]，1个prompt可以返回0～N条
        """
        raise NotImplementedError()

    def set_global_step(self, global_step: int):
        """
        设置当前trainer开始的step。需要讲global_step传到每个AgentWorker里
        """
        raise NotImplementedError()


class RayActorExecutor(ExecutorBase):

    def __init__(self, name, config, tokenizer, processor, host, port, request_manager_name, loop):
        self.name = name
        self.max_workers = config.rollout_server.agent.max_workers
        self.worker_max_concurrency = config.rollout_server.agent.worker_max_concurrency
        worker_oob_concurrency = 10  # 允许worker额外的并发度，用于控制指令和其他非rollout调用
        resources = {}
        stable_pool_names = config.elastic.resource_pools.stable_pool_names
        stable_pool_name = stable_pool_names[0] if stable_pool_names else ''
        if stable_pool_name and not is_local_ray_instance():
            resources = {stable_pool_name: 1}
        RemoteAgentWorker = ray.remote(AgentWorker)
        self.workers = [
            RemoteAgentWorker.options(scheduling_strategy="SPREAD",
                                      max_concurrency=self.worker_max_concurrency + worker_oob_concurrency,
                                      resources=resources,
                                      name=f"{name}-agent_worker_{idx}").remote(idx, config, tokenizer, processor, host,
                                                                                port, request_manager_name, None)
            for idx in range(self.max_workers)
        ]

        self.worker_in_band_concurrency_limits = [
            asyncio.Semaphore(self.worker_max_concurrency) for idx in range(self.max_workers)
        ]
        self.worker_pointer = cycle(range(self.max_workers))

    async def submit(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent] | callable, /, *args, **kwargs):
        worker_idx = next(self.worker_pointer)
        worker = self.workers[worker_idx]
        # 兼容旧的functional handler
        if inspect.isfunction(agent_cls):
            agent_cls = functional_agent(agent_cls)

        # 这里统一控制ray actor in-band并发度，预留一小部分给控制
        async with self.worker_in_band_concurrency_limits[worker_idx]:
            return await worker.execute.remote(agent_cls, *args, **kwargs)

    def set_global_step(self, global_step: int):
        refs = []
        for w in self.workers:
            ref = w.set_global_step.remote(global_step)
            refs.append(ref)
        ray.get(refs)


class LocalExecutor(ExecutorBase):

    def __init__(self, name, config, tokenizer, processor, host, port, request_manager_name, loop):
        self.name = name
        self.max_workers = config.rollout_server.agent.max_workers
        self.worker_max_concurrency = config.rollout_server.agent.worker_max_concurrency
        self.loop = loop
        self.workers = [
            AgentWorker(idx, config, tokenizer, processor, host, port, request_manager_name, loop)
            for idx in range(self.max_workers)
        ]

        self.worker_pointer = cycle(range(self.max_workers))

    async def submit(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent] | callable, /, *args, **kwargs):
        worker_idx = next(self.worker_pointer)
        worker = self.workers[worker_idx]
        # 兼容旧的functional handler
        if inspect.isfunction(agent_cls):
            agent_cls = functional_agent(agent_cls)
        return await worker.execute(agent_cls, *args, **kwargs)

    def set_global_step(self, global_step: int):
        for w in self.workers:
            w.set_global_step(global_step)
