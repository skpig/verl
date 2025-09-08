import asyncio
import inspect
import time
import traceback
import uuid
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
from alpha_seed.workers.agents.trajectory import Trajectory, TrajectoryFactory, get_agent_trajectory_collector, \
    AgentIdentity
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
        self.collector = get_agent_metrics_collector()
        self.traj_collector = get_agent_trajectory_collector()

        # 暂存running tasks
        self.tasks: Dict[str, AsyncAgent | ThreadedAgent] = {}  # uid -> agent

        # 添加监控
        self.worker_id = worker_id
        self.worker_name = f"{request_manager_name}.w{worker_id}"
        self.monitor = AgentWorkerMonitor(
            worker_id=self.worker_name,
            enabled=config.rollout_server.agent.enable_monitoring,
        )

        self._metrics_enabled = config.rollout_server.agent.enable_monitoring
        self._last_metrics_emit = 0
        self._metrics_emit_interval = 2.0  # 至少间隔N秒提交一次
        self._traj_emit_interval = 5.0

        # collector tasks
        self._metrics_task = self.get_event_loop().create_task(self._metrics_collection_loop())
        self._traj_task = self.get_event_loop().create_task(self._trajectory_collection_loop())

    async def execute(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent], /, item: DataProto, context: TaskContext,
                      **kwargs):
        # 兼容旧的functional handler，保持task context中有tokenizer赋值
        context.tokenizer = self.tokenizer

        # 用来跟踪整个trajectory
        uid = item.non_tensor_batch['uid'][0]

        # 给item增加必要的agent worker相关的元数据
        item.meta_info.update({
            'agent_worker_name': self.worker_name,
            'agent_class': agent_cls.__name__,
        })

        agent_init_kwargs = self._make_essential_init_kwargs(uid, agent_cls.__name__, context.global_step)

        if issubclass(agent_cls, AsyncAgent):
            tracker = self.monitor.task_tracker()
            with tracker.creation(agent_cls):
                agent = agent_cls(self.async_tokenizer, self.llm, **agent_init_kwargs)
                self.tasks[agent.uid] = agent
            async with self.concurrency_limit:
                with tracker.execution(agent):
                    ret = await agent.run_task(item, context, **kwargs)

        elif issubclass(agent_cls, ThreadedAgent):
            tracker = self.monitor.task_tracker()
            with tracker.creation(agent_cls):
                agent = agent_cls(self.tokenizer, self.sync_llm, **agent_init_kwargs)
                self.tasks[agent.uid] = agent
            async with self.concurrency_limit:
                # 不要漏了threaded agent也算在总并发度里
                with tracker.execution(agent):
                    agent_task = partial(agent.run_task, **kwargs) if kwargs else agent.run_task
                    loop = self.get_event_loop()
                    ret = await loop.run_in_executor(self._thread_executor, agent_task, item, context)
        else:
            raise TypeError(f"agent_cls must be a subclass of AsyncAgent or ThreadedAgent. got {type(agent_cls)}")

        self.tasks.pop(agent.uid, None)

        # flush metrics and trace
        agent.trajectory_factory.finish(self.tokenizer)
        await self.traj_collector.collect.remote(agent.trajectory_factory.trajectories)

        return ret

    def _make_essential_init_kwargs(self, uid: str, agent_name: str, global_step: int) -> dict:
        """
        :param uid: 追踪整个trajectory的id，会一路传到train那边，数据集那边也可以加上，构造来源见
                    alpha_seed.trainer.ppo.RayPPOTrainer._preprocess_batch_before_gen
        :param agent_name: agent class name
        """
        return {
            'uid': uid,
            'trajectory_factory': TrajectoryFactory(uid, agent_name, global_step),
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

    async def _trajectory_collection_loop(self):
        if not self._metrics_enabled:
            return
        sleep_interval = self._traj_emit_interval
        while True:
            await asyncio.sleep(sleep_interval)
            try:
                staging_segs = {}
                idents_map = {}
                for agent in list(self.tasks.values()):
                    segs = agent.trajectory_factory.get_staging_segments(self.tokenizer)
                    ident = agent.trajectory_factory.agent_ident
                    staging_segs[agent.uid] = segs
                    idents_map[agent.uid] = ident
                await self.traj_collector.collect_segments.remote(staging_segs, idents_map)
            except Exception as e:
                traceback.print_exc()
                print(f"Failed to submit traj: {e}, ignoring")

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

    def stop(self):
        self._metrics_task.cancel()
        self._traj_task.cancel()


class ExecutorBase:

    async def submit(self, cls: Type[callable], /, item: DataProto, *args, **kwargs) -> DataProto | List[DataProto]:
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

    async def stop(self):
        """
        executor结束当前task和workers，给executor内部清场用
        """
        raise NotImplementedError()


class RayActorExecutor(ExecutorBase):

    def __init__(self, name, config, tokenizer, processor, host, port, request_manager_name, loop):
        self.name = name
        self.max_workers = config.rollout_server.agent.max_workers
        self.worker_max_concurrency = config.rollout_server.agent.worker_max_concurrency
        resources = {}
        stable_pool_names = config.elastic.resource_pools.stable_pool_names
        stable_pool_name = stable_pool_names[0] if stable_pool_names else ''
        if stable_pool_name and not is_local_ray_instance():
            resources = {stable_pool_name: 1}
        RemoteAgentWorker = ray.remote(AgentWorker)
        self.workers = [
            RemoteAgentWorker.options(
                scheduling_strategy="SPREAD",
                max_concurrency=999999999,  # 非常大，不可能超过的数字就行，内部有另外的并发限制
                resources=resources,
                name=f"{name}-agent_worker_{idx}").remote(idx, config, tokenizer, processor, host, port,
                                                          request_manager_name, None) for idx in range(self.max_workers)
        ]
        self.worker_pointer = cycle(range(self.max_workers))

    async def submit(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent], /, item: DataProto, *args, **kwargs):
        worker_idx = next(self.worker_pointer)
        worker = self.workers[worker_idx]
        # 兼容旧的functional handler
        if inspect.isfunction(agent_cls):
            agent_cls = functional_agent(agent_cls)
        return await worker.execute.remote(agent_cls, item, *args, **kwargs)

    def set_global_step(self, global_step: int):
        refs = []
        for w in self.workers:
            ref = w.set_global_step.remote(global_step)
            refs.append(ref)
        ray.get(refs)

    def stop(self):
        refs = []
        for w in self.workers:
            ref = w.stop.remote()
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

    async def submit(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent], /, item: DataProto, *args, **kwargs):
        worker_idx = next(self.worker_pointer)
        worker = self.workers[worker_idx]
        # 兼容旧的functional handler
        if inspect.isfunction(agent_cls):
            agent_cls = functional_agent(agent_cls)
        return await worker.execute(agent_cls, item, *args, **kwargs)

    def set_global_step(self, global_step: int):
        for w in self.workers:
            w.set_global_step(global_step)

    def stop(self):
        for w in self.workers:
            w.stop()
