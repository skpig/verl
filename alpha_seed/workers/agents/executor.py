import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import cycle
from typing import Type, List

import ray
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, AutoProcessor

from alpha_seed.utils.server_client import is_local_ray_instance
from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import TaskContext, GlobalState
from alpha_seed.workers.agents.handlers.base import AsyncAgent, functional_agent, ThreadedAgent
from alpha_seed.workers.agents.llm import OpenAIAsyncClient, OpenAIClient, DirectAsyncClient, DirectClient
from mono_rl import DataProto


class AgentWorker:

    def __init__(self, config: DictConfig, tokenizer: PreTrainedTokenizer, processor: AutoProcessor, host, port,
                 request_manager_name, worker_id: int):
        """
        config: root config
        host: llm server host
        port: llm server port
        request_manager_name: the corresponding request manager of rollout
        worker_id: worker id to identify different workers
        """
        self.worker_max_concurrency = config.rollout_server.agent.worker_max_concurrency
        self.llm_request_concurrency = config.rollout_server.agent.llm_request_concurrency
        self.llm_timeout = config.rollout_server.timeout
        self._thread_executor = ThreadPoolExecutor(max_workers=self.worker_max_concurrency,
                                                   thread_name_prefix=f"agent-worker-{worker_id}")
        self.tokenizer = tokenizer
        self.async_tokenizer = AsyncTokenizer(tokenizer)
        self.processor = processor
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

    async def execute(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent], /, *args, **kwargs):
        # 兼容旧的functional handler，保持task context中有tokenizer赋值
        for a in args:
            if isinstance(a, TaskContext):
                a.tokenizer = self.tokenizer
        agent_init_kwargs = self._make_essential_init_kwargs()
        if issubclass(agent_cls, AsyncAgent):
            agent = agent_cls(self.async_tokenizer, self.llm, **agent_init_kwargs)
            async with self.concurrency_limit:
                return await agent(*args, **kwargs)
        else:
            agent = agent_cls(self.tokenizer, self.sync_llm, **agent_init_kwargs)
            loop = asyncio.get_event_loop()
            if kwargs:
                agent = partial(agent, **kwargs)
            return await loop.run_in_executor(self._thread_executor, agent, *args)

    def set_global_step(self, global_step: int):
        self.global_state.set_global_step(global_step)

    def _make_essential_init_kwargs(self):
        return {
            'config': self.config,
            'executor': self._thread_executor,
            'global_state': self.global_state,
            'processor': self.processor
        }


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

    def __init__(self, name, config, tokenizer, processor, host, port, request_manager_name):
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
                                      name=f"{name}-agent_worker_{idx}").remote(config, tokenizer, processor, host,
                                                                                port, request_manager_name, idx)
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

    def __init__(self, name, config, tokenizer, processor, host, port, request_manager_name):
        self.name = name
        self.max_workers = config.rollout_server.agent.max_workers
        self.worker_max_concurrency = config.rollout_server.agent.worker_max_concurrency
        self.workers = [
            AgentWorker(config, tokenizer, processor, host, port, request_manager_name, idx)
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
