import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import cycle
from typing import Type

import ray
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import TaskContext
from alpha_seed.workers.agents.handlers.base import AsyncAgent, functional_agent, ThreadedAgent
from alpha_seed.workers.agents.llm import OpenAIAsyncClient, OpenAIClient


class AgentWorker:

    def __init__(self, config: DictConfig, tokenizer: PreTrainedTokenizer, host, port, worker_id: int):
        """
        config: root config
        host: llm server host
        port: llm server port
        worker_id: worker id to identify different workers
        """
        self.worker_max_concurrency = config.rollout_server.agent.worker_max_concurrency
        self.llm_request_concurrency = config.rollout_server.agent.llm_request_concurrency
        self._thread_executor = ThreadPoolExecutor(max_workers=self.worker_max_concurrency,
                                                   thread_name_prefix=f"agent-worker-{worker_id}")
        self.tokenizer = tokenizer
        self.async_tokenizer = AsyncTokenizer(tokenizer)
        self.llm = OpenAIAsyncClient(host, port, self.llm_request_concurrency)
        self.sync_llm = OpenAIClient(host, port)
        self.concurrency_limit = asyncio.Semaphore(self.worker_max_concurrency)
        self.config = config

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

    def _make_essential_init_kwargs(self):
        return {
            'config': self.config,
            'executor': self._thread_executor,
        }


class ExecutorBase:

    async def submit(self, cls: Type[callable], /, *args, **kwargs):
        raise NotImplementedError()


class RayActorExecutor(ExecutorBase):

    def __init__(self, name, config, tokenizer, host, port):
        self.name = name
        self.max_workers = config.rollout_server.agent.max_workers
        self.worker_max_concurrency = config.rollout_server.agent.worker_max_concurrency
        RemoteAgentWorker = ray.remote(AgentWorker)
        self.workers = [
            RemoteAgentWorker.options(scheduling_strategy="SPREAD",
                                      max_concurrency=self.worker_max_concurrency,
                                      name=f"{name}-agent_worker_{idx}").remote(config, tokenizer, host, port, idx)
            for idx in range(self.max_workers)
        ]
        self.worker_pointer = cycle(range(self.max_workers))

    async def submit(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent] | callable, /, *args, **kwargs):
        worker_idx = next(self.worker_pointer)
        worker = self.workers[worker_idx]
        # 兼容旧的functional handler
        if inspect.isfunction(agent_cls):
            agent_cls = functional_agent(agent_cls)
        return await worker.execute.remote(agent_cls, *args, **kwargs)


class LocalExecutor(ExecutorBase):

    def __init__(self, name, config, tokenizer, host, port):
        self.name = name
        self.max_workers = config.rollout_server.agent.max_workers
        self.worker_max_concurrency = config.rollout_server.agent.worker_max_concurrency
        self.workers = [AgentWorker(config, tokenizer, host, port, idx) for idx in range(self.max_workers)]
        self.worker_pointer = cycle(range(self.max_workers))

    async def submit(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent] | callable, /, *args, **kwargs):
        worker_idx = next(self.worker_pointer)
        worker = self.workers[worker_idx]
        # 兼容旧的functional handler
        if inspect.isfunction(agent_cls):
            agent_cls = functional_agent(agent_cls)
        return await worker.execute(agent_cls, *args, **kwargs)
