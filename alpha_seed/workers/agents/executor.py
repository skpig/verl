import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import cycle
from typing import Type

import ray
from transformers import AutoTokenizer, PreTrainedTokenizer

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import TaskContext
from alpha_seed.workers.agents.handlers.base import AsyncAgent, functional_agent, OpenAIAsyncClient, ThreadedAgent


class AgentWorker:

    def __init__(self, tokenizer: PreTrainedTokenizer, host, port, worker_max_concurrency):
        self.worker_max_concurrency = worker_max_concurrency
        self._thread_executor = ThreadPoolExecutor(max_workers=worker_max_concurrency,
                                                   thread_name_prefix="agent-worker")
        self.tokenizer = tokenizer
        self.async_tokenizer = AsyncTokenizer(tokenizer)
        self.llm = OpenAIAsyncClient(host, port)
        self.concurrency_limit = asyncio.Semaphore(worker_max_concurrency)

    async def execute(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent], /, *args, **kwargs):
        # 兼容旧的functional handler，保持task context中有tokenizer赋值
        for a in args:
            if isinstance(a, TaskContext):
                a.tokenizer = self.tokenizer
        if issubclass(agent_cls, AsyncAgent):
            agent = agent_cls(self.async_tokenizer, self.llm)
            async with self.concurrency_limit:
                return await agent(*args, **kwargs)
        else:
            agent = agent_cls(self.tokenizer, self.llm)
            loop = asyncio.get_event_loop()
            if kwargs:
                agent = partial(agent, **kwargs)
            return await loop.run_in_executor(self._thread_executor, agent, *args)


class ExecutorBase:

    async def submit(self, cls: Type[callable], /, *args, **kwargs):
        raise NotImplementedError()


class RayActorExecutor(ExecutorBase):

    def __init__(self, tokenizer, host, port, max_workers=1, worker_max_concurrency=1):
        RemoteAgentWorker = ray.remote(AgentWorker)
        self.workers = [
            RemoteAgentWorker.options(scheduling_strategy="SPREAD",
                                      max_concurrency=worker_max_concurrency,
                                      name=f"ray_executor_{idx}").remote(tokenizer, host, port, worker_max_concurrency)
            for idx in range(max_workers)
        ]
        self.worker_pointer = cycle(range(max_workers))

    async def submit(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent] | callable, /, *args, **kwargs):
        worker_idx = next(self.worker_pointer)
        worker = self.workers[worker_idx]
        # 兼容旧的functional handler
        if inspect.isfunction(agent_cls):
            agent_cls = functional_agent(agent_cls)
        return await worker.execute.remote(agent_cls, *args, **kwargs)


class LocalExecutor(ExecutorBase):

    def __init__(self, tokenizer, host, port, max_workers=1, worker_max_concurrency=1):
        self.workers = [AgentWorker(tokenizer, host, port, worker_max_concurrency) for idx in range(max_workers)]
        self.worker_pointer = cycle(range(max_workers))

    async def submit(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent] | callable, /, *args, **kwargs):
        worker_idx = next(self.worker_pointer)
        worker = self.workers[worker_idx]
        # 兼容旧的functional handler
        if inspect.isfunction(agent_cls):
            agent_cls = functional_agent(agent_cls)
        return await worker.execute(agent_cls, *args, **kwargs)
