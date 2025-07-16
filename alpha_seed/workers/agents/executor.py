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
from alpha_seed.workers.agents.handlers.base import AsyncAgent, functional_agent, OpenAIAsyncClient, ThreadedAgent


class AgentWorker:

    def __init__(self,
                 config: DictConfig,
                 tokenizer: PreTrainedTokenizer,
                 host,
                 port,
                 worker_max_concurrency,
                 worker_id=0):
        """
        config: root config
        host: llm server host
        port: llm server port
        worker_max_concurrency: the worker can handle numbers of concurrent tasks/threads
        worker_id: worker id to identify different workers
        """
        self.worker_max_concurrency = worker_max_concurrency
        self._thread_executor = ThreadPoolExecutor(max_workers=worker_max_concurrency,
                                                   thread_name_prefix=f"agent-worker-{worker_id}")
        self.tokenizer = tokenizer
        self.async_tokenizer = AsyncTokenizer(tokenizer)
        self.llm = OpenAIAsyncClient(host, port)
        self.concurrency_limit = asyncio.Semaphore(worker_max_concurrency)
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
            agent = agent_cls(self.tokenizer, self.llm, **agent_init_kwargs)
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

    def __init__(self, config, tokenizer, host, port, max_workers=1, worker_max_concurrency=1):
        RemoteAgentWorker = ray.remote(AgentWorker)
        self.workers = [
            RemoteAgentWorker.options(scheduling_strategy="SPREAD",
                                      max_concurrency=worker_max_concurrency,
                                      name=f"ray_executor_{idx}").remote(config, tokenizer, host, port,
                                                                         worker_max_concurrency, idx)
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

    def __init__(self, config, tokenizer, host, port, max_workers=1, worker_max_concurrency=1):
        self.workers = [
            AgentWorker(config, tokenizer, host, port, worker_max_concurrency, idx) for idx in range(max_workers)
        ]
        self.worker_pointer = cycle(range(max_workers))

    async def submit(self, agent_cls: Type[AsyncAgent] | Type[ThreadedAgent] | callable, /, *args, **kwargs):
        worker_idx = next(self.worker_pointer)
        worker = self.workers[worker_idx]
        # 兼容旧的functional handler
        if inspect.isfunction(agent_cls):
            agent_cls = functional_agent(agent_cls)
        return await worker.execute(agent_cls, *args, **kwargs)
