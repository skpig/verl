import contextvars
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import TaskContext, GlobalState
from alpha_seed.workers.agents.llm import AsyncLLMInterface, SyncLLMInterface
from alpha_seed.workers.agents.monitor_ctx import current_agent, set_current_agent
from alpha_seed.workers.agents.monitoring import AgentWorkerMonitor
from mono_rl import DataProto


class AsyncAgent:

    def __new__(cls, *args, **kwargs):
        config: DictConfig = kwargs.pop('config')
        executor: ThreadPoolExecutor = kwargs.pop('executor')
        global_state: GlobalState = kwargs.pop('global_state')
        monitor: AgentWorkerMonitor = kwargs.pop('monitor')
        instance = super().__new__(cls)
        instance.config = config
        instance.executor = executor
        instance.global_state = global_state
        instance.monitor = monitor
        return instance

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        # async tokenizer can be used as the normal pretrained tokenizer
        self.tokenizer: AsyncTokenizer = tokenizer
        self.llm = llm

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs) -> DataProto | List[DataProto]:
        """
        在这里采样出完整的trajectory
        可以返回0个1个或多个trajectory，返回0个时请返回[]
        :param item: input prompt
        :pram context: agent task可能用到的一些环境信息
        """
        raise NotImplementedError

    async def run_task(self, item: DataProto, *args, **kwargs) -> DataProto:
        current_agent.set(self.__class__.__name__)
        return await self.__call__(item, *args, **kwargs)

    def get_name(self):
        """
        返回agent的名字，主要用来区分agent种类
        """
        return f"{self.__class__.__name__}.async"


class ThreadedAgent:

    def __new__(cls, *args, **kwargs):
        config: DictConfig = kwargs.pop('config')
        executor: ThreadPoolExecutor = kwargs.pop('executor')
        global_state: GlobalState = kwargs.pop('global_state')
        monitor: AgentWorkerMonitor = kwargs.pop('monitor')
        instance = super().__new__(cls)
        instance.config = config
        instance.executor = executor
        instance.global_state = global_state
        instance.monitor = monitor
        return instance

    def __init__(self, tokenizer: PreTrainedTokenizer, llm: SyncLLMInterface, **kwargs):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.llm = llm

    def __call__(self, item: DataProto, context: TaskContext, **kwargs) -> DataProto | List[DataProto]:
        raise NotImplementedError

    def run_task(self, item: DataProto, *args, **kwargs) -> DataProto:
        set_current_agent(self.__class__.__name__)
        return self.__call__(item, *args, **kwargs)

    def get_name(self):
        """
        返回agent的名字，主要用来区分agent种类
        """
        return f"{self.__class__.__name__}.threaded"


def functional_agent(func):

    class FunctionAsyncAgent(AsyncAgent):

        def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
            super().__init__(tokenizer, llm, **kwargs)
            self.func = func

        async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
            return await self.func(item, context, **kwargs)

    return FunctionAsyncAgent
