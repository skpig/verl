from concurrent.futures.thread import ThreadPoolExecutor

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import TaskContext
from alpha_seed.workers.agents.llm import AsyncLLMInterface, SyncLLMInterface
from mono_rl import DataProto


class AsyncAgent:

    def __new__(cls, *args, **kwargs):
        config: DictConfig = kwargs.pop('config')
        executor: ThreadPoolExecutor = kwargs.pop('executor')
        instance = super().__new__(cls)
        instance.config = config
        instance.executor = executor
        return instance

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        # async tokenizer can be used as the normal pretrained tokenizer
        self.tokenizer: AsyncTokenizer = tokenizer
        self.llm = llm

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        raise NotImplementedError


class ThreadedAgent:

    def __new__(cls, *args, **kwargs):
        config: DictConfig = kwargs.pop('config')
        executor: ThreadPoolExecutor = kwargs.pop('executor')
        instance = super().__new__(cls)
        instance.config = config
        instance.executor = executor
        return instance

    def __init__(self, tokenizer: PreTrainedTokenizer, llm: SyncLLMInterface, **kwargs):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.llm = llm

    def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        raise NotImplementedError


def functional_agent(func):

    class FunctionAsyncAgent(AsyncAgent):

        def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
            super().__init__(tokenizer, llm, **kwargs)
            self.func = func

        async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
            return await self.func(item, context, **kwargs)

    return FunctionAsyncAgent
