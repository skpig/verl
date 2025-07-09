import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from transformers import PreTrainedTokenizer


class AsyncTokenizer:

    def __init__(self, tokenizer: PreTrainedTokenizer, executor: ThreadPoolExecutor):
        self.tokenizer = tokenizer
        self.executor = executor

    # 这两个方法改写成async模式，不占用asyncio loop
    async def encode_async(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        encode = self.tokenizer.encode
        if kwargs:
            encode = partial(encode, **kwargs)
        return await loop.run_in_executor(self.executor, encode, *args)

    # batch encode可以释放gil，配合run_in_executor进一步减少python thread时间占用
    async def batch_encode_plus_async(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        batch_encode_plus = self.tokenizer.batch_encode_plus
        if kwargs:
            batch_encode_plus = partial(batch_encode_plus, **kwargs)
        return await loop.run_in_executor(self.executor, batch_encode_plus, *args)

    def __getattr__(self, item):
        return getattr(self.tokenizer, item)
