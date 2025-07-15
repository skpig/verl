import asyncio
import copy
import threading
from concurrent.futures import ThreadPoolExecutor

from transformers import PreTrainedTokenizer


class AsyncTokenizer:

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self._thread_local = threading.local()
        # tokenizer用自己的小线程池，避免local executor模式下把task runner线程数占满
        # linux(L20)测试下来8个线程性能足够好了，更多线程反而不好
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix='async-tokenizer')

    def get_local_tokenizer(self):
        if not hasattr(self._thread_local, "tokenizer"):
            self._thread_local.tokenizer = copy.deepcopy(self.tokenizer)
        return self._thread_local.tokenizer

    def _encode_in_thread_local(self, args, kwargs):
        tokenizer = self.get_local_tokenizer()
        encode = tokenizer.encode
        return encode(*args, **kwargs)

    # 这两个方法改写成async模式，不占用asyncio loop
    async def encode_async(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._encode_in_thread_local, args, kwargs)

    def _batch_encode_in_thread_local(self, args, kwargs):
        tokenizer = self.get_local_tokenizer()
        batch_encode_plus = tokenizer.batch_encode_plus
        return batch_encode_plus(*args, **kwargs)

    # batch encode可以释放gil，配合run_in_executor进一步减少python thread时间占用
    async def batch_encode_plus_async(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._batch_encode_in_thread_local, args, kwargs)

    def __getattr__(self, item):
        return getattr(self.tokenizer, item)
