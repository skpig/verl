import asyncio
import copy
import threading
import time
import traceback
import statistics
from queue import Queue

import pytest
from transformers import AutoTokenizer

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer


def get_tokenizer(model_name='Qwen/Qwen2.5-1.5B'):
    # 创建tokenizer实例
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建async tokenizer
    async_tokenizer = AsyncTokenizer(tokenizer)
    return async_tokenizer


def get_prompts():
    base_texts = [
        "Hello world",
        "This is a longer sentence with more tokens to test tokenization behavior.",
        "Short",
        "A" * 100,  # 长文本
        "Multiple words here for testing tokenization with various content.",
        "中文测试文本",
        "Mixed 中文 and English text",
        "Special characters: !@#$%^&*()_+-=[]{}|;:,.<>?",
        "Numbers: 123456789 and dates: 2024-01-01",
        "",  # 空字符串
    ]
    return base_texts


def get_prompts_by_length(length: int) -> str:
    # 生成指定长度的测试文本
    base_text = "This is a test sentence for tokenizer performance benchmark. " * 2000
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
    # 编码然后截断到指定长度
    tokens = tokenizer.encode(base_text)[:length]
    assert len(tokens) == length
    return tokenizer.decode(tokens)


@pytest.mark.asyncio
async def test_async_tokenizer_encode_concurrent():
    async_tokenizer = get_tokenizer()
    prompts = [p * 100 for p in get_prompts()[0:1]]
    concurrency = 32
    tasks = []
    ground_truth = async_tokenizer.tokenizer.batch_encode_plus(prompts,
                                                               add_special_tokens=False,
                                                               max_length=20048,
                                                               truncation=True)
    for i in range(concurrency):
        coro = async_tokenizer.batch_encode_plus_async(prompts,
                                                       add_special_tokens=False,
                                                       max_length=20048,
                                                       truncation=True)
        tasks.append(asyncio.create_task(coro))

    results = await asyncio.gather(*tasks)
    for result in results:
        assert result == ground_truth


@pytest.mark.asyncio
async def test_tokenizer_encode_concurrent_race_async():
    async_tokenizer = get_tokenizer()
    prompt = "Hello world"
    await asyncio.gather(*[
        async_tokenizer.batch_encode_plus_async(
            [prompt], max_length=1024, truncation=True, padding_side="right", padding=True),
        async_tokenizer.batch_encode_plus_async(
            [prompt], max_length=2048, truncation=True, padding_side="right", padding=True),
    ])


@pytest.mark.asyncio
async def test_tokenizer_encode_concurrent_race():
    model_name = 'Qwen/Qwen2.5-1.5B'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Hello world " * 100

    def tokenize(prompt, max_length, exc: Queue):
        try:
            ret = tokenizer.batch_encode_plus([prompt], max_length=max_length, truncation=True)
            exc.put(None)
            return ret
        except Exception as e:
            traceback.print_exc()
            time.sleep(2)
            exc.put(e)

    exc_queue = Queue()
    t1 = threading.Thread(target=tokenize, args=(prompt, 1024, exc_queue))
    t2 = threading.Thread(target=tokenize, args=(prompt, 2048, exc_queue))
    t1.start(), t2.start()
    t1.join(), t2.join()
    # 预期会有报错的
    print(exc_queue.get(), exc_queue.get())


@pytest.mark.asyncio
async def test_tokenizer_benchmark():
    """
    完成tokenizer的throughput benchmark，bsz和seqlen都给定了，按batchsize=1一条一条encode，用batch_encode_plus_async函数
    """
    bsz, seqlen = 1024 * 4, 8192
    async_tokenizer = get_tokenizer()
    prompt = get_prompts_by_length(seqlen)

    # 测试参数
    batch_size = 1  # 按要求使用batchsize=1
    num_requests = bsz  # 使用传入的bsz作为请求数

    print(f"Benchmark Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seqlen}")
    print(f"  Total requests: {num_requests}")
    print(f"  Test text length: {len(prompt)} chars")

    async def run_benchmark():
        latencies = []
        start_time = time.time()

        tasks = []

        # 按bs=1 encode
        for i in range(num_requests):
            # 使用batch_encode_plus_async，但每次只传一个文本（batchsize=1）
            coro = async_tokenizer.batch_encode_plus_async(
                [prompt],  # 单个文本作为列表传入
                max_length=seqlen,
                padding='max_length',
                truncation=True,
            )
            tasks.append(asyncio.create_task(coro))

        await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # 计算统计信息
        total_tokens = num_requests * seqlen
        throughput_tokens = total_tokens / total_time

        # 打印结果
        print(f"\nBenchmark Results:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput_tokens:.1f} tokens/second")

        return {
            'total_time': total_time,
            'throughput_tokens': throughput_tokens,
            'num_requests': num_requests,
            'batch_size': batch_size,
            'seq_len': seqlen,
            'total_tokens': total_tokens
        }

    # 运行异步benchmark
    await run_benchmark()


def test_tokenizer_getattr_perf():
    async_tokenizer = get_tokenizer()
    exponential = list(range(8))
    seqlen_base = 128
    batch_size = 128
    for exp in exponential:
        seqlen = seqlen_base * 2**exp

        sample_list = [0] * seqlen
        t0 = time.time()
        # 访问async_tokenizer.pad_token_id会很慢
        for _ in range(batch_size):
            _ = [v for v in sample_list if v != async_tokenizer.pad_token_id]
        t1 = time.time()
        pad = async_tokenizer.pad_token_id
        for _ in range(batch_size):
            _ = [v for v in sample_list if v != pad]
        t2 = time.time()
        print()
        print(f'cost 1 {seqlen=}: {(t1 - t0) / (seqlen * batch_size) * 1e6:.3f}us')
        print(f'cost 2 {seqlen=}: {(t2 - t1) / (seqlen * batch_size) * 1e6:.3f}us')
