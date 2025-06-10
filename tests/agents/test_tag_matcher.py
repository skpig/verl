from typing import List
from dataclasses import dataclass
from alpha_seed.workers.agents.plugins.tag_matcher import TagMatcher
from transformers import AutoTokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from utils import get_bbpe_tokenizer


def test_tag_matcher():

    @dataclass
    class TestData:
        start_tag: str
        end_tag: str
        test_text: str
        expect_results: List[str]

    test_data_list = [
        TestData(
            start_tag='<Begin>',
            end_tag='<End>',
            test_text='before begin tag<Begin>between two begin tag<Begin>content_to_match<End>invalid end tags<End>',
            expect_results=['content_to_match']),
        TestData(start_tag='',
                 end_tag='<End>',
                 test_text='first block<End>second block<End>\n<End>',
                 expect_results=['first block', 'second block', '\n']),
        TestData(
            start_tag='<｜tool▁call_begin｜>',
            end_tag='<｜tool▁call_end｜>',
            test_text='other text<｜tool▁call_begin｜><｜tool▁call_begin｜>tool▁calling<｜tool▁call_end｜>\n<｜tool▁call_end｜>',
            expect_results=['tool▁calling']),
    ]

    tokenizer = get_bbpe_tokenizer()

    for test_data in test_data_list:
        matcher = TagMatcher(start_tag=test_data.start_tag, end_tag=test_data.end_tag)
        tokens = tokenizer.tokenize(test_data.test_text)
        results = []
        # add_token_match
        for token in tokens:
            ret = matcher.add_token_match(token, tokenizer=tokenizer)
            if ret is not None:
                results.append(ret)
        assert results == test_data.expect_results
        # add_string_match
        matcher = TagMatcher(start_tag=test_data.start_tag, end_tag=test_data.end_tag)
        results = matcher.add_string_match(test_data.test_text, tokenizer=tokenizer)
        assert results == test_data.expect_results


def test_perf():
    import time
    import random
    random.seed(2025)
    tokenizer = get_bbpe_tokenizer()
    num_tokens = len(tokenizer)

    start_tag = '<|FuncCallBegin|>'
    end_tag = '<|FuncCallEnd|>'

    def gen_random_tokens(length):
        token_ids = []
        for _ in range(length):
            token_id = random.randint(0, num_tokens - 1)
            token_ids.append(token_id)
        random_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        start_tokens = tokenizer.tokenize(start_tag)
        end_tokens = tokenizer.tokenize(end_tag)
        return start_tokens + random_tokens + end_tokens

    total_time = 0
    num_iters = 100
    seq_len = 16384
    for _ in range(num_iters):
        tokens = gen_random_tokens(seq_len)
        matcher = TagMatcher(start_tag=start_tag, end_tag=end_tag)
        start_time = time.time()
        matched = []
        for token in tokens:
            content = matcher.add_token_match(token, tokenizer)
            if content is not None:
                matched.append(content)
        assert len(matched) > 0

        elapsed = time.time() - start_time
        total_time += elapsed
    print(f"per token avg elapsed: {total_time * 1e6 / num_iters / seq_len:.3f}us")
