import pytest
import random
from typing import *
from alpha_seed.workers.xperf_rollout.component.cache_manager import CacheManager
from alpha_seed.workers.xperf_rollout.component.query import Query
from dataclasses import dataclass


@dataclass
class QueryConfig:
    prompt_len: int
    resp_len: int
    model_output_mask: List[bool]

    @staticmethod
    def get_random(max_prompt_len: int, max_resp_len: int):
        prompt_len = random.randint(1, max_prompt_len)
        resp_len = random.randint(1, max_resp_len)
        model_output_mask = [bool(random.random() < 0.98) for _ in range(resp_len)]
        return QueryConfig(prompt_len=prompt_len, resp_len=resp_len, model_output_mask=model_output_mask)


@pytest.mark.parametrize('use_vllm', [True, False])
@pytest.mark.parametrize('schedule_strategy', ['default', 'fifo'])
def test_scheduler(use_vllm, schedule_strategy: str):
    slot_num = 288
    slot_block_size = 1024
    max_batch_size = 100 if use_vllm else 16
    context_batchsize_limit = 8
    mgr = CacheManager(slot_num=slot_num,
                       max_batch_size=max_batch_size,
                       use_vllm=use_vllm,
                       slot_block_size=slot_block_size,
                       context_batchsize_limit=context_batchsize_limit,
                       schedule_strategy=schedule_strategy)
    mgr.empty_cache()

    # print("Available slots", mgr.get_available_slot_num())

    def _check_queues(running: List[Query], waiting: List[Query], paused: List[Query]):
        running_ids = set()
        waiting_ids = set()
        paused_ids = set()
        seen_slots = set()

        def _check_kv_slot_enough(query: Query):
            token_len = len(query.input_ids) + len(query.new_token_ids)
            kv_slot_num = len(query.kv_slot_ids)
            if use_vllm:
                if (token_len + slot_block_size - 1) // slot_block_size > kv_slot_num:
                    breakpoint()
                assert (token_len + slot_block_size - 1) // slot_block_size <= kv_slot_num
            else:
                assert kv_slot_num == 1

        for query in running:
            running_ids.add(query.idx)
            _check_kv_slot_enough(query)
        for query in waiting:
            waiting_ids.add(query.idx)
            assert not query.is_kv_cache_slot_allocated()
        for query in paused:
            paused_ids.add(query.idx)
            if query.is_kv_cache_slot_allocated():
                _check_kv_slot_enough(query)
        import itertools
        for query in itertools.chain(running, waiting, paused):
            for slot_id in query.kv_slot_ids:
                assert slot_id not in seen_slots
                seen_slots.add(slot_id)

        assert len(running_ids & waiting_ids) == 0
        assert len(running_ids & paused_ids) == 0
        assert len(waiting_ids & paused_ids) == 0

    running: List[Query] = []
    waiting: List[Query] = []
    paused: List[Query] = []

    random.seed(2025)
    max_prompt_len = 2048
    max_resp_len = 16384

    query_confs = [QueryConfig.get_random(max_prompt_len, max_resp_len) for _ in range(100)]

    for idx, conf in enumerate(query_confs):
        query = Query(input_ids=[0] * conf.prompt_len, idx=idx, input_prompt='')
        waiting.append(query)

    def output_len(query: Query):
        return len(query.input_ids) + len(query.new_token_ids)

    # mock inference session execute
    step = 0
    page_swap_bs = 0
    page_swap_tokens = 0
    while len(running) + len(waiting) + len(paused) > 0:
        step += 1
        # print(f"#{step}, {len(running)}, {len(waiting)}, {len(paused)}")
        new_paused = []
        for query in paused:
            resume = random.random() < 0.2
            if resume:
                conf = query_confs[query.idx]
                while True:
                    out_len = output_len(query)
                    if out_len < conf.prompt_len + conf.resp_len and not conf.model_output_mask[out_len -
                                                                                                conf.prompt_len]:
                        query.add_token(0)
                    else:
                        break
                if output_len(query) == conf.prompt_len + conf.resp_len:
                    mgr.release_query(query)
                    query.reset_compute()
                    continue

                if query.is_kv_cache_slot_allocated():
                    running.append(query)
                else:
                    waiting.append(query)
            else:
                new_paused.append(query)
        paused = new_paused
        paused_ids_before_update = set([query.idx for query in paused])
        running, waiting = mgr.update_queries(running, waiting, paused)
        if len(running) == 0 and len(paused) == 0:
            assert len(waiting) == 0, f"KV util: {mgr.get_kv_cache_utils()}"
        paused_ids_after_update = set([query.idx for query in paused])
        _check_queues(running, waiting, paused)
        assert paused_ids_before_update == paused_ids_after_update, "paused queue should not be modified"
        page_swap_bs += mgr.page_swap_out_bs
        page_swap_tokens += mgr.page_swap_out_token
        paused_swapped = 0
        for query in paused:
            if not query.is_kv_cache_slot_allocated():
                paused_swapped += 1

        new_paused = []
        new_running = []
        for query in running:
            query.add_token(0)
            out_len = output_len(query)
            conf = query_confs[query.idx]
            if out_len >= conf.prompt_len + conf.resp_len:
                mgr.release_query(query)
                query.reset_compute()
            else:
                is_model_output = conf.model_output_mask[out_len - conf.prompt_len]
                if is_model_output:
                    new_running.append(query)
                else:
                    new_paused.append(query)
        running = new_running
        paused.extend(new_paused)
    assert mgr.get_kv_cache_utils() == 0
    print(f"use_vllm={use_vllm}, schedule_strategy={schedule_strategy}, steps={step}, "
          f"swap_out_bs={page_swap_bs}, swap_out_tokens={page_swap_tokens}")


if __name__ == '__main__':
    test_scheduler(use_vllm=True, schedule_strategy='default')
