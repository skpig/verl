from typing import *
from dataclasses import dataclass
from enum import Enum
import time
import uuid
from collections import deque
import logging
from alpha_seed.workers.xperf_rollout.component.query import Query
from xperf_gpt.utils import logging_rank_only


@dataclass
class SlotStatus:
    id: int
    is_occupied: bool = False
    occupied_by: Optional[uuid.UUID] = None
    occupied_duration: int = 0  # UNIX timestamp


class UpdateQueryStatus(Enum):
    SUCCESS = 0
    REACH_MAX_BS = 1
    REACH_MAX_CTX_BS = 2
    ALLOC_INSUFFICIENT_SLOT = 3
    NEED_SWAP_OUT = 4


class CacheManager:

    def __init__(
            self,
            slot_num,
            max_batch_size,
            pp_size,
            use_vllm,
            slot_block_size=256,
            context_batchsize_limit=1,
            enable_ngrams_decoding=False,
            num_pred_tokens=0,
            moving_avg_length=1024,
            schedule_strategy="default"  # ['default','fifo']
    ):
        self.slot_num = slot_num
        self.slot_block_size = slot_block_size
        self.max_batch_size = max_batch_size
        self.pp_size = pp_size
        self.micro_max_batch_size = max(int(self.max_batch_size / self.pp_size), 1)
        self.use_vllm = use_vllm
        self.slot_table_status: List[SlotStatus] = [SlotStatus(id=i) for i in range(slot_num)]
        self.available_slot_table: deque = deque([i for i in range(slot_num)])
        self.max_context_len_this_run = 0
        self.cur_context_bs_this_run = 0
        self.cur_bs_this_run = 0
        self.context_batchsize_limit = context_batchsize_limit
        self.micro_context_batchsize_limit = max(int(context_batchsize_limit / self.pp_size), 1)
        self.page_swap_out_bs = 0
        self.page_swap_out_token = 0
        self.enable_ngrams_decoding = enable_ngrams_decoding
        self.num_pred_tokens = num_pred_tokens
        self.moving_avg_len = moving_avg_length
        self.schedule_strategy = schedule_strategy.lower()
        assert self.schedule_strategy in ['default', 'fifo'], f"invalid schedule_strategy: {self.schedule_strategy}"

    def empty_cache(self):
        self.slot_table_status: List[SlotStatus] = [SlotStatus(id=i) for i in range(self.slot_num)]
        self.available_slot_table: deque = deque([i for i in range(self.slot_num)])
        self.max_context_len_this_run = 0
        self.cur_context_bs_this_run = 0
        self.cur_bs_this_run = 0
        self.page_swap_out_bs = 0
        self.page_swap_out_token = 0

    def get_available_slot_num(self):
        return len(self.available_slot_table)

    def update_queries(self, running_queries: List[Query], waiting_queries: List[Query]):
        if self.schedule_strategy == 'default':
            return self._update_queries_default(running_queries, waiting_queries)
        elif self.schedule_strategy == 'fifo':
            return self._update_queries_fifo(running_queries, waiting_queries)
        else:
            raise NotImplementedError(f"unsupported schedule_strategy: {self.schedule_strategy}")

    def _update_queries_default(self, running_queries: List[Query], waiting_queries: List[Query]):
        phase0_running = []
        phase1_running = []
        waiting = []
        self.max_context_len_this_run = 0
        self.cur_context_bs_this_run = 0
        self.cur_bs_this_run = 0
        self.page_swap_out_bs = 0
        self.page_swap_out_token = 0

        # See if any query can be continued
        for idx, query in enumerate(running_queries):
            self.moving_avg_len = int(self.moving_avg_len * (idx + 1) // (idx + 2) +
                                      (len(query.input_ids) + len(query.new_token_ids)) // (idx + 2))
            status = self._update_query(query)
            if status == UpdateQueryStatus.SUCCESS:
                phase1_running.append(query)
                self.cur_context_bs_this_run += int(query.is_context_computing)
                self.cur_bs_this_run += 1
            else:
                if status == UpdateQueryStatus.NEED_SWAP_OUT:
                    self.page_swap_out_bs += 1
                    self.page_swap_out_token += len(query.input_ids) + len(query.new_token_ids)
                self.release_query(query)
                query.kv_slot_ids = []
                query.is_context_computing = True
                query.input_ids.extend(query.new_token_ids)
                query.new_token_ids = []
                query.context_shift = 0
                query.prefix_already_computed_len = 0
                waiting.append(query)

        # See if any query from the waiting-list can be activated
        for query in waiting_queries:
            if (self._update_query(query,
                                   thresold=self.moving_avg_len // self.slot_block_size) == UpdateQueryStatus.SUCCESS):
                phase0_running.append(query)
                self.cur_context_bs_this_run += int(query.is_context_computing)
                self.cur_bs_this_run += 1
                query.first_scheduled_time = time.time() * 1000
            else:
                waiting.append(query)

        return phase0_running + phase1_running, waiting

    def _update_queries_fifo(self, running_queries: List[Query], waiting_queries: List[Query]):

        def sort_queue(lis: List[Query], reverse=False):
            return sorted(lis, key=lambda x: x.idx, reverse=reverse)

        phase1_running = sort_queue(running_queries)
        waiting = sort_queue(waiting_queries)

        self.max_context_len_this_run = 0
        self.cur_context_bs_this_run = 0
        self.cur_bs_this_run = 0
        self.page_swap_out_bs = 0
        self.page_swap_out_token = 0

        def release_running_query(query):
            self.release_query(query)
            query.kv_slot_ids = []
            query.is_context_computing = True
            query.input_ids.extend(query.new_token_ids)
            query.new_token_ids = []
            query.context_shift = 0
            query.prefix_already_computed_len = 0

        # see if any queries can be continued
        idx = 0
        while idx < len(phase1_running):
            query = phase1_running[idx]
            self.moving_avg_len = int(self.moving_avg_len * (idx + 1) // (idx + 2) +
                                      (len(query.input_ids) + len(query.new_token_ids)) // (idx + 2))
            cur_idx = idx
            idx += 1

            status = self._update_query(query)
            while status == UpdateQueryStatus.NEED_SWAP_OUT and len(phase1_running) > cur_idx + 1:
                # if need swap_out, swap out the latest running queries and retry
                to_swap_query = phase1_running.pop()
                waiting.append(to_swap_query)
                self.page_swap_out_bs += 1
                self.page_swap_out_token += len(to_swap_query.input_ids) + len(to_swap_query.new_token_ids)
                release_running_query(to_swap_query)
                status = self._update_query(query)

            if status == UpdateQueryStatus.SUCCESS:
                self.cur_context_bs_this_run += int(query.is_context_computing)
                self.cur_bs_this_run += 1
            else:
                # release all queries afterwards
                while len(phase1_running) > cur_idx:
                    query = phase1_running.pop()
                    waiting.append(query)
                    self.page_swap_out_bs += 1
                    self.page_swap_out_token += len(query.input_ids) + len(query.new_token_ids)
                    release_running_query(query)

        waiting = sort_queue(waiting, reverse=True)
        phase0_running = []
        # See if any query from the waiting-list can be activated
        while len(waiting) > 0:
            query = waiting[-1]
            if self._update_query(query,
                                  thresold=self.moving_avg_len // self.slot_block_size) == UpdateQueryStatus.SUCCESS:
                if not query.first_scheduled_time:
                    query.first_scheduled_time = time.time() * 1000
                # move to running
                phase0_running.append(query)
                waiting.pop()
                self.cur_context_bs_this_run += int(query.is_context_computing)
                self.cur_bs_this_run += 1
            else:
                break

        phase0_running = sort_queue(phase0_running)
        phase1_running = sort_queue(phase1_running)
        waiting = sort_queue(waiting)

        return phase0_running + phase1_running, waiting

    def release_query(self, query: Query):
        if len(query.kv_slot_ids) > 0:
            # clear cache
            for slot in query.kv_slot_ids:
                self.available_slot_table.appendleft(slot)
            query.kv_slot_ids.clear()
            logging_rank_only(logging.debug, 0, "kv utils {}".format(
                (self.slot_num - len(self.available_slot_table)) / self.slot_num))

    def _update_query(self, query: Query, thresold: int = 0) -> UpdateQueryStatus:
        if self.cur_bs_this_run == self.micro_max_batch_size:
            return UpdateQueryStatus.REACH_MAX_BS

        if self.micro_context_batchsize_limit == self.cur_context_bs_this_run and query.is_context_computing:
            return UpdateQueryStatus.REACH_MAX_CTX_BS

        # Context stage: Allocate kv_slot_ids for query for the first time
        if not query.is_kv_cache_slot_allocated():
            if self.use_vllm:
                context_slots_num = (len(query.input_ids) + self.slot_block_size - 1) // self.slot_block_size
                if self.get_available_slot_num() < context_slots_num + thresold:
                    return UpdateQueryStatus.ALLOC_INSUFFICIENT_SLOT
                query.kv_slot_ids.extend([self.available_slot_table.popleft() for i in range(context_slots_num)])
            else:
                if self.get_available_slot_num() < 1:
                    return UpdateQueryStatus.ALLOC_INSUFFICIENT_SLOT
                query.kv_slot_ids.extend([self.available_slot_table.popleft()])
            return UpdateQueryStatus.SUCCESS
        # Decode stage: Allocate kv_slot_ids for vllm if needed
        elif self.use_vllm:
            tokens_num = len(query.input_ids) + len(query.new_token_ids) + self.num_pred_tokens
            if tokens_num >= len(query.kv_slot_ids) * self.slot_block_size:
                if self.get_available_slot_num() < 1:
                    return UpdateQueryStatus.NEED_SWAP_OUT
                query.kv_slot_ids.append(self.available_slot_table.popleft())

        return UpdateQueryStatus.SUCCESS

    def get_kv_cache_utils(self):
        return (self.slot_num - len(self.available_slot_table)) / self.slot_num
