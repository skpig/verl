from typing import *
from dataclasses import dataclass
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


class CacheManager:

    def __init__(self,
                 slot_num,
                 max_batch_size,
                 pp_size,
                 use_vllm,
                 slot_block_size=256,
                 context_batchsize_limit=1,
                 enable_ngrams_decoding=False,
                 num_pred_tokens=0,
                 moving_avg_length=1024):
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
            if (self._update_query(query)):
                phase1_running.append(query)
                self.cur_context_bs_this_run += int(query.is_context_computing)
                self.cur_bs_this_run += 1
            else:
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
            if (self._update_query(query, thresold=self.moving_avg_len // self.slot_block_size)):
                phase0_running.append(query)
                self.cur_context_bs_this_run += int(query.is_context_computing)
                self.cur_bs_this_run += 1
                query.first_scheduled_time = time.time() * 1000
            else:
                waiting.append(query)

        return phase0_running + phase1_running, waiting

    def release_query(self, query: Query):
        if len(query.kv_slot_ids) > 0:
            # clear cache
            for slot in query.kv_slot_ids:
                self.available_slot_table.appendleft(slot)
            query.kv_slot_ids.clear()
            logging_rank_only(logging.debug, 0, "kv utils {}".format(
                (self.slot_num - len(self.available_slot_table)) / self.slot_num))

    def _update_query(self, query: Query, thresold: int = 0):
        if self.cur_bs_this_run == self.micro_max_batch_size:
            return False

        if self.micro_context_batchsize_limit == self.cur_context_bs_this_run and query.is_context_computing:
            return False

        # Context stage: Allocate kv_slot_ids for query for the first time
        if not query.is_kv_cache_slot_allocated():
            if self.use_vllm:
                context_slots_num = (len(query.input_ids) + self.slot_block_size - 1) // self.slot_block_size
                if self.get_available_slot_num() < context_slots_num + thresold:
                    return False
                query.kv_slot_ids.extend([self.available_slot_table.popleft() for i in range(context_slots_num)])
            else:
                if self.get_available_slot_num() < 1:
                    return False
                query.kv_slot_ids.extend([self.available_slot_table.popleft()])
            return True
        # Decode stage: Allocate kv_slot_ids for vllm if needed
        elif self.use_vllm:
            tokens_num = len(query.input_ids) + len(query.new_token_ids) + self.num_pred_tokens
            if tokens_num >= len(query.kv_slot_ids) * self.slot_block_size:
                if self.get_available_slot_num() < 1:
                    self.page_swap_out_bs += 1
                    self.page_swap_out_token += len(query.input_ids) + len(query.new_token_ids)
                    return False
                query.kv_slot_ids.append(self.available_slot_table.popleft())

        return True

    def get_kv_cache_utils(self):
        return (self.slot_num - len(self.available_slot_table)) / self.slot_num
