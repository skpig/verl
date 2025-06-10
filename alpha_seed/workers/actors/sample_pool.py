import numpy as np
from collections import defaultdict, deque
import ray
import queue
import random
import logging
import math
from mono_rl import DataProto

logger = logging.getLogger(__file__)


class SamplePool:
    name = "sample_pool"

    def __init__(self, config):
        self.config = config
        self.sample_pool = []
        self.sample_pool_uid = set()
        self.priority_dict = {}
        self.TD_priority_dict = {}
        self.global_step = 0

    def rearrange_sample_pool(self):
        priority_sample_list = []
        for idx, item in enumerate(self.sample_pool):
            uid = item.non_tensor_batch['index'][0]
            mean_score = sum(self.priority_dict[uid]['scores'][-100:]) / max(
                1, len(self.priority_dict[uid]['scores'][-100:]))
            exploit = 1.0 - abs(mean_score)
            explore = (math.log(self.global_step) / max(1, len(self.priority_dict[uid]['scores'])))**0.5
            upper_confidence_bound = exploit + 0.25 * explore
            priority_sample_list.append((upper_confidence_bound, item))
        priority_sample_list = sorted(priority_sample_list, key=lambda x: x[0], reverse=True)
        self.sample_pool = [i[1] for i in priority_sample_list]

    def rearrange_TD_sample_pool(self, insert_batch_size):
        TD_priority_sample_list = [(k, v) for k, v in self.TD_priority_dict.items() if v > 0]
        TD_priority_sample_list = sorted(TD_priority_sample_list, key=lambda x: x[1], reverse=True)
        TD_priority_sample_list = TD_priority_sample_list[:insert_batch_size]
        TD_priority_sample_list = [x[0] for x in TD_priority_sample_list]
        priority_sample_list = []
        for uid in TD_priority_sample_list:
            for item in self.sample_pool:
                if uid == item.non_tensor_batch['index'][0]:
                    priority_sample_list.append(item)
        for item in self.sample_pool:
            if item.non_tensor_batch['index'][0] not in TD_priority_sample_list:
                priority_sample_list.append(item)
        self.sample_pool = [i for i in priority_sample_list]

    def fill_sample_pool(self, batch):
        self.global_step += 1
        batch_lst = batch.chunk(len(batch))
        for item in batch_lst:
            uid = item.non_tensor_batch['index'][0]
            if uid not in self.priority_dict:
                self.priority_dict[uid] = {'scores': []}
            if uid not in self.priority_dict:
                self.TD_priority_dict[uid] = 0.0
            if uid not in self.sample_pool_uid:
                self.sample_pool_uid.add(uid)
                self.sample_pool.append(item)
        print("[SamplePool] fill_batch:", len(batch_lst), "sample_pool size:", len(self.sample_pool))

    def get_gen_batch(self, return_batch_size):
        return_batch = []
        while len(return_batch) < return_batch_size:
            item = self.sample_pool[0]
            uid = item.non_tensor_batch['index'][0]
            self.sample_pool = self.sample_pool[1:]
            self.sample_pool_uid.remove(uid)
            return_batch.append(item)
        return DataProto.concat(return_batch)

    def update_priority_dict(self, batch):
        batch_lst = batch.chunk(len(batch))
        for item in batch_lst:
            uid = item.non_tensor_batch['index'][0]
            reward = item.batch['token_level_scores'].sum().item()
            reward = 1 if reward > 0 else -1
            self.priority_dict[uid]['scores'].append(reward)

    def update_TD_priority_dict(self, batch):
        batch_lst = batch.chunk(len(batch))
        for item in batch_lst:
            uid = item.non_tensor_batch['index'][0]
            self.TD_priority_dict[uid] = 0.0
        for item in batch_lst:
            uid = item.non_tensor_batch['index'][0]
            TD_error = item.batch['seq_vf'].item()
            self.TD_priority_dict[uid] = max(self.TD_priority_dict[uid], TD_error)
