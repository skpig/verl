import numpy as np
from collections import defaultdict
import ray
import random
import queue
import logging
import copy

from mono_rl import DataProto
from alpha_seed.utils.dataset.dist_data_util import release_ref_counts, get_image_manager

logger = logging.getLogger(__file__)
'''
The replay buffer backed by an in-mem dict.
It's NOT thread safe.
'''


class VanillaReplayBufferClient():

    def __init__(self):
        self.__pool = dict()

    def push(self, key, batches):
        if not isinstance(batches, list):
            batches = [batches]
        if key not in self.__pool:
            self.__pool[key] = []
        for batch in batches:
            self.__pool[key].append(batch)

    def get(self, key: str):
        return self.__pool.get(key, None)

    def sample(self):
        keys = list(self.__pool.keys())
        random.shuffle(keys)
        for key in keys:
            yield key

    def delete(self, key: str):
        return self.__pool.pop(key)


class RolloutPool:
    name = "rollout_pool"

    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.num_bon = self.config.actor_rollout_ref.rollout.get("num_bon", 1)
        self.fn_map = {"default": self.get_train_batch_default}
        self.strategy = self.config.actor_rollout_ref.rollout.get("strategy", "default")
        assert (self.strategy in self.fn_map), "strategy {} not in fn_map, expected in [{}]".format(
            self.strategy, self.fn_map.keys())
        self.replay_buffer_type = self.config.actor_rollout_ref.rollout.get("replay_buffer_type", "default")
        assert self.replay_buffer_type in ["default", "persistable"]
        if self.replay_buffer_type == "default":
            self.pool = VanillaReplayBufferClient()
        elif self.replay_buffer_type == "persistable":
            # requires verl verion >= 1.0.0.366
            from mono_rl.utils.replay_buffer.persistable_replay_buffer_client import PersistableReplayBufferClient
            replay_buffer_name = self.config.actor_rollout_ref.rollout.get("replay_buffer_name", "replay_buffer")
            assert len(replay_buffer_name) != 0
            cache_size_limit_in_mb = self.config.actor_rollout_ref.rollout.get("replay_buffer_in_memory_cache_limit_mb",
                                                                               1024)
            hdfs_path = self.config.actor_rollout_ref.rollout.get("replay_buffer_hdfs_path", None)
            from mono_rl.utils.replay_buffer.samplers.uniform_key_sampler import UniformKeySampler
            self.samplers = [UniformKeySampler()]  # uniform sampling
            self.pool = PersistableReplayBufferClient(replay_buffer_name, cache_size_limit_in_mb, hdfs_path,
                                                      self.samplers)

        self.pool_size = 0
        self.history_pool = dict()

        # index of bon_ready_batch and rollout_id2uid should be poped out simutaneously
        self.bon_ready_batch = queue.Queue()
        self.rollout_id2uid = defaultdict(set)

        self.pool_with_grad = queue.Queue()
        self.image_manager = get_image_manager()
        # self.pool_with_grad_ready_batch = queue.Queue()

    def get_train_batch(self):
        return self.fn_map[self.strategy]()

    def fill_rollout_pool(self, batch_lst):

        if len(batch_lst) != 0:
            batch_lst = DataProto.concat(batch_lst)
            batch_lst = batch_lst.chunk(len(batch_lst))

            for batch in batch_lst:
                rollout_id = batch.non_tensor_batch['rollout_id'][0]
                uid = batch.non_tensor_batch['uid'][0]
                batch.meta_info = copy.deepcopy(batch.meta_info)
                self.pool.push(rollout_id, batch)
                self.pool_size += 1

                self.rollout_id2uid[rollout_id].add(uid)
                if len(self.rollout_id2uid[rollout_id]) >= self.num_bon:
                    self.bon_ready_batch.put(rollout_id)

        print("[fill_rollout_pool] fill_batch:", len(batch_lst), "bon_ready_batch:",
              self.bon_ready_batch.qsize() * self.num_bon, "pool_size:", self.pool_size)

    def fill_rollout_pool_dynamic_sampling(self, batch):
        batch_lst = batch.chunk(len(batch))
        # score，根据score来判定要不要进pool_with_grad
        id2data = defaultdict(list)
        id2acc = defaultdict(list)

        size_before_fill = self.pool_with_grad.qsize()

        # get acc
        for item in batch_lst:
            score = item.batch['token_level_scores'].sum(-1).item()
            item.meta_info = copy.deepcopy(item.meta_info)
            id2acc[item.non_tensor_batch['rollout_id'][0]].append(score)
            id2data[item.non_tensor_batch['rollout_id'][0]].append(item)
        for k, v in id2acc.items():
            id2acc[k] = np.mean(v)

        for k, v in id2acc.items():
            if self.config.algorithm.dynamic_sampling.strategy == 'v1':
                if v != self.config.algorithm.dynamic_sampling.min_score and v != self.config.algorithm.dynamic_sampling.max_score:
                    for item in id2data[k]:
                        self.pool_with_grad.put(item)
            elif self.config.algorithm.dynamic_sampling.strategy == 'v2':
                if v >= self.config.algorithm.dynamic_sampling.min_score and v <= self.config.algorithm.dynamic_sampling.max_score:
                    for item in id2data[k]:
                        self.pool_with_grad.put(item)
            elif self.config.algorithm.dynamic_sampling.strategy == 'v3':
                if v > self.config.algorithm.dynamic_sampling.min_score and v < self.config.algorithm.dynamic_sampling.max_score:
                    for item in id2data[k]:
                        self.pool_with_grad.put(item)

        print("[fill_rollout_pool_dynamic_sampling] fill_batch:", len(batch_lst), "pool_size before fill:",
              size_before_fill, "pool_size after fill:", self.pool_with_grad.qsize())
        return self.pool_with_grad.qsize() - size_before_fill, self.pool_with_grad.qsize()

    def get_dynamic_sampling_pool_size(self):
        return self.pool_with_grad.qsize()

    def pool_with_grad_clear(self):
        self.pool_with_grad = queue.Queue()

    def get_train_batch_default(self):
        return_batch_size = self.config.data.train_batch_size * self.config.trainer.league_training_config.buffer_size * self.config.actor_rollout_ref.rollout.get(
            "num_bon", 1)
        return_batch = []
        while not self.bon_ready_batch.empty() and (
                self.config.actor_rollout_ref.rollout.rollout_pool.clear_rollout_pool or
                len(return_batch) < return_batch_size):
            index = self.bon_ready_batch.get()
            ready_batch = self.pool.get(index)
            if ready_batch is None:
                continue
            if (len(return_batch) + len(ready_batch)
                    > return_batch_size) and not self.config.actor_rollout_ref.rollout.rollout_pool.clear_rollout_pool:
                self.bon_ready_batch.put(index)
                break
            return_batch.extend(ready_batch)
            uids = self.rollout_id2uid.pop(index)
            # self.history_pool[index] = ready_batch
            # currently ready_batch in history_pool is not used
            # ready_batch will occupy very large memory, especially in vlm tasks
            empty_ready_batch = {}
            self.history_pool[index] = empty_ready_batch
            delete_batch = self.pool.delete(index)
            release_ref_counts(self.image_manager, delete_batch)
            self.pool_size -= self.num_bon
        complete_bon_bsz = len(return_batch)

        # TODO(qiying): default behaviour should not have replay buffer sampling
        # if self.replay_buffer_type == "persistable":
        #     sampler = self.samplers[0].sample()
        # else:  # default
        #     sampler = self.pool.sample()

        # upsample
        mini_bsz = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        if self.config.actor_rollout_ref.rollout.rollout_pool.upsample_mode == 'batch':
            if len(return_batch) < return_batch_size and len(return_batch) > 0:
                return_batch.extend([random.choice(return_batch) for _ in range(return_batch_size - len(return_batch))])
        elif self.config.actor_rollout_ref.rollout.rollout_pool.upsample_mode == 'mini_batch':
            # upsample to multiple of mini_bsz
            if len(return_batch) % mini_bsz != 0:
                return_batch.extend(
                    [random.choice(return_batch) for _ in range(mini_bsz - len(return_batch) % mini_bsz)])
        else:
            assert False
        incomplete_bon_bsz = len(return_batch) - complete_bon_bsz

        # drop to max = max_batch_size
        if self.config.actor_rollout_ref.actor.max_ppo_mini_batch > 0 and len(
                return_batch) > mini_bsz * self.config.actor_rollout_ref.actor.max_ppo_mini_batch:
            random.shuffle(return_batch)
            return_batch = return_batch[:mini_bsz * self.config.actor_rollout_ref.actor.max_ppo_mini_batch]

        print("[get_train_batch] total_train_bsz:", len(return_batch), "complete_bon_bsz:", complete_bon_bsz,
              "incomplete_bon_bsz:", incomplete_bon_bsz, "pool size:", self.pool_size, "history_pool size:",
              len(self.history_pool) * self.num_bon)
        return return_batch

    def get_train_batch_grad(self, return_batch_size):
        return_batch = []
        while return_batch_size > 0:
            item = self.pool_with_grad.get()
            return_batch.append(item)
            return_batch_size -= 1
        return return_batch

    @staticmethod
    def get_or_create_actor(config, mode="local"):
        # TODO: support distributed-ray mode
        if mode == "local":
            return RolloutPool(config, mode)
        rollout_pool = None
        try:
            rollout_pool = ray.get_actor(name=RolloutPool.name)
        except Exception as e:
            rollout_pool = ray.remote(RolloutPool).options(name=RolloutPool.name).remote(config, mode)
        return rollout_pool

    @staticmethod
    def dynamic_call(obj, method_name, *args, **kwargs):
        if obj.mode == "ray":
            method_ref = getattr(obj, method_name).remote(*args, **kwargs)
            return ray.get(method_ref)
        else:
            method = getattr(obj, method_name)
            return method(*args, **kwargs)
