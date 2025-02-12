import ray
import queue
import random
import logging

logger = logging.getLogger(__file__)


class RolloutPool:
    name = "rollout_pool"

    def __init__(self, config):
        self.config = config
        self.num_bon = self.config.actor_rollout_ref.rollout.get("num_bon", 1)
        self.fn_map = {"default": self.get_train_batch_default}
        self.strategy = self.config.actor_rollout_ref.rollout.get("strategy", "default")
        assert (self.strategy in self.fn_map), "strategy {} not in fn_map, expected in [{}]".format(
            self.strategy, self.fn_map.keys())
        self.pool = dict()
        self.pool_size = 0
        self.history_pool = dict()
        self.bon_ready_batch = queue.Queue()

    def get_train_batch(self, return_batch_size):
        return self.fn_map[self.strategy](return_batch_size)

    def fill_rollout_pool(self, batch_lst):
        for batch in batch_lst:
            index = batch.non_tensor_batch['rollout_id'][0]
            if index not in self.pool:
                self.pool[index] = queue.Queue()
            self.pool[index].put(batch)
            self.pool_size += 1

            if self.pool[index].qsize() >= self.num_bon:
                self.bon_ready_batch.put(index)
        print("[fill_rollout_pool] fill_batch:", len(batch_lst), "bon_ready_batch:",
              self.bon_ready_batch.qsize() * self.num_bon, "pool_size:", self.pool_size)

    def get_train_batch_default(self, return_batch_size):
        return_batch = []
        while not self.bon_ready_batch.empty() and len(return_batch) < return_batch_size:
            index = self.bon_ready_batch.get()
            ready_batch = self.pool[index]
            if len(return_batch) + ready_batch.qsize() > return_batch_size:
                self.bon_ready_batch.put(index)
                break
            return_batch.extend(list(ready_batch.queue))
            self.history_pool[index] = self.pool.pop(index)
            self.pool_size -= self.num_bon
        complete_bon_bsz = len(return_batch)

        while len(return_batch) < return_batch_size and len(self.pool) > 0:
            index = random.choice(list(self.pool.keys()))
            ready_batch = self.pool[index]
            if len(return_batch) + ready_batch.qsize() > return_batch_size:
                break
            return_batch.extend(list(ready_batch.queue))

        if len(return_batch) < return_batch_size:
            return_batch.extend([random.choice(return_batch) for _ in range(return_batch_size - len(return_batch))])
        incomplete_bon_bsz = len(return_batch) - complete_bon_bsz

        print("[get_train_batch] total_train_bsz:", return_batch_size, "complete_bon_bsz:", complete_bon_bsz,
              "incomplete_bon_bsz:", incomplete_bon_bsz, "pool size:", self.pool_size, "history_pool size:",
              len(self.history_pool) * self.num_bon)
        return return_batch

    @staticmethod
    def get_or_create_actor(config):
        rollout_pool = None
        try:
            rollout_pool = ray.get_actor(name=RolloutPool.name)
        except Exception as e:
            rollout_pool = ray.remote(RolloutPool).options(name=RolloutPool.name).remote(config)
        return rollout_pool
