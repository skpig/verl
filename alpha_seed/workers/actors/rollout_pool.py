import numpy as np
from collections import defaultdict
import ray
import random
import queue
import logging
import copy
import torch

from mono_rl import DataProto
from alpha_seed.utils.dataset.dist_data_util import release_ref_counts, add_ref_counts, init_or_get_image_manager
from alpha_seed.utils.reward_score import NON_AGENT_PLACE_HOLDER_SCORE

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
        self.pre_process_stats_fn_map = {
            "default": lambda batch_lst, cur_step, num_fillin_th: (batch_lst, {}),  # 默认策略什么都不做，直接返回
            "swalm": self.pre_process_stats_swalm
        }
        self.post_process_stats_fn_map = {
            "default": lambda batch, rollout_id: batch,  # 默认策略什么都不做，直接返回
            "swalm": self.post_process_stats_swalm
        }
        self.strategy = self.config.actor_rollout_ref.rollout.get("strategy", "default")
        self.process_stats_strategy = self.config.data.get("process_stats_strategy", "default")
        assert (self.strategy in self.fn_map), "strategy {} not in fn_map, expected in [{}]".format(
            self.strategy, self.fn_map.keys())
        assert (
            self.process_stats_strategy
            in self.pre_process_stats_fn_map), "strategy {} not in pre_process_stats_fn_map, expected in [{}]".format(
                self.process_stats_strategy, self.pre_process_stats_fn_map.keys())
        assert (
            self.process_stats_strategy
            in self.post_process_stats_fn_map), "strategy {} not in post_process_stats_fn_map, expected in [{}]".format(
                self.process_stats_strategy, self.post_process_stats_fn_map.keys())
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

        self.uid2score = dict()
        self.rid2score_mean = dict()
        self.rid2score_std = dict()

        self.pool_with_grad = queue.Queue()
        stable_pool_names = self.config.elastic.resource_pools.stable_pool_names
        stable_pool_name = stable_pool_names[0] if stable_pool_names else ''
        self.image_manager = init_or_get_image_manager(stable_pool_name)
        # self.pool_with_grad_ready_batch = queue.Queue()

    def get_ready_pool_size(self):
        return self.bon_ready_batch.qsize() * self.num_bon

    def get_train_batch(self):
        return self.fn_map[self.strategy]()

    def pre_process_stats_swalm(self, batch_lst, cur_step, num_fillin_th=None):
        ready_bon_num = 0
        ready_traj_num = 0
        dropped_env_failure_traj_num = 0
        dropped_dynamic_sampling_traj_num = 0
        dropped_offpolicy_sample_num = 0
        success_to_fail_traj_num = 0
        total_raw_score = []

        if num_fillin_th is None:
            num_fillin_th = self.num_bon

        # drop offpolocy
        rollout_offpolicy_step_th = self.config.trainer.get("rollout_agent_offpolicy_step_th", -1)
        if rollout_offpolicy_step_th >= 0:
            keep_bon_ready_batch = []
            while not self.bon_ready_batch.empty():
                index = self.bon_ready_batch.get()
                ready_batch = self.pool.get(index)
                if ready_batch is None:
                    continue
                ready_uids = set()
                for batch in ready_batch:
                    start_step = batch.meta_info["cur_step"]
                    if cur_step - start_step <= rollout_offpolicy_step_th:
                        ready_uids.add(batch.non_tensor_batch['uid'][0])
                if len(ready_uids) >= num_fillin_th:
                    keep_bon_ready_batch.append(index)
                else:
                    dropped_offpolicy_sample_num += len(
                        self.rollout_id2uid[index])  # agent metric: dropped_offpolicy_sample_num
                    self.rollout_id2uid.pop(index)
            for index in keep_bon_ready_batch:
                self.bon_ready_batch.put(index)

        bon_ready_rollout_id = set()
        keep_batch_lst = []
        if len(batch_lst) != 0:
            for batch in batch_lst:
                rollout_id = batch.non_tensor_batch['rollout_id'][0]
                uid = batch.non_tensor_batch['uid'][0]
                if batch.non_tensor_batch['ability'][0] == "swalm_env":
                    swalm_agent_score = batch.batch["swalm_agent_score"][0].item()
                    from alpha_seed.workers.agents.handlers.swalm.swalm_handler import SWALM_ENV_FAIL_SCORE
                    if swalm_agent_score != SWALM_ENV_FAIL_SCORE:
                        self.uid2score[uid] = swalm_agent_score
                        keep_batch_lst.append(batch)
                        total_raw_score.append(swalm_agent_score)
                    else:
                        dropped_env_failure_traj_num += 1  # agent metric: dropped_env_failure_traj_num
                        continue
                else:
                    batch.batch["swalm_agent_score"] = torch.tensor(NON_AGENT_PLACE_HOLDER_SCORE).repeat(len(batch))
                    if batch.non_tensor_batch["extra_info"][0].get('is_success_to_fail', False):
                        success_to_fail_traj_num += 1
                    keep_batch_lst.append(batch)

                self.rollout_id2uid[rollout_id].add(uid)
                if len(self.rollout_id2uid[rollout_id]) >= num_fillin_th:
                    self.bon_ready_batch.put(rollout_id)
                    bon_ready_rollout_id.add(rollout_id)

            agent_bon_strategy = self.config.actor_rollout_ref.rollout.get("agent_bon_strategy", "all")
            for rollout_id in bon_ready_rollout_id:
                ready_bon_num += 1  # agent metric: ready_bon_num
                ready_traj_num += len(self.rollout_id2uid[rollout_id])  # agent metric: ready_traj_num
                swalm_agent_scores = []
                for uid in self.rollout_id2uid[rollout_id]:
                    if uid in self.uid2score:
                        swalm_agent_scores.append(self.uid2score[uid])

                if len(swalm_agent_scores) == 1:
                    self.rid2score_mean[rollout_id] = torch.tensor(0.0)
                    self.rid2score_std[rollout_id] = torch.tensor(1.0)
                elif len(swalm_agent_scores) > 1:
                    swalm_agent_scores = torch.tensor(swalm_agent_scores)
                    mean_score = torch.mean(swalm_agent_scores)
                    std_score = torch.std(swalm_agent_scores)
                    if (agent_bon_strategy != "bon_filter") or ((agent_bon_strategy == "bon_filter") and
                                                                (std_score == 0)):
                        self.rid2score_mean[rollout_id] = mean_score
                        self.rid2score_std[rollout_id] = std_score
                    else:
                        dropped_dynamic_sampling_traj_num += len(
                            self.rollout_id2uid[rollout_id])  # agent metric: dropped_dynamic_sampling_traj_num
                        self.rollout_id2uid.pop(
                            rollout_id
                        )  # attention: drop rollout_id in rollout_id2uid, will drop rollout_id in self.bon_ready_batch laterly
        metrics = {
            "rollout/agent/ready_bon_num": ready_bon_num,
            "rollout/agent/ready_traj_num": ready_traj_num,
            "rollout/agent/dropped_env_failure_traj_num": dropped_env_failure_traj_num,
            "rollout/agent/dropped_dynamic_sampling_traj_num": dropped_dynamic_sampling_traj_num,
            "rollout/agent/dropped_offpolicy_sample_num": dropped_offpolicy_sample_num,
            "rollout/agent/success_to_fail_traj_num": success_to_fail_traj_num,
            "score/agent/total_raw": sum(total_raw_score) / max(1, len(total_raw_score))
        }
        return keep_batch_lst, metrics

    def fill_rollout_pool(self, batch_lst, cur_step):

        metrics = {}
        fill_sample_num = 0
        rollout_fillin_bon_rate = self.config.data.get("rollout_fillin_bon_rate", -1)
        if rollout_fillin_bon_rate > 0:
            num_fillin_th = max(int(rollout_fillin_bon_rate * self.num_bon), 1)
        else:
            num_fillin_th = self.num_bon

        batch_lst, pre_process_metrics = self.pre_process_stats_fn_map[self.process_stats_strategy](batch_lst, cur_step,
                                                                                                    num_fillin_th)
        if len(batch_lst) != 0:
            batch_lst = DataProto.concat(batch_lst)
            batch_lst = batch_lst.chunk(len(batch_lst))
            for batch in batch_lst:
                rollout_id = batch.non_tensor_batch['rollout_id'][0]
                uid = batch.non_tensor_batch['uid'][0]
                batch.meta_info = copy.deepcopy(batch.meta_info)
                self.pool.push(rollout_id, batch)
                self.pool_size += 1
                fill_sample_num += 1

                if self.process_stats_strategy == "default":
                    self.rollout_id2uid[rollout_id].add(uid)
                    if len(self.rollout_id2uid[rollout_id]) >= num_fillin_th:
                        self.bon_ready_batch.put(rollout_id)
            add_ref_counts(self.image_manager, batch_lst)

        print("[fill_rollout_pool] fill_batch:", len(batch_lst), "bon_ready_batch:",
              self.bon_ready_batch.qsize() * self.num_bon, "pool_size:", self.pool_size)

        metrics.update({"rollout/fill_sample_num": fill_sample_num})
        metrics.update(pre_process_metrics)
        return metrics

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

    def post_process_stats_swalm(self, ready_batch, rollout_id):
        for batch in ready_batch:
            if rollout_id in self.rid2score_mean:
                batch.batch['token_level_scores_mean'] = self.rid2score_mean[rollout_id].repeat(len(batch))
                batch.batch['token_level_scores_std'] = self.rid2score_std[rollout_id].repeat(len(batch))
            else:
                batch.batch['token_level_scores_mean'] = torch.tensor(NON_AGENT_PLACE_HOLDER_SCORE).repeat(len(batch))
                batch.batch['token_level_scores_std'] = torch.tensor(NON_AGENT_PLACE_HOLDER_SCORE).repeat(len(batch))
        return ready_batch

    def get_agent_bon_ready_bsz(self):
        batch_size = 0
        rollout_fillin_bon_rate = self.config.data.get("rollout_fillin_bon_rate", -1)
        if rollout_fillin_bon_rate > 0:
            num_fillin_th = max(int(rollout_fillin_bon_rate * self.num_bon), 1)
        else:
            num_fillin_th = self.num_bon
        for rollout_id in self.rollout_id2uid:
            if len(self.rollout_id2uid[rollout_id]) >= num_fillin_th:
                batch_size += len(self.pool.get(rollout_id))
        return batch_size

    def get_train_batch_default(self):
        rollout_return_bsz = self.config.data.get("rollout_return_bsz", None)
        if rollout_return_bsz is None:
            return_batch_size = self.config.data.train_batch_size * self.config.trainer.league_training_config.buffer_size * self.config.actor_rollout_ref.rollout.get(
                "num_bon", 1)
        else:
            return_batch_size = rollout_return_bsz
        total_bon_ready_batch_size = self.get_agent_bon_ready_bsz()
        print(f"[get_train_batch] total_bon_ready_bsz: {total_bon_ready_batch_size}")
        if (rollout_return_bsz is not None) and (total_bon_ready_batch_size < return_batch_size):
            return [], {}
        return_batch = []
        while not self.bon_ready_batch.empty() and (
                self.config.actor_rollout_ref.rollout.rollout_pool.clear_rollout_pool or
                len(return_batch) < return_batch_size):
            index = self.bon_ready_batch.get()
            if index not in self.rollout_id2uid:  # attention: drop rollout_id in self.bon_ready_batch for dynamic sampling
                continue
            ready_batch = self.pool.get(index)
            if ready_batch is None:
                continue
            if (len(return_batch) + len(ready_batch)
                    > return_batch_size) and not self.config.actor_rollout_ref.rollout.rollout_pool.clear_rollout_pool:
                self.bon_ready_batch.put(index)
                break
            ready_batch = self.post_process_stats_fn_map[self.process_stats_strategy](ready_batch, index)
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
        metrics = {
            "rollout/total_bon_ready_batch_size": total_bon_ready_batch_size,
            "rollout/complete_bon_bsz": complete_bon_bsz,
        }
        return return_batch, metrics

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
