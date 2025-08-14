# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractmethod
from collections import deque
from collections.abc import Sized
import pprint
from regex import F
import torch
from omegaconf import DictConfig
from torch.utils.data import Sampler
from traitlets import default
import numpy as np
from typing import Deque, List, Dict
import traceback
from torch.utils.data import RandomSampler, SequentialSampler

from verl import DataProto
from verl.utils.dataset.rl_dataset import TreeNode


class AbstractSampler(Sampler[int]):
    """Abstract interface for custom samplers."""

    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        pass


class AbstractCurriculumSampler(AbstractSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto, step_num: int) -> None:
        pass

class AbstractBatchSampler(Sampler[List[int]]):
    @abstractmethod
    def __init__(*args, **kargs):
        pass

class AbstractCurriculumBatchSampler(AbstractBatchSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto, step_num: int) -> None:
        pass

class TreeSampler(AbstractCurriculumSampler):
    def __init__(self, data_source: Sized, data_config: DictConfig):
        super().__init__(data_source, data_config)
    
        self.data_source = data_source
        self.original_len = len(data_source)
        self.root = self.data_source.root
        self.item2node: Dict[int, TreeNode] = self.data_source.item2node
    
    def __iter__(self):
        self.idx = 0
        while self.idx < self.original_len:
            # \epsilon greedy sampling
            if np.random.rand() < 0.5 or len(self.item2node[self.idx].children) == 0:
                yield self.idx
            else:
                # 从当前节点的children中随机选择一个
                children = self.item2node[self.idx].children
                child_node = np.random.choice(children)
                yield child_node.item
            self.idx += 1
    
    def update(self, batch: DataProto, step_num: int) -> None:
        self.data_source.update(batch, step_num=step_num)
        return

class MoPPSSampler(AbstractCurriculumBatchSampler):
    """
    MoPPS 版 BatchSampler:
    - 每次 __iter__ 直接返回一个长度 bsz 的索引列表
    - 仍然暴露 update / state_dict / load_state_dict，方便训练闭环
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        super().__init__(data_source)
        self.data_source = data_source
        self.bsz = data_config.train_batch_size
        self.temporal_decay = data_config.sampler.temporal_decay

        # 后验参数
        self.alpha = torch.ones(len(data_source))
        self.beta  = torch.ones(len(data_source))
        print("[Sampler] Initializing MoPPSampler with batch size:", self.bsz, 
                "dataset length:", len(data_source))

        # 这里 queue 还是存“单条索引”，方便 fill_queue 逻辑复用
        self.queue: Deque[int] = deque(maxlen=self.bsz)
        self.index2acc = {}
        self.fill_queue()

    # ---------- 训练后更新 ----------
    def update(self, batch: DataProto, step_num: int) -> None:
        """batch 内必须带 'item' (索引) 和 'score' (0/1 or 回归分数)"""
        indices = torch.tensor(batch.non_tensor_batch["item"].astype(np.int32))
        scores  = torch.tensor(batch.non_tensor_batch["score"])
        print("[Sampler] Update with indices:", indices.tolist())

        unique_idx, inverse = torch.unique(indices, return_inverse=True)
        counts = torch.bincount(inverse, minlength=len(unique_idx))
        score_sum = torch.bincount(inverse, weights=scores, minlength=len(unique_idx))
        score_comp = counts.to(torch.float32) - score_sum

        # 指数衰减 + 观测更新
        self.alpha[unique_idx] = (
            self.temporal_decay * self.alpha[unique_idx]
            + (1 - self.temporal_decay) * 1.0
            + score_sum
        ).float()

        self.beta[unique_idx] = (
            self.temporal_decay * self.beta[unique_idx]
            + (1 - self.temporal_decay) * 1.0
            + score_comp
        ).float()

        # update the accuracy tracking with the new indices and scores
        for index, score_sum, count in zip(unique_idx.tolist(), score_sum.tolist(), counts.tolist()):
            self.index2acc[index] = score_sum / count

        # 补货
        self.fill_queue()


    # ---------- 采样核心 ----------
    def fill_queue(self):
        k = self.bsz - len(self.queue)
        print(f"[Sampler] fill queue from {len(self.queue)} to {self.bsz} items")
        if k <= 0:
            return

        print("[Sampler] fill queue")
        # 1) 抽 posterior
        posterior = torch.distributions.Beta(self.alpha, self.beta)
        rates = posterior.sample()                 # [N]
        weights = (rates - 0.5).abs()              # 不确定度

        # 2) 选最小权重的 k 个样本补入队列
        #    注意：可能出现重复，为避免刷屏可随机打乱或加去重
        new_indices = torch.topk(weights, k=k, largest=False).indices
        # print("[Sampler] rates:")
        # pprint.pprint(rates[new_indices].tolist())
        # print("[Sampler] acc:")
        # pprint.pprint([self.index2acc.get(idx, -1) for idx in new_indices.tolist()])
        print("[Sampler] rates/acc")
        print([
            (f'{r:.2f}', f'{self.index2acc.get(idx, -1):.2f}')
            for r, idx in zip(rates[new_indices].tolist(),new_indices.tolist())
        ])
        self.queue.extend(new_indices.tolist())

    # ---------- 迭代 ----------
    def __iter__(self):
        while True:
            # # 若不够一个 batch 就补货
            if len(self.queue) < self.bsz:
                print("[Sampler] Current queue size: ", len(self.queue))
                print("[Sampler] Not enough items in queue, filling queue...")
                # print current runtime stack
                traceback.print_stack()
                self.fill_queue()
            # assert len(self.queue) >= self.bsz

            # 组装一个 batch
            print("[Sampler] Pop queue")
            batch = [self.queue.popleft() for _ in range(self.bsz)]
            print("[Sampler] Current Train Batch: ", batch)
            yield batch

    # ---------- 状态保存 / 恢复 ----------
    def state_dict(self):
        return {
            "alpha": self.alpha,
            "beta":  self.beta,
            "queue": list(self.queue),
            "bsz":   self.bsz,
            "temporal_decay": self.temporal_decay,
        }

    def load_state_dict(self, state):
        self.alpha = state["alpha"]
        self.beta  = state["beta"]
        self.queue = deque(state["queue"], maxlen=self.bsz)
        # 其余字段按需加载
        # self.bsz = state['bsz']
        # self.temporal_decay = state['temporal_decay']

class PrioritySampler(AbstractBatchSampler):
    """Priority-based BatchSampler that returns batches of indices."""

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        self.bsz = data_config.train_batch_size
        print("Initializing Priority BatchSampler with batch size:", self.bsz)

        # we always assert drop last
        self.data_item_num = len(data_source)
        self.temporal_decay = data_config.sampler.temporal_decay
        
        # Initialize accuracy tensor with -1 (indicating uninitialized)
        self.index2acc = torch.full((self.data_item_num,), -1.0, dtype=torch.float32)

        # initialize priority queue
        self.queue = deque()
        for i in range(self.data_item_num):
            self.queue.append(i)
    

    def update(self, batch: DataProto, step_num: int) -> None:
        """Update the sampler with the current batch."""
        indices = torch.tensor(batch.non_tensor_batch['item'].astype(np.int32))  # item is the index passed to the dataset.__getitem__
        scores = torch.tensor(batch.non_tensor_batch['score'])

        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)

        counts = torch.bincount(inverse_indices, minlength=len(unique_indices))

        score_sums = torch.bincount(inverse_indices, weights=scores, minlength=len(unique_indices))
        score_complements = counts.to(torch.float32) - score_sums

        # update the accuracy tracking with the new indices and scores using tensor operations
        new_acc = score_sums.float() / counts.float()  # compute new accuracy for each unique index
        
        # Create masks for first-time and existing indices
        first_time_mask = self.index2acc[unique_indices] == -1.0
        existing_mask = ~first_time_mask
        
        # Update first-time indices
        self.index2acc[unique_indices[first_time_mask]] = new_acc[first_time_mask]
        
        # Update existing indices with EMA
        self.index2acc[unique_indices[existing_mask]] = (
            self.temporal_decay * new_acc[existing_mask] + 
            (1 - self.temporal_decay) * self.index2acc[unique_indices[existing_mask]]
        )
        
        self.fill_queue()
        
    def fill_queue(self) -> None:
        """Fill the queue with the highest priority items."""
        k = self.bsz - len(self.queue)
        if k <= 0:
            return

        # Check that all indices have been initialized (no -1 values)
        assert torch.all(self.index2acc != -1.0), "Should not fill queue before calculate acc for all indices"

        # Use tensor operations directly
        reverse_acc_tensor = 1.0 - self.index2acc
        prob_dist = reverse_acc_tensor / (reverse_acc_tensor.sum() + 1e-8)

        # Sample k indices proportional to their reverse accuracy
        sampled_indices = torch.multinomial(prob_dist, num_samples=k, replacement=False)

        # Add sampled indices to queue
        for idx in sampled_indices:
            self.queue.append(idx.item())

    def __iter__(self):
        """Iterate over the sampler, yielding batches of indices."""
        while True:
            # Ensure we have enough items in the queue for a full batch
            # if len(self.queue) < self.bsz:
            #     self.fill_queue()
            assert len(self.queue) >= self.bsz
            
            # Create a batch by popping bsz items from the queue
            batch = [self.queue.popleft() for _ in range(self.bsz)]
            yield batch
    
    def state_dict(self):
        return {
            "index2acc": self.index2acc,
            "queue": list(self.queue),
            "data_item_num": self.data_item_num,
            "bsz": self.bsz,
            "temporal_decay": self.temporal_decay
        }

    def load_state_dict(self, state):
        self.index2acc = state['index2acc']
        self.queue = deque(state['queue'])
        self.data_item_num = state['data_item_num']
        self.bsz = state['bsz']
        self.temporal_decay = state['temporal_decay']