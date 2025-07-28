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
from regex import F
import torch
from omegaconf import DictConfig
from torch.utils.data import Sampler
from traitlets import default
import numpy as np

from verl import DataProto


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
    def update(self, batch: DataProto) -> None:
        pass


class MoPPSSampler(AbstractCurriculumSampler):
    """Experimental interface for MoPPS samplers."""

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        super().__init__(data_source, data_config)
        self.bsz = data_config.train_batch_size
        print("Initializing MoPPS sampler with batch size:", self.bsz)

        # some hyper
        self.temporal_decay = data_config.sampler.temporal_decay

        # # assert each item has a unique index
        # index_lst = [i['index'] for i in self.data_source]
        # assert len(index_lst) == len(set(index_lst)), "Each item must have a unique index."

        # initialize posterior weights for all prompts
        self.alpha = torch.ones(len(data_source))
        self.beta = torch.ones(len(data_source))


        self.queue = deque(maxlen=self.bsz)
        self.fill_queue()


    
    def update(self, batch: DataProto) -> None:
        """Update the sampler with the current batch."""
        # breakpoint()

        indices = torch.tensor(batch.non_tensor_batch['item'].astype(np.int32)) # item is the index passed to the dataset.__getitem__
        scores = torch.tensor(batch.non_tensor_batch['score'])

        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)

        counts = torch.bincount(inverse_indices, minlength=len(unique_indices))

        score_sums = torch.bincount(inverse_indices, weights=scores, minlength=len(unique_indices))
        score_complements = counts.to(torch.float32) - score_sums

        self.alpha[unique_indices] = (
            self.temporal_decay * self.alpha[unique_indices]
            + (1 - self.temporal_decay) * 1.0
            + score_sums
        ).float()

        self.beta[unique_indices] = (
            self.temporal_decay * self.beta[unique_indices]
            + (1 - self.temporal_decay) * 1.0
            + score_complements
        ).float()

        self.fill_queue()
    
    def fill_queue(self) -> None:
        k = self.bsz - len(self.queue) 

        # sample from the posterior distribution
        posterior = torch.distributions.Beta(self.alpha, self.beta)
        rates = posterior.sample()
        weights = (rates - 0.5).abs()

        # sample the smallest k indices based on the weights
        sorted_indices = torch.topk(weights, k=k, largest=False).indices

        self.queue.extend(sorted_indices.tolist())

    def __iter__(self):
        """Iterate over the sampler."""
        # breakpoint()
        def dynamic_iter():
            while True:
                assert len(self.queue) > 0, "Queue should not be empty."
                yield self.queue.popleft()
        
        return iter(dynamic_iter())
    
    def state_dict(self):
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "queue": list(self.queue),
            "bsz": self.bsz,
            "temporal_decay": self.temporal_decay
        }
    
    def load_state_dict(self, state):
        self.alpha = state['alpha']
        self.beta = state['beta']
        self.queue = deque(state['queue'], maxlen=self.bsz)
        self.bsz = state['bsz']
        self.temporal_decay = state['temporal_decay']


class PrioritySampler(AbstractCurriculumSampler):
    """Experimental interface for priority samplers."""

    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        super().__init__(data_source, data_config)
        self.bsz = data_config.train_batch_size
        print("Initializing Priority sampler with batch size:", self.bsz)

        self.index2acc = {}
        # we always assert drop last
        self.data_item_num = len(data_source)
        self.temporal_decay = data_config.sampler.temporal_decay

        # initialize priority queue
        self.queue = deque()
        for i in range(self.data_item_num):
            self.queue.append(i)
    

    def update(self, batch: DataProto) -> None:
        """Update the sampler with the current batch."""
        indices = torch.tensor(batch.non_tensor_batch['item'].astype(np.int32))  # item is the index passed to the dataset.__getitem__
        scores = torch.tensor(batch.non_tensor_batch['score'])

        unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)

        counts = torch.bincount(inverse_indices, minlength=len(unique_indices))

        score_sums = torch.bincount(inverse_indices, weights=scores, minlength=len(unique_indices))
        score_complements = counts.to(torch.float32) - score_sums

        # update the queue with the new indices and scores
        for index, score_sum, count in zip(unique_indices.tolist(), score_sums.tolist(), counts.tolist()):
            if index not in self.index2acc:
                self.index2acc[index] = score_sum / count
            else:
                self.index2acc[index] = self.temporal_decay * (score_sum / count) + (1 - self.temporal_decay) * self.index2acc[index] # EMA update
        
        self.fill_queue()
        
    def fill_queue(self) -> None:
        """Fill the queue with the highest priority items."""

        if len(self.queue) >= self.bsz:
            return

        k = self.bsz - len(self.queue)

        id_tensor = np.array(list(self.index2acc.keys()))
        acc_tensor = np.array(list(self.index2acc.values()))
        reverse_acc_tensor = 1.0 - acc_tensor
        prob_dist = reverse_acc_tensor / (reverse_acc_tensor.sum() + 1e-8)

        # Sample k indices proportional to their reverse accuracy
        sampled_indices = np.random.choice(
            len(reverse_acc_tensor),  # total number of indices
            size=k,              # number of samples to draw
            replace=False,            # no replacement
            p=prob_dist               # sampling probability
        )

        # Get the actual indices from the sampled indices
        sampled_items = id_tensor[sampled_indices]

        for item in sampled_items:
            self.queue.append(item.item())

    def __iter__(self):
        """Iterate over the sampler."""
        # breakpoint()
        def dynamic_iter():
            while True:
                assert len(self.queue) > 0, "Queue should not be empty."
                yield self.queue.popleft()
        
        return iter(dynamic_iter())
    
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