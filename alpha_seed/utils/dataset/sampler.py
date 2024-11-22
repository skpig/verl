"""
Sampler that support pickles
"""

from typing import Sized, Optional, Iterator
from torch.utils.data import Sampler

import torch


class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source
        self.current_index = 0

    def __iter__(self) -> Iterator[int]:
        current_list = list(range(len(self.data_source)))
        for i in range(self.current_index, len(current_list)):
            self.current_index = (i + 1) % len(current_list)
            yield i

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(self,
                 data_source: Sized,
                 replacement: bool = False,
                 num_samples: Optional[int] = None,
                 generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        self.current_index = 0
        self.current_list = None

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        assert self.num_samples == n, f'Changing the dataset size during iteration is not allwowed.'
        assert not self.replacement, f'RandomSampler with replacement=True is not supported.'

        if self.current_index == 0:
            self.current_list = torch.randperm(n, generator=generator).tolist()

        for i in range(self.current_index, self.num_samples):
            idx = self.current_list[i]
            self.current_index = (i + 1) % self.num_samples
            yield idx

    def __len__(self) -> int:
        return self.num_samples
