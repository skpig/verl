import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
import os
import random


def torchrun(ngpus, test_fn, *args, **kwargs):
    """
    Usage for pytest

    test_xxx = functools.partial(torchrun, 2, example)
    """
    assert len(kwargs) == 0, f"kwargs not supported"
    if ngpus == 1:
        return test_fn(*args)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(random.randint(10000, 60000))
        spawn(
            entry_fn,
            args=(ngpus, test_fn, *args),
            nprocs=ngpus,
        )


def entry_fn(rank, world_size, fn, *args, **kwargs):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    try:
        fn(*args, **kwargs)
    finally:
        dist.destroy_process_group()
