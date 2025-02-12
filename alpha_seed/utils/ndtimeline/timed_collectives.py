from typing import Optional
import os

import torch
import torch.distributed as dist

import contextlib
try:
    from bytedance.ndtimeline import ndtimeit_coll
    from bytedance.ndtimeline import NDTIMELINE_SEQ_ID_KEY, NDTIMELINE_PG_ID_KEY
except ImportError:

    @contextlib.contextmanager
    def ndtimeit_coll(name, group, tensor, tag):
        yield
        return

    NDTIMELINE_SEQ_ID_KEY = "seq_id"
    NDTIMELINE_PG_ID_KEY = "pg_id"

from alpha_seed.models.transformers.parallel import timed_collectives
from alpha_seed.models.transformers.parallel import collectives


def timed_all_reduce_impl(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False, name: Optional[str] = None):
    if name is None:
        return dist.all_reduce(tensor, op, group, async_op)
    tag = {}
    if group is not None:
        tag = {
            NDTIMELINE_SEQ_ID_KEY: group._get_sequence_number_for_group(),
            NDTIMELINE_PG_ID_KEY: group.group_name,
        }
    with ndtimeit_coll(name, group, tensor, tag):
        work = dist.all_reduce(tensor, op, group, async_op)
    return work


def all_reduce_identity_forward_impl(ctx, itensor: torch.Tensor, group: dist.ProcessGroup, name: Optional[str] = None):
    if name is None:
        dist.all_reduce(itensor, group=group)
        return itensor
    tag = {}
    if group is not None:
        tag = {
            NDTIMELINE_SEQ_ID_KEY: group._get_sequence_number_for_group(),
            NDTIMELINE_PG_ID_KEY: group.group_name,
        }
    with ndtimeit_coll(name, group, itensor, tag):
        dist.all_reduce(itensor, group=group)
    return itensor


def identity_all_reduce_forward_impl(ctx, itensor: torch.Tensor, group: dist.ProcessGroup, name: Optional[str] = None):
    ctx._group = group
    ctx._name = name
    return itensor


def identity_all_reduce_backward_impl(ctx, grad: torch.Tensor):
    group = ctx._group
    name = ctx._name
    if name is None:
        dist.all_reduce(grad, group=group)
        return grad, None, None
    tag = {}
    if group is not None:
        tag = {
            NDTIMELINE_SEQ_ID_KEY: group._get_sequence_number_for_group(),
            NDTIMELINE_PG_ID_KEY: group.group_name,
        }
    with ndtimeit_coll(name, group, grad, tag):
        dist.all_reduce(grad, group=group)
    return grad, None, None


def patch_coll_ops():
    enable_all = (os.getenv("NDTIMELINE_ENABLE_EP_TP_SP_COLL", "0") == "1")

    if enable_all or os.getenv("NDTIMELINE_ENABLE_GGEMM_EP_AR", "0") == "1":
        timed_collectives.TimedDistOP.all_reduce = timed_all_reduce_impl

    if enable_all or os.getenv("NDTIMELINE_ENABLE_ARI", "0") == "1":
        collectives.AllReduceIdentity.forward = all_reduce_identity_forward_impl

    if enable_all or os.getenv("NDTIMELINE_ENABLE_IAR", "0") == "1":
        collectives.IdentityAllreduce.forward = identity_all_reduce_forward_impl
        collectives.IdentityAllreduce.backward = identity_all_reduce_backward_impl
