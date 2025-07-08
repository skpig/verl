from typing import Optional
import torch
import torch.distributed as dist


class AllReduceIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, group: dist.ProcessGroup, name: Optional[str] = None):
        dist.all_reduce(itensor, group=group)
        return itensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# wangchenyuan.99: add `name` for compability with timer's monkey patch
def allreduce_identity(tensor: torch.Tensor, group: dist.ProcessGroup, name: Optional[str] = None) -> torch.Tensor:
    return AllReduceIdentity.apply(tensor, group, name)


class IdentityAllreduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, group: dist.ProcessGroup, name: Optional[str] = None):
        ctx._group = group
        return itensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        group = ctx._group
        dist.all_reduce(grad, group=group)
        return grad, None, None


# wangchenyuan.99: add `name` for compability with timer's monkey patch
def identity_allreduce(tensor: torch.Tensor, group: dist.ProcessGroup, name: Optional[str] = None) -> torch.Tensor:
    return IdentityAllreduce.apply(tensor, group, name)


def get_memory(group=None):
    max_memory_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=torch.cuda.current_device())
    max_memory_reserved = torch.tensor(torch.cuda.max_memory_reserved() / 2**30, device=torch.cuda.current_device())
    if dist.is_initialized():
        dist.all_reduce(max_memory_allocated, op=dist.ReduceOp.MAX, group=group, async_op=False)
        dist.all_reduce(max_memory_reserved, op=dist.ReduceOp.MAX, group=group, async_op=False)
    return max_memory_allocated.item(), max_memory_reserved.item()
