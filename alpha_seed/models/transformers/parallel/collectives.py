import torch
import torch.distributed as dist


class AllReduceIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, group: dist.ProcessGroup):
        dist.all_reduce(itensor, group=group)
        return itensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def allreduce_identity(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return AllReduceIdentity.apply(tensor, group)


class IdentityAllreduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, group: dist.ProcessGroup):
        ctx._group = group
        return itensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        group = ctx._group
        dist.all_reduce(grad, group=group)
        return grad, None


def identity_allreduce(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return IdentityAllreduce.apply(tensor, group)


def get_memory():
    max_memory_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=torch.cuda.current_device())
    max_memory_reserved = torch.tensor(torch.cuda.max_memory_reserved() / 2**30, device=torch.cuda.current_device())
    if dist.is_initialized():
        dist.all_reduce(max_memory_allocated, op=dist.ReduceOp.MAX, group=None, async_op=False)
        dist.all_reduce(max_memory_reserved, op=dist.ReduceOp.MAX, group=None, async_op=False)
    return max_memory_allocated.item(), max_memory_reserved.item()
