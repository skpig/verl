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
