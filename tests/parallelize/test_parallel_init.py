import torch
from torch.distributed._tensor import Shard
import torch.distributed as dist
from alpha_seed.models.transformers.parallel.parallelize import parallelize_module
from alpha_seed.workers.actors.initialize import meta_device_init
from torch.distributed.device_mesh import init_device_mesh
from alpha_seed.models.transformers.parallel.collectives import identity_allreduce, allreduce_identity

from ..launch import torchrun
from functools import partial


class MLP(torch.nn.Module):

    def __init__(self, dim: int = 1024):
        super().__init__()
        # linears
        self.linear1 = torch.nn.Linear(2, dim, bias=False)
        self.linear2 = torch.nn.Linear(dim, 2, bias=False)

    def forward(self, x):
        if hasattr(self, "_tp_mesh") and self._tp_mesh is not None:
            x = identity_allreduce(x, group=self._tp_mesh.get_group())
        x = self.linear1(x)
        x = self.linear2(x)
        if hasattr(self, "_tp_mesh") and self._tp_mesh is not None:
            x = allreduce_identity(x, group=self._tp_mesh.get_group())
        return x


class DummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Embedding(1024, 2)
        # linears
        self.mlp1 = MLP()
        self.mlp2 = MLP()

        self.lm_head = torch.nn.Linear(2, 1024, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, x):
        x = self.wte(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.lm_head(x)
        return x


tp_plan = {"linear1.weight": Shard(0), "linear2.weight": Shard(1)}


def parallelize_init():
    world_size = dist.get_world_size()
    tp_mesh = init_device_mesh("cuda", (world_size,))

    with meta_device_init():
        model = DummyModel()

    parallelize_module(model, tp_plan, tp_mesh)

    assert model.lm_head.weight is model.wte.weight
    assert model.wte.weight.numel() == 1024 * 2

    assert model.mlp1.linear1.weight.numel() == 1024 * 2 // world_size
    assert model.mlp1.linear1.weight.size() == torch.Size([1024 // world_size, 2])
    assert model.mlp1.linear2.weight.size() == torch.Size([2, 1024 // world_size])
    assert model.mlp1.linear2.weight.numel() == 1024 * 2 // world_size

    assert model.mlp2.linear1.weight.numel() == 1024 * 2 // world_size
    assert model.mlp2.linear1.weight.size() == torch.Size([1024 // world_size, 2])
    assert model.mlp2.linear2.weight.size() == torch.Size([2, 1024 // world_size])
    assert model.mlp2.linear2.weight.numel() == 1024 * 2 // world_size

    nparam = sum(p.numel() for p in model.parameters())
    assert nparam == (1024 * 2 + 1024 * 2 * 4 // world_size), nparam


test_parallelize_init = partial(torchrun, 4, parallelize_init)
