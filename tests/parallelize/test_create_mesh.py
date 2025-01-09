from alpha_seed.workers.actors.initialize import create_mesh
import torch.distributed as dist
from ..launch import torchrun
from functools import partial


def create_mesh_test(fsdp_size, tp_size, sp_size):
    world_size = dist.get_world_size()

    fsdp_mesh, tp_mesh, sp_mesh, gather_mesh = create_mesh(fsdp_size, tp_size, sp_size)

    fsdp_size = world_size // tp_size

    assert fsdp_mesh.ndim <= 2
    assert fsdp_mesh.size() == world_size // tp_size

    assert tp_mesh.size() == tp_size
    assert tp_mesh.ndim == 1

    assert sp_mesh.size() == sp_size
    assert sp_mesh.ndim == 1

    assert gather_mesh.size() == tp_size * sp_size


# fsdp
test_mesh_create_1 = partial(torchrun, 4, create_mesh_test, -1, 1, 1)
# fsdp + sp
test_mesh_create_2 = partial(torchrun, 4, create_mesh_test, -1, 1, 2)
# hsdp
test_mesh_create_3 = partial(torchrun, 4, create_mesh_test, 2, 1, 1)
# hsdp + sp
test_mesh_create_4 = partial(torchrun, 4, create_mesh_test, 2, 1, 4)
# fsdp + tp
test_mesh_create_5 = partial(torchrun, 4, create_mesh_test, 2, 2, 1)
# fsdp + tp + sp
test_mesh_create_6 = partial(torchrun, 4, create_mesh_test, -1, 2, 2)
# hsdp + tp
test_mesh_create_7 = partial(torchrun, 4, create_mesh_test, 1, 2, 1)
# hsdp + tp + sp
test_mesh_create_8 = partial(torchrun, 4, create_mesh_test, 1, 2, 1)
