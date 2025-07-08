from alpha_seed.workers.fsdp.initialize import create_mesh
import torch.distributed as dist
from ..launch import torchrun
from functools import partial


def create_mesh_test(fsdp_size, tp_size, sp_size):
    world_size = dist.get_world_size()
    fsdp_mesh, tp_mesh, _, sp_mesh, gather_mesh = create_mesh(fsdp_size, tp_size, 1, sp_size)

    assert fsdp_mesh.ndim <= 2
    assert fsdp_mesh.size() == world_size // tp_size

    assert tp_mesh.size() == tp_size
    assert tp_mesh.ndim == 1

    assert sp_mesh.size() == sp_size
    assert sp_mesh.ndim == 1

    assert gather_mesh.size() == tp_size * sp_size


def create_duplicate_mesh(fsdp_size, tp_size, sp_size):
    fsdp_mesh1, tp_mesh1, _, sp_mesh1, gather_mesh1 = create_mesh(fsdp_size, tp_size, 1, sp_size)
    fsdp_mesh2, tp_mesh2, _, sp_mesh2, gather_mesh2 = create_mesh(fsdp_size, tp_size, 1, sp_size)
    assert fsdp_mesh1.get_group(mesh_dim=0) is fsdp_mesh2.get_group(mesh_dim=0)
    assert tp_mesh1.get_group(0) is tp_mesh2.get_group(0)
    assert sp_mesh1.get_group(0) is sp_mesh2.get_group(0)
    assert gather_mesh1.get_group(0) is gather_mesh2.get_group(0)


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
# duplicate create: hsdp + tp + sp
test_duplicate_mesh_create = partial(torchrun, 4, create_duplicate_mesh, 1, 2, 1)
