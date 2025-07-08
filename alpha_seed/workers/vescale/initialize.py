import torch.distributed as dist
from vescale.initialize.mesh import create_mesh_with_names


def create_mesh(fsdp_size: int, tp_size: int, oe_size: int, sp_size: int, tp_outside: bool = False):
    """
    Create device meshes for fsdp, tp, and sp.

    Returns:
        fsdp_mesh: DeviceMesh for FSDP/HSDP (can be 1-D or 2-D)
        ep_mesh: DeviceMesh for TP (1-D)
        oe_mesh: DeviceMesh for EP (1-D)
        sp_mesh: DeviceMesh for SP (1-D)
        gather_mesh: DeviceMesh for data replication group (1-D)
    """
    world_size = dist.get_world_size()
    fsdp_size = world_size if fsdp_size <= 0 else fsdp_size
    assert world_size % sp_size == 0, f'{world_size=} {sp_size=}'
    # train mesh
    fsdp_mesh = create_mesh_with_names("cuda", dp=-1, fsdp=fsdp_size, mp=1)["dp", "fsdp"]
    ep_mesh = create_mesh_with_names("cuda", dp=-1, ep=tp_size)["ep"]
    oe_mesh = create_mesh_with_names("cuda", dp=-1, oe=oe_size)["oe"]
    sp_mesh = create_mesh_with_names("cuda", dp=-1, sp=sp_size)["sp"]
    gather_mesh = sp_mesh
    return fsdp_mesh, ep_mesh, oe_mesh, sp_mesh, gather_mesh
