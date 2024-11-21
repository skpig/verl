import torch.distributed
from torch.distributed.device_mesh import init_device_mesh


def create_device_mesh(fsdp_size, role):
    world_size = torch.distributed.get_world_size()

    if fsdp_size > 0 and (world_size // fsdp_size) > 1:
        # if dp_size > 1, use HSDP
        assert world_size % fsdp_size == 0, "world_size must be divisible by fsdp_size"
        dp_size = world_size // fsdp_size
        device_mesh = init_device_mesh('cuda', mesh_shape=(dp_size, fsdp_size), mesh_dim_names=['dp', 'fsdp'])
        print(f"Using HSDP {dp_size} {device_mesh} for {role}")
    else:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
        print(f"using FSDP {world_size} {device_mesh} for {role}")
    return device_mesh