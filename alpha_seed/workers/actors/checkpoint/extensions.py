import torch
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard


class FlexDTensor(FSDPExtensions):

    def chunk_dtensor(self, tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> torch.Tensor:
        """Shards a tensor/DTensor to DTensor and returns the local DTensor."""
        # We need to explicitly call .detach() to return a new tensor detached from the current graph.
        tensor = tensor.clone().detach()
        fsdp_size = device_mesh.size(-1)
        dimlens = tuple(tensor.size())
        # by default we use the max-len dimension for sharding
        selected_dim = dimlens.index(max(dimlens))
        for dim, dimlen in enumerate(dimlens):
            if dimlen % fsdp_size == 0:
                selected_dim = dim
                break
        # HSDP placements: [Replicate(), ..., Shard(selected_dim)]
        replicate_placements = [Replicate() for _ in range(device_mesh.ndim)]
        shard_placements = [Replicate() for _ in range(device_mesh.ndim)]
        shard_placements[-1] = Shard(selected_dim)  # type: ignore[call-overload]
        return DTensor.from_local(tensor, device_mesh, replicate_placements, run_check=False).redistribute(
            placements=shard_placements,
        )

    def chunk_tensor(self, tensor, rank, world_size, num_devices_per_node, pg, device=None):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor
        return _ext_chunk_tensor(tensor, rank, world_size, num_devices_per_node, pg)

    def pre_flatten_transform(self, tensor):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_flatten_transform
        return _ext_pre_flatten_transform(tensor)

    def pre_load_state_dict_transform(self, tensor):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_flatten_transform
        return _ext_pre_flatten_transform(tensor)

    def post_unflatten_transform(self, tensor, param_extension):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform
        return _ext_post_unflatten_transform(tensor, param_extension)

    def all_gather_dtensor(self, tensor: DTensor, parent_mesh):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_all_gather_dtensor
        return _ext_all_gather_dtensor(tensor, parent_mesh)


def register_dtensor_save_hook(fsdp_module: FSDP):
    extension = FlexDTensor()
    for fsdp_module in FSDP.fsdp_modules(fsdp_module):
        assert fsdp_module._fsdp_extension is None
        fsdp_module._fsdp_extension = extension
    return fsdp_module
