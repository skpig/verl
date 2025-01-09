from typing import Dict
import torch
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from alpha_seed.models.transformers.parallel import TPSpec
import copy


class FlexDTensor(FSDPExtensions):

    def __init__(self, shard_plan: Dict):
        super().__init__()
        self.tp_mesh = None
        self.fqn2spec: Dict[str, TPSpec] = shard_plan
        for spec in self.fqn2spec.values():
            if self.tp_mesh is not None:
                assert self.tp_mesh is spec.mesh
            self.tp_mesh = spec.mesh

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
        raise NotImplementedError("Please init FSDP with device mesh")
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor
        return _ext_chunk_tensor(tensor, rank, world_size, num_devices_per_node, pg)

    def pre_flatten_transform(self, tensor):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_flatten_transform
        return _ext_pre_flatten_transform(tensor)

    def pre_load_state_dict_transform(self, tensor):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_load_state_dict_transform
        return _ext_pre_load_state_dict_transform(tensor)

    def post_unflatten_transform(self, tensor, param_extension):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform
        return _ext_post_unflatten_transform(tensor, param_extension)

    def all_gather_dtensor(self, tensor: DTensor, parent_mesh):
        # this is required during loading checkpoint (model.load_state_dict)
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_all_gather_dtensor
        if self.tp_mesh is None:
            return _ext_all_gather_dtensor(tensor, None)
        # replicate the fsdp dimension while keeps tp sharding
        placements = list(copy.deepcopy(tensor.placements))
        assert len(placements) >= 2 and parent_mesh.mesh_dim_names[-1] == "tp"
        # no need to touch the tp dimension
        placements[-2] = Replicate()
        local_tensor = tensor.redistribute(placements=placements, async_op=False)._local_tensor
        return local_tensor

    @torch.no_grad()
    def post_state_dict_with_tp_spec(self, module, state_dict, prefix, local_metadata):
        if self.tp_mesh is None:
            return
        # [dp, fsdp, tp] or [fsdp, tp]
        global_device_mesh = self.tp_mesh._parent_mesh
        assert global_device_mesh.ndim in (2, 3)
        keys = list(state_dict.keys())
        for name in sorted(keys):
            tensor = state_dict[name]
            if isinstance(tensor, DTensor):
                orig_device = tensor.device
                shard = self.fqn2spec[name].shard
                placements = list(tensor.placements)
                # if the tensor parallelism and fsdp cuts the same dimension,
                # torch has bugs in handling this case due to the mismatched order
                # in device mesh and parallelism order, more details refer to
                # https://github.com/pytorch/pytorch/issues/129206.
                # Here we bypass this bug by switching the sharding dimension to
                # make sure FSDP doesn't have same sharding dimension with TP.
                if isinstance(shard, Shard) and shard.dim == placements[-1].dim:
                    dimlens = list(tensor.size())
                    dimlens[shard.dim] = 0  # filer out already sharded dim
                    fsdp_size = tensor.device_mesh.size(-1)
                    selected = dimlens.index(max(dimlens))
                    for dim, dimlen in enumerate(dimlens):
                        if dim == shard.dim:
                            continue
                        if dimlen % fsdp_size == 0:
                            selected = dim
                            break
                    placements[-1] = Shard(selected)
                    tensor = tensor.cuda()
                    if tensor.device_mesh.size() >= 128:
                        # FXIME(zhiqi.0): for large device number, we cannot create large-scope
                        # all2all due to the port number limitation for each device.
                        # Therefore, we choose to first replicate the tensor and then chunk.
                        replicates = [Replicate() for _ in range(len(placements))]
                        tensor = tensor.redistribute(placements=replicates, async_op=False)
                        tensor = tensor.redistribute(placements=placements, async_op=False)
                    else:
                        tensor = tensor.redistribute(placements=placements, async_op=False)
                # add tensor parallel shard
                placements.append(shard)
                shape = list(tensor.size())
                if isinstance(shard, Shard):
                    shape[shard.dim] *= self.tp_mesh.size()
                    assert placements[-1] != placements[-2], f"{placements[-1]} vs. {placements[-2]}"
                # shape must be tuple. Give a list will cause unhashable error
                # in torch. This is a bug in torch.
                tensor = DTensor.from_local(tensor._local_tensor,
                                            device_mesh=global_device_mesh,
                                            placements=placements,
                                            shape=torch.Size(shape),
                                            stride=tensor.stride(),
                                            run_check=False)
                tensor = tensor.to(orig_device)
                state_dict[name] = tensor


def register_dtensor_save_hook(fsdp_model: FSDP, shard_plan: Dict = None):
    """
    Register dtensor-based hooks for FSDP+TP

    This will:

    1. Customize the FSDP extension for save / load hooks in TP scenarios.
    2. Equip each module with attribute `_tp_mesh` for access.
    """
    shard_plan = {} if shard_plan is None else shard_plan
    extension = FlexDTensor(shard_plan)
    for fsdp_module in FSDP.fsdp_modules(fsdp_model):
        fsdp_module._fsdp_extension = extension
        fsdp_module._handle._fsdp_extension = extension
    # make sure the root module is also registered
    fsdp_model._fsdp_extension = extension
    fsdp_module._handle._fsdp_extension = extension

    # register load / save hook for tp
    fsdp_model._register_state_dict_hook(extension.post_state_dict_with_tp_spec)

    # register tp mesh in each module so that we can manage the
    # model parallel context inside each module
    for name, module in fsdp_model.named_modules():
        assert not hasattr(module, "_tp_mesh"), f"{name} already gots _tp_mesh field"
        module._tp_mesh = extension.tp_mesh
    return fsdp_model
