from typing import Dict
import torch
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed._tensor.placement_types import Placement
from alpha_seed.models.transformers.parallel import TPSpec
import copy
import warnings

orig_optim_state_dict = FSDP.optim_state_dict
orig_optim_state_dict_to_load = FSDP.optim_state_dict_to_load


def _append_state_with_tp_spec(tensor: DTensor, shard: Placement, tp_mesh: DeviceMesh, tp_outside: bool):
    global_device_mesh = tp_mesh._parent_mesh
    assert global_device_mesh is not None, f"The tp_mesh must be a sub_mesh to global device mesh"
    orig_device = tensor.device
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
            # FIXME(zhiqi.0): for large device number, we cannot create large-scope
            # all2all due to the port number limitation for each device.
            # Therefore, we choose to first replicate the tensor and then chunk.
            replicates = [Replicate() for _ in range(len(placements))]
            tensor = tensor.redistribute(placements=replicates, async_op=False)
            tensor = tensor.redistribute(placements=placements, async_op=False)
        else:
            tensor = tensor.redistribute(placements=placements, async_op=False)
    # add tensor parallel shard
    shape = list(tensor.size())
    if isinstance(shard, Shard):
        shape[shard.dim] *= tp_mesh.size()
        # for the case that a tensor has only one dimension, we cannot avoid
        # having FSDP and TP sharding on a same dimension. In this case,
        # we change FSDP placement to replicate.
        if shard == placements[-1]:
            assert len(shape) == 1, f"got unexpected ndim ({len(shape)}) > 1"
            placements[-1] = Replicate()
            tensor = tensor.redistribute(placements=placements, async_op=False)
            # print(f"after reshard: {tensor.size()}")
        assert placements[-1] != shard, f"{placements[-1]} vs. {shard}"
    placements = [shard] + placements if tp_outside else placements + [shard]
    # shape must be tuple. Give a list will cause unhashable error
    # in torch. This is a bug in torch.
    tensor = DTensor.from_local(tensor._local_tensor,
                                device_mesh=global_device_mesh,
                                placements=placements,
                                shape=torch.Size(shape),
                                stride=tensor.stride(),
                                run_check=False)
    tensor = tensor.to(orig_device)
    return tensor


class FlexDTensor(FSDPExtensions):

    def __init__(self, shard_plan: Dict, tp_outside: bool):
        super().__init__()
        self.tp_outside = tp_outside
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
        if "tp" not in parent_mesh.mesh_dim_names:
            warnings.warn(
                "Cannot detect tp mesh when loading model, this can only happen when the checkpoint is saved before tp support."
            )
            placements[-1] = Replicate()
        elif self.tp_outside:
            assert len(placements) >= 2 and parent_mesh.mesh_dim_names[0] == "tp"
            # no need to touch the tp dimension
            placements[-1] = Replicate()
        else:
            assert len(placements) >= 2 and parent_mesh.mesh_dim_names[-1] == "tp"
            # no need to touch the tp dimension
            placements[-2] = Replicate()
        local_tensor = tensor.redistribute(placements=placements, async_op=False)._local_tensor
        return local_tensor

    @torch.no_grad()
    def post_state_dict_hook(self, module, state_dict, prefix, local_metadata):
        """
        Post state dict when calling `model.state_dict()` for TP cases.

        This will append TP placements to the FSDP DTensor state dicts
        """
        if self.tp_mesh is None:
            return
        # [dp, fsdp, tp] or [fsdp, tp]
        global_device_mesh = self.tp_mesh._parent_mesh
        assert global_device_mesh.ndim in (2, 3)
        keys = list(state_dict.keys())
        for name in sorted(keys):
            tensor = state_dict[name]
            if isinstance(tensor, DTensor):
                shard = self.fqn2spec[name].shard
                tensor = _append_state_with_tp_spec(tensor, shard, self.tp_mesh, self.tp_outside)
                state_dict[name] = tensor

    def register_post_optim_hook(self):
        """
        post optimizer state dict hook when calling `FSDP.optim_state_dict(model, optimizer)`

        This will extend the DTensors in optimizer state dict with TP placements
        """

        def optim_state_post_hook_patch(model, optim, optim_state_dict=None):
            fsdp_mesh = model._device_mesh
            assert fsdp_mesh is not None, f"Please init FSDP module with device_mesh"
            # NOTE we don't support diverse process group for different FSDP sub-modules
            fsdp_pg = model.process_group
            optim_state = orig_optim_state_dict(model, optim, optim_state_dict, fsdp_pg)
            if self.tp_mesh is None:
                return optim_state

            global_device_mesh = self.tp_mesh._parent_mesh
            assert global_device_mesh.ndim in (2, 3)
            # extend placements by adding TP placement
            for fqn in sorted(optim_state["state"].keys()):
                fqn_state = {}
                for key, val in optim_state["state"][fqn].items():
                    if isinstance(val, DTensor):
                        shard = self.fqn2spec[fqn].shard
                        val = _append_state_with_tp_spec(val, shard, self.tp_mesh, self.tp_outside)
                    fqn_state[key] = val
                optim_state["state"][fqn] = fqn_state
            return optim_state

        # monkey patch
        FSDP.optim_state_dict = staticmethod(optim_state_post_hook_patch)

        def optim_state_load_pre_hook(model,
                                      optim,
                                      optim_state_dict,
                                      is_named_optimizer=False,
                                      load_directly=False,
                                      group=None):
            """
            At this point, the `optim_state_dict` is correctly resharded to the current device mesh by `dcp.load`
            """
            fsdp_mesh = model._device_mesh
            assert fsdp_mesh is not None, f"Please init FSDP module with device_mesh"

            # NOTE we don't support diverse process group for different FSDP sub-modules
            if self.tp_mesh is not None:
                global_device_mesh = self.tp_mesh._parent_mesh
                assert global_device_mesh.ndim in (2, 3)
                for fqn in sorted(optim_state_dict["state"].keys()):
                    fqn_state = {}
                    for key, val in optim_state_dict["state"][fqn].items():
                        if isinstance(val, DTensor):
                            device_mesh = val.device_mesh
                            placements = copy.deepcopy(val.placements)[:-1]
                            mesh_dim_names = device_mesh.mesh_dim_names
                            if "tp" not in mesh_dim_names:
                                warnings.warn(
                                    "Cannot detect tp mesh when loading optimizer, this can only happen when the checkpoint is saved before tp support."
                                )
                                fsdp_mesh = device_mesh
                            elif self.tp_outside:
                                assert mesh_dim_names[0] == "tp"
                                fsdp_mesh = device_mesh[mesh_dim_names[1:]]
                            else:
                                assert mesh_dim_names[-1] == "tp"
                                fsdp_mesh = device_mesh[mesh_dim_names[:-1]]
                            assert fsdp_mesh.ndim <= 2
                            val = DTensor.from_local(
                                local_tensor=val._local_tensor,
                                device_mesh=fsdp_mesh,
                                placements=placements,
                                run_check=False,
                                shape=val.size(),
                                stride=val.stride(),
                            )
                        fqn_state[key] = val
                    optim_state_dict["state"][fqn] = fqn_state

            fsdp_pg = model.process_group
            optim_state = orig_optim_state_dict_to_load(model, optim, optim_state_dict, is_named_optimizer,
                                                        load_directly, fsdp_pg)
            return optim_state

        # monkey patch
        FSDP.optim_state_dict_to_load = staticmethod(optim_state_load_pre_hook)


def register_dtensor_save_hook(fsdp_model: FSDP, shard_plan: Dict = None, tp_outside: bool = False):
    """
    Register dtensor-based hooks for FSDP+TP

    This will:

    1. Customize the FSDP extension for save / load hooks in TP scenarios.
    2. Equip each module with attribute `_tp_mesh` for access.
    """
    shard_plan = {} if shard_plan is None else shard_plan
    extension = FlexDTensor(shard_plan, tp_outside)
    for fsdp_module in FSDP.fsdp_modules(fsdp_model):
        fsdp_module._fsdp_extension = extension
        fsdp_module._handle._fsdp_extension = extension
    # make sure the root module is also registered
    fsdp_model._fsdp_extension = extension
    fsdp_module._handle._fsdp_extension = extension

    # register load / save hook for tp
    fsdp_model._register_state_dict_hook(extension.post_state_dict_hook)
    extension.register_post_optim_hook()

    # register tp mesh in each module so that we can manage the
    # model parallel context inside each module
    for name, module in fsdp_model.named_modules():
        assert not hasattr(module, "_tp_mesh"), f"{name} already gots _tp_mesh field"
        module._tp_mesh = extension.tp_mesh
    return fsdp_model
