from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import (
    _get_default_group,
    _get_group_tag,
    get_world_size,
    new_group,
    ProcessGroup,
)

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
from torch.utils._typing_utils import not_none
import torch

from torch.distributed import is_available

original_init_process_groups = DeviceMesh._init_process_groups


# patch
def _patched_init_process_groups(self):
    # tag/ranks/group_name associated with each mesh dimension, each
    # mesh dimension should have one sub-group per rank
    #
    # TODO(yifu): remove tag and ranks once we fully migrate to native
    # functional collectives. See details in:
    # https://github.com/pytorch/pytorch/issues/93173#issuecomment-1907095208
    dim_group_infos: List[Tuple[str, List[int], str]] = []

    if self.mesh.ndim == 1 and self.mesh.numel() == get_world_size():
        # if the mesh is the same as world_pg, we just append the default
        # pg to the first dim groups, as new_group cannot have the exact
        # same ranks as world
        dim_group_infos.append((
            _get_group_tag(_get_default_group()),
            list(range(get_world_size())),
            _get_default_group().group_name,
        ))
    else:
        # create sub pgs base on the mesh argument specified
        for dim in range(self.mesh.ndim):
            # swap the current dim to the last dim
            # then reshape to flatten out other dims
            pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
            # multi-dim mesh, create subgroups by looping over the pg_ranks
            # for each dim and append the groups
            for dim_mesh in pg_ranks_by_dim:
                subgroup_ranks = dim_mesh.tolist()

                # Respect dim group options specified via _MeshEnv.set_dim_group_options().
                # Inherit from the parent group if no options are specified for the group.
                if dim in _mesh_resources.mesh_dim_group_options:
                    (
                        backend,
                        pg_options,
                    ) = _mesh_resources.mesh_dim_group_options[dim]
                else:
                    backend, pg_options = None, None

                # We temporarily revert the re-use subgroup, since it breaks two internal tests.
                # Temporarily reverting to resolve test timeout while root-causing.
                # TODO: Add two tests to cover internal tests scenarios and re-enable reuse subgroup if exists.
                from datetime import timedelta
                import os
                timeout = timedelta(seconds=int(os.getenv('NCCL_TIMEOUT', 3600)))
                dim_group = new_group(ranks=subgroup_ranks, backend=backend, pg_options=pg_options, timeout=timeout)

                # only add to dim_groups if the current rank in the subgroup
                if self.get_rank() in subgroup_ranks:
                    if len(dim_group_infos) > dim:
                        raise RuntimeError(
                            f"Each device mesh dimension should get only one process group, but got {self.get_rank} "
                            f"in {subgroup_ranks}!")
                    dim_group_infos.append((
                        _get_group_tag(not_none(dim_group)),
                        subgroup_ranks,
                        dim_group.group_name,
                    ))
    self._dim_group_infos = dim_group_infos


DeviceMesh._init_process_groups = _patched_init_process_groups
