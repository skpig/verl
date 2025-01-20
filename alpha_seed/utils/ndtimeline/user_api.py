from typing import Tuple
from .api import set_cuda_timer_option, init_ndtimers
from .nccl_trace import set_nccl_trace_option, init_emergency_server


def init_with_ray(use_cuda_timer: bool, mesh_shape: Tuple[int, ...], ray_class_instance):
    import ray
    import os
    set_cuda_timer_option(use_cuda_timer)
    if len(mesh_shape) == 1 and mesh_shape[0] == -1:
        print("unsupported role, skipped init")
        set_cuda_timer_option(False)
    if len(mesh_shape) == 2:  # align with ndtimeline internal settings
        mesh_shape = (mesh_shape[1], mesh_shape[0])
    init_ndtimers(mesh_shape=mesh_shape, ray_class_instance=ray_class_instance)

    local_rank = int(os.getenv("RAY_LOCAL_RANK", "0"))
    set_nccl_trace_option()
    init_emergency_server(local_rank=local_rank, actor_name=ray.get_runtime_context().get_actor_name())
