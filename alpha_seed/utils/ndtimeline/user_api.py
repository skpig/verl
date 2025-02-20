from .api import set_cuda_timer_option, init_ndtimers
from .nccl_trace import set_nccl_trace_option, init_emergency_server


def init_flight_recorder():
    import os
    local_rank = int(os.getenv("RAY_LOCAL_RANK", "0"))
    set_nccl_trace_option()
    import ray
    init_emergency_server(local_rank=local_rank, actor_name=ray.get_runtime_context().get_actor_name())


def init_with_ray(use_cuda_timer: bool, wg):
    set_cuda_timer_option(use_cuda_timer)  # this will be removed soon
    init_flight_recorder()
    if not use_cuda_timer:
        return
    if hasattr(wg, "actor_strategy") and wg.actor_strategy in ['megatron']:
        # TODO: support megatron strategy
        set_cuda_timer_option(False)
        return
    if (hasattr(wg, "_is_actor") and wg._is_actor) or (hasattr(wg, "_is_rollout") and wg._is_rollout):
        mocked_fsdp_shape = list(wg.actor_fsdp_mesh.shape)
        mocked_fsdp_shape[-1] *= wg.actor_tp_mesh.size()
        mocked_fsdp_shape = tuple(mocked_fsdp_shape)
    elif hasattr(wg, "_is_ref") and wg._is_ref:
        mocked_fsdp_shape = list(wg.ref_fsdp_mesh.shape)
        mocked_fsdp_shape[-1] *= wg.ref_tp_mesh.size()
        mocked_fsdp_shape = tuple(mocked_fsdp_shape)
    elif hasattr(wg, "_is_standalone_rollout") and wg._is_standalone_rollout:
        mocked_fsdp_shape = (wg.config.streaming_rollout_args.n_gpus_per_node *
                             wg.config.streaming_rollout_args.nnodes,)
    elif hasattr(wg, "_is_standalone_validator") and wg._is_standalone_validator:
        mocked_fsdp_shape = (wg.config.streaming_validator_args.n_gpus_per_node *
                             wg.config.streaming_validator_args.nnodes,)
    elif wg.role in ["critic", "rm"]:
        mocked_fsdp_shape = list(wg.fsdp_mesh.shape)
        mocked_fsdp_shape[-1] *= wg.tp_mesh.size()
        mocked_fsdp_shape = tuple(mocked_fsdp_shape)
    else:
        set_cuda_timer_option(False)
        return
    if len(mocked_fsdp_shape) == 2:  # align with ndtimeline internal settings
        mocked_fsdp_shape = (mocked_fsdp_shape[1], mocked_fsdp_shape[0])
    init_ndtimers(mesh_shape=mocked_fsdp_shape, ray_class_instance=wg)
