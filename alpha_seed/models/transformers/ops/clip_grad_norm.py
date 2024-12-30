import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import _get_grad_norm, _lazy_init
from ..parallel.parallelize import TPSpec, Replicate
import math
import torch.distributed as dist
import functools
import warnings


def clip_grad_norm_(fsdp_model: FSDP, max_norm, norm_type=2.0) -> torch.Tensor:

    extension = fsdp_model._fsdp_extension
    tp_mesh = extension.tp_mesh
    tp_group = None if tp_mesh is None else tp_mesh.get_group()

    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return fsdp_model.clip_grad_norm_(max_norm, norm_type)

    assert fsdp_model._is_root
    # use dict as ordered set to make param order consistent among
    # dp (hsdp) ranks to avoid gnorm difference due to reduction order
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    sharded_params_for_gnorm = {}
    fsdp_managed_params = set()
    nonsharded_params = {}
    grads_for_clip = []
    for handle in fsdp_model._all_handles:
        assert handle.uses_sharded_strategy
        assert handle._use_orig_params, \
            f"tensor parallelism can only work with FSDP using `use_orig_params=True`"
        for param in handle.flat_param._params:
            assert hasattr(param, "_spec")
            tp_spec: TPSpec = param._spec
            fsdp_managed_params.add(param)
            if param.grad is not None:
                grads_for_clip.append(param.grad)
            # for replicated parameters across tensor parallelism group,
            # we only need to compute the norm on the first rank
            if isinstance(tp_spec.shard, Replicate) and not _is_first_tp_rank(tp_group):
                continue
            sharded_params_for_gnorm.setdefault(param, None)
    for param in fsdp_model.parameters():
        not_fsdp_managed = (param not in fsdp_managed_params and param not in nonsharded_params)
        if not_fsdp_managed:
            assert hasattr(param, "_spec")
            raise NotImplementedError(f"param {param._spec.fqn} is not managed by FSDP")
            nonsharded_params.setdefault(param, None)
            if param.grad is not None:
                grads_for_clip.append(param.grad)
    # Compute local norms (forced to be in FP32)
    local_sharded_norm = _get_grad_norm(sharded_params_for_gnorm, norm_type).to(fsdp_model.compute_device)
    local_nonsharded_norm = (_get_grad_norm(nonsharded_params, norm_type).to(fsdp_model.compute_device)
                             if nonsharded_params else None)
    # Reconstruct the total gradient norm depending on the norm type
    if norm_type == math.inf:
        total_norm = (torch.maximum(local_sharded_norm, local_nonsharded_norm)
                      if local_nonsharded_norm is not None else local_sharded_norm)
        dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=fsdp_model.process_group)
        # allreduce across tp group
        dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=tp_group)
    else:
        total_norm = local_sharded_norm**norm_type
        dist.all_reduce(total_norm, group=fsdp_model.process_group)
        # all reduce across tp group
        dist.all_reduce(total_norm, group=tp_group)
        # All-reducing the local non-sharded norm would count it an extra
        # world-size-many times
        if local_nonsharded_norm is not None:
            total_norm += local_nonsharded_norm**norm_type
        total_norm = total_norm**(1.0 / norm_type)
    if fsdp_model.cpu_offload.offload_params:
        total_norm = total_norm.cpu()

    clip_coef = max_norm / (total_norm + 1e-6)
    # Multiplying by the clamped coefficient is meaningless when it is
    # equal to 1, but it avoids the host-device sync that would result from
    # `if clip_coef < 1`
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in grads_for_clip:
        grad.mul_(clip_coef_clamped.to(grad.device, grad.dtype))
    # Use the "largest" dtype by type promotion semantics to use the same
    # dtype as if we did not force local norm computation to be in FP32
    if len(grads_for_clip) == 0:
        # If this rank has no gradients, then we must default to FP32
        # unless we use additional communication, which we prefer to avoid
        # since `clip_grad_norm_()` is called in the training loop
        warnings.warn(f"Called FSDP.clip_grad_norm_() on rank {fsdp_model.rank} with no "
                      "gradients -- returning the total norm in the default dtype "
                      f"{total_norm.dtype}")  # warn since this is generally unexpected
        return total_norm
    total_norm_dtype = functools.reduce(
        torch.promote_types,
        [grad.dtype for grad in grads_for_clip],
    )
    return total_norm.to(total_norm_dtype)


def _is_first_tp_rank(tp_group: dist.ProcessGroup):
    assert tp_group is not None
    return dist.get_rank(tp_group) == 0
