"""
Offload utilities for FSDP
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
import gc

from typing import Optional


@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, f"Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert flat_param.data.data_ptr() == flat_param._local_shard.data_ptr() and \
            id(flat_param.data) != id(flat_param._local_shard) and \
            flat_param.data.size() == flat_param._local_shard.size()
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)
    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, f"Only support root model loading to GPU"
    device_id = torch.cuda.current_device()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device(f"cuda:{device_id}"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


"""
Add mariana megatron model load/offload
"""

from verl.utils.debug import log_gpu_memory_usage


@torch.no_grad()
def offload_megatron_model_to_cpu(models, empty_cache: bool = False):
    """
    In megatron, the model and optimizer storage are:
    - bf16 parameter data chunked in model parallel group
    - fp32 grad chunked in model parallel group
    - fp32 main_parameter chunked in model and dp group
    - fp32 optimizer state chunked in model and dp group
    Here, we assume that main_parameter and optimizer state are taken care of by mariana
    """
    for model_chunk in models:
        for buffer in model_chunk.buffers:
            # offload parameters
            if buffer.param_data.storage().size() > 0:
                buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                buffer.param_data_size = buffer.param_data.storage().size()
                buffer.param_data.storage().resize_(0)

            assert buffer.param_data_size == buffer.param_data.cpu_data.storage().size()

            if buffer.grad_data.storage().size() > 0:
                # if the grad_data size is already zero, we assume that it is already offloaded
                buffer.grad_data_size = buffer.grad_data.storage().size()
                buffer.grad_data.storage().resize_(0)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_megatron_model_to_gpu(models, load_grad=True):
    for model_chunk in models:
        for buffer in model_chunk.buffers:
            # sometimes, we don't want to load grad for pure inference
            if load_grad:
                buffer.grad_data.storage().resize_(buffer.grad_data_size)
                buffer.grad_data.zero_()

            if buffer.param_data.storage().size() == 0:
                buffer.param_data.storage().resize_(buffer.param_data_size)
                # copy data from cpu to cuda
                buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)


@torch.no_grad()
def offload_megatron_optimizer(optimizers):
    opt_state_dict_values = optimizers[0].optimizer.state.values()
    for v in opt_state_dict_values:
        v['exp_avg'] = v['exp_avg'].to('cpu', non_blocking=False)
        v['exp_avg_sq'] = v['exp_avg_sq'].to('cpu', non_blocking=False)


@torch.no_grad()
def load_megatron_optimizer(optimizers, device_id):
    opt_state_dict_values = optimizers[0].optimizer.state.values()
    for v in opt_state_dict_values:
        v['exp_avg'] = v['exp_avg'].to(torch.cuda.current_device(), non_blocking=False)
        v['exp_avg_sq'] = v['exp_avg_sq'].to(torch.cuda.current_device(), non_blocking=False)
