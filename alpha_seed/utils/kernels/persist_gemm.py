import argparse

import torch
from torch import Tensor
import triton
import triton.language as tl
import triton.profiler as proton
from contextlib import contextmanager
import time
from typing import Optional

origin_torch_matmul = None

persist_gemm_configs = {
    torch.float8_e4m3fn: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 8
    },
    torch.float16: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_stages": 2,
        "num_warps": 8
    },
    torch.bfloat16: {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 8
    }
}


@triton.jit
def matmul_kernel_persistent(
        a_ptr,
        b_ptr,
        c_ptr,  #
        M,
        N,
        K,  #
        stride_am,
        stride_ak,  #
        stride_bk,
        stride_bn,  #
        stride_cm,
        stride_cn,  #
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        NUM_SMS: tl.constexpr,  #
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            if (c_ptr.dtype.element_ty == tl.float8e4nv):
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_persistent(a, b, c=None, sm_margin=4):

    global persist_gemm_configs
    configs = persist_gemm_configs
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count - sm_margin
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=dtype)
    else:
        assert c.shape == (M, N)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS,
                             triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],  #
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],  #
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],  #
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],  #
        NUM_SMS=NUM_SMS,  #
        num_stages=configs[dtype]["num_stages"],  #
        num_warps=configs[dtype]["num_warps"],  #
    )
    return c


def get_squeezed_dim(tensor: Tensor):
    squeezed_dim = tensor.squeeze().dim()
    return squeezed_dim


def deploy_persist_gemm(sm_margin):
    print("Will replace the torch.matmul with persist gemm")
    global origin_torch_matmul
    origin_torch_matmul = torch.matmul

    def persist_matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None):
        if get_squeezed_dim(input) == 2 and get_squeezed_dim(other) == 2 and input.dtype in persist_gemm_configs:
            return matmul_persistent(input, other, out, sm_margin=sm_margin)
        return origin_torch_matmul(input, other, out=out)

    setattr(torch, 'matmul', persist_matmul)


def undelopy_persist_gemm():
    assert origin_torch_matmul is not None
    setattr(torch, 'matmul', origin_torch_matmul)


if __name__ == '__main__':
    t_a = torch.rand(7092, 8192, dtype=torch.bfloat16).cuda() - 0.5
    t_b = torch.rand(28672, 8192, dtype=torch.bfloat16).cuda() - 0.5
    torch.cuda.synchronize()
    ctx = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ))

    with ctx:
        for i in range(20):
            out = matmul_persistent(t_a, t_b.t())
    warmup_iters = 5
    for i in range(warmup_iters):
        output = matmul_persistent(t_a, t_b.t())
    torch.cuda.synchronize()
    iters = 100
    total_time = 0
    start = time.time()
    for i in range(iters):
        output = matmul_persistent(t_a, t_b.t())
    torch.cuda.synchronize()
    end = time.time()
    total_time = end - start
    gemm_time_ms = total_time / iters * 1000
    print(f"gemm_time: {gemm_time_ms}ms")
    output_torch = torch.matmul(t_a, t_b.t())
    diff = output - output_torch
    print(diff)
    print(output)
    print(output_torch)
    print(torch.abs(diff).max())
    assert torch.allclose(output_torch, output, atol=1e-3, rtol=1e-2)
    torch.cuda.synchronize()
    ctx.export_chrome_trace(f"matmul_persist.json.gz")
