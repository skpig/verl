import argparse
import torch
from torch import Tensor
import triton
import triton.language as tl
import triton.profiler as proton
from contextlib import contextmanager
import time
import sys
from typing import Any, Optional

origin_torch_matmul = None
origin_torch_linear = None
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
        bias_ptr,  #
        M,
        N,
        K,  #
        stride_am,
        stride_ak,  #
        stride_bk,
        stride_bn,  #
        stride_cm,
        stride_cn,  #
        with_bias: tl.constexpr,  #
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
            if with_bias:
                bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N).to(tl.float32)
                accumulator += tl.broadcast_to(tl.expand_dims(bias, 0), accumulator.shape)
            if (c_ptr.dtype.element_ty == tl.float8e4nv):
                c = accumulator.to(tl.float8e4nv)
            else:
                c = accumulator.to(tl.bfloat16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def matmul_persistent(a, b, c=None, bias=None, sm_margin=4):
    global persist_gemm_configs
    configs = persist_gemm_configs
    # Check constraints.
    assert a.shape[-1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, f"Incompatible dtypes a:{a.dtype} b:{b.dtype}"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count - sm_margin
    K = a.shape[-1]
    M = a.numel() // K
    K, N = b.shape
    dtype = a.dtype
    assert a.dim() == 2 and b.dim() == 2
    # Allocates output.
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=dtype, requires_grad=a.requires_grad)
    else:
        assert c.numel() == M * N
    if bias is not None:
        assert bias.dim() == 1 and bias.size(0) == N, f"bias:{bias.shape} != ({N})"
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS,
                             triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        bias is not None,
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


class PersistLinearFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None, sm_margin=0):
        K = input.shape[-1]
        M = input.numel() // K
        N = weight.shape[0]
        ctx.MKN = (M, K, N)
        ctx.save_for_backward(input, weight, bias)
        ctx.sm_margin = sm_margin
        ctx.input_shape = input.shape
        ctx.output_shape = list(input.shape)[:-1] + [N]
        if input.dim() != 2:
            input = input.view(M, K)
        out = matmul_persistent(input, weight.t(), None, bias, sm_margin=sm_margin)
        return out.view(ctx.output_shape)

    @staticmethod
    def backward(ctx, grad_output):
        M, K, N = ctx.MKN
        input, weight, bias = ctx.saved_tensors
        sm_margin = ctx.sm_margin
        grad_input = grad_weight = grad_bias = None
        if grad_output.dim() != 2:
            grad_output = grad_output.view(M, N)
            input = input.view(M, K)
        if ctx.needs_input_grad[0]:
            grad_input = matmul_persistent(grad_output, weight, sm_margin=sm_margin)
        if ctx.needs_input_grad[1]:
            grad_weight = matmul_persistent(grad_output.t(), input, sm_margin=sm_margin)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        return grad_input.view(ctx.input_shape), grad_weight, grad_bias, None


def deploy_persist_gemm(sm_margin):
    print("Will replace the torch.matmul with persist gemm")
    global origin_torch_matmul
    global origin_torch_linear
    origin_torch_matmul = torch.matmul
    origin_torch_linear = torch.nn.functional.linear
    CPP_INT_MAX = 2**31

    def persist_matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None):
        if get_squeezed_dim(input) == 2 and get_squeezed_dim(other) == 2 and input.dtype in persist_gemm_configs:
            return matmul_persistent(input, other, out, None, sm_margin=sm_margin)
        return origin_torch_matmul(input, other, out=out)

    def persist_linear(input, weight, bias: Optional[Tensor] = None):
        K = input.shape[-1]
        M = input.numel() // K
        N = weight.shape[0]
        if input.dtype in persist_gemm_configs and M * K < CPP_INT_MAX and M * N < CPP_INT_MAX and K * N < CPP_INT_MAX:
            return PersistLinearFunc.apply(input, weight, bias, sm_margin)
        return origin_torch_linear(input, weight, bias)

    setattr(torch, 'matmul', persist_matmul)
    setattr(torch.nn.functional, 'linear', persist_linear)


def undelopy_persist_gemm():
    global origin_torch_matmul
    global origin_torch_linear
    if origin_torch_matmul is not None:
        print("restore original torch gemms")
        assert origin_torch_matmul is not None
        assert origin_torch_linear is not None
        setattr(torch, 'matmul', origin_torch_matmul)
        setattr(torch.nn.functional, 'linear', origin_torch_linear)
        origin_torch_linear = None
        origin_torch_matmul = None


def torch_allclose(x, y, rtol, atol, verbose=True):
    if not torch.allclose(x, y, rtol=rtol, atol=atol):
        print(f"shape of x: {x.shape}")
        print(f"shape of y: {y.shape}")

        print("x:", file=sys.stderr)
        print(x, file=sys.stderr)
        print("y:", file=sys.stderr)
        print(y, file=sys.stderr)
        print("x-y", x - y, file=sys.stderr)
        diff_loc = torch.isclose(x, y, rtol=rtol, atol=atol) == False
        print("x diff:", file=sys.stderr)
        print(x[diff_loc], file=sys.stderr)
        print("y diff:", file=sys.stderr)
        print(y[diff_loc], file=sys.stderr)
        num_diff = torch.sum(diff_loc)

        if len(y.shape) == 1:
            diff_rate = num_diff / y.shape[0]
        else:
            diff_rate = num_diff / (y.shape[0] * y.shape[1])
        print(f"diff count: {num_diff} ({diff_rate*100:.3f}%), {list(y.shape)}", file=sys.stderr)
        max_diff = torch.max(torch.abs(x - y))
        rtol_abs = rtol * torch.min(torch.abs(y))
        print(f"diff max: {max_diff}, atol: {atol}, rtol_abs: {rtol_abs}", file=sys.stderr)
        diff_indices = (diff_loc == True).nonzero(as_tuple=False)
        print(f"diff locations:\n{diff_indices}", file=sys.stderr)
        print("--------------------------------------------------------------\n", file=sys.stderr)
        raise RuntimeError

    if verbose:
        print("all close!")


def stress_test(iters):
    print("perform stress test")
    undelopy_persist_gemm()
    assert origin_torch_linear is None
    import random
    while iters != 0:
        iters -= 1
        M = random.randint(1024, 10240)
        K = random.randint(1024, 10240)
        N = random.randint(1024, 10240)
        print("M,K,N:", M, K, N)
        t_a1 = ((torch.rand(1, M, K, dtype=torch.bfloat16, requires_grad=True).cuda() - 0.5) / 10).detach()
        t_a1.requires_grad_()
        t_a2 = torch.empty_like(t_a1, requires_grad=True)
        t_a2.data.copy_(t_a1)
        t_b = (torch.rand(N, K, dtype=torch.bfloat16, requires_grad=True, device='cuda') - 0.5).detach()
        t_b.requires_grad_()
        t_bias = torch.zeros(N, dtype=torch.bfloat16, requires_grad=True, device='cuda')
        linear = torch.nn.Linear(in_features=K, out_features=N, bias=True).to(torch.bfloat16).cuda()
        linear.weight.data.copy_(t_b.data)
        linear.bias.data.copy_(t_bias.data)
        linear.requires_grad_()
        ref_out = linear(t_a1)
        out = PersistLinearFunc.apply(t_a2, t_b, t_bias, 4)
        loss = torch.rand_like(ref_out)
        ref_out.backward(loss)
        out.backward(loss)
        print(out)
        print(ref_out)
        torch_allclose(ref_out, out, atol=3e-2, rtol=3e-2)
        torch_allclose(t_a1.grad, t_a2.grad, atol=3e-2, rtol=3e-2)
        torch_allclose(t_b.grad, linear.weight.grad, atol=3e-2, rtol=3e-2)
        if linear.bias is not None:
            torch_allclose(t_bias.grad, linear.bias.grad, atol=2e-2, rtol=2e-2)
        torch.cuda.synchronize()


if __name__ == '__main__':
    t_a = torch.rand(1, 7092, 8192, dtype=torch.bfloat16).cuda() - 0.5
    linear = torch.nn.Linear(in_features=8192, out_features=28672, bias=True).to(torch.bfloat16).cuda()
    deploy_persist_gemm(4)
    t_b = torch.rand(28672, 8192, dtype=torch.bfloat16).cuda() - 0.5
    with torch.no_grad():
        linear.weight.data.copy_(t_b.data)
    torch.cuda.synchronize()
    ctx = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ))
    with ctx:
        for i in range(20):
            out = linear(t_a)
    #  measure the performance
    iters = 100
    total_time = 0
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        output = linear(t_a)
    torch.cuda.synchronize()
    end = time.time()
    total_time = end - start
    gemm_time_ms = total_time / iters * 1000
    print(f"gemm_time: {gemm_time_ms}ms")
    undelopy_persist_gemm()
    torch.cuda.synchronize()
    ctx.export_chrome_trace(f"matmul_persist.json.gz")
    # stress_test(100)
