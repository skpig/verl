"""
torchrun --nproc_per_node=4 --master-port=12322 tests/parallelize/test_group_gemm_ep.py \
    2>&1 | tee test_op.log
"""
import os

os.environ['TRITON_CACHE_MANAGER'] = 'triton.runtime.cache:RemoteCacheManager'
os.environ['TRITON_REMOTE_CACHE_BACKEND'] = 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend'
os.environ['BPEX_NO_WARN_ON_UNTUNED_CASE'] = '1'
os.environ['NCCL_DEBUG'] = '0'

from mono_rl.models.seed_models.ops.group_gemm_ep import FusedMoeExpertFunctionEP
from seed_models.utils.modeling_fused_moe import FusedMoeExpertFunction

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tests.launch import torchrun
import functools


def prepare_inputs():

    num_experts = 128
    ffn_dim = 1024
    hidden_dim = 2048
    topk = 2
    ntokens = 16384

    torch.manual_seed(42)

    # weights
    fc1_1 = torch.nn.Parameter(
        torch.randn(num_experts, ffn_dim, hidden_dim),
        requires_grad=True,
    )
    fc1_2 = torch.nn.Parameter(
        torch.randn(num_experts, ffn_dim, hidden_dim),
        requires_grad=True,
    )
    fc2 = torch.nn.Parameter(
        torch.randn(num_experts, hidden_dim, ffn_dim),
        requires_grad=True,
    )
    # activations
    gate_weights = torch.randn((ntokens, topk), requires_grad=True)
    expert_index = torch.randint(0, num_experts, (ntokens, topk), dtype=torch.int)
    hidden_states = torch.randn(ntokens, hidden_dim, requires_grad=True)
    return (fc1_1, fc1_2, fc2), (num_experts, gate_weights, expert_index, hidden_states)


def compare_moe_expert_parallel(ep_size: int):

    world_size = dist.get_world_size()
    tp_mesh = init_device_mesh("cuda", (world_size // ep_size, ep_size), mesh_dim_names=("dp", "ep"))["ep"]
    tp_group = tp_mesh.get_group()
    tp_size = tp_mesh.size()
    tp_rank = tp_mesh.get_local_rank()

    torch.set_default_dtype(torch.bfloat16)
    with torch.device(torch.cuda.current_device()):
        weights, inputs = prepare_inputs()

    fc1_1, fc1_2, fc2 = weights
    num_experts, gate_weights, expert_index, hidden_states = inputs

    # baseline
    if dist.get_rank() == 0:
        print(f"start computing baseline...")

    ref_output = FusedMoeExpertFunction.apply(num_experts, gate_weights, expert_index, hidden_states, fc1_1, fc1_2, fc2)
    ref_output.sum().backward()
    ref_grads = [
        fc1_1.grad,
        fc1_2.grad,
        fc2.grad,
        gate_weights.grad,
        hidden_states.grad,
    ]
    fc1_1.grad = fc1_2.grad = fc2.grad = gate_weights.grad = hidden_states.grad = None
    assert not torch.isnan(ref_output).any()
    assert not torch.isinf(ref_output).any()

    if dist.get_rank() == 0:
        print(f"finished computing baseline...")

    dist.barrier()
    # expert parallel result

    from alpha_seed.utils.ndtimeline.timed_collectives import patch_coll_ops
    patch_coll_ops()

    if dist.get_rank() == 0:
        print(f"start computing expert parallel...")

    fc1_1_local = fc1_1.chunk(tp_size, dim=0)[tp_rank]
    fc1_2_local = fc1_2.chunk(tp_size, dim=0)[tp_rank]
    fc2_local = fc2.chunk(tp_size, dim=0)[tp_rank]

    # the first time maybe slow as triton kernel may need JIT compile
    output, handle, _ = FusedMoeExpertFunctionEP.apply(
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_local,
        fc1_2_local,
        fc2_local,
        tp_group,
    )
    # handle.wait()
    output.sum().backward()

    dist.all_reduce(fc1_1.grad, group=tp_group)
    dist.all_reduce(fc1_2.grad, group=tp_group)
    dist.all_reduce(fc2.grad, group=tp_group)
    grads = [
        fc1_1.grad,
        fc1_2.grad,
        fc2.grad,
        gate_weights.grad,
        hidden_states.grad,
    ]
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    if dist.get_rank() == 0:
        print(f"finished computing expert parallel...")
    dist.barrier()

    if dist.get_rank() == 0:
        print(f"comparing forward output...")
        torch.testing.assert_close(ref_output, output, atol=0, rtol=0)
        for idx, (ref_grad, grad) in enumerate(zip(ref_grads, grads)):
            print(f"comapring {idx}-th grad...")
            torch.testing.assert_close(ref_grad, grad, atol=0, rtol=0)
        print(f"bitwise test passed")


test_group_gemm_no_ep = functools.partial(torchrun, 4, compare_moe_expert_parallel, 1)
test_group_gemm_partial_ep = functools.partial(torchrun, 4, compare_moe_expert_parallel, 2)
test_group_gemm_full_ep = functools.partial(torchrun, 4, compare_moe_expert_parallel, 4)

if __name__ == '__main__':
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    compare_moe_expert_parallel()
    dist.destroy_process_group()
