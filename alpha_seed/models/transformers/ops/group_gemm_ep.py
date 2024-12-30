import torch
import torch.distributed as dist
from seed_models.integrations.bpex_triton.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
from seed_models.integrations.bpex_triton.kernel.moe import expert_histogram, moe_gather, moe_scatter


class FusedMoeExpertFunctionEP(torch.autograd.Function):
    """
    forward output:
        expert_output ([batch_size * seqlen, hidden_size]):
            full value of hidden states, matching with
            single-device result.

    backward output:
        gate_weights_grad ([batch_size * seqlen, topk]):
            full value of gate weights gradients, matching with
            single-device result.
        grad_hidden_states ([batch_size * seqlen, hidden_size]):
            full value of gate weight gradients
        fc1_1_weight_grad (num_experts // tp_size, ...):
            full value of partitioned fc1_1 weight gradients
    """

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,  # [ntoken, topk]
        expert_index,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        ep_group: dist.ProcessGroup,
    ):

        # ==================== EP Region: init ======================
        # get owned expert range
        ctx._ep_group = ep_group
        ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
        ep_rank = 1 if ep_group is None else dist.get_rank(ep_group)
        assert num_experts % ep_size == 0
        local_num_experts = num_experts // ep_size
        expert_start = local_num_experts * ep_rank
        expert_end = expert_start + local_num_experts
        # ==================== EP Region: init ======================

        # MOE Step 3: dispatch input tokens to the experts
        # result shape is (batch_size * sequence_len * topk, hidden_size)
        # MOE Step 3-1: compute the token num for each expert
        # splits shape (num_experts)
        splits = expert_histogram(expert_index, num_experts)

        # MOE Step 3-2: compute the each token's index in result
        # scatter_index shape (batch_size * sequence_len, topk)
        # TODO(wenyawei): opt it
        # [ntoken, topk] -> [ntoken * topk]
        scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)

        # MOE Step 3-3: compute the result, select tokens by scatter_index, and put them together
        # scatter_output shape (batch_size * sequence_len * topk, hidden_size)
        scatter_output = moe_scatter(hidden_states, scatter_index)

        # =============== EP Region: get owned hidden dimension =================
        if ep_size > 1:
            token_start = torch.sum(splits[:expert_start])
            token_end = torch.sum(splits[:expert_end])
            splits = splits[expert_start:expert_end]
            scatter_output = scatter_output[token_start:token_end]
        # =============== EP Region: get owned hidden dimension =================

        # MOE Step 4: compute linear layer 1-1
        # Not consistent.
        cumsum_t = torch.cumsum(splits, dim=0)
        fc1_1_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_1_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 6: compute linear layer 1-2
        # fc1_2_output shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_2_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 5: compute the actication of linear layer 1-1
        # TODO(wenyawei): act function
        # fc1_1_activation shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7: compute final result of linear layer 1
        fc1_activation = fc1_1_activation * fc1_2_output

        # MOE Step 8: compute the the weighted linear layer 1 result
        # MOE Step 8-1: compute scattered_gate_weight, shape is (batch_size * sequence_len * topk)
        reshaped_gate_weight = gate_weights.reshape(-1, 1)  # [ntoken * topk, 1]
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        # =============== EP Region: get owned gate weight =================
        if ep_size > 1:
            scattered_gate_weight = scattered_gate_weight[token_start:token_end]
        # =============== EP Region: get owned gate weight =================

        # MOE Step 8-2: multiply activate with scattered_gate_weight
        # fc1_weighted_output shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_weighted_output = fc1_activation * scattered_gate_weight

        # MOE Step 9: compute linear layer 2
        # result shape is (batch_size * sequence_len * topk, hidden_size)
        fc2_output = group_gemm_same_nk(
            a=fc1_weighted_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 10: gather the final token result by averate the the topk token results
        # =============== EP Region: get overall expert output =================
        if ep_size > 1:
            full_fc2_output = torch.zeros(size=(expert_index.numel(), fc2_output.size(1)),
                                          dtype=fc2_output.dtype,
                                          device=fc2_output.device)
            full_fc2_output[token_start:token_end] = fc2_output
            # TODO(zhiqi.0): optimize this all_reduce by moving it
            # after moe_gather (directly moving it will fail in bitwise correct)
            # dist.all_reduce(full_fc2_output, group=ep_group)
            fc2_output = full_fc2_output
        # =============== EP Region: get overall expert output =================

        expert_output = moe_gather(fc2_output, scatter_index)
        dist.all_reduce(expert_output, group=ep_group)

        # reshape the output with input shape
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.save_for_backward(
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        )
        if ep_size > 1:
            ctx.start_end_tokens = (token_start, token_end)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        ) = ctx.saved_tensors
        ep_group = ctx._ep_group
        ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
        if ep_size > 1:
            token_start, token_end = ctx.start_end_tokens

        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # MOE Step 10
        grad_fc2_output = moe_scatter(grad_output, scatter_index)

        # =============== EP Region ================
        if ep_size > 1:
            grad_fc2_output = grad_fc2_output[token_start:token_end]
        # =============== EP Region ================

        # MOE Step 9
        # grad_fc1_weighted_output = torch.empty_like(fc1_weighted_output)

        # dgrad
        grad_fc1_weighted_output = group_gemm_same_nk(
            a=grad_fc2_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_fc2_output,
                b=fc1_weighted_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 8
        # MOE Step 8-2
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight

        # MOE Step 8-1
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)

        # ==================== EP Region ====================
        # [ntoken * topk, hidden_dim]
        if ep_size > 1:
            full_grad_gate_weight = torch.zeros(
                size=(scatter_index.numel(),),
                dtype=grad_scattered_gate_weight.dtype,
                device=grad_scattered_gate_weight.device,
            )
            full_grad_gate_weight[token_start:token_end] = grad_scattered_gate_weight
            grad_scattered_gate_weight = full_grad_gate_weight
        # ==================== EP Region ====================

        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # ==================== EP Region ====================
        if ep_size > 1:
            grad_gate_handle = dist.all_reduce(grad_gate_weight, group=ep_group, async_op=True)
        # ==================== EP Region ====================

        # recompute during backward
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation

        # MOE Step 6
        # grad_scatter_output_2 = torch.empty_like(scatter_output)

        # dgrad
        grad_scatter_output_2 = group_gemm_same_nk(
            a=grad_fc1_2_output,
            b=fc1_2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_2_output,
                b=scatter_output,
                c=grad_fc1_2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 5
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        # MOE Step 4
        # grad_scatter_output_1 = torch.empty_like(scatter_output)

        # dgrad
        grad_scatter_output_1 = group_gemm_same_nk(
            a=grad_fc1_1_output,
            b=fc1_1_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            group_gemm_same_mn(
                a=grad_fc1_1_output,
                b=scatter_output,
                c=grad_fc1_1_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 3
        # MOE Step 3-3
        grad_scatter_output = grad_scatter_output_1 + grad_scatter_output_2

        # =============== EP Region ================
        if ep_size > 1:
            full_grad_scatter_output = torch.zeros(
                size=(scatter_index.numel(), hidden_dim),
                dtype=grad_scatter_output.dtype,
                device=grad_scatter_output.device,
            )
            full_grad_scatter_output[token_start:token_end] = grad_scatter_output
            grad_gate_handle.wait()
            # TODO(zhiqi.0): optimize this all_reduce to put after moe_gather
            # (directly moving it will fail in bitwise correct)
            # dist.all_reduce(full_grad_scatter_output, group=ep_group)
            grad_scatter_output = full_grad_scatter_output
        # =============== EP Region ================

        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        dist.all_reduce(grad_hidden_states, group=ep_group)

        # MOE Step 3-2: no grad
        # MOE Step 3-1: no grad

        # reshape the result with input shape
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
            None,  # ep_group
        )
