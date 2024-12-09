import torch
import os
from torch import nn


def cross_entropy_fn(lm_head, hidden_states, labels_rmpad):
    logits = lm_head(hidden_states)
    # logits_rmpad.div_(temperature)
    logits_rmpad = logits.view(-1, logits.shape[-1]).squeeze(0)  # (total_nnz, vocab_size)

    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    log_probs = -cross_entropy_loss(logits_rmpad, labels_rmpad, inplace_backward=True)[0]  # (total_nnz,)

    return log_probs


def fused_cross_entropy_fn(lm_head, hidden_states, labels_rmpad, version=1):
    hidden_states_rmpad = hidden_states.view(-1, hidden_states.shape[-1])
    if version == 1:
        from bumi.function.flash_cross_entropy import FlashCrossEntropy
        recompute_level = 2  # not saving logits
        nll, _ = FlashCrossEntropy.apply(hidden_states_rmpad, lm_head.weight, labels_rmpad, recompute_level)
    return -1 * nll


def test_cross_entropy_fusion():
    # TODO(haibin.lin): test with temperature
    torch.manual_seed(0)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    # Example inputs
    batch_size = 3
    seq_len = 1536
    vocab_size = 10240  # 155136
    hidden_size = 1536

    input_ids_rmpad = torch.randint(0, vocab_size, (batch_size * seq_len,), device='cuda')
    labels_rmpad = torch.roll(input_ids_rmpad, shifts=-1, dims=-1)

    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', requires_grad=True).bfloat16()
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False).cuda().bfloat16()
    manual_lm_head = nn.Linear(hidden_size, vocab_size, bias=False).cuda().bfloat16()
    manual_lm_head.weight.data.copy_(lm_head.weight.data)

    log_prob = cross_entropy_fn(lm_head, x, labels_rmpad)
    print(f'log_prob = {log_prob}')
    log_prob.sum().backward()

    manual_log_prob = fused_cross_entropy_fn(manual_lm_head, x, labels_rmpad, version=1)
    print(f"manual autograd with fused ops:")
    print(f'log_prob = {manual_log_prob}')
    manual_log_prob.sum().backward()

    # Verify that the gradients match
    summary = compare_two_tensors(manual_lm_head.weight.grad, lm_head.weight.grad)
    print(f'================ grad diff summary ================')
    for k, v in summary.items():
        print(f'{k}: {v}')
    print(f'================ grad diff summary ================')

    from test_utils import check_close
    # only raise Exception if mismatch percentage is > 1%
    check_close(manual_lm_head.weight.grad, lm_head.weight.grad, equal_nan=True, percent_threshold=0.01)
    torch.cuda.synchronize()
    print("passed test_cross_entropy_fusion")


def allocated_gigabytes():
    return torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)


def reserved_gigabytes():
    return torch.cuda.memory_reserved() / (1024 * 1024 * 1024)


def benchmark_cross_entropy():
    torch.manual_seed(0)
    # Example inputs
    batch_size = 2
    seq_len = 16384
    vocab_size = 155136
    hidden_size = 1536

    input_ids_rmpad = torch.randint(0, vocab_size, (batch_size * seq_len,), device='cuda')
    labels_rmpad = torch.roll(input_ids_rmpad, shifts=-1, dims=-1)

    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', requires_grad=True).bfloat16()
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False).cuda().bfloat16()
    print(f"Max memory allocated: {allocated_gigabytes()} memory reserved: {reserved_gigabytes()}")
    log_prob = cross_entropy_fn(lm_head, x, labels_rmpad)
    print(f"Max memory allocated: {allocated_gigabytes()} memory reserved: {reserved_gigabytes()}")
    log_prob.sum().backward()
    print(f"Max memory allocated: {allocated_gigabytes()} memory reserved: {reserved_gigabytes()}")

    torch.cuda.synchronize()


def benchmark_fused_cross_entropy():
    torch.manual_seed(0)
    # Example inputs
    batch_size = 2
    seq_len = 16384
    vocab_size = 155136
    hidden_size = 1536

    input_ids_rmpad = torch.randint(0, vocab_size, (batch_size * seq_len,), device='cuda')
    labels_rmpad = torch.roll(input_ids_rmpad, shifts=-1, dims=-1)

    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', requires_grad=True).bfloat16()
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False).cuda().bfloat16()
    print(f"Max memory allocated: {allocated_gigabytes()} memory reserved: {reserved_gigabytes()}")
    log_prob = fused_cross_entropy_fn(lm_head, x, labels_rmpad, version=1)
    print(f"Max memory allocated: {allocated_gigabytes()} memory reserved: {reserved_gigabytes()}")
    log_prob.sum().backward()
    print(f"Max memory allocated: {allocated_gigabytes()} memory reserved: {reserved_gigabytes()}")

    torch.cuda.synchronize()


def compare_two_tensors(expected_2d, actual_2d, verbose=False):
    expected = expected_2d.reshape(-1)
    actual = actual_2d.reshape(-1)
    diff = actual - expected
    abs_diff = diff.abs()
    expected_ref = expected.clone()
    abs_max = abs_diff.max()
    abs_max_idx = abs_diff.argmax()
    rel_diff = abs_diff / expected_ref
    rel_diff_2d = rel_diff.reshape(expected_2d.shape)
    abs_diff_2d = abs_diff.reshape(expected_2d.shape)
    if verbose:
        for i in range(expected_2d.shape[0]):
            for j in range(expected_2d.shape[1]):
                if rel_diff_2d[i, j] > 1e-2 and abs_diff_2d[i, j] > 1e-5:
                    print(f'expected[{i}, {j}] = {expected_2d[i, j]} actual[{i}, {j}] = {actual_2d[i, j]}')

    rel_max = rel_diff.max()
    rel_max_idx = rel_diff.argmax()
    rel_max_idx_cpu = rel_max_idx.detach().cpu().numpy()
    rel_max_indices = rel_max_idx_cpu // expected_2d.shape[1], rel_max_idx_cpu % expected_2d.shape[1]
    return {
        'abs_max': abs_max,
        'expected[abs_idx]': expected[abs_max_idx],
        'actual[abs_idx]': actual[abs_max_idx],
        'expected[rel_idx]': expected[rel_max_idx],
        'actual[rel_idx]': actual[rel_max_idx],
        'rel_max_idx': rel_max_indices,
        'rel_max': rel_max
    }


if __name__ == '__main__':
    test_cross_entropy_fusion()
    # benchmark_cross_entropy()
    # benchmark_fused_cross_entropy()
