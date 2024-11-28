# Utils for testing only. No public APIs exposed.
import torch

__all__ = []


def _compute_entropy_loss(logits, eos_mask):
    """Reference: categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    entropy = _entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = _masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def _entropy_from_logits(logits: torch.Tensor):
    """Reference: entropy loss from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def _masked_mean(values, mask, axis=None):
    """Reference: Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def compare_tensors(expected_2d, actual_2d, verbose=False):
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


def prepare_data():
    from flash_attn.bert_padding import index_first_axis, rearrange
    # sample tensor available at hdfs://haruna/home/byte_data_seed/lf_lq/user/haibin.lin/rl/ppo_actor_rmpad_v3.pt
    rmpad_data = torch.load('ppo_actor_rmpad_v3.pt', map_location='cuda:0')

    response_length = rmpad_data['response_length']
    attention_mask = rmpad_data['attention_mask']
    input_ids_rmpad = rmpad_data['input_ids']
    position_ids_rmpad = rmpad_data['position_ids']
    indices = rmpad_data['indices']

    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

    # hidden_states: (nnz, hidden_size) after rmpad
    full_response_mask = attention_mask.clone()
    full_response_mask[:, :-response_length] = 0  # set the prompt part to zero
    full_response_mask_rmpad = index_first_axis(rearrange(full_response_mask.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                indices).squeeze(-1)  # (total_nnz,)

    return input_ids_rmpad, input_ids_rmpad_rolled, full_response_mask_rmpad, position_ids_rmpad


def ref_loss_fn(output, input_ids_rmpad_rolled, full_response_mask_rmpad):
    # input_ids_rmpad_rolled: (total_nnz,)
    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
    # logits_rmpad.div_(temperature)
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    full_log_probs_rmpad = -cross_entropy_loss(logits_rmpad, input_ids_rmpad_rolled,
                                               inplace_backward=False)[0]  # (total_nnz,)

    # compute entropy loss
    # logits_rmpad: (total_nnz, vocab_size)
    # full_response_mask_rmpad: (total_nnz,)
    entropy_loss = _compute_entropy_loss(logits_rmpad, full_response_mask_rmpad)
    # entropy_loss: scaler
    return entropy_loss, full_log_probs_rmpad
