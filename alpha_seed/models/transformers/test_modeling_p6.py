import seed_models
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from seed_models.models.p6.modeling_p6 import P6ForCausalLM


def build_model():
    # must set dropout to 0.0 for consistency
    # hdfs path available at tasks_scripts/ci/run_p6_400m_math_v1.sh
    config = AutoConfig.from_pretrained('p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf/config.json',
                                        attn_implementation='flash_attention_2')
    model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
    model.apply(model._init_weights)
    return model


def compute_entropy_loss(logits, eos_mask):
    """Reference: categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    entropy = entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def entropy_from_logits(logits: torch.Tensor):
    """Reference: entropy loss from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def masked_mean(values, mask, axis=None):
    """Reference: Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


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


def loss_fn(output, input_ids_rmpad_rolled, full_response_mask_rmpad):
    # input_ids_rmpad_rolled: (total_nnz,)
    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
    # logits_rmpad.div_(temperature)
    # import pdb; pdb.set_trace()
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    full_log_probs_rmpad = -cross_entropy_loss(logits_rmpad, input_ids_rmpad_rolled,
                                               inplace_backward=False)[0]  # (total_nnz,)
    # compute entropy loss

    # logits_rmpad: (total_nnz, vocab_size)
    # full_response_mask_rmpad: (total_nnz,)
    entropy_loss = compute_entropy_loss(logits_rmpad, full_response_mask_rmpad)
    # entropy_loss: scaler
    return entropy_loss, full_log_probs_rmpad


def test_logits_cross_entropy_fusion():
    model = build_model()
    # v2 is the model with bumi cross entropy
    model_v2 = build_model()
    model_v2.load_state_dict(model.state_dict())

    # apply monkey patch
    from alpha_seed.models.transformers.modeling_p6 import p6_model_forward
    P6ForCausalLM.forward = p6_model_forward

    input_ids_rmpad, input_ids_rmpad_rolled, full_response_mask_rmpad, position_ids_rmpad = prepare_data()
    # print(input_ids_rmpad.shape, input_ids_rmpad_rolled.shape, full_response_mask_rmpad.shape, full_response_mask_rmpad)

    kwargs = {'input_ids': input_ids_rmpad, 'position_ids': position_ids_rmpad}
    outputs_v1 = model(**kwargs, use_cache=False, output_hidden_states=True)

    loss, log_prob = loss_fn(outputs_v1, input_ids_rmpad_rolled, full_response_mask_rmpad)
    print(f'loss = {loss} log_prob = {log_prob}')

    kwargs_v2 = {
        'input_ids': input_ids_rmpad.detach(),
        'position_ids': position_ids_rmpad.detach(),
        'labels': input_ids_rmpad_rolled.detach(),
        'fuse_lm_head_ce_loss': True
    }
    # TODO: add entropy loss and test
    outputs_v2 = model_v2(**kwargs_v2, use_cache=False, output_hidden_states=True)
    log_prob_v2 = outputs_v2.loss * -1
    print(f'log_prob v2 = {log_prob_v2}')

    torch.testing.assert_close(log_prob, log_prob_v2)
    print("passed")


if __name__ == "__main__":
    test_logits_cross_entropy_fusion()
