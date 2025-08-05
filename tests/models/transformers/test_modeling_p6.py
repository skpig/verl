import seed_models
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from seed_models.models.p6.modeling_p6 import P6ForCausalLM
from test_utils import ref_loss_fn, compare_tensors, prepare_data


def build_model():
    # hdfs path available at tasks_scripts/ci/run_p6_400m_math_v1.sh
    config = AutoConfig.from_pretrained('p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf/config.json',
                                        attn_implementation='flash_attention_2')
    config.num_hidden_layers = 8
    # must set dropout to 0.0 for consistency
    config.resid_pdrop = 0
    config.attention_dropout = 0
    model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
    model.apply(model._init_weights)
    return model


def test_logits_cross_entropy_fusion():
    model = build_model()
    # v2 is the model with bumi cross entropy
    model_v2 = build_model()
    model_v2.load_state_dict(model.state_dict())

    # apply monkey patch
    from mono_rl.models.seed_models.modeling_p6 import p6_model_forward
    P6ForCausalLM.forward = p6_model_forward

    input_ids_rmpad, input_ids_rmpad_rolled, full_response_mask_rmpad, position_ids_rmpad = prepare_data()

    kwargs = {'input_ids': input_ids_rmpad, 'position_ids': position_ids_rmpad}
    outputs_v1 = model(**kwargs, use_cache=False, output_hidden_states=True)

    loss, log_prob = ref_loss_fn(outputs_v1, input_ids_rmpad_rolled, full_response_mask_rmpad)
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

    (log_prob * 10 + log_prob).sum().backward()
    (log_prob_v2 * 10 + log_prob_v2).sum().backward()
    summary = compare_tensors(model.lm_head.weight.grad, model_v2.lm_head.weight.grad)
    print(f'================ grad diff summary ================')
    for k, v in summary.items():
        print(f'{k}: {v}')
    print(f'================ grad diff summary ================')
    # sometimes a small number of elements mismatches:
    # Mismatched elements: 1666189 / 238288896 (0.7%)
    assert torch.testing.assert_close(model.lm_head.weight.grad, model_v2.lm_head.weight.grad)
    print("passed")


if __name__ == "__main__":
    test_logits_cross_entropy_fusion()
