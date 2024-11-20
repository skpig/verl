import torch
import torch.distributed

from functools import partial

from transformers import PretrainedConfig

from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from seed_models import P6Config
from alpha_seed.workers.xperf_rollout.utils.layout_convert_helper import _fix_qkv_ordering, _fix_o_ordering, load_to_cuda


def convert_fp8_weights_from_bf16(qkv_proj_weight, attention_proj_weight, FFN0_weight, FFN1_weight, experts_num=32):
    # qkv w
    fp8_qkv_scale_converted = 1 / torch.classes.XGPT.Fp8GemmTestOp().GetPerTensorScale(qkv_proj_weight, 0)
    fp8_qkvw_converted = torch.classes.XGPT.Fp8GemmTestOp().Quant(qkv_proj_weight, 1 / fp8_qkv_scale_converted)

    # qkv out
    fp8_attention_proj_weight_scale_converted = 1 / torch.classes.XGPT.Fp8GemmTestOp().GetPerTensorScale(
        attention_proj_weight, 0)
    fp8_attention_proj_weight_converted = torch.classes.XGPT.Fp8GemmTestOp().Quant(
        attention_proj_weight, 1 / fp8_attention_proj_weight_scale_converted)

    # FFN0
    fp8_FFN0_weight_scale_converted_all = []
    fp8_FFN0_weight_converted_all = []
    for i in range(experts_num):
        FFN0_weight = FFN0_weight.reshape(experts_num, -1)
        local_FFN0_weight = FFN0_weight[i]
        fp8_FFN0_weight_scale_converted = 1 / torch.classes.XGPT.Fp8GemmTestOp().GetPerTensorScale(local_FFN0_weight, 0)
        fp8_FFN0_weight_converted = torch.classes.XGPT.Fp8GemmTestOp().Quant(local_FFN0_weight,
                                                                             1 / fp8_FFN0_weight_scale_converted)
        fp8_FFN0_weight_scale_converted_all.append(fp8_FFN0_weight_scale_converted)
        fp8_FFN0_weight_converted_all.append(fp8_FFN0_weight_converted.cpu())

    fp8_FFN0_weight_scale_converted_all = torch.cat(fp8_FFN0_weight_scale_converted_all)
    fp8_FFN0_weight_converted_all = torch.cat(fp8_FFN0_weight_converted_all).cuda()

    # FFN1
    fp8_FFN1_weight_scale_converted_all = []
    fp8_FFN1_weight_converted_all = []
    for i in range(experts_num):
        FFN1_weight = FFN1_weight.reshape(experts_num, -1)
        local_FFN1_weight = FFN1_weight[i]
        fp8_FFN1_weight_scale_converted = 1 / torch.classes.XGPT.Fp8GemmTestOp().GetPerTensorScale(local_FFN1_weight, 0)
        fp8_FFN1_weight_converted = torch.classes.XGPT.Fp8GemmTestOp().Quant(local_FFN1_weight,
                                                                             1 / fp8_FFN1_weight_scale_converted)
        fp8_FFN1_weight_scale_converted_all.append(fp8_FFN1_weight_scale_converted)
        fp8_FFN1_weight_converted_all.append(fp8_FFN1_weight_converted.cpu())

    fp8_FFN1_weight_scale_converted_all = torch.cat(fp8_FFN1_weight_scale_converted_all)
    fp8_FFN1_weight_converted_all = torch.cat(fp8_FFN1_weight_converted_all).cuda()

    return fp8_qkv_scale_converted, fp8_qkvw_converted, fp8_attention_proj_weight_scale_converted, fp8_attention_proj_weight_converted, fp8_FFN0_weight_scale_converted_all, fp8_FFN0_weight_converted_all, fp8_FFN1_weight_scale_converted_all, fp8_FFN1_weight_converted_all


def _reshard_fsdp_state_dict_to_xperf_p6_fp8(tp_model, state_dict, device_mesh: DeviceMesh, model_config: P6Config):
    assert isinstance(model_config, P6Config)

    # checking
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
        assert tp_size <= model_config.num_key_value_heads
        assert model_config.num_key_value_heads % tp_size == 0
    else:
        tp_size = 1
        tp_rank = 0

    assert tp_size == model_config.num_key_value_heads or tp_size == 1

    head_dim = model_config.hidden_size // model_config.num_attention_heads

    ln_f_weight = state_dict.pop('transformer.ln_f.weight').full_tensor().to(torch.bfloat16)
    ln_f_bias = state_dict.pop('transformer.ln_f.bias').full_tensor().to(torch.bfloat16)
    ln_f = torch.stack((ln_f_weight, ln_f_bias)).contiguous()

    tp_model.layernorm_weight.data = ln_f

    del ln_f_weight, ln_f_bias

    # TODO: use xperf vocab_tp
    wte: DTensor = state_dict.pop('transformer.wte.weight').to(torch.bfloat16)
    state_dict.pop('lm_head.weight')

    wte_weight = wte.full_tensor()

    if device_mesh is not None and tp_model.wte_weight.data.shape != wte_weight.shape:
        # TODO: we may need to do full first
        wte_weight = DTensor.from_local(wte_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
        wte_weight_tp = wte_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                     Shard(1)])._local_tensor
        lm_head_tp = wte_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(0)])._local_tensor
        # del wte
    else:
        wte_weight_tp = wte_weight
        lm_head_tp = wte_weight

    assert wte_weight_tp.shape == tp_model.wte_weight.data.shape
    assert lm_head_tp.shape == tp_model.lm_head_weight.data.shape

    tp_model.wte_weight.data = wte_weight_tp.contiguous()
    tp_model.lm_head_weight.data = lm_head_tp.contiguous()  # tied weights

    for layer_index, (pre_gamma_beta, key_gamma_beta, context_gamma_beta, qkv_proj_weight, qkv_proj_bias,
                      attention_proj_weight, attention_proj_bias, ffn_gamma_beta, moe_gate_weight, moe_gate_bias,
                      FFN0_weight, FFN0_bias, FFN1_weight, FFN1_bias, norm0_out_scale, qkv_proj_in_inv_scale,
                      qkv_proj_weight_inv_scale, attn_out_scale, attn_proj_in_inv_scale, attn_proj_weight_inv_scale,
                      norm1_out_scale, FFN0_in_inv_scale, FFN0_weight_inv_scale, FFN0_out_scale, FFN1_in_inv_scale,
                      FFN1_weight_inv_scale, *_) in enumerate(tp_model.layers_weight):
        # torch.distributed.breakpoint()
        ln_1_weight = state_dict.pop(f'transformer.h.{layer_index}.ln_1.weight').full_tensor()
        ln_1_bias = state_dict.pop(f'transformer.h.{layer_index}.ln_1.bias').full_tensor()
        ln_1_weight = torch.stack((ln_1_weight, ln_1_bias), dim=0).to(torch.bfloat16)
        assert pre_gamma_beta.data.shape == ln_1_weight.shape
        pre_gamma_beta.data = ln_1_weight.contiguous()

        del ln_1_bias

        key_norm_weight = state_dict[f'transformer.h.{layer_index}.attn.key_layernorm.weight'].full_tensor()
        key_norm_bias = state_dict[f'transformer.h.{layer_index}.attn.key_layernorm.bias'].full_tensor()
        key_norm_weight = torch.stack((key_norm_weight, key_norm_bias), dim=0).to(torch.bfloat16)
        assert key_gamma_beta.data.shape == key_norm_weight.shape
        key_gamma_beta.data = key_norm_weight.contiguous()

        del key_norm_bias

        context_norm_weight = state_dict[f'transformer.h.{layer_index}.attn.context_norm.weight'].full_tensor()
        context_norm_bias = state_dict[f'transformer.h.{layer_index}.attn.context_norm.bias'].full_tensor()
        context_norm_weight = torch.stack((context_norm_weight, context_norm_bias), dim=0).to(torch.bfloat16)
        assert context_gamma_beta.data.shape == context_norm_weight.shape
        context_gamma_beta.data = context_norm_weight.contiguous()

        del context_norm_bias

        q_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.q_proj.weight').full_tensor().to(
            torch.bfloat16)  # (num_q_head, head_dim, hidden_size)
        k_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.k_proj.weight').full_tensor().to(
            torch.bfloat16)  # (num_kv_head, head_dim, hidden_size)
        v_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.v_proj.weight').full_tensor().to(
            torch.bfloat16)  # (num_kv_head, head_dim, hidden_size)

        num_heads = q_proj_weight.shape[0] // head_dim
        mqa_kv_heads = k_proj_weight.shape[0] // head_dim

        qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0)
        if device_mesh is not None:
            qkv_weight, q_heads_list = _fix_qkv_ordering(qkv_weight,
                                                         tp_size=tp_size,
                                                         num_heads=num_heads,
                                                         mqa_kv_heads=mqa_kv_heads,
                                                         interleaved_kv_shared=model_config.interleaved_kv_shared)
            qkv_weight = qkv_weight[tp_rank]

        q_proj_bias = state_dict.pop(f'transformer.h.{layer_index}.attn.q_proj.bias').full_tensor().to(torch.bfloat16)
        k_proj_bias = state_dict.pop(f'transformer.h.{layer_index}.attn.k_proj.bias').full_tensor().to(torch.bfloat16)
        v_proj_bias = state_dict.pop(f'transformer.h.{layer_index}.attn.v_proj.bias').full_tensor().to(torch.bfloat16)

        qkv_bias = torch.cat((q_proj_bias, k_proj_bias, v_proj_bias), dim=0)
        if device_mesh is not None:
            qkv_bias, q_heads_list = _fix_qkv_ordering(qkv_bias,
                                                       tp_size=tp_size,
                                                       num_heads=num_heads,
                                                       mqa_kv_heads=mqa_kv_heads,
                                                       interleaved_kv_shared=model_config.interleaved_kv_shared)
            qkv_bias = qkv_bias[tp_rank]

        assert qkv_bias.shape == qkv_proj_bias.shape
        qkv_proj_bias.data = qkv_bias.contiguous()

        o_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.o_proj.weight').full_tensor().to(
            torch.bfloat16)

        if device_mesh is not None:
            o_proj_weight = _fix_o_ordering(o_proj_weight, num_heads=num_heads, q_heads_list=q_heads_list)[tp_rank]

        o_proj_bias = state_dict.pop(f'transformer.h.{layer_index}.attn.o_proj.bias').full_tensor().to(torch.bfloat16)
        assert o_proj_bias.shape == attention_proj_bias.shape
        attention_proj_bias.data = o_proj_bias.contiguous()

        ln_2_weight = state_dict.pop(f'transformer.h.{layer_index}.ln_2.weight').full_tensor().to(torch.bfloat16)
        ln_2_bias = state_dict.pop(f'transformer.h.{layer_index}.ln_2.bias').full_tensor().to(torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight, ln_2_bias), dim=0)
        assert ln_2_weight.shape == ffn_gamma_beta.shape
        ffn_gamma_beta.data = ln_2_weight.contiguous()

        del ln_2_bias

        gate_wg = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.gate.wg').full_tensor().T.contiguous().float()
        gate_wg_ema = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.gate.wg_ema').T.contiguous().float()

        gate_wg = (gate_wg + gate_wg_ema) * 0.5

        assert gate_wg.shape == moe_gate_weight.shape
        moe_gate_weight.data = gate_wg.contiguous()

        use_grouped_gemm_weight = getattr(model_config, '_moe_implementation', 'eager') == 'fused'

        if f'transformer.h.{layer_index}.mlp.moe.experts.fc1_1_weight' in state_dict:
            raise ValueError(
                '请使用最新的依赖。并根据文档 https://bytedance.larkoffice.com/docx/SBuXdoDpgoCwiDxV4Hwco1ahnug 重新转p6 checkpoint')

        if use_grouped_gemm_weight:
            fc1_1_weight = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts.fc1_1').to(
                torch.bfloat16).full_tensor()
            if device_mesh is not None:
                fc1_1_weight = DTensor.from_local(fc1_1_weight,
                                                  device_mesh=device_mesh,
                                                  placements=[Replicate(), Replicate()])
                fc1_1_weight = fc1_1_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                              Shard(1)])._local_tensor

            fc1_2_weight = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts.fc1_2').to(
                torch.bfloat16).full_tensor()

            if device_mesh is not None:
                fc1_2_weight = DTensor.from_local(fc1_2_weight,
                                                  device_mesh=device_mesh,
                                                  placements=[Replicate(), Replicate()])
                fc1_2_weight = fc1_2_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                              Shard(1)])._local_tensor
        else:
            fc1_1_list = []
            fc1_2_list = []
            # breakpoint()
            for expert_index in range(model_config.moe_num_expert):
                fc1_1 = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts.{expert_index}.fc1_1.weight').to(
                    torch.bfloat16).full_tensor()
                fc1_2 = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts.{expert_index}.fc1_2.weight').to(
                    torch.bfloat16).full_tensor()

                if device_mesh is not None:
                    fc1_1 = DTensor.from_local(fc1_1, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
                    fc1_1 = fc1_1.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                    Shard(0)])._local_tensor

                    fc1_2 = DTensor.from_local(fc1_2, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
                    fc1_2 = fc1_2.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                    Shard(0)])._local_tensor

                fc1_1_list.append(fc1_1)
                fc1_2_list.append(fc1_2)

            # (num_experts, intermediate_size // tp, hidden_size)
            fc1_1_weight = torch.stack(fc1_1_list, dim=0)
            fc1_2_weight = torch.stack(fc1_2_list, dim=0)

            del fc1_1_list, fc1_2_list

        fc1_weight = torch.cat((fc1_1_weight, fc1_2_weight), dim=1).contiguous().flatten()

        if use_grouped_gemm_weight:
            fc2_weight = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts.fc2').to(
                torch.bfloat16).full_tensor()
            if device_mesh is not None:
                fc2_weight = DTensor.from_local(fc2_weight,
                                                device_mesh=device_mesh,
                                                placements=[Replicate(), Replicate()])
                fc2_weight = fc2_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                          Shard(2)])._local_tensor
            fc2_weight = fc2_weight.contiguous().flatten()
        else:
            fc2_list = []
            for expert_index in range(model_config.moe_num_expert):
                fc2 = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts.{expert_index}.fc2.weight').to(
                    torch.bfloat16).full_tensor()
                if device_mesh is not None:
                    fc2 = DTensor.from_local(fc2, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
                    fc2 = fc2.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(1)])._local_tensor
                fc2_list.append(fc2)

            fc2_weight = torch.stack(fc2_list, dim=0).contiguous().flatten()

            del fc2_list

        # (num_experts, intermediate_size // tp, hidden_size)
        # breakpoint()
        fp8_qkv_scale_converted, fp8_qkvw_converted, fp8_attention_proj_weight_scale_converted, fp8_attention_proj_weight_converted, fp8_FFN0_weight_scale_converted_all, fp8_FFN0_weight_converted_all, fp8_FFN1_weight_scale_converted_all, fp8_FFN1_weight_converted_all = convert_fp8_weights_from_bf16(
            qkv_weight, o_proj_weight, fc1_weight, fc2_weight)
        qkv_proj_weight.data = fp8_qkvw_converted  # FP8
        qkv_proj_weight_inv_scale.data = fp8_qkv_scale_converted
        attention_proj_weight.data = fp8_attention_proj_weight_converted  # FP8
        attn_proj_weight_inv_scale.data = fp8_attention_proj_weight_scale_converted
        FFN0_weight.data = fp8_FFN0_weight_converted_all  # FP8
        FFN0_weight_inv_scale.data = fp8_FFN0_weight_scale_converted_all
        FFN1_weight.data = fp8_FFN1_weight_converted_all  # FP8
        FFN1_weight_inv_scale.data = fp8_FFN1_weight_scale_converted_all
        norm0_out_scale.fill_(1.0)
        qkv_proj_in_inv_scale.fill_(1.0)
        attn_out_scale.fill_(1.0)
        attn_proj_in_inv_scale.fill_(1.0)
        norm1_out_scale.fill_(1.0)
        FFN0_in_inv_scale.fill_(1.0)
        FFN0_out_scale.fill_(1.0)
        FFN1_in_inv_scale.fill_(1.0)

    load_to_cuda(tp_model=tp_model)

    # assert len(state_dict) == 0

    torch.cuda.empty_cache()
