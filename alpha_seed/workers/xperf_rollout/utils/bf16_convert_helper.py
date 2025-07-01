# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains utilities to bind weights to XPerfGPT. It is model agnostic
"""

import os
import torch
import torch.distributed

from functools import partial

from transformers import PretrainedConfig

from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh

from seed_models import P4Config, P5Config, P6Config

from alpha_seed.workers.xperf_rollout.utils.layout_convert_helper import _fix_qkv_ordering, _fix_o_ordering, load_to_cuda


def assert_not_nan(tensor: torch.Tensor):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if os.getenv('XPERF_CHECK_NAN', '1') == '1':
        assert not torch.any(torch.isnan(tensor)).item(), f'Got nan in parameter {tensor} on rank {rank}'


def _reshard_fsdp_state_dict_to_xperf_p4(tp_model, state_dict, device_mesh: DeviceMesh, model_config: P4Config):
    assert isinstance(model_config, P4Config)

    # checking
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
        assert tp_size <= model_config.num_key_value_heads
        assert model_config.num_key_value_heads % tp_size == 0
    else:
        tp_size = 1
        tp_rank = 0

    assert tp_size == model_config.num_key_value_heads or tp_size == 1 or \
        model_config.num_attention_heads == model_config.num_key_value_heads

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

    for layer_index, (ln_1, _, qkv_w, qkv_b, dense_w, dense_b, ln_2, fc1_w, fc1_b, fc2_w, fc2_b,
                      *_) in enumerate(tp_model.layers_weight):
        # torch.distributed.breakpoint()
        ln_1_weight = state_dict.pop(f'transformer.h.{layer_index}.ln_1.weight').full_tensor()
        ln_1_bias = state_dict.pop(f'transformer.h.{layer_index}.ln_1.bias').full_tensor()
        ln_1_weight = torch.stack((ln_1_weight, ln_1_bias), dim=0).to(torch.bfloat16)
        ln_1.data = ln_1_weight.contiguous()

        del ln_1_bias

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

        assert qkv_weight.shape == qkv_w.shape
        qkv_w.data = qkv_weight.contiguous()

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

        assert qkv_bias.shape == qkv_b.shape
        qkv_b.data = qkv_bias.contiguous()

        o_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.o_proj.weight').full_tensor().to(
            torch.bfloat16)
        if device_mesh is not None:
            o_proj_weight = _fix_o_ordering(o_proj_weight, num_heads=num_heads, q_heads_list=q_heads_list)[tp_rank]

        assert o_proj_weight.shape == dense_w.shape
        dense_w.data = o_proj_weight.contiguous()

        o_proj_bias = state_dict.pop(f'transformer.h.{layer_index}.attn.o_proj.bias').full_tensor().to(torch.bfloat16)
        assert o_proj_bias.shape == dense_b.shape
        dense_b.data = o_proj_bias.contiguous()

        ln_2_weight = state_dict.pop(f'transformer.h.{layer_index}.ln_2.weight').full_tensor().to(torch.bfloat16)
        ln_2_bias = state_dict.pop(f'transformer.h.{layer_index}.ln_2.bias').full_tensor().to(torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight, ln_2_bias), dim=0)
        assert ln_2_weight.shape == ln_2.shape
        ln_2.data = ln_2_weight.contiguous()

        del ln_2_bias

        fc1_weight = state_dict.pop(f'transformer.h.{layer_index}.mlp.c_fc.weight').to(torch.bfloat16).full_tensor()

        if device_mesh is not None:
            fc1_weight = DTensor.from_local(fc1_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            fc1_weight = fc1_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                      Shard(0)])._local_tensor

        assert fc1_w.shape == fc1_weight.shape
        fc1_w.data = fc1_weight

        fc1_bias = state_dict.pop(f'transformer.h.{layer_index}.mlp.c_fc.bias').to(torch.bfloat16).full_tensor()

        if device_mesh is not None:
            fc1_bias = DTensor.from_local(fc1_bias, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            fc1_bias = fc1_bias.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(0)])._local_tensor

        assert fc1_bias.shape == fc1_b.shape
        fc1_b.data = fc1_bias

        fc2_weight = state_dict.pop(f'transformer.h.{layer_index}.mlp.c_proj.weight').to(torch.bfloat16).full_tensor()

        if device_mesh is not None:
            fc2_weight = DTensor.from_local(fc2_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            fc2_weight = fc2_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                      Shard(1)])._local_tensor

        assert fc2_w.shape == fc2_weight.shape
        fc2_w.data = fc2_weight

        fc2_bias = state_dict.pop(f'transformer.h.{layer_index}.mlp.c_proj.bias').to(torch.bfloat16).full_tensor()
        assert fc2_b.shape == fc2_bias.shape
        fc2_b.data = fc2_bias

    load_to_cuda(tp_model=tp_model)

    # assert len(state_dict) == 0

    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_p5(tp_model, state_dict, device_mesh: DeviceMesh, model_config: P5Config):
    # TODO: add key norm

    assert isinstance(model_config, P5Config)

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

    for layer_index, (ln_1, _, qkv_w, qkv_b, dense_w, dense_b, ln_2, gate_w, _, fc1_w, _, fc2_w, _,
                      *_) in enumerate(tp_model.layers_weight):
        # torch.distributed.breakpoint()
        ln_1_weight = state_dict.pop(f'transformer.h.{layer_index}.ln_1.weight').full_tensor()
        ln_1_bias = state_dict.pop(f'transformer.h.{layer_index}.ln_1.bias').full_tensor()
        ln_1_weight = torch.stack((ln_1_weight, ln_1_bias), dim=0).to(torch.bfloat16)
        ln_1.data = ln_1_weight.contiguous()

        del ln_1_bias

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

        assert qkv_weight.shape == qkv_w.shape
        qkv_w.data = qkv_weight.contiguous()

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

        assert qkv_bias.shape == qkv_b.shape
        qkv_b.data = qkv_bias.contiguous()

        o_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.o_proj.weight').full_tensor().to(
            torch.bfloat16)
        if device_mesh is not None:
            o_proj_weight = _fix_o_ordering(o_proj_weight, num_heads=num_heads, q_heads_list=q_heads_list)[tp_rank]

        assert o_proj_weight.shape == dense_w.shape
        dense_w.data = o_proj_weight.contiguous()

        o_proj_bias = state_dict.pop(f'transformer.h.{layer_index}.attn.o_proj.bias').full_tensor().to(torch.bfloat16)
        assert o_proj_bias.shape == dense_b.shape
        dense_b.data = o_proj_bias.contiguous()

        ln_2_weight = state_dict.pop(f'transformer.h.{layer_index}.ln_2.weight').full_tensor().to(torch.bfloat16)
        ln_2_bias = state_dict.pop(f'transformer.h.{layer_index}.ln_2.bias').full_tensor().to(torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight, ln_2_bias), dim=0)
        assert ln_2_weight.shape == ln_2.shape
        ln_2.data = ln_2_weight.contiguous()

        del ln_2_bias

        gate_wg = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.gate.wg').full_tensor().T.contiguous().float()
        gate_wg_ema = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.gate.wg_ema').T.contiguous().float()

        gate_wg = (gate_wg + gate_wg_ema) * 0.5
        assert gate_wg.shape == gate_w.shape
        gate_w.data = gate_wg.contiguous()

        fc1_list = []
        for expert_index in range(model_config.moe_num_expert):
            fc1 = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts.{expert_index}.fc1.weight').to(
                torch.bfloat16).full_tensor()

            if device_mesh is not None:
                fc1 = DTensor.from_local(fc1, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
                fc1 = fc1.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(0)])._local_tensor

            fc1_list.append(fc1)

        # (num_experts, intermediate_size // tp, hidden_size)
        fc1_weight = torch.stack(fc1_list, dim=0).contiguous().flatten()

        assert fc1_weight.shape == fc1_w.shape
        fc1_w.data = fc1_weight.contiguous()

        del fc1_list

        fc2_list = []
        for expert_index in range(model_config.moe_num_expert):
            fc2 = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts.{expert_index}.fc2.weight').to(
                torch.bfloat16).full_tensor()
            if device_mesh is not None:
                fc2 = DTensor.from_local(fc2, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
                fc2 = fc2.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(1)])._local_tensor
            fc2_list.append(fc2)

        # (num_experts, intermediate_size // tp, hidden_size)
        fc2_weight = torch.stack(fc2_list, dim=0).contiguous().flatten()
        assert fc2_weight.shape == fc2_w.shape
        fc2_w.data = fc2_weight.contiguous()

        del fc2_list

    load_to_cuda(tp_model=tp_model)

    # assert len(state_dict) == 0

    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_p6(tp_model,
                                         state_dict,
                                         device_mesh: DeviceMesh,
                                         model_config: P6Config,
                                         prefix=""):
    assert isinstance(model_config, P6Config)

    # checking
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
    else:
        tp_size = 1
        tp_rank = 0

    head_dim = model_config.hidden_size // model_config.num_attention_heads

    ln_f_weight = state_dict.pop(prefix + 'transformer.ln_f.weight').full_tensor().to(torch.bfloat16)
    ln_f_bias = state_dict.pop(prefix + 'transformer.ln_f.bias').full_tensor().to(torch.bfloat16)
    ln_f = torch.stack((ln_f_weight, ln_f_bias)).contiguous()

    tp_model.layernorm_weight.data = ln_f

    assert_not_nan(tp_model.layernorm_weight.data)

    del ln_f_weight, ln_f_bias

    # TODO: use xperf vocab_tp
    wte: DTensor = state_dict.pop(prefix + 'transformer.wte.weight').to(torch.bfloat16)
    state_dict.pop(prefix + 'lm_head.weight')

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

    assert_not_nan(tp_model.wte_weight.data)
    assert_not_nan(tp_model.lm_head_weight.data)

    for layer_index, (ln_1, key_norm, context_norm, qkv_w, qkv_b, dense_w, dense_b, ln_2, gate_w, _, fc1_w, _, fc2_w, _,
                      *_) in enumerate(tp_model.layers_weight):
        # torch.distributed.breakpoint()
        ln_1_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.ln_1.weight').full_tensor()
        ln_1_bias = state_dict.pop(prefix + f'transformer.h.{layer_index}.ln_1.bias').full_tensor()
        ln_1_weight = torch.stack((ln_1_weight, ln_1_bias), dim=0).to(torch.bfloat16)
        assert ln_1.data.shape == ln_1_weight.shape
        ln_1.data = ln_1_weight.contiguous()

        assert_not_nan(ln_1.data)

        del ln_1_bias

        key_norm_weight = state_dict[prefix + f'transformer.h.{layer_index}.attn.key_layernorm.weight'].full_tensor()
        key_norm_bias = state_dict[prefix + f'transformer.h.{layer_index}.attn.key_layernorm.bias'].full_tensor()
        key_norm_weight = torch.stack((key_norm_weight, key_norm_bias), dim=0).to(torch.bfloat16)
        assert key_norm.data.shape == key_norm_weight.shape
        key_norm.data = key_norm_weight.contiguous()

        assert_not_nan(key_norm.data)

        del key_norm_bias

        context_norm_weight = state_dict[prefix + f'transformer.h.{layer_index}.attn.context_norm.weight'].full_tensor()
        context_norm_bias = state_dict[prefix + f'transformer.h.{layer_index}.attn.context_norm.bias'].full_tensor()
        context_norm_weight = torch.stack((context_norm_weight, context_norm_bias), dim=0).to(torch.bfloat16)
        assert context_norm.data.shape == context_norm_weight.shape
        context_norm.data = context_norm_weight.contiguous()

        assert_not_nan(context_norm.data)

        del context_norm_bias

        q_proj_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.q_proj.weight').full_tensor().to(
            torch.bfloat16)  # (num_q_head, head_dim, hidden_size)
        k_proj_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.k_proj.weight').full_tensor().to(
            torch.bfloat16)  # (num_kv_head, head_dim, hidden_size)
        v_proj_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.v_proj.weight').full_tensor().to(
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

        assert qkv_weight.shape == qkv_w.shape
        qkv_w.data = qkv_weight.contiguous()

        # check nan
        assert_not_nan(qkv_w.data)

        q_proj_bias = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.q_proj.bias').full_tensor().to(
            torch.bfloat16)
        k_proj_bias = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.k_proj.bias').full_tensor().to(
            torch.bfloat16)
        v_proj_bias = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.v_proj.bias').full_tensor().to(
            torch.bfloat16)

        qkv_bias = torch.cat((q_proj_bias, k_proj_bias, v_proj_bias), dim=0)
        if device_mesh is not None:
            qkv_bias, q_heads_list = _fix_qkv_ordering(qkv_bias,
                                                       tp_size=tp_size,
                                                       num_heads=num_heads,
                                                       mqa_kv_heads=mqa_kv_heads,
                                                       interleaved_kv_shared=model_config.interleaved_kv_shared)
            qkv_bias = qkv_bias[tp_rank]

        assert qkv_bias.shape == qkv_b.shape
        qkv_b.data = qkv_bias.contiguous()

        assert_not_nan(qkv_b.data)

        o_proj_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.o_proj.weight').full_tensor().to(
            torch.bfloat16)
        if device_mesh is not None:
            o_proj_weight = _fix_o_ordering(o_proj_weight, num_heads=num_heads, q_heads_list=q_heads_list)[tp_rank]

        assert o_proj_weight.shape == dense_w.shape
        dense_w.data = o_proj_weight.contiguous()

        assert_not_nan(dense_w.data)

        o_proj_bias = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.o_proj.bias').full_tensor().to(
            torch.bfloat16)
        assert o_proj_bias.shape == dense_b.shape
        dense_b.data = o_proj_bias.contiguous()

        assert_not_nan(dense_b.data)

        ln_2_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.ln_2.weight').full_tensor().to(
            torch.bfloat16)
        ln_2_bias = state_dict.pop(prefix + f'transformer.h.{layer_index}.ln_2.bias').full_tensor().to(torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight, ln_2_bias), dim=0)
        assert ln_2_weight.shape == ln_2.shape
        ln_2.data = ln_2_weight.contiguous()

        assert_not_nan(ln_2.data)

        del ln_2_bias

        gate_wg = state_dict.pop(prefix +
                                 f'transformer.h.{layer_index}.mlp.moe.gate.wg').full_tensor().T.contiguous().float()
        gate_wg_ema = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.gate.wg_ema').T.contiguous().float()

        gate_wg = (gate_wg + gate_wg_ema) * 0.5

        assert gate_wg.shape == gate_w.shape
        gate_w.data = gate_wg.contiguous()

        assert_not_nan(gate_w.data)

        use_grouped_gemm_weight = getattr(model_config, '_moe_implementation', 'eager') == 'fused'

        if prefix + f'transformer.h.{layer_index}.mlp.moe.experts.fc1_1_weight' in state_dict:
            raise ValueError(
                '请使用最新的依赖。并根据文档 https://bytedance.larkoffice.com/docx/SBuXdoDpgoCwiDxV4Hwco1ahnug 重新转p6 checkpoint')

        if use_grouped_gemm_weight:
            fc1_1_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.experts.fc1_1').to(
                torch.bfloat16).full_tensor()
            if device_mesh is not None:
                fc1_1_weight = DTensor.from_local(fc1_1_weight,
                                                  device_mesh=device_mesh,
                                                  placements=[Replicate(), Replicate()])
                fc1_1_weight = fc1_1_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                              Shard(1)])._local_tensor

            fc1_2_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.experts.fc1_2').to(
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
                fc1_1 = state_dict.pop(prefix +
                                       f'transformer.h.{layer_index}.mlp.moe.experts.{expert_index}.fc1_1.weight').to(
                                           torch.bfloat16).full_tensor()
                fc1_2 = state_dict.pop(prefix +
                                       f'transformer.h.{layer_index}.mlp.moe.experts.{expert_index}.fc1_2.weight').to(
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

        assert fc1_weight.shape == fc1_w.shape
        fc1_w.data = fc1_weight.contiguous()

        assert_not_nan(fc1_w.data)

        if use_grouped_gemm_weight:
            fc2_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.experts.fc2').to(
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
                fc2 = state_dict.pop(prefix +
                                     f'transformer.h.{layer_index}.mlp.moe.experts.{expert_index}.fc2.weight').to(
                                         torch.bfloat16).full_tensor()
                if device_mesh is not None:
                    fc2 = DTensor.from_local(fc2, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
                    fc2 = fc2.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(1)])._local_tensor
                fc2_list.append(fc2)

            fc2_weight = torch.stack(fc2_list, dim=0).contiguous().flatten()

            del fc2_list

        # (num_experts, intermediate_size // tp, hidden_size)
        assert fc2_weight.shape == fc2_w.shape, f'{fc2_weight.shape=}, {fc2_w.shape=}'
        fc2_w.data = fc2_weight.contiguous()

        assert_not_nan(fc2_w.data)

    load_to_cuda(tp_model=tp_model)

    # assert len(state_dict) == 0

    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_p6dense(tp_model, state_dict, device_mesh: DeviceMesh, model_config):
    from seed_models import P6DenseConfig
    assert isinstance(model_config, P6DenseConfig)

    # checking
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
        assert tp_size <= model_config.num_key_value_heads
        assert model_config.num_key_value_heads % tp_size == 0
    else:
        tp_size = 1
        tp_rank = 0

    assert model_config.num_key_value_heads % tp_size == 0

    head_dim = getattr(model_config, 'head_dim', None) or (model_config.hidden_size // model_config.num_attention_heads)
    hidden_size = model_config.hidden_size
    num_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads

    norm = state_dict.pop('model.norm.weight').full_tensor().to(torch.bfloat16).unsqueeze(0)
    assert norm.shape == tp_model.layernorm_weight.shape
    tp_model.layernorm_weight.data = norm

    assert_not_nan(tp_model.layernorm_weight.data)

    wte_weight = state_dict.pop('model.embed_tokens.weight').full_tensor().to(torch.bfloat16)
    lm_head = state_dict.pop('lm_head.weight').full_tensor().to(torch.bfloat16)

    # TODO: implement vocab_tp

    if device_mesh is not None and tp_model.wte_weight.data.shape != wte_weight.shape:
        assert False, "Not tested yet"
        # TODO: we may need to do full first
        wte_weight = DTensor.from_local(wte_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
        wte_weight_tp = wte_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                     Shard(1)])._local_tensor

        lm_head = DTensor.from_local(lm_head, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
        lm_head_tp = wte_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(0)])._local_tensor
    else:
        wte_weight_tp = wte_weight
        lm_head_tp = lm_head

    wte_weight_tp = wte_weight_tp.contiguous()
    lm_head_tp = lm_head_tp.contiguous()

    assert tp_model.wte_weight.data.shape == wte_weight_tp.shape
    assert tp_model.lm_head_weight.data.shape == lm_head_tp.shape

    tp_model.wte_weight.data = wte_weight_tp
    tp_model.lm_head_weight.data = lm_head_tp

    assert_not_nan(tp_model.wte_weight.data)
    assert_not_nan(tp_model.lm_head_weight.data)

    for layer_index, (ln_1, qkv_w, qkv_bias, dense_w, dense_bias, ln_2, fc12, _, fc2, _,
                      *extra_params) in enumerate(tp_model.layers_weight):
        # model.layers.0.input_layernorm.weight
        ln_1_weight = state_dict.pop(f'model.layers.{layer_index}.input_layernorm.weight').full_tensor()
        ln_1_weight = torch.unsqueeze(ln_1_weight, dim=0).to(torch.bfloat16)
        assert ln_1.data.shape == ln_1_weight.shape
        ln_1.data = ln_1_weight.contiguous()

        assert_not_nan(ln_1.data)

        # model.layers.0.post_attention_layernorm.weight
        ln_2_weight = state_dict.pop(f'model.layers.{layer_index}.post_attention_layernorm.weight').full_tensor()
        ln_2_weight = torch.unsqueeze(ln_2_weight, dim=0).to(torch.bfloat16)
        assert ln_2.data.shape == ln_2_weight.shape
        ln_2.data = ln_2_weight.contiguous()

        assert_not_nan(ln_2.data)

        # model.layers.0.self_attn.q_proj.weight
        # model.layers.0.self_attn.k_proj.weight
        # model.layers.0.self_attn.v_proj.weight

        q_proj_weight = state_dict.pop(f'model.layers.{layer_index}.self_attn.q_proj.weight').full_tensor().to(
            torch.bfloat16)
        q_proj_weight = q_proj_weight.view(num_kv_heads, -1, head_dim, hidden_size).transpose(0, 1).contiguous()
        k_proj_weight = state_dict.pop(f'model.layers.{layer_index}.self_attn.k_proj.weight').full_tensor().to(
            torch.bfloat16)
        k_proj_weight = k_proj_weight.view(1, num_kv_heads, head_dim, hidden_size)
        v_proj_weight = state_dict.pop(f'model.layers.{layer_index}.self_attn.v_proj.weight').full_tensor().to(
            torch.bfloat16)
        v_proj_weight = v_proj_weight.view(1, num_kv_heads, head_dim, hidden_size)

        # optional has qkv bias and dense_bias
        if qkv_bias is not None:
            assert model_config.attention_bias
            q_proj_bias = state_dict.pop(f'model.layers.{layer_index}.self_attn.q_proj.bias').full_tensor().to(
                torch.bfloat16)
            q_proj_bias = q_proj_bias.view(num_kv_heads, -1, head_dim).transpose(0, 1).contiguous()
            k_proj_bias = state_dict.pop(f'model.layers.{layer_index}.self_attn.k_proj.bias').full_tensor().to(
                torch.bfloat16)
            k_proj_bias = k_proj_bias.view(1, num_kv_heads, head_dim)
            v_proj_bias = state_dict.pop(f'model.layers.{layer_index}.self_attn.v_proj.bias').full_tensor().to(
                torch.bfloat16)
            v_proj_bias = v_proj_bias.view(1, num_kv_heads, head_dim)

        # shard qkv and concat
        if device_mesh is not None:
            q_proj_weight = DTensor.from_local(q_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            q_proj_weight = q_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            k_proj_weight = DTensor.from_local(k_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            k_proj_weight = k_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            v_proj_weight = DTensor.from_local(v_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            v_proj_weight = v_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            if qkv_bias is not None:
                q_proj_bias = DTensor.from_local(q_proj_bias,
                                                 device_mesh=device_mesh,
                                                 placements=[Replicate(), Replicate()])
                q_proj_bias = q_proj_bias.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

                k_proj_bias = DTensor.from_local(k_proj_bias,
                                                 device_mesh=device_mesh,
                                                 placements=[Replicate(), Replicate()])
                k_proj_bias = k_proj_bias.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

                v_proj_bias = DTensor.from_local(v_proj_bias,
                                                 device_mesh=device_mesh,
                                                 placements=[Replicate(), Replicate()])
                v_proj_bias = v_proj_bias.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

        qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0).contiguous().view(-1, hidden_size)

        assert qkv_w.data.shape == qkv_weight.shape, f"{qkv_w.data.shape} != {qkv_weight.shape}"
        qkv_w.data = qkv_weight.contiguous()

        assert_not_nan(qkv_w.data)

        if qkv_bias is not None:
            qkv_bias_hf = torch.cat((q_proj_bias, k_proj_bias, v_proj_bias), dim=0).contiguous().view(-1)
            assert qkv_bias.data.shape == qkv_bias_hf.shape
            qkv_bias.data = qkv_bias_hf.contiguous()
            assert_not_nan(qkv_bias.data)

        if getattr(model_config, 'use_qk_rmsnorm', False):
            q_norm, k_norm, *_ = extra_params
            q_norm_weight = state_dict.pop(f'model.layers.{layer_index}.self_attn.q_norm.weight').full_tensor().to(
                torch.bfloat16)[None, :]
            k_norm_weight = state_dict.pop(f'model.layers.{layer_index}.self_attn.k_norm.weight').full_tensor().to(
                torch.bfloat16)[None, :]
            assert q_norm.data.shape == q_norm_weight.shape
            assert k_norm.data.shape == k_norm_weight.shape
            q_norm.data = q_norm_weight.contiguous()
            k_norm.data = k_norm_weight.contiguous()

        # model.layers.0.self_attn.o_proj.weight
        o_proj_weight = state_dict.pop(f'model.layers.{layer_index}.self_attn.o_proj.weight').full_tensor().to(
            torch.bfloat16)

        # the XPerfGPT has different ordering
        o_proj_weight = o_proj_weight.view(hidden_size, num_kv_heads, -1,
                                           head_dim).transpose(1, 2)  # (hidden_size, -1, num_kv_heads, head_dim)

        if dense_bias is not None:
            assert model_config.attention_bias
            if hasattr(model_config, 'o_bias') and not model_config.o_bias:
                o_proj_bias = torch.zeros_like(dense_bias)
            elif hasattr(model_config, 'attention_out_bias') and not model_config.attention_out_bias:
                o_proj_bias = torch.zeros_like(dense_bias)
            else:
                o_proj_bias = state_dict.pop(f'model.layers.{layer_index}.self_attn.o_proj.bias').full_tensor().to(
                    torch.bfloat16)
                o_proj_bias = o_proj_bias.view(num_kv_heads, -1, head_dim).transpose(1,
                                                                                     2)  # (-1, num_kv_heads, head_dim)

        if device_mesh is not None:
            o_proj_weight = DTensor.from_local(o_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            o_proj_weight = o_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(2)])._local_tensor

        o_proj_weight = o_proj_weight.contiguous().view(hidden_size, -1).contiguous()

        assert dense_w.data.shape == o_proj_weight.shape
        dense_w.data = o_proj_weight

        assert_not_nan(dense_w.data)

        if dense_bias is not None:
            # Note that o_proj_bias is NOT chunked in tp group
            o_proj_bias = o_proj_bias.contiguous().view(-1).contiguous()
            assert dense_bias.data.shape == o_proj_bias.shape
            dense_bias.data = o_proj_bias
            assert_not_nan(dense_bias.data)

        # model.layers.0.mlp.gate_proj.weight

        gate_proj_weight = state_dict.pop(f'model.layers.{layer_index}.mlp.gate_proj.weight').full_tensor().to(
            torch.bfloat16)

        if device_mesh is not None:
            gate_proj_weight = DTensor.from_local(gate_proj_weight,
                                                  device_mesh=device_mesh,
                                                  placements=[Replicate(), Replicate()])
            gate_proj_weight = gate_proj_weight.redistribute(device_mesh=device_mesh,
                                                             placements=[Replicate(), Shard(0)])._local_tensor

        gate_proj_weight = gate_proj_weight.contiguous()

        # model.layers.0.mlp.up_proj.weight
        up_proj_weight = state_dict.pop(f'model.layers.{layer_index}.mlp.up_proj.weight').full_tensor().to(
            torch.bfloat16)

        if device_mesh is not None:
            up_proj_weight = DTensor.from_local(up_proj_weight,
                                                device_mesh=device_mesh,
                                                placements=[Replicate(), Replicate()])
            up_proj_weight = up_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                              Shard(0)])._local_tensor

        up_proj_weight = up_proj_weight.contiguous()

        gate_up_proj_weight = torch.cat((gate_proj_weight, up_proj_weight), dim=0).contiguous()

        assert fc12.data.shape == gate_up_proj_weight.shape
        fc12.data = gate_up_proj_weight.contiguous()

        assert_not_nan(fc12.data)

        # model.layers.0.mlp.down_proj.weight

        down_proj_weight = state_dict.pop(f'model.layers.{layer_index}.mlp.down_proj.weight').full_tensor().to(
            torch.bfloat16)

        if device_mesh is not None:
            down_proj_weight = DTensor.from_local(down_proj_weight,
                                                  device_mesh=device_mesh,
                                                  placements=[Replicate(), Replicate()])
            down_proj_weight = down_proj_weight.redistribute(device_mesh=device_mesh,
                                                             placements=[Replicate(), Shard(1)])._local_tensor

        down_proj_weight = down_proj_weight.contiguous()

        assert fc2.data.shape == down_proj_weight.shape
        fc2.data = down_proj_weight.contiguous()

        assert_not_nan(fc2.data)

    load_to_cuda(tp_model=tp_model)

    assert len(state_dict) == 0

    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_p7(tp_model, state_dict, device_mesh: DeviceMesh, model_config):
    from seed_models import P7Config
    assert isinstance(model_config, P7Config)

    # checking
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
        assert tp_size <= model_config.num_key_value_heads
        assert model_config.num_key_value_heads % tp_size == 0
    else:
        tp_size = 1
        tp_rank = 0

    # assert tp_size == model_config.num_key_value_heads or tp_size == 1

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    num_kv_heads = model_config.num_key_value_heads

    ln_f_weight = state_dict.pop('transformer.ln_f.weight').full_tensor().to(torch.bfloat16)
    ln_f_weight = ln_f_weight.reshape(1, ln_f_weight.shape[-1])
    tp_model.layernorm_weight.data = ln_f_weight.contiguous()
    del ln_f_weight

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

    for layer_index, (ln_1, key_norm, context_norm, qkv_w, qkv_b, dense_w, dense_b, ln_2, gate_w, _, fc1_w, _, fc2_w, _,
                      *_) in enumerate(tp_model.layers_weight):
        ln_1_weight = state_dict.pop(f'transformer.h.{layer_index}.ln_1.weight').full_tensor()
        ln_1_weight = torch.stack((ln_1_weight,), dim=0).to(torch.bfloat16).reshape(1, ln_1_weight.shape[-1])
        assert ln_1.data.shape == ln_1_weight.shape
        ln_1.data = ln_1_weight.contiguous()

        key_norm_weight = state_dict[f'transformer.h.{layer_index}.attn.key_layernorm.weight'].full_tensor()
        key_norm_weight = torch.stack((key_norm_weight,),
                                      dim=0).to(torch.bfloat16).reshape(1, key_norm_weight.shape[-1])
        assert key_norm.data.shape == key_norm_weight.shape
        key_norm.data = key_norm_weight.contiguous()

        context_norm_weight = state_dict[f'transformer.h.{layer_index}.attn.context_norm.weight'].full_tensor()
        context_norm_weight = torch.stack((context_norm_weight,),
                                          dim=0).to(torch.bfloat16).reshape(1, context_norm_weight.shape[-1])
        assert context_norm.data.shape == context_norm_weight.shape
        context_norm.data = context_norm_weight.contiguous()

        q_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.q_proj.weight').full_tensor().to(
            torch.bfloat16)
        q_proj_weight = q_proj_weight.view(num_kv_heads, -1, head_dim, hidden_size).transpose(0, 1).contiguous()
        k_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.k_proj.weight').full_tensor().to(
            torch.bfloat16)
        k_proj_weight = k_proj_weight.view(1, num_kv_heads, head_dim, hidden_size)
        v_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.v_proj.weight').full_tensor().to(
            torch.bfloat16)
        v_proj_weight = v_proj_weight.view(1, num_kv_heads, head_dim, hidden_size)

        # shard qkv and concat
        if device_mesh is not None:
            q_proj_weight = DTensor.from_local(q_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            q_proj_weight = q_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            k_proj_weight = DTensor.from_local(k_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            k_proj_weight = k_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            v_proj_weight = DTensor.from_local(v_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            v_proj_weight = v_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

        qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0).contiguous().view(-1, hidden_size)

        assert qkv_w.data.shape == qkv_weight.shape
        qkv_w.data = qkv_weight.contiguous()

        o_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.o_proj.weight').full_tensor().to(
            torch.bfloat16)

        # the XPerfGPT has different ordering
        o_proj_weight = o_proj_weight.view(hidden_size, num_kv_heads, -1,
                                           head_dim).transpose(1, 2)  # (hidden_size, -1, num_kv_heads, head_dim)

        if device_mesh is not None:
            o_proj_weight = DTensor.from_local(o_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            o_proj_weight = o_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(2)])._local_tensor

        o_proj_weight = o_proj_weight.contiguous().view(hidden_size, -1).contiguous()

        assert o_proj_weight.shape == dense_w.shape
        dense_w.data = o_proj_weight.contiguous()

        ln_2_weight = state_dict.pop(f'transformer.h.{layer_index}.ln_2.weight').full_tensor().to(torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight,), dim=0).reshape(1, ln_2_weight.shape[-1])
        assert ln_2_weight.shape == ln_2.shape
        ln_2.data = ln_2_weight.contiguous()

        gate_wg = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.gate.wg').full_tensor().T.contiguous().float()
        gate_wg_ema = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.gate.wg_ema').T.contiguous().float()

        gate_wg = (gate_wg + gate_wg_ema) * 0.5

        assert gate_wg.shape == gate_w.shape
        gate_w.data = gate_wg.contiguous()

        # use_grouped_gemm_weight = getattr(model_config, '_moe_implementation', 'eager') == 'fused'
        # assert model_config._moe_implementation == fused
        # assert use_grouped_gemm_weight
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

        base_fc1_weight = torch.cat((fc1_1_weight, fc1_2_weight), dim=1)
        del fc1_1_weight
        del fc1_2_weight

        share_fc1_1_weight = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts_share.fc1_1').to(
            torch.bfloat16).full_tensor()
        share_fc1_1_weight = share_fc1_1_weight.reshape(2, -1, share_fc1_1_weight.shape[-1])  ##
        if device_mesh is not None:
            share_fc1_1_weight = DTensor.from_local(share_fc1_1_weight,
                                                    device_mesh=device_mesh,
                                                    placements=[Replicate(), Replicate()])
            share_fc1_1_weight = share_fc1_1_weight.redistribute(device_mesh=device_mesh,
                                                                 placements=[Replicate(), Shard(1)])._local_tensor

        share_fc1_2_weight = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts_share.fc1_2').to(
            torch.bfloat16).full_tensor()
        share_fc1_2_weight = share_fc1_2_weight.reshape(2, -1, share_fc1_2_weight.shape[-1])

        if device_mesh is not None:
            share_fc1_2_weight = DTensor.from_local(share_fc1_2_weight,
                                                    device_mesh=device_mesh,
                                                    placements=[Replicate(), Replicate()])
            share_fc1_2_weight = share_fc1_2_weight.redistribute(device_mesh=device_mesh,
                                                                 placements=[Replicate(), Shard(1)])._local_tensor
        share_fc1_weight = torch.cat((share_fc1_1_weight, share_fc1_2_weight), dim=1)
        del share_fc1_1_weight
        del share_fc1_2_weight

        fc1_weight = torch.cat((base_fc1_weight, share_fc1_weight), dim=0).contiguous().flatten()
        del base_fc1_weight
        del share_fc1_weight

        assert fc1_weight.shape == fc1_w.shape
        fc1_w.data = fc1_weight.contiguous()

        fc2_weight = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts.fc2').to(torch.bfloat16).full_tensor()
        if device_mesh is not None:
            fc2_weight = DTensor.from_local(fc2_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            fc2_weight = fc2_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                      Shard(2)])._local_tensor

        share_fc2_weight = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.experts_share.fc2').to(
            torch.bfloat16).full_tensor()
        share_fc2_weight = share_fc2_weight.reshape(share_fc2_weight.shape[-2], 2, -1).transpose(0, 1)
        if device_mesh is not None:

            share_fc2_weight = DTensor.from_local(share_fc2_weight,
                                                  device_mesh=device_mesh,
                                                  placements=[Replicate(), Replicate()])
            share_fc2_weight = share_fc2_weight.redistribute(device_mesh=device_mesh,
                                                             placements=[Replicate(), Shard(2)])._local_tensor

        fc2_weight_merge = torch.cat((fc2_weight, share_fc2_weight), dim=0).contiguous().flatten()
        del fc2_weight
        del share_fc2_weight

        # (num_experts, intermediate_size // tp, hidden_size)
        assert fc2_weight_merge.shape == fc2_w.shape, f'{fc2_weight_merge.shape=}, {fc2_w.shape=}'
        fc2_w.data = fc2_weight_merge.contiguous()

    # enforce check nan
    assert_not_nan(tp_model.layernorm_weight.data)
    assert_not_nan(tp_model.wte_weight.data)
    assert_not_nan(tp_model.lm_head_weight.data)

    for layer_index, weights in enumerate(tp_model.layers_weight):
        for weight in weights:
            if isinstance(weight, torch.Tensor):
                assert_not_nan(weight)

    load_to_cuda(tp_model=tp_model)
    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_m8(tp_model, state_dict, device_mesh: DeviceMesh, model_config, prefix=''):
    from seed_models import M8Config
    assert isinstance(model_config, M8Config)

    # checking
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
        # assert tp_size <= model_config.num_key_value_heads
        assert model_config.num_key_value_heads % tp_size == 0 or tp_size % model_config.num_key_value_heads == 0
    else:
        tp_size = 1
        tp_rank = 0

    # assert tp_size == model_config.num_key_value_heads or tp_size == 1

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    num_kv_heads = model_config.num_key_value_heads
    ln_f_weight = state_dict.pop(prefix + 'transformer.norm.weight').full_tensor().to(torch.bfloat16)
    ln_f_weight = ln_f_weight.reshape(1, ln_f_weight.shape[-1])
    tp_model.layernorm_weight.data = ln_f_weight.contiguous()
    del ln_f_weight

    # TODO: use xperf vocab_tp
    wte: DTensor = state_dict.pop(prefix + 'transformer.wte.weight').to(torch.bfloat16)
    state_dict.pop(prefix + 'lm_head.weight')

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

    for layer_index, (ln_1, key_norm, context_norm, qkv_w, qkv_b, dense_w, dense_b, ln_2, gate_w, _, fc1_w, _, fc2_w, _,
                      share_fc1_w, share_fc2_w, *_) in enumerate(tp_model.layers_weight):
        ln_1_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.input_layernorm.weight').full_tensor()
        ln_1_weight = torch.stack((ln_1_weight,), dim=0).to(torch.bfloat16).reshape(1, ln_1_weight.shape[-1])
        assert ln_1.data.shape == ln_1_weight.shape
        ln_1.data = ln_1_weight.contiguous()

        key_norm_weight = state_dict[prefix + f'transformer.h.{layer_index}.attn.key_layernorm.weight'].full_tensor()
        key_norm_weight = torch.stack((key_norm_weight,),
                                      dim=0).to(torch.bfloat16).reshape(1, key_norm_weight.shape[-1])
        assert key_norm.data.shape == key_norm_weight.shape
        key_norm.data = key_norm_weight.contiguous()

        context_norm_weight = state_dict[prefix + f'transformer.h.{layer_index}.attn.context_norm.weight'].full_tensor()
        context_norm_weight = torch.stack((context_norm_weight,),
                                          dim=0).to(torch.bfloat16).reshape(1, context_norm_weight.shape[-1])
        assert context_norm.data.shape == context_norm_weight.shape
        context_norm.data = context_norm_weight.contiguous()

        q_proj_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.q_proj.weight').full_tensor().to(
            torch.bfloat16)
        q_proj_weight = q_proj_weight.view(num_kv_heads, -1, head_dim, hidden_size).transpose(0, 1).contiguous()
        k_proj_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.k_proj.weight').full_tensor().to(
            torch.bfloat16)
        k_proj_weight = k_proj_weight.view(1, num_kv_heads, head_dim, hidden_size)
        v_proj_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.v_proj.weight').full_tensor().to(
            torch.bfloat16)
        v_proj_weight = v_proj_weight.view(1, num_kv_heads, head_dim, hidden_size)

        kv_replicate = tp_size // model_config.num_key_value_heads
        # duplicate kv
        if kv_replicate > 1:
            q_proj_weight = q_proj_weight.view(-1, tp_size, head_dim, hidden_size).contiguous()
            k_proj_weight = torch.tile(k_proj_weight, (1, kv_replicate, 1, 1))
            v_proj_weight = torch.tile(v_proj_weight, (1, kv_replicate, 1, 1))

        # shard qkv and concat
        if device_mesh is not None:
            q_proj_weight = DTensor.from_local(q_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            q_proj_weight = q_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            k_proj_weight = DTensor.from_local(k_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            k_proj_weight = k_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            v_proj_weight = DTensor.from_local(v_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            v_proj_weight = v_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

        qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0).contiguous().view(-1, hidden_size)

        assert qkv_w.data.shape == qkv_weight.shape
        qkv_w.data = qkv_weight.contiguous()

        o_proj_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.attn.o_proj.weight').full_tensor().to(
            torch.bfloat16)

        # the XPerfGPT has different ordering
        o_proj_weight = o_proj_weight.view(hidden_size, num_kv_heads, -1, head_dim).transpose(
            1, 2).contiguous()  # (hidden_size, -1, num_kv_heads, head_dim)

        if kv_replicate > 1:
            o_proj_weight = o_proj_weight.view(hidden_size, -1, tp_size, head_dim)

        if device_mesh is not None:
            o_proj_weight = DTensor.from_local(o_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            o_proj_weight = o_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(2)])._local_tensor

        o_proj_weight = o_proj_weight.contiguous().view(hidden_size, -1).contiguous()

        assert o_proj_weight.shape == dense_w.shape
        dense_w.data = o_proj_weight.contiguous()

        ln_2_weight = state_dict.pop(prefix +
                                     f'transformer.h.{layer_index}.post_attention_layernorm.weight').full_tensor().to(
                                         torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight,), dim=0).reshape(1, ln_2_weight.shape[-1])
        assert ln_2_weight.shape == ln_2.shape
        ln_2.data = ln_2_weight.contiguous()

        gate_wg = state_dict.pop(prefix +
                                 f'transformer.h.{layer_index}.mlp.moe.gate.wg').full_tensor().T.contiguous().float()
        gate_wg_ema = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.gate.wg_ema').T.contiguous().float()

        gate_wg = (gate_wg + gate_wg_ema) * 0.5

        assert gate_wg.shape == gate_w.shape
        gate_w.data = gate_wg.contiguous()

        # use_grouped_gemm_weight = getattr(model_config, '_moe_implementation', 'eager') == 'fused'
        # assert model_config._moe_implementation == fused
        # assert use_grouped_gemm_weight
        fc1_1_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.experts.fc1_1').to(
            torch.bfloat16).full_tensor()

        if device_mesh is not None:
            fc1_1_weight = DTensor.from_local(fc1_1_weight,
                                              device_mesh=device_mesh,
                                              placements=[Replicate(), Replicate()])
            fc1_1_weight = fc1_1_weight.redistribute(device_mesh=device_mesh,
                                                     placements=[Replicate(),
                                                                 Shard(0 if tp_model.use_ep else 1)])._local_tensor

        fc1_2_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.experts.fc1_2').to(
            torch.bfloat16).full_tensor()

        if device_mesh is not None:
            fc1_2_weight = DTensor.from_local(fc1_2_weight,
                                              device_mesh=device_mesh,
                                              placements=[Replicate(), Replicate()])
            fc1_2_weight = fc1_2_weight.redistribute(device_mesh=device_mesh,
                                                     placements=[Replicate(),
                                                                 Shard(0 if tp_model.use_ep else 1)])._local_tensor

        base_fc1_weight = torch.cat((fc1_1_weight, fc1_2_weight), dim=1)
        del fc1_1_weight
        del fc1_2_weight

        share_fc1_1_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.experts_share.fc1_1').to(
            torch.bfloat16).full_tensor()
        share_fc1_1_weight = share_fc1_1_weight.reshape(2, -1, share_fc1_1_weight.shape[-1])  ##
        if device_mesh is not None:
            share_fc1_1_weight = DTensor.from_local(share_fc1_1_weight,
                                                    device_mesh=device_mesh,
                                                    placements=[Replicate(), Replicate()])
            share_fc1_1_weight = share_fc1_1_weight.redistribute(device_mesh=device_mesh,
                                                                 placements=[Replicate(), Shard(1)])._local_tensor

        share_fc1_2_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.experts_share.fc1_2').to(
            torch.bfloat16).full_tensor()
        share_fc1_2_weight = share_fc1_2_weight.reshape(2, -1, share_fc1_2_weight.shape[-1])

        if device_mesh is not None:
            share_fc1_2_weight = DTensor.from_local(share_fc1_2_weight,
                                                    device_mesh=device_mesh,
                                                    placements=[Replicate(), Replicate()])
            share_fc1_2_weight = share_fc1_2_weight.redistribute(device_mesh=device_mesh,
                                                                 placements=[Replicate(), Shard(1)])._local_tensor
        share_fc1_weight = torch.cat((share_fc1_1_weight, share_fc1_2_weight), dim=1)
        del share_fc1_1_weight
        del share_fc1_2_weight

        if tp_model.use_ep:
            fc1_weight = base_fc1_weight.contiguous().flatten()
            s_fc1_weight = share_fc1_weight.reshape(share_fc1_weight.shape[0], 2, -1,
                                                    share_fc1_weight.shape[-1]).transpose(0, 1).reshape(
                                                        -1, share_fc1_weight.shape[-1])
        else:
            fc1_weight = torch.cat((base_fc1_weight, share_fc1_weight), dim=0).contiguous().flatten()
        del base_fc1_weight
        del share_fc1_weight

        assert fc1_weight.shape == fc1_w.shape, f'{fc1_weight.shape=}, {fc1_w.shape=}'
        fc1_w.data = fc1_weight.contiguous()
        if tp_model.use_ep:
            assert s_fc1_weight.shape == share_fc1_w.shape, f'{s_fc1_weight.shape=}, {share_fc1_w.shape=}'
            share_fc1_w.data = s_fc1_weight.contiguous()

        fc2_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.experts.fc2').to(
            torch.bfloat16).full_tensor()
        if device_mesh is not None:
            fc2_weight = DTensor.from_local(fc2_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            fc2_weight = fc2_weight.redistribute(device_mesh=device_mesh,
                                                 placements=[Replicate(),
                                                             Shard(0 if tp_model.use_ep else 2)])._local_tensor

        share_fc2_weight = state_dict.pop(prefix + f'transformer.h.{layer_index}.mlp.moe.experts_share.fc2').to(
            torch.bfloat16).full_tensor()
        share_fc2_weight = share_fc2_weight.reshape(share_fc2_weight.shape[-2], 2, -1).transpose(0, 1)
        if device_mesh is not None:

            share_fc2_weight = DTensor.from_local(share_fc2_weight,
                                                  device_mesh=device_mesh,
                                                  placements=[Replicate(), Replicate()])
            share_fc2_weight = share_fc2_weight.redistribute(device_mesh=device_mesh,
                                                             placements=[Replicate(), Shard(2)])._local_tensor

        if tp_model.use_ep:
            fc2_weight_merge = fc2_weight.contiguous().flatten()
            s_fc2_weight_merge = share_fc2_weight.transpose(0, 1).reshape(share_fc2_weight.shape[1], -1)
        else:
            fc2_weight_merge = torch.cat((fc2_weight, share_fc2_weight), dim=0).contiguous().flatten()
        del fc2_weight
        del share_fc2_weight

        # (num_experts, intermediate_size // tp, hidden_size)
        assert fc2_weight_merge.shape == fc2_w.shape, f'{fc2_weight_merge.shape=}, {fc2_w.shape=}'
        fc2_w.data = fc2_weight_merge.contiguous()
        if tp_model.use_ep:
            assert s_fc2_weight_merge.shape == share_fc2_w.shape, f'{s_fc2_weight_merge.shape=}, {share_fc2_w.shape=}'
            share_fc2_w.data = s_fc2_weight_merge.contiguous()

    # enforce check nan
    assert_not_nan(tp_model.layernorm_weight.data)
    assert_not_nan(tp_model.wte_weight.data)
    assert_not_nan(tp_model.lm_head_weight.data)

    for layer_index, weights in enumerate(tp_model.layers_weight):
        for i, weight in enumerate(weights):
            if isinstance(weight, torch.Tensor):
                assert_not_nan(weight)

    load_to_cuda(tp_model=tp_model)
    torch.cuda.empty_cache()


from .megatron_helper import allgather_from_megatron_tp, broadcast_from_megatron_pp


def _reshard_fsdp_state_dict_to_xperf_m8_megatron(tp_model, state_dict: dict, device_mesh: DeviceMesh, model_config):
    from megatron.core import parallel_state as mpu

    from seed_models import M8Config
    assert isinstance(model_config, M8Config)

    train_tp_size = mpu.get_tensor_model_parallel_world_size()

    # checking
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
        assert tp_size <= model_config.num_key_value_heads
        assert model_config.num_key_value_heads % tp_size == 0 or tp_size % model_config.num_key_value_heads == 0
    else:
        tp_size = 1
        tp_rank = 0

    # assert tp_size == model_config.num_key_value_heads or tp_size == 1

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    num_kv_heads = model_config.num_key_value_heads

    ln_f_weight = state_dict.pop('transformer.ln_f.weight', None)
    ln_f_weight = broadcast_from_megatron_pp(ln_f_weight)
    ln_f_weight = ln_f_weight.to(torch.bfloat16)
    ln_f_weight = ln_f_weight.reshape(1, ln_f_weight.shape[-1])
    tp_model.layernorm_weight.data = ln_f_weight.contiguous().clone()  # clone to avoid offload by trainer
    del ln_f_weight

    # TODO: use xperf vocab_tp
    wte = state_dict.pop('transformer.wte.weight', None)
    wte = broadcast_from_megatron_pp(wte)
    wte = allgather_from_megatron_tp(wte, dim=0).to(torch.bfloat16)
    # allgather from tp

    wte_weight = wte

    if device_mesh is not None and tp_model.wte_weight.data.shape != wte_weight.shape:
        # TODO: we may need to do full first
        raise NotImplementedError
        # del wte
    else:
        wte_weight_tp = wte_weight
        lm_head_tp = wte_weight

    assert wte_weight_tp.shape == tp_model.wte_weight.data.shape
    assert lm_head_tp.shape == tp_model.lm_head_weight.data.shape

    tp_model.wte_weight.data = wte_weight_tp.contiguous()
    tp_model.lm_head_weight.data = lm_head_tp.contiguous()  # tied weights

    for layer_index, (ln_1, key_norm, context_norm, qkv_w, qkv_b, dense_w, dense_b, ln_2, gate_w, _, fc1_w, _, fc2_w, _,
                      share_fc1_w, share_fc2_w, *_) in enumerate(tp_model.layers_weight):

        ln_1_weight = state_dict.pop(f'transformer.h.layers.{layer_index}.input_layernorm.weight', None)
        ln_1_weight = broadcast_from_megatron_pp(ln_1_weight).clone()  # clone to avoid offload by trainer
        ln_1_weight = torch.stack((ln_1_weight,), dim=0).to(torch.bfloat16).reshape(1, ln_1_weight.shape[-1])
        assert ln_1.data.shape == ln_1_weight.shape
        ln_1.data = ln_1_weight.contiguous()

        key_norm_weight = state_dict.pop(f'transformer.h.layers.{layer_index}.self_attention.key_layernorm.weight',
                                         None)
        key_norm_weight = broadcast_from_megatron_pp(key_norm_weight).clone()  # clone to avoid offload by trainer
        key_norm_weight = torch.stack((key_norm_weight,),
                                      dim=0).to(torch.bfloat16).reshape(1, key_norm_weight.shape[-1])
        assert key_norm.data.shape == key_norm_weight.shape
        key_norm.data = key_norm_weight.contiguous()

        context_norm_weight = state_dict.pop(
            f'transformer.h.layers.{layer_index}.self_attention.context_groupnorm.weight', None)
        context_norm_weight = broadcast_from_megatron_pp(
            context_norm_weight).clone()  # clone to avoid offload by trainer
        context_norm_weight = torch.stack((context_norm_weight,),
                                          dim=0).to(torch.bfloat16).reshape(1, context_norm_weight.shape[-1])
        assert context_norm.data.shape == context_norm_weight.shape
        context_norm.data = context_norm_weight.contiguous()

        qkv: torch.Tensor = state_dict.pop(f'transformer.h.layers.{layer_index}.self_attention.query_key_value.weight',
                                           None)
        qkv = broadcast_from_megatron_pp(qkv)

        q_dim_per_tp = head_dim * model_config.num_attention_heads * model_config.query_head_scale_factor // train_tp_size
        kv_dim_per_tp = head_dim * model_config.num_key_value_heads // train_tp_size
        assert qkv.shape[
            0] == q_dim_per_tp + kv_dim_per_tp * 2, f'Got {q_dim_per_tp=}, {kv_dim_per_tp=}, {qkv.shape=}, {train_tp_size=}'

        q, k, v = torch.split(qkv, split_size_or_sections=[q_dim_per_tp, kv_dim_per_tp, kv_dim_per_tp], dim=0)

        q_proj_weight = allgather_from_megatron_tp(q, dim=0)
        k_proj_weight = allgather_from_megatron_tp(k, dim=0)
        v_proj_weight = allgather_from_megatron_tp(v, dim=0)

        # q_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.q_proj.weight').full_tensor().to(
        #     torch.bfloat16)
        q_proj_weight = q_proj_weight.view(num_kv_heads, -1, head_dim, hidden_size).transpose(0, 1).contiguous()
        # k_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.k_proj.weight').full_tensor().to(
        #     torch.bfloat16)
        k_proj_weight = k_proj_weight.view(1, num_kv_heads, head_dim, hidden_size)
        # v_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.v_proj.weight').full_tensor().to(
        #     torch.bfloat16)
        v_proj_weight = v_proj_weight.view(1, num_kv_heads, head_dim, hidden_size)

        kv_replicate = tp_size // model_config.num_key_value_heads
        # duplicate kv
        if kv_replicate > 1:
            q_proj_weight = q_proj_weight.view(-1, tp_size, head_dim, hidden_size).contiguous()
            k_proj_weight = torch.tile(k_proj_weight, (1, kv_replicate, 1, 1))
            v_proj_weight = torch.tile(v_proj_weight, (1, kv_replicate, 1, 1))

        # shard qkv and concat
        if device_mesh is not None:
            q_proj_weight = DTensor.from_local(q_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            q_proj_weight = q_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            k_proj_weight = DTensor.from_local(k_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            k_proj_weight = k_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            v_proj_weight = DTensor.from_local(v_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            v_proj_weight = v_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

        qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0).contiguous().view(-1, hidden_size)

        assert qkv_w.data.shape == qkv_weight.shape
        qkv_w.data = qkv_weight.contiguous()

        o_proj_weight = state_dict.pop(f'transformer.h.layers.{layer_index}.self_attention.proj.weight', None)
        o_proj_weight = broadcast_from_megatron_pp(o_proj_weight)
        o_proj_weight = o_proj_weight.to(torch.bfloat16)
        o_proj_weight = allgather_from_megatron_tp(o_proj_weight, dim=1)

        # the XPerfGPT has different ordering
        o_proj_weight = o_proj_weight.view(hidden_size, num_kv_heads, -1, head_dim).transpose(
            1, 2).contiguous()  # (hidden_size, -1, num_kv_heads, head_dim)

        if kv_replicate > 1:
            o_proj_weight = o_proj_weight.view(hidden_size, -1, tp_size, head_dim)

        if device_mesh is not None:
            o_proj_weight = DTensor.from_local(o_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            o_proj_weight = o_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(2)])._local_tensor

        o_proj_weight = o_proj_weight.contiguous().view(hidden_size, -1).contiguous()

        assert o_proj_weight.shape == dense_w.shape
        dense_w.data = o_proj_weight.contiguous()

        ln_2_weight = state_dict.pop(f'transformer.h.layers.{layer_index}.post_attention_layernorm.weight', None)
        ln_2_weight = broadcast_from_megatron_pp(ln_2_weight).clone()  # clone to avoid offload by trainer
        ln_2_weight = ln_2_weight.to(torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight,), dim=0).reshape(1, ln_2_weight.shape[-1])
        assert ln_2_weight.shape == ln_2.shape
        ln_2.data = ln_2_weight.contiguous()

        gate_wg = state_dict.pop(f'transformer.h.layers.{layer_index}.mlp.moe.gate.wg', None)
        gate_wg = broadcast_from_megatron_pp(gate_wg)
        gate_wg = gate_wg.T.contiguous().float().clone()  # clone to avoid offload by trainer

        gate_wg_ema = state_dict.pop(f'transformer.h.layers.{layer_index}.mlp.moe.gate.wg_ema', None)
        gate_wg_ema = broadcast_from_megatron_pp(gate_wg_ema)
        gate_wg_ema = gate_wg_ema.T.contiguous().float().clone()  # clone to avoid offload by trainer

        gate_wg = (gate_wg + gate_wg_ema) * 0.5

        assert gate_wg.shape == gate_w.shape
        gate_w.data = gate_wg.contiguous()

        # remove useless ce_ema
        state_dict.pop(f'transformer.h.layers.{layer_index}.mlp.moe.gate.ce_ema', None)

        # use_grouped_gemm_weight = getattr(model_config, '_moe_implementation', 'eager') == 'fused'
        # assert model_config._moe_implementation == fused
        # assert use_grouped_gemm_weight
        fc1_1_weight = state_dict.pop(f'transformer.h.layers.{layer_index}.mlp.moe.experts.fc1_1', None)
        fc1_1_weight = broadcast_from_megatron_pp(fc1_1_weight)
        fc1_1_weight = allgather_from_megatron_tp(fc1_1_weight.to(torch.bfloat16), dim=0)

        if device_mesh is not None:
            fc1_1_weight = DTensor.from_local(fc1_1_weight,
                                              device_mesh=device_mesh,
                                              placements=[Replicate(), Replicate()])
            fc1_1_weight = fc1_1_weight.redistribute(device_mesh=device_mesh,
                                                     placements=[Replicate(),
                                                                 Shard(0 if tp_model.use_ep else 1)])._local_tensor

        fc1_2_weight = state_dict.pop(f'transformer.h.layers.{layer_index}.mlp.moe.experts.fc1_2', None)
        fc1_2_weight = broadcast_from_megatron_pp(fc1_2_weight)
        fc1_2_weight = allgather_from_megatron_tp(fc1_2_weight.to(torch.bfloat16), dim=0)

        if device_mesh is not None:
            fc1_2_weight = DTensor.from_local(fc1_2_weight,
                                              device_mesh=device_mesh,
                                              placements=[Replicate(), Replicate()])
            fc1_2_weight = fc1_2_weight.redistribute(device_mesh=device_mesh,
                                                     placements=[Replicate(),
                                                                 Shard(0 if tp_model.use_ep else 1)])._local_tensor

        base_fc1_weight = torch.cat((fc1_1_weight, fc1_2_weight), dim=1)
        del fc1_1_weight
        del fc1_2_weight

        share_fc1_1_weight = state_dict.pop(f'transformer.h.layers.{layer_index}.mlp.moe.pr_expert.fc1_1', None)
        share_fc1_1_weight = broadcast_from_megatron_pp(share_fc1_1_weight)
        share_fc1_1_weight = allgather_from_megatron_tp(share_fc1_1_weight.to(torch.bfloat16), dim=0)
        share_fc1_1_weight = share_fc1_1_weight.reshape(2, -1, share_fc1_1_weight.shape[-1])  ##
        if device_mesh is not None:
            share_fc1_1_weight = DTensor.from_local(share_fc1_1_weight,
                                                    device_mesh=device_mesh,
                                                    placements=[Replicate(), Replicate()])
            share_fc1_1_weight = share_fc1_1_weight.redistribute(device_mesh=device_mesh,
                                                                 placements=[Replicate(), Shard(1)])._local_tensor

        share_fc1_2_weight = state_dict.pop(f'transformer.h.layers.{layer_index}.mlp.moe.pr_expert.fc1_2', None)
        share_fc1_2_weight = broadcast_from_megatron_pp(share_fc1_2_weight)
        share_fc1_2_weight = allgather_from_megatron_tp(share_fc1_2_weight.to(torch.bfloat16), dim=0)
        share_fc1_2_weight = share_fc1_2_weight.reshape(2, -1, share_fc1_2_weight.shape[-1])

        if device_mesh is not None:
            share_fc1_2_weight = DTensor.from_local(share_fc1_2_weight,
                                                    device_mesh=device_mesh,
                                                    placements=[Replicate(), Replicate()])
            share_fc1_2_weight = share_fc1_2_weight.redistribute(device_mesh=device_mesh,
                                                                 placements=[Replicate(), Shard(1)])._local_tensor
        share_fc1_weight = torch.cat((share_fc1_1_weight, share_fc1_2_weight), dim=1)
        del share_fc1_1_weight
        del share_fc1_2_weight

        if tp_model.use_ep:
            fc1_weight = base_fc1_weight.contiguous().flatten()
            s_fc1_weight = share_fc1_weight.reshape(share_fc1_weight.shape[0], 2, -1,
                                                    share_fc1_weight.shape[-1]).transpose(0, 1).reshape(
                                                        -1, share_fc1_weight.shape[-1])
        else:
            fc1_weight = torch.cat((base_fc1_weight, share_fc1_weight), dim=0).contiguous().flatten()
        del base_fc1_weight
        del share_fc1_weight

        assert fc1_weight.shape == fc1_w.shape, f'{fc1_weight.shape=}, {fc1_w.shape=}'
        fc1_w.data = fc1_weight.contiguous()
        if tp_model.use_ep:
            assert s_fc1_weight.shape == share_fc1_w.shape, f'{s_fc1_weight.shape=}, {share_fc1_w.shape=}'
            share_fc1_w.data = s_fc1_weight.contiguous()

        fc2_weight = state_dict.pop(f'transformer.h.layers.{layer_index}.mlp.moe.experts.fc2', None)
        fc2_weight = broadcast_from_megatron_pp(fc2_weight)
        fc2_weight = allgather_from_megatron_tp(fc2_weight.to(torch.bfloat16), dim=0)
        if device_mesh is not None:
            fc2_weight = DTensor.from_local(fc2_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            fc2_weight = fc2_weight.redistribute(device_mesh=device_mesh,
                                                 placements=[Replicate(),
                                                             Shard(0 if tp_model.use_ep else 2)])._local_tensor

        share_fc2_weight = state_dict.pop(f'transformer.h.layers.{layer_index}.mlp.moe.pr_expert.fc2', None)
        share_fc2_weight = broadcast_from_megatron_pp(share_fc2_weight)
        share_fc2_weight = allgather_from_megatron_tp(share_fc2_weight.to(torch.bfloat16), dim=1)
        share_fc2_weight = share_fc2_weight.reshape(share_fc2_weight.shape[-2], 2, -1).transpose(0, 1)
        if device_mesh is not None:

            share_fc2_weight = DTensor.from_local(share_fc2_weight,
                                                  device_mesh=device_mesh,
                                                  placements=[Replicate(), Replicate()])
            share_fc2_weight = share_fc2_weight.redistribute(device_mesh=device_mesh,
                                                             placements=[Replicate(), Shard(2)])._local_tensor

        if tp_model.use_ep:
            fc2_weight_merge = fc2_weight.contiguous().flatten()
            s_fc2_weight_merge = share_fc2_weight.transpose(0, 1).reshape(share_fc2_weight.shape[1], -1)
        else:
            fc2_weight_merge = torch.cat((fc2_weight, share_fc2_weight), dim=0).contiguous().flatten()
        del fc2_weight
        del share_fc2_weight

        # (num_experts, intermediate_size // tp, hidden_size)
        assert fc2_weight_merge.shape == fc2_w.shape, f'{fc2_weight_merge.shape=}, {fc2_w.shape=}'
        fc2_w.data = fc2_weight_merge.contiguous()
        if tp_model.use_ep:
            assert s_fc2_weight_merge.shape == share_fc2_w.shape, f'{s_fc2_weight_merge.shape=}, {share_fc2_w.shape=}'
            share_fc2_w.data = s_fc2_weight_merge.contiguous()

    # enforce check nan
    assert_not_nan(tp_model.layernorm_weight.data)
    assert_not_nan(tp_model.wte_weight.data)
    assert_not_nan(tp_model.lm_head_weight.data)

    for layer_index, weights in enumerate(tp_model.layers_weight):
        for i, weight in enumerate(weights):
            if isinstance(weight, torch.Tensor):
                assert_not_nan(weight)

    load_to_cuda(tp_model=tp_model)
    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_m8_vision(tp_model, state_dict, device_mesh: DeviceMesh, model_config):
    from functools import reduce

    def get_attr(obj, attr_path):
        return reduce(getattr, attr_path.split('.'), obj)

    def assign_data(param_in_tp_model, key):
        if isinstance(key, str):
            param_in_state_dict = state_dict.pop(key).to(torch.bfloat16).full_tensor()
        else:
            param_in_state_dict = key
        assert param_in_tp_model.shape == param_in_state_dict.shape, f'{key=}, {param_in_tp_model.shape=}, {param_in_state_dict.shape=}'
        param_in_tp_model.data = param_in_state_dict.contiguous()
        assert_not_nan(param_in_tp_model.data)

    for layer_index, layer in enumerate(tp_model.visual_encoder.module.custom_decoder.layers):
        prefix = f'vision_encoder.blocks.{layer_index}.'
        assign_data(layer.norm1.weight, prefix + 'norm1.weight')
        assign_data(layer.norm1.bias, prefix + 'norm1.bias')
        assign_data(layer.norm2.weight, prefix + 'norm2.weight')
        assign_data(layer.norm2.bias, prefix + 'norm2.bias')
        assign_data(layer.mlp.fc1.weight, prefix + 'mlp.fc1.weight')
        assign_data(layer.mlp.fc1.bias, prefix + 'mlp.fc1.bias')
        assign_data(layer.mlp.fc2.weight, prefix + 'mlp.fc2.weight')
        assign_data(layer.mlp.fc2.bias, prefix + 'mlp.fc2.bias')
        assign_data(layer.attn.proj.weight, prefix + 'attn.proj.weight')
        assign_data(layer.attn.proj.bias, prefix + 'attn.proj.bias')
        assign_data(layer.attn.qkv.weight, prefix + 'attn.qkv.weight')
        assign_data(layer.attn.q_bias, prefix + 'attn.q_bias')
        assign_data(layer.attn.v_bias, prefix + 'attn.v_bias')

    assign_data(tp_model.visual_encoder.module.custom_decoder.patch_embed.proj.weight,
                'vision_encoder.patch_embed.proj.weight')
    assign_data(tp_model.visual_encoder.module.custom_decoder.patch_embed.proj.bias,
                'vision_encoder.patch_embed.proj.bias')

    def parameter_pack(t):
        return torch.nn.Parameter(t, requires_grad=False)

    assign_data(tp_model.visual_encoder.module.weights['ln_vision'][0], 'ln_vision.weight')
    assign_data(tp_model.visual_encoder.module.weights['ln_vision'][1], 'ln_vision.bias')
    tp_model.ln_vision.weight = parameter_pack(tp_model.visual_encoder.module.weights['ln_vision'][0])
    tp_model.ln_vision.bias = parameter_pack(tp_model.visual_encoder.module.weights['ln_vision'][1])
    assign_data(tp_model.visual_encoder.module.weights['seed_proj'][0], 'multi_modal_projector.0.weight')
    assign_data(tp_model.visual_encoder.module.weights['seed_proj'][1], 'multi_modal_projector.0.bias')
    assign_data(tp_model.visual_encoder.module.weights['seed_proj'][2], 'multi_modal_projector.2.weight')
    assign_data(tp_model.visual_encoder.module.weights['seed_proj'][3], 'multi_modal_projector.2.bias')

    tp_model.seed_proj[0].weight = parameter_pack(tp_model.visual_encoder.module.weights['seed_proj'][0])
    tp_model.seed_proj[0].bias = parameter_pack(tp_model.visual_encoder.module.weights['seed_proj'][1])
    tp_model.seed_proj[2].weight = parameter_pack(tp_model.visual_encoder.module.weights['seed_proj'][2])
    tp_model.seed_proj[2].bias = parameter_pack(tp_model.visual_encoder.module.weights['seed_proj'][3])
    # load_to_cuda(tp_model=tp_model)
    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_m8_vision_tp(tp_model, state_dict, device_mesh: DeviceMesh, model_config):
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
    else:
        tp_size = 1
        tp_rank = 0

    if model_config.embed_dim < 2048:  # disable small vit tp
        tp_size = 1
        tp_rank = 0
        device_mesh = None

    def assign_data(param_in_tp_model, param_in_state_dict):
        assert param_in_tp_model.shape == param_in_state_dict.shape, f"shape mismatch {param_in_tp_model.shape} vs {param_in_state_dict.shape}"
        param_in_tp_model.data = param_in_state_dict.contiguous()

    ffn_intermediate_dim = (int(model_config.embed_dim * model_config.mlp_ratio) + 63) // 64 * 64
    padding_size = ffn_intermediate_dim - int(model_config.embed_dim * model_config.mlp_ratio)
    for layer_index, (norm1_wb, qkv_w, qkv_b, out_w, out_b, norm2_wb, fc1_w, fc1_b, fc2_w, fc2_b,
                      *_) in enumerate(tp_model.visual_encoder.module.layers_weight):
        prefix = f'vision_encoder.blocks.{layer_index}.'
        norm1_weight = state_dict.pop(prefix + 'norm1.weight').full_tensor()
        norm1_bias = state_dict.pop(prefix + 'norm1.bias').full_tensor()
        state_norm1_wb = torch.stack((norm1_weight, norm1_bias), dim=0).to(torch.bfloat16)
        assign_data(norm1_wb, state_norm1_wb)
        assert_not_nan(norm1_wb.data)
        # del norm1_weight
        # del norm1_bias

        norm2_weight = state_dict.pop(prefix + 'norm2.weight').full_tensor()
        norm2_bias = state_dict.pop(prefix + 'norm2.bias').full_tensor()
        state_norm2_wb = torch.stack((norm2_weight, norm2_bias), dim=0).to(torch.bfloat16)
        assign_data(norm2_wb, state_norm2_wb)
        assert_not_nan(norm2_wb.data)
        # del norm2_weight
        # del norm2_bias

        fc1_weight = state_dict.pop(prefix + 'mlp.fc1.weight').to(torch.bfloat16).full_tensor()

        if device_mesh is not None:
            fc1_weight = DTensor.from_local(fc1_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            fc1_weight = fc1_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                      Shard(0)])._local_tensor
        if padding_size > 0:
            fc1_weight = torch.nn.functional.pad(fc1_weight, (0, 0, 0, padding_size))
        assign_data(fc1_w, fc1_weight)

        fc1_bias = state_dict.pop(prefix + 'mlp.fc1.bias').to(torch.bfloat16).full_tensor()

        if device_mesh is not None:
            fc1_bias = DTensor.from_local(fc1_bias, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            fc1_bias = fc1_bias.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(0)])._local_tensor
        if padding_size > 0:
            fc1_bias = torch.nn.functional.pad(fc1_bias, (0, padding_size))
        assign_data(fc1_b, fc1_bias)

        fc2_weight = state_dict.pop(prefix + 'mlp.fc2.weight').to(torch.bfloat16).full_tensor()

        if device_mesh is not None:
            fc2_weight = DTensor.from_local(fc2_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            fc2_weight = fc2_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                      Shard(1)])._local_tensor
        if padding_size > 0:
            fc2_weight = torch.nn.functional.pad(fc2_weight, (0, padding_size))
        assign_data(fc2_w, fc2_weight)

        fc2_bias = state_dict.pop(prefix + 'mlp.fc2.bias').to(torch.bfloat16).full_tensor()
        assign_data(fc2_b, fc2_bias)

        attn_proj_weight = state_dict.pop(prefix + 'attn.proj.weight').to(torch.bfloat16).full_tensor()
        if device_mesh is not None:
            attn_proj_weight = DTensor.from_local(attn_proj_weight,
                                                  device_mesh=device_mesh,
                                                  placements=[Replicate(), Replicate()])
            attn_proj_weight = attn_proj_weight.redistribute(device_mesh=device_mesh,
                                                             placements=[Replicate(), Shard(1)])._local_tensor
        assign_data(out_w, attn_proj_weight)

        attn_proj_bias = state_dict.pop(prefix + 'attn.proj.bias').to(torch.bfloat16).full_tensor()
        assign_data(out_b, attn_proj_bias)

        def qkv_split(src, dim, tp_size):
            src_split = torch.split(src.data, src.shape[dim] // 3, dim=dim)
            qkv_split = [torch.split(src_s, src_s.shape[dim] // tp_size, dim=dim) for src_s in src_split]
            qkv_cat = [torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=dim) for i in range(len(qkv_split[0]))]
            return qkv_cat

        qkv_weight = state_dict.pop(prefix + 'attn.qkv.weight').full_tensor().to(torch.bfloat16)
        qkv_weight = qkv_split(qkv_weight, 0, tp_size)[tp_rank]
        assign_data(qkv_w, qkv_weight)

        q_bias = state_dict.pop(prefix + 'attn.q_bias').full_tensor().to(torch.bfloat16)
        v_bias = state_dict.pop(prefix + 'attn.v_bias').full_tensor().to(torch.bfloat16)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
        qkv_bias = qkv_split(qkv_bias, 0, tp_size)[tp_rank]
        assign_data(qkv_b, qkv_bias)

    patch_emb_proj_weight = state_dict.pop('vision_encoder.patch_embed.proj.weight').to(torch.bfloat16).full_tensor()
    assign_data(tp_model.visual_encoder.module.patch_embed.proj.weight, patch_emb_proj_weight)
    patch_emb_proj_bias = state_dict.pop('vision_encoder.patch_embed.proj.bias').to(torch.bfloat16).full_tensor()
    assign_data(tp_model.visual_encoder.module.patch_embed.proj.bias, patch_emb_proj_bias)

    def parameter_pack(t):
        return torch.nn.Parameter(t, requires_grad=False)

    ln_vision_weight = state_dict.pop('ln_vision.weight').to(torch.bfloat16).full_tensor()
    assign_data(tp_model.visual_encoder.module.weights['ln_vision'][0], ln_vision_weight)
    ln_vision_bias = state_dict.pop('ln_vision.bias').to(torch.bfloat16).full_tensor()
    assign_data(tp_model.visual_encoder.module.weights['ln_vision'][1], ln_vision_bias)
    tp_model.ln_vision.weight = parameter_pack(tp_model.visual_encoder.module.weights['ln_vision'][0])
    tp_model.ln_vision.bias = parameter_pack(tp_model.visual_encoder.module.weights['ln_vision'][1])

    seed_proj0_weight = state_dict.pop('multi_modal_projector.0.weight').to(torch.bfloat16).full_tensor()
    assign_data(tp_model.visual_encoder.module.weights['seed_proj'][0], seed_proj0_weight)
    seed_proj0_bias = state_dict.pop('multi_modal_projector.0.bias').to(torch.bfloat16).full_tensor()
    assign_data(tp_model.visual_encoder.module.weights['seed_proj'][1], seed_proj0_bias)

    seed_proj2_weight = state_dict.pop('multi_modal_projector.2.weight').to(torch.bfloat16).full_tensor()
    assign_data(tp_model.visual_encoder.module.weights['seed_proj'][2], seed_proj2_weight)
    seed_proj2_bias = state_dict.pop('multi_modal_projector.2.bias').to(torch.bfloat16).full_tensor()
    assign_data(tp_model.visual_encoder.module.weights['seed_proj'][3], seed_proj2_bias)

    tp_model.seed_proj[0].weight = parameter_pack(tp_model.visual_encoder.module.weights['seed_proj'][0])
    tp_model.seed_proj[0].bias = parameter_pack(tp_model.visual_encoder.module.weights['seed_proj'][1])
    tp_model.seed_proj[2].weight = parameter_pack(tp_model.visual_encoder.module.weights['seed_proj'][2])
    tp_model.seed_proj[2].bias = parameter_pack(tp_model.visual_encoder.module.weights['seed_proj'][3])
    # seed_proj[0] -> 'multi_modal_projector.0'
    load_to_cuda(tp_model=tp_model)
    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_p6_vision(tp_model, state_dict, device_mesh: DeviceMesh, model_config):
    raise NotImplementedError


def _reshard_fsdp_state_dict_to_xperf_vl(tp_model, vit_tp_model, state_dict, device_mesh: DeviceMesh, model_config):
    if model_config.text_config.architectures[0] == "M8ForCausalLM":
        _reshard_fsdp_state_dict_to_xperf_m8(tp_model,
                                             state_dict,
                                             device_mesh,
                                             model_config.text_config,
                                             prefix="language_model.")
        if hasattr(vit_tp_model.visual_encoder.module, "layers_weight"):
            _reshard_fsdp_state_dict_to_xperf_m8_vision_tp(vit_tp_model, state_dict, device_mesh,
                                                           model_config.vision_config)
        else:
            _reshard_fsdp_state_dict_to_xperf_m8_vision(vit_tp_model, state_dict, device_mesh,
                                                        model_config.vision_config)
    elif model_config.text_config.architectures[0] == "P6ForCausalLM":
        _reshard_fsdp_state_dict_to_xperf_p6(tp_model,
                                             state_dict,
                                             device_mesh,
                                             model_config.text_config,
                                             prefix="language_model.")
        _reshard_fsdp_state_dict_to_xperf_p6_vision(vit_tp_model, state_dict, device_mesh, model_config.vision_config)
    else:
        raise NotImplementedError(f"{model_config.text_config.architectures[0]} is not supported")


def _reshard_fsdp_state_dict_to_xperf_deepseek_v3(tp_model, state_dict, device_mesh: DeviceMesh, model_config):
    """
    Reshard the state dict of FSDP model to XPerf DeepSeek V3 model.
    """
    from seed_models import DeepseekV3Config
    assert isinstance(model_config, DeepseekV3Config)

    # checking
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
        assert model_config.num_key_value_heads % tp_size == 0 or tp_size % model_config.num_key_value_heads == 0
    else:
        tp_size = 1
        tp_rank = 0

    # assert tp_size == model_config.num_key_value_heads or tp_size == 1

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    num_kv_heads = model_config.num_key_value_heads

    ln_f_weight = state_dict.pop('model.norm.weight').full_tensor().to(torch.bfloat16)
    ln_f_weight = ln_f_weight.reshape(1, ln_f_weight.shape[-1])
    tp_model.layernorm_weight.data = ln_f_weight.contiguous()

    # TODO: use xperf vocab_tp
    wte: DTensor = state_dict.pop('model.embed_tokens.weight').to(torch.bfloat16)
    lm_head: DTensor = state_dict.pop('lm_head.weight').to(torch.bfloat16)

    wte_weight = wte.full_tensor()
    lm_head = lm_head.full_tensor()
    if device_mesh is not None and tp_model.wte_weight.data.shape != wte_weight.shape:
        # TODO: we may need to do full first
        wte_weight = DTensor.from_local(wte_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
        wte_weight_tp = wte_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                     Shard(1)])._local_tensor
        lm_head = DTensor.from_local(lm_head, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
        lm_head_tp = lm_head.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(0)])._local_tensor
        # del wte
    else:
        wte_weight_tp = wte_weight
        lm_head_tp = lm_head

    assert wte_weight_tp.shape == tp_model.wte_weight.data.shape
    assert lm_head_tp.shape == tp_model.lm_head_weight.data.shape

    tp_model.wte_weight.data = wte_weight_tp.contiguous()
    tp_model.lm_head_weight.data = lm_head_tp.contiguous()  # untied weights
    '''  TensorView inference_tp_ep(...
                             const TensorView& norm0_gamma, const TensorView& query_gamma, const TensorView& key_gamma,
                             const TensorView& q_a_proj_weight, const TensorView& q_b_proj_weight,
                             const TensorView& kv_a_proj_weight, const TensorView& k_b_proj_weight, 
                             const TensorView& v_b_proj_weight, const TensorView& attention_proj_weight, 
                             const TensorView& norm1_gamma, const TensorView& moe_gate_weight, 
                             const TensorView& moe_gate_bias, const TensorView& FFN0_weight, const TensorView& FFN1_weight, 
                             const TensorView& share_expert_FFN0_weight, const TensorView& share_expert_FFN1_weight, '''

    for layer_index, (norm0_gamma, query_gamma, key_gamma, q_a_proj_weight, q_b_proj_weight, kv_a_proj_weight,
                      k_b_proj_weight, v_b_proj_weight, attention_proj_weight, norm1_gamma, moe_gate_weight,
                      moe_gate_bias, FFN0_weight, FFN1_weight, share_expert_FFN0_weight, share_expert_FFN1_weight,
                      *_) in enumerate(tp_model.layers_weight):
        ln_1_weight = state_dict.pop(f'model.layers.{layer_index}.input_layernorm.weight').full_tensor()
        ln_1_weight = torch.stack((ln_1_weight,), dim=0).to(torch.bfloat16).reshape(1, ln_1_weight.shape[-1])
        assert norm0_gamma.data.shape == ln_1_weight.shape
        norm0_gamma.data = ln_1_weight.cpu().contiguous()

        query_ln_weight = state_dict[f'model.layers.{layer_index}.self_attn.q_a_layernorm.weight'].full_tensor()
        query_ln_weight = torch.stack((query_ln_weight,),
                                      dim=0).to(torch.bfloat16).reshape(1, query_ln_weight.shape[-1])
        assert query_gamma.data.shape == query_ln_weight.shape
        query_gamma.data = query_ln_weight.cpu().contiguous()

        kv_ln_weight = state_dict[f'model.layers.{layer_index}.self_attn.kv_a_layernorm.weight'].full_tensor()
        kv_ln_weight = torch.stack((kv_ln_weight,), dim=0).to(torch.bfloat16).reshape(1, kv_ln_weight.shape[-1])
        assert key_gamma.data.shape == kv_ln_weight.shape
        key_gamma.data = kv_ln_weight.cpu().contiguous()

        # query
        q_a_proj_weight_src = state_dict.pop(f'model.layers.{layer_index}.self_attn.q_a_proj.weight').full_tensor().to(
            torch.bfloat16)
        q_b_proj_weight_src = state_dict.pop(f'model.layers.{layer_index}.self_attn.q_b_proj.weight').full_tensor().to(
            torch.bfloat16)

        # shard qkv and concat
        if device_mesh is not None:
            q_b_proj_weight_src = DTensor.from_local(q_b_proj_weight_src,
                                                     device_mesh=device_mesh,
                                                     placements=[Replicate(), Replicate()])
            q_b_proj_weight_src = q_b_proj_weight_src.redistribute(device_mesh=device_mesh,
                                                                   placements=[Replicate(), Shard(0)])._local_tensor

        # breakpoint()
        assert q_a_proj_weight.data.shape == q_a_proj_weight_src.shape
        q_a_proj_weight.data = q_a_proj_weight_src.cpu().contiguous()

        assert q_b_proj_weight.data.shape == q_b_proj_weight_src.shape
        q_b_proj_weight.data = q_b_proj_weight_src.cpu().contiguous()

        # key and value
        kv_a_proj_weight_src = state_dict.pop(
            f'model.layers.{layer_index}.self_attn.kv_a_proj_with_mqa.weight').full_tensor().to(torch.bfloat16)
        assert kv_a_proj_weight.data.shape == kv_a_proj_weight_src.shape
        kv_a_proj_weight.data = kv_a_proj_weight_src.cpu().contiguous()

        kv_b_proj_weight_src = state_dict.pop(
            f'model.layers.{layer_index}.self_attn.kv_b_proj.weight').full_tensor().to(torch.bfloat16)
        w_kv_b = kv_b_proj_weight_src.reshape(model_config.num_attention_heads,
                                              model_config.qk_nope_head_dim + model_config.v_head_dim, -1)
        w_k_b, w_v_b = torch.split(w_kv_b, [model_config.qk_nope_head_dim, model_config.v_head_dim], dim=1)
        w_k_b = w_k_b.reshape(model_config.num_attention_heads, model_config.qk_nope_head_dim, -1)
        w_v_b = w_v_b.reshape(model_config.num_attention_heads, model_config.v_head_dim, -1)

        if device_mesh is not None:
            w_k_b = DTensor.from_local(w_k_b, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            w_k_b = w_k_b.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(0)])._local_tensor

            w_v_b = DTensor.from_local(w_v_b, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            w_v_b = w_v_b.redistribute(device_mesh=device_mesh, placements=[Replicate(), Shard(0)])._local_tensor

        w_k_b = w_k_b.reshape(-1, w_k_b.shape[-1])
        w_v_b = w_v_b.reshape(-1, w_v_b.shape[-1])

        assert k_b_proj_weight.data.shape == w_k_b.shape
        assert v_b_proj_weight.data.shape == w_v_b.shape

        k_b_proj_weight.data = w_k_b.cpu().contiguous()
        v_b_proj_weight.data = w_v_b.cpu().contiguous()

        o_proj_weight_src = state_dict.pop(f'model.layers.{layer_index}.self_attn.o_proj.weight').full_tensor().to(
            torch.bfloat16)

        if device_mesh is not None:
            o_proj_weight_src = DTensor.from_local(o_proj_weight_src,
                                                   device_mesh=device_mesh,
                                                   placements=[Replicate(), Replicate()])
            o_proj_weight_src = o_proj_weight_src.redistribute(device_mesh=device_mesh,
                                                               placements=[Replicate(), Shard(1)])._local_tensor

        o_proj_weight_src = o_proj_weight_src.cpu().contiguous().view(hidden_size, -1).cpu().contiguous()

        assert o_proj_weight_src.shape == attention_proj_weight.shape
        attention_proj_weight.data = o_proj_weight_src.cpu().contiguous()

        ln_2_weight = state_dict.pop(f'model.layers.{layer_index}.post_attention_layernorm.weight').full_tensor().to(
            torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight,), dim=0).reshape(1, ln_2_weight.shape[-1])
        assert ln_2_weight.shape == norm1_gamma.shape
        norm1_gamma.data = ln_2_weight.cpu().contiguous()

        if layer_index < model_config.first_k_dense_replace:
            gate_proj_src = state_dict.pop(f'model.layers.{layer_index}.mlp.gate_proj.weight').to(
                torch.bfloat16).full_tensor()
            up_proj_src = state_dict.pop(f'model.layers.{layer_index}.mlp.up_proj.weight').to(
                torch.bfloat16).full_tensor()
            down_proj_src = state_dict.pop(f'model.layers.{layer_index}.mlp.down_proj.weight').to(
                torch.bfloat16).full_tensor()

            if device_mesh is not None:
                gate_proj_src = DTensor.from_local(gate_proj_src,
                                                   device_mesh=device_mesh,
                                                   placements=[Replicate(), Replicate()])
                gate_proj_src = gate_proj_src.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                                Shard(0)])._local_tensor

                up_proj_src = DTensor.from_local(up_proj_src,
                                                 device_mesh=device_mesh,
                                                 placements=[Replicate(), Replicate()])
                up_proj_src = up_proj_src.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(0)])._local_tensor

                down_proj_src = DTensor.from_local(down_proj_src,
                                                   device_mesh=device_mesh,
                                                   placements=[Replicate(), Replicate()])
                down_proj_src = down_proj_src.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                                Shard(1)])._local_tensor

            FFN0_weight_src = torch.cat((gate_proj_src, up_proj_src), dim=0)
            del gate_proj_src
            del up_proj_src
            assert FFN0_weight_src.shape == FFN0_weight.shape
            FFN0_weight.data = FFN0_weight_src.cpu().contiguous()

            assert down_proj_src.shape == FFN1_weight.shape
            FFN1_weight.data = down_proj_src.cpu().contiguous()

        else:
            gate_w_src = state_dict.pop(f'model.layers.{layer_index}.mlp.gate.weight').to(torch.bfloat16).full_tensor()
            assert moe_gate_weight.data.shape == gate_w_src.shape
            moe_gate_weight.data = gate_w_src.cpu().contiguous()

            gate_score_src = state_dict.pop(f'model.layers.{layer_index}.mlp.gate.e_score_correction_bias').to(
                torch.bfloat16)
            assert moe_gate_bias.data.shape == gate_score_src.shape
            moe_gate_bias.data = gate_score_src.cpu().contiguous()

            fc1_1_src = state_dict.pop(f'model.layers.{layer_index}.mlp.fc1_1').to(torch.bfloat16).full_tensor()
            fc1_2_src = state_dict.pop(f'model.layers.{layer_index}.mlp.fc1_2').to(torch.bfloat16).full_tensor()
            fc2_src = state_dict.pop(f'model.layers.{layer_index}.mlp.fc2').to(torch.bfloat16).full_tensor()
            fc_1_src = torch.cat((fc1_1_src, fc1_2_src), dim=1)

            if device_mesh is not None:
                fc_1_src = DTensor.from_local(fc_1_src, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
                fc_1_src = fc_1_src.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                      Shard(0)])._local_tensor
                fc2_src = DTensor.from_local(fc2_src, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
                fc2_src = fc2_src.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                    Shard(0)])._local_tensor

            assert fc_1_src.shape == FFN0_weight.data.shape
            FFN0_weight.data = fc_1_src.cpu().contiguous()

            assert fc2_src.shape == FFN1_weight.data.shape
            FFN1_weight.data = fc2_src.cpu().contiguous()

            shared_gate_weight_src = state_dict.pop(
                f'model.layers.{layer_index}.mlp.shared_experts.gate_proj.weight').to(torch.bfloat16).full_tensor()

            shared_up_weight_src = state_dict.pop(f'model.layers.{layer_index}.mlp.shared_experts.up_proj.weight').to(
                torch.bfloat16).full_tensor()

            shared_down_weight_src = state_dict.pop(
                f'model.layers.{layer_index}.mlp.shared_experts.down_proj.weight').to(torch.bfloat16).full_tensor()

            if device_mesh is not None:
                shared_gate_weight_src = DTensor.from_local(shared_gate_weight_src,
                                                            device_mesh=device_mesh,
                                                            placements=[Replicate(), Replicate()])
                shared_gate_weight_src = shared_gate_weight_src.redistribute(device_mesh=device_mesh,
                                                                             placements=[Replicate(),
                                                                                         Shard(0)])._local_tensor

                shared_up_weight_src = DTensor.from_local(shared_up_weight_src,
                                                          device_mesh=device_mesh,
                                                          placements=[Replicate(), Replicate()])
                shared_up_weight_src = shared_up_weight_src.redistribute(device_mesh=device_mesh,
                                                                         placements=[Replicate(),
                                                                                     Shard(0)])._local_tensor

                shared_down_weight_src = DTensor.from_local(shared_down_weight_src,
                                                            device_mesh=device_mesh,
                                                            placements=[Replicate(), Replicate()])
                shared_down_weight_src = shared_down_weight_src.redistribute(device_mesh=device_mesh,
                                                                             placements=[Replicate(),
                                                                                         Shard(1)])._local_tensor

            shared_FFN0_weight_src = torch.cat((shared_gate_weight_src, shared_up_weight_src), dim=0)
            del shared_gate_weight_src
            del shared_up_weight_src

            assert share_expert_FFN0_weight.shape == shared_FFN0_weight_src.shape
            share_expert_FFN0_weight.data = shared_FFN0_weight_src.cpu().contiguous()

            assert share_expert_FFN1_weight.shape == shared_down_weight_src.shape
            share_expert_FFN1_weight.data = shared_down_weight_src.cpu().contiguous()

    # enforce check nan
    assert_not_nan(tp_model.layernorm_weight.data)
    assert_not_nan(tp_model.wte_weight.data)
    assert_not_nan(tp_model.lm_head_weight.data)

    for layer_index, weights in enumerate(tp_model.layers_weight):
        for weight in weights:
            if isinstance(weight, torch.Tensor):
                assert_not_nan(weight)

    load_to_cuda(tp_model=tp_model)
    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_m10(tp_model, state_dict, device_mesh: DeviceMesh, model_config, prefix=''):
    from seed_models import M10Config
    assert isinstance(model_config, M10Config)

    # checking
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
        # assert tp_size <= model_config.num_key_value_heads
        assert model_config.num_key_value_heads % tp_size == 0 or tp_size % model_config.num_key_value_heads == 0
    else:
        tp_size = 1
        tp_rank = 0

    # assert tp_size == model_config.num_key_value_heads or tp_size == 1

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    num_kv_heads = model_config.num_key_value_heads
    if model_config.mtp_mode == "recurrent":
        ln_f_weight = state_dict.pop('transformer.model.mtp_ce_norms.0.head_ln.weight').full_tensor().to(torch.bfloat16)
    else:
        ln_f_weight = state_dict.pop(prefix + 'transformer.norm.weight').full_tensor().to(torch.bfloat16)
    ln_f_weight = ln_f_weight.reshape(1, ln_f_weight.shape[-1])
    tp_model.layernorm_weight.data = ln_f_weight.contiguous()

    # NOTE: lm_head == wte for m10, tie embedding
    wte: DTensor = state_dict.pop('transformer.embed_tokens.weight').to(torch.bfloat16)
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

    for layer_index, (ln_1, key_norm, context_norm, qkv_w, qkv_b, dense_w, dense_b, ln_2, gate_w, _, q_norm, v_norm,
                      attn_outnorm, ffn_outnorm, fc1_w, _, fc2_w, _, share_fc1_w, share_fc2_w,
                      *_) in enumerate(tp_model.layers_weight):
        # if os.getenv("LOCAL_RANK", "0") == "0":
        #     import ipdb; ipdb.set_trace()
        # else:
        #     import time
        #     time.sleep(1000)
        ln_1_weight = state_dict.pop(prefix +
                                     f'transformer.model.layers.{layer_index}.input_layernorm.weight').full_tensor()
        ln_1_weight = torch.stack((ln_1_weight,), dim=0).to(torch.bfloat16).reshape(1, ln_1_weight.shape[-1])
        assert ln_1.data.shape == ln_1_weight.shape
        ln_1.data = ln_1_weight.contiguous()

        # k norm
        key_norm_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.self_attention.k_norm.weight').full_tensor()
        key_norm_weight = torch.stack((key_norm_weight,),
                                      dim=0).to(torch.bfloat16).reshape(1, key_norm_weight.shape[-1])
        assert key_norm.data.shape == key_norm_weight.shape
        key_norm.data = key_norm_weight.contiguous()

        # '''q norm'''
        query_norm_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.self_attention.q_norm.weight').full_tensor()
        query_norm_weight = torch.stack((query_norm_weight,),
                                        dim=0).to(torch.bfloat16).reshape(1, query_norm_weight.shape[-1])
        assert q_norm.data.shape == query_norm_weight.shape
        q_norm.data = query_norm_weight.contiguous()

        assert v_norm is None
        '''post_attention_layernorm'''
        post_attention_layernorm_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.self_attention.o_norm.weight').full_tensor()
        post_attention_layernorm_weight = torch.stack(
            (post_attention_layernorm_weight,),
            dim=0).to(torch.bfloat16).reshape(1, post_attention_layernorm_weight.shape[-1])
        assert attn_outnorm.data.shape == post_attention_layernorm_weight.shape
        attn_outnorm.data = post_attention_layernorm_weight.contiguous()
        '''ffn_output_layernorm '''
        ffn_output_layernorm_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.ffn_output_layernorm.weight').full_tensor()
        ffn_output_layernorm_weight = torch.stack(
            (ffn_output_layernorm_weight,), dim=0).to(torch.bfloat16).reshape(1, ffn_output_layernorm_weight.shape[-1])
        assert ffn_outnorm.data.shape == ffn_output_layernorm_weight.shape
        ffn_outnorm.data = ffn_output_layernorm_weight.contiguous()

        if context_norm is not None:
            context_norm_weight = state_dict.pop(
                prefix + f'transformer.model.layers.{layer_index}.self_attention.c_norm.weight').full_tensor()
            context_norm_weight = torch.stack((context_norm_weight,),
                                              dim=0).to(torch.bfloat16).reshape(1, context_norm_weight.shape[-1])
            assert context_norm.data.shape == context_norm_weight.shape
            context_norm.data = context_norm_weight.contiguous()

        q_proj_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.self_attention.q_proj.weight').full_tensor().to(
                torch.bfloat16)
        q_proj_weight = q_proj_weight.view(num_kv_heads, -1, head_dim, hidden_size).transpose(0, 1).contiguous()
        k_proj_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.self_attention.k_proj.weight').full_tensor().to(
                torch.bfloat16)
        k_proj_weight = k_proj_weight.view(1, num_kv_heads, head_dim, hidden_size)
        v_proj_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.self_attention.v_proj.weight').full_tensor().to(
                torch.bfloat16)
        v_proj_weight = v_proj_weight.view(1, num_kv_heads, head_dim, hidden_size)

        kv_replicate = tp_size // model_config.num_key_value_heads
        # duplicate kv
        if kv_replicate > 1:
            q_proj_weight = q_proj_weight.view(-1, tp_size, head_dim, hidden_size).contiguous()
            k_proj_weight = torch.tile(k_proj_weight, (1, kv_replicate, 1, 1))
            v_proj_weight = torch.tile(v_proj_weight, (1, kv_replicate, 1, 1))

        # shard qkv and concat
        if device_mesh is not None:
            q_proj_weight = DTensor.from_local(q_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            q_proj_weight = q_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            k_proj_weight = DTensor.from_local(k_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            k_proj_weight = k_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

            v_proj_weight = DTensor.from_local(v_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            v_proj_weight = v_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(1)])._local_tensor

        qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0).contiguous().view(-1, hidden_size)

        assert qkv_w.data.shape == qkv_weight.shape
        qkv_w.data = qkv_weight.contiguous()

        o_proj_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.self_attention.o_proj.weight').full_tensor().to(
                torch.bfloat16)

        # the XPerfGPT has different ordering
        o_proj_weight = o_proj_weight.view(hidden_size, num_kv_heads, -1, head_dim).transpose(
            1, 2).contiguous()  # (hidden_size, -1, num_kv_heads, head_dim)

        if kv_replicate > 1:
            o_proj_weight = o_proj_weight.view(hidden_size, -1, tp_size, head_dim)

        if device_mesh is not None:
            o_proj_weight = DTensor.from_local(o_proj_weight,
                                               device_mesh=device_mesh,
                                               placements=[Replicate(), Replicate()])
            o_proj_weight = o_proj_weight.redistribute(device_mesh=device_mesh, placements=[Replicate(),
                                                                                            Shard(2)])._local_tensor

        o_proj_weight = o_proj_weight.contiguous().view(hidden_size, -1).contiguous()

        assert o_proj_weight.shape == dense_w.shape
        dense_w.data = o_proj_weight.contiguous()

        ln_2_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.post_attention_layernorm.weight').full_tensor().to(
                torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight,), dim=0).reshape(1, ln_2_weight.shape[-1])
        assert ln_2_weight.shape == ln_2.shape
        ln_2.data = ln_2_weight.contiguous()

        gate_wg = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.mlp.moe.gate.wg').full_tensor().T.contiguous().float()
        gate_wg_ema = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.mlp.moe.gate.wg_ema').T.contiguous().float()

        # remove useless ce_ema
        state_dict.pop(f'transformer.model.layers.{layer_index}.mlp.moe.gate.ce_ema', None)

        gate_wg = (gate_wg + gate_wg_ema) * 0.5

        assert gate_wg.shape == gate_w.shape
        gate_w.data = gate_wg.contiguous()

        # use_grouped_gemm_weight = getattr(model_config, '_moe_implementation', 'eager') == 'fused'
        # assert model_config._moe_implementation == fused
        # assert use_grouped_gemm_weight
        fc1_1_weight = state_dict.pop(prefix + f'transformer.model.layers.{layer_index}.mlp.moe.experts.gate_proj').to(
            torch.bfloat16).full_tensor()

        if device_mesh is not None:
            fc1_1_weight = DTensor.from_local(fc1_1_weight,
                                              device_mesh=device_mesh,
                                              placements=[Replicate(), Replicate()])
            fc1_1_weight = fc1_1_weight.redistribute(device_mesh=device_mesh,
                                                     placements=[Replicate(),
                                                                 Shard(0 if tp_model.use_ep else 1)])._local_tensor

        fc1_2_weight = state_dict.pop(prefix + f'transformer.model.layers.{layer_index}.mlp.moe.experts.up_proj').to(
            torch.bfloat16).full_tensor()

        if device_mesh is not None:
            fc1_2_weight = DTensor.from_local(fc1_2_weight,
                                              device_mesh=device_mesh,
                                              placements=[Replicate(), Replicate()])
            fc1_2_weight = fc1_2_weight.redistribute(device_mesh=device_mesh,
                                                     placements=[Replicate(),
                                                                 Shard(0 if tp_model.use_ep else 1)])._local_tensor

        base_fc1_weight = torch.cat((fc1_1_weight, fc1_2_weight), dim=1)
        del fc1_1_weight
        del fc1_2_weight
        share_fc1_1_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.mlp.moe.shared_experts.gate_proj').to(
                torch.bfloat16).full_tensor()
        share_fc1_1_weight = share_fc1_1_weight.reshape(2, -1, share_fc1_1_weight.shape[-1])  ##
        if device_mesh is not None:
            share_fc1_1_weight = DTensor.from_local(share_fc1_1_weight,
                                                    device_mesh=device_mesh,
                                                    placements=[Replicate(), Replicate()])
            share_fc1_1_weight = share_fc1_1_weight.redistribute(device_mesh=device_mesh,
                                                                 placements=[Replicate(), Shard(1)])._local_tensor

        share_fc1_2_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.mlp.moe.shared_experts.up_proj').to(
                torch.bfloat16).full_tensor()
        share_fc1_2_weight = share_fc1_2_weight.reshape(2, -1, share_fc1_2_weight.shape[-1])

        if device_mesh is not None:
            share_fc1_2_weight = DTensor.from_local(share_fc1_2_weight,
                                                    device_mesh=device_mesh,
                                                    placements=[Replicate(), Replicate()])
            share_fc1_2_weight = share_fc1_2_weight.redistribute(device_mesh=device_mesh,
                                                                 placements=[Replicate(), Shard(1)])._local_tensor
        share_fc1_weight = torch.cat((share_fc1_1_weight, share_fc1_2_weight), dim=1)
        del share_fc1_1_weight
        del share_fc1_2_weight

        if tp_model.use_ep:
            fc1_weight = base_fc1_weight.contiguous().flatten()
            s_fc1_weight = share_fc1_weight.reshape(share_fc1_weight.shape[0], 2, -1,
                                                    share_fc1_weight.shape[-1]).transpose(0, 1).reshape(
                                                        -1, share_fc1_weight.shape[-1])
        else:
            fc1_weight = torch.cat((base_fc1_weight, share_fc1_weight), dim=0).contiguous().flatten()
        del base_fc1_weight
        del share_fc1_weight

        assert fc1_weight.shape == fc1_w.shape, f'{fc1_weight.shape=}, {fc1_w.shape=}'
        fc1_w.data = fc1_weight.contiguous()
        if tp_model.use_ep:
            assert s_fc1_weight.shape == share_fc1_w.shape, f'{s_fc1_weight.shape=}, {share_fc1_w.shape=}'
            share_fc1_w.data = s_fc1_weight.contiguous()

        fc2_weight = state_dict.pop(prefix + f'transformer.model.layers.{layer_index}.mlp.moe.experts.down_proj').to(
            torch.bfloat16).full_tensor()
        if device_mesh is not None:
            fc2_weight = DTensor.from_local(fc2_weight, device_mesh=device_mesh, placements=[Replicate(), Replicate()])
            fc2_weight = fc2_weight.redistribute(device_mesh=device_mesh,
                                                 placements=[Replicate(),
                                                             Shard(0 if tp_model.use_ep else 2)])._local_tensor

        share_fc2_weight = state_dict.pop(
            prefix + f'transformer.model.layers.{layer_index}.mlp.moe.shared_experts.down_proj').to(
                torch.bfloat16).full_tensor()
        share_fc2_weight = share_fc2_weight.reshape(share_fc2_weight.shape[-2], 2, -1).transpose(0, 1)
        if device_mesh is not None:

            share_fc2_weight = DTensor.from_local(share_fc2_weight,
                                                  device_mesh=device_mesh,
                                                  placements=[Replicate(), Replicate()])
            share_fc2_weight = share_fc2_weight.redistribute(device_mesh=device_mesh,
                                                             placements=[Replicate(), Shard(2)])._local_tensor

        if tp_model.use_ep:
            fc2_weight_merge = fc2_weight.contiguous().flatten()
            s_fc2_weight_merge = share_fc2_weight.transpose(0, 1).reshape(share_fc2_weight.shape[1], -1)
        else:
            fc2_weight_merge = torch.cat((fc2_weight, share_fc2_weight), dim=0).contiguous().flatten()
        del fc2_weight
        del share_fc2_weight

        # (num_experts, intermediate_size // tp, hidden_size)
        assert fc2_weight_merge.shape == fc2_w.shape, f'{fc2_weight_merge.shape=}, {fc2_w.shape=}'
        fc2_w.data = fc2_weight_merge.contiguous()
        if tp_model.use_ep:
            assert s_fc2_weight_merge.shape == share_fc2_w.shape, f'{s_fc2_weight_merge.shape=}, {share_fc2_w.shape=}'
            share_fc2_w.data = s_fc2_weight_merge.contiguous()

    # assert len(state_dict) == 0, f'left state_dict: {state_dict.keys()}'

    # enforce check nan
    assert_not_nan(tp_model.layernorm_weight.data)
    assert_not_nan(tp_model.wte_weight.data)
    assert_not_nan(tp_model.lm_head_weight.data)

    for layer_index, weights in enumerate(tp_model.layers_weight):
        for i, weight in enumerate(weights):
            if isinstance(weight, torch.Tensor):
                assert_not_nan(weight)

    load_to_cuda(tp_model=tp_model)
    torch.cuda.empty_cache()
