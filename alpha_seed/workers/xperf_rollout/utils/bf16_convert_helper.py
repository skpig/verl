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

    if os.getenv('XPERF_CHECK_NAN', '0') == '1':
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


def _reshard_fsdp_state_dict_to_xperf_p6(tp_model, state_dict, device_mesh: DeviceMesh, model_config: P6Config):
    assert isinstance(model_config, P6Config)

    # checking
    if device_mesh is not None:
        tp_size = device_mesh['tp'].size()
        tp_rank = device_mesh['tp'].get_local_rank()
    else:
        tp_size = 1
        tp_rank = 0

    head_dim = model_config.hidden_size // model_config.num_attention_heads

    ln_f_weight = state_dict.pop('transformer.ln_f.weight').full_tensor().to(torch.bfloat16)
    ln_f_bias = state_dict.pop('transformer.ln_f.bias').full_tensor().to(torch.bfloat16)
    ln_f = torch.stack((ln_f_weight, ln_f_bias)).contiguous()

    tp_model.layernorm_weight.data = ln_f

    assert_not_nan(tp_model.layernorm_weight.data)

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

    assert_not_nan(tp_model.wte_weight.data)
    assert_not_nan(tp_model.lm_head_weight.data)

    for layer_index, (ln_1, key_norm, context_norm, qkv_w, qkv_b, dense_w, dense_b, ln_2, gate_w, _, fc1_w, _, fc2_w, _,
                      *_) in enumerate(tp_model.layers_weight):
        # torch.distributed.breakpoint()
        ln_1_weight = state_dict.pop(f'transformer.h.{layer_index}.ln_1.weight').full_tensor()
        ln_1_bias = state_dict.pop(f'transformer.h.{layer_index}.ln_1.bias').full_tensor()
        ln_1_weight = torch.stack((ln_1_weight, ln_1_bias), dim=0).to(torch.bfloat16)
        assert ln_1.data.shape == ln_1_weight.shape
        ln_1.data = ln_1_weight.contiguous()

        assert_not_nan(ln_1.data)

        del ln_1_bias

        key_norm_weight = state_dict[f'transformer.h.{layer_index}.attn.key_layernorm.weight'].full_tensor()
        key_norm_bias = state_dict[f'transformer.h.{layer_index}.attn.key_layernorm.bias'].full_tensor()
        key_norm_weight = torch.stack((key_norm_weight, key_norm_bias), dim=0).to(torch.bfloat16)
        assert key_norm.data.shape == key_norm_weight.shape
        key_norm.data = key_norm_weight.contiguous()

        assert_not_nan(key_norm.data)

        del key_norm_bias

        context_norm_weight = state_dict[f'transformer.h.{layer_index}.attn.context_norm.weight'].full_tensor()
        context_norm_bias = state_dict[f'transformer.h.{layer_index}.attn.context_norm.bias'].full_tensor()
        context_norm_weight = torch.stack((context_norm_weight, context_norm_bias), dim=0).to(torch.bfloat16)
        assert context_norm.data.shape == context_norm_weight.shape
        context_norm.data = context_norm_weight.contiguous()

        assert_not_nan(context_norm.data)

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

        assert qkv_weight.shape == qkv_w.shape
        qkv_w.data = qkv_weight.contiguous()

        # check nan
        assert_not_nan(qkv_w.data)

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

        assert_not_nan(qkv_b.data)

        o_proj_weight = state_dict.pop(f'transformer.h.{layer_index}.attn.o_proj.weight').full_tensor().to(
            torch.bfloat16)
        if device_mesh is not None:
            o_proj_weight = _fix_o_ordering(o_proj_weight, num_heads=num_heads, q_heads_list=q_heads_list)[tp_rank]

        assert o_proj_weight.shape == dense_w.shape
        dense_w.data = o_proj_weight.contiguous()

        assert_not_nan(dense_w.data)

        o_proj_bias = state_dict.pop(f'transformer.h.{layer_index}.attn.o_proj.bias').full_tensor().to(torch.bfloat16)
        assert o_proj_bias.shape == dense_b.shape
        dense_b.data = o_proj_bias.contiguous()

        assert_not_nan(dense_b.data)

        ln_2_weight = state_dict.pop(f'transformer.h.{layer_index}.ln_2.weight').full_tensor().to(torch.bfloat16)
        ln_2_bias = state_dict.pop(f'transformer.h.{layer_index}.ln_2.bias').full_tensor().to(torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight, ln_2_bias), dim=0)
        assert ln_2_weight.shape == ln_2.shape
        ln_2.data = ln_2_weight.contiguous()

        assert_not_nan(ln_2.data)

        del ln_2_bias

        gate_wg = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.gate.wg').full_tensor().T.contiguous().float()
        gate_wg_ema = state_dict.pop(f'transformer.h.{layer_index}.mlp.moe.gate.wg_ema').T.contiguous().float()

        gate_wg = (gate_wg + gate_wg_ema) * 0.5

        assert gate_wg.shape == gate_w.shape
        gate_w.data = gate_wg.contiguous()

        assert_not_nan(gate_w.data)

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

        assert fc1_weight.shape == fc1_w.shape
        fc1_w.data = fc1_weight.contiguous()

        assert_not_nan(fc1_w.data)

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

    head_dim = model_config.hidden_size // model_config.num_attention_heads
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

    for layer_index, (ln_1, qkv_w, _, dense_w, _, ln_2, fc12, _, fc2, *_) in enumerate(tp_model.layers_weight):
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

        assert_not_nan(qkv_w.data)

        # model.layers.0.self_attn.o_proj.weight
        o_proj_weight = state_dict.pop(f'model.layers.{layer_index}.self_attn.o_proj.weight').full_tensor().to(
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

        assert dense_w.data.shape == o_proj_weight.shape
        dense_w.data = o_proj_weight

        assert_not_nan(dense_w.data)

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

    load_to_cuda(tp_model=tp_model)
    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_m8(tp_model, state_dict, device_mesh: DeviceMesh, model_config):
    from seed_models import M8Config
    assert isinstance(model_config, M8Config)

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

    ln_f_weight = state_dict.pop('transformer.norm.weight').full_tensor().to(torch.bfloat16)
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
        ln_1_weight = state_dict.pop(f'transformer.h.{layer_index}.input_layernorm.weight').full_tensor()
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

        ln_2_weight = state_dict.pop(f'transformer.h.{layer_index}.post_attention_layernorm.weight').full_tensor().to(
            torch.bfloat16)
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

    load_to_cuda(tp_model=tp_model)
    torch.cuda.empty_cache()
