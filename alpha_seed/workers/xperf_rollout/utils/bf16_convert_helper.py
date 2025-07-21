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
from alpha_seed.workers.xperf_rollout.utils.weights_adapter import WeightsAdapter


def assert_not_nan(tensor: torch.Tensor):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if os.getenv('XPERF_CHECK_NAN', '1') == '1':
        assert not torch.any(torch.isnan(tensor)).item(), f'Got nan in parameter {tensor} on rank {rank}'


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
