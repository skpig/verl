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

import torch
import torch.distributed

from functools import partial

from transformers import PretrainedConfig

from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh


def init_meta(tp_model):
    param_list = [tp_model.layernorm_weight, tp_model.wte_weight, tp_model.lm_head_weight] + \
                    [p for layer in tp_model.layers_weight for p in layer if isinstance(p, torch.Tensor)]
    if hasattr(tp_model, 'wpe'):
        param_list.append(tp_model.wpe.weight)
    for param in param_list:
        new_param = torch.empty_like(param, device='cpu')
        torch.utils.swap_tensors(param, new_param)


def offload_to_cpu(tp_model):
    param_list = [tp_model.layernorm_weight, tp_model.wte_weight, tp_model.lm_head_weight] + \
                    [p for layer in tp_model.layers_weight for p in layer if isinstance(p, torch.Tensor)]
    if hasattr(tp_model, 'wpe'):
        param_list.append(tp_model.wpe.weight)
    for param in param_list:
        param.data = param.data.cpu()
    if hasattr(tp_model, '_free_kv_cache'):
        tp_model._free_kv_cache()

    torch.cuda.empty_cache()


def load_to_cuda(tp_model):
    param_list = [tp_model.layernorm_weight, tp_model.wte_weight, tp_model.lm_head_weight] + \
                    [p for layer in tp_model.layers_weight for p in layer if isinstance(p, torch.Tensor)]
    if hasattr(tp_model, 'wpe'):
        param_list.append(tp_model.wpe.weight)
    for param in param_list:
        # make sure the param is not a DTensor
        assert not isinstance(param.data, DTensor)
        param.data = param.data.cuda()


def _fix_qkv_ordering(param, tp_size, num_heads, mqa_kv_heads, interleaved_kv_shared, dim=0):
    q_heads_list = None
    kv_heads_list = None

    assert tp_size % mqa_kv_heads == 0 or mqa_kv_heads % tp_size == 0, \
        "can't split mqa kv head {} to {} GPUs".format(mqa_kv_heads, tp_size)
    if mqa_kv_heads % tp_size == 0:
        heads_per_rank = mqa_kv_heads // tp_size
        kv_heads_list = [[idx for idx in range(mp * heads_per_rank, (mp + 1) * heads_per_rank)] for mp in range(tp_size)
                        ]

        q_heads_list = list()
        repeat_size = num_heads // mqa_kv_heads
        for kv_heads in kv_heads_list:
            if interleaved_kv_shared:
                q_heads_list.append(
                    [idx + mqa_kv_heads * repeat_idx for repeat_idx in range(repeat_size) for idx in kv_heads])
            else:
                q_heads_list.append(
                    [kv_idx * repeat_size + repeat_idx for repeat_idx in range(repeat_size) for kv_idx in kv_heads])
    elif tp_size % mqa_kv_heads == 0:
        rank_per_head = tp_size // mqa_kv_heads
        kv_heads_list = [[mp // rank_per_head] for mp in range(tp_size)]

        q_heads_list = list()
        repeat_size = num_heads // mqa_kv_heads // rank_per_head
        for mp, kv_heads in enumerate(kv_heads_list):
            if interleaved_kv_shared:
                q_heads_list.append([
                    kv_heads[0] + mqa_kv_heads * (repeat_idx + (mp % rank_per_head) * repeat_size)
                    for repeat_idx in range(repeat_size)
                ])
            else:
                q_heads_list.append([
                    kv_idx * rank_per_head * repeat_size + (mp % rank_per_head) * repeat_size + repeat_idx
                    for repeat_idx in range(repeat_size)
                    for kv_idx in kv_heads
                ])

    head_dim = param.shape[dim] // (num_heads + mqa_kv_heads * 2)
    src_split = torch.split(param.data, [num_heads * head_dim, mqa_kv_heads * head_dim, mqa_kv_heads * head_dim],
                            dim=dim)
    qkv_split = [torch.split(src_s, head_dim, dim=dim) for src_s in src_split]

    q_split = [torch.cat([qkv_split[0][i] for i in q_heads], axis=dim) for q_heads in q_heads_list]
    k_split = [torch.cat([qkv_split[1][i] for i in k_heads], axis=dim) for k_heads in kv_heads_list]
    v_split = [torch.cat([qkv_split[2][i] for i in v_heads], axis=dim) for v_heads in kv_heads_list]

    qkv_cat = [torch.cat([q, k, v], axis=dim) for q, k, v in zip(q_split, k_split, v_split)]
    return qkv_cat, q_heads_list


def _fix_o_ordering(param, num_heads, q_heads_list, dim=1):
    assert q_heads_list is not None, "q_heads_list should not be None"

    head_dim = param.shape[dim] // num_heads
    src_split = torch.split(param, head_dim, dim=dim)
    src_split = [torch.cat([src_split[i] for i in q_heads], axis=dim) for q_heads in q_heads_list]
    return src_split
