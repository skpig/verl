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

from seed_models import P4Config, P5Config, P6Config

from alpha_seed.workers.xperf_rollout.utils.bf16_convert_helper import (
    _reshard_fsdp_state_dict_to_xperf_p4, _reshard_fsdp_state_dict_to_xperf_p5, _reshard_fsdp_state_dict_to_xperf_p6,
    _reshard_fsdp_state_dict_to_xperf_p6dense, _reshard_fsdp_state_dict_to_xperf_p7,
    _reshard_fsdp_state_dict_to_xperf_m8, _reshard_fsdp_state_dict_to_xperf_vl)
from alpha_seed.workers.xperf_rollout.utils.fp8_convert_helper import (_reshard_fsdp_state_dict_to_xperf_p6_fp8,
                                                                       _reshard_fsdp_state_dict_to_xperf_p6dense_fp8,
                                                                       _reshard_fsdp_state_dict_to_xperf_m8_fp8)


def get_xperf_gpt_weight_bind_fn(model_config: PretrainedConfig, quant_mode: str = "NO_QUANT"):
    if quant_mode == "WFP8":
        if model_config.model_type == 'seed_p6':
            return partial(_reshard_fsdp_state_dict_to_xperf_p6_fp8, model_config=model_config)
        elif model_config.model_type == 'seed_p6dense':
            return partial(_reshard_fsdp_state_dict_to_xperf_p6dense_fp8, model_config=model_config)
        elif model_config.model_type == 'seed_m8':
            return partial(_reshard_fsdp_state_dict_to_xperf_m8_fp8, model_config=model_config)
        else:
            raise NotImplementedError(f'Unsupported model type {model_config.model_type} in WFP8 quant mode')
    if model_config.model_type == 'seed_p4':
        return partial(_reshard_fsdp_state_dict_to_xperf_p4, model_config=model_config)
    elif model_config.model_type == 'seed_p5':
        return partial(_reshard_fsdp_state_dict_to_xperf_p5, model_config=model_config)
    elif model_config.model_type == 'seed_p6':
        return partial(_reshard_fsdp_state_dict_to_xperf_p6, model_config=model_config)
    elif model_config.model_type == 'seed_p6dense':
        return partial(_reshard_fsdp_state_dict_to_xperf_p6dense, model_config=model_config)
    elif model_config.model_type == 'seed_p7':
        return partial(_reshard_fsdp_state_dict_to_xperf_p7, model_config=model_config)
    elif model_config.model_type == 'seed_m8':
        return partial(_reshard_fsdp_state_dict_to_xperf_m8, model_config=model_config)
    elif model_config.model_type == 'seed_vl':
        return partial(_reshard_fsdp_state_dict_to_xperf_vl, model_config=model_config)
    else:
        raise NotImplementedError(f'Unsupported model type {model_config.model_type}')
