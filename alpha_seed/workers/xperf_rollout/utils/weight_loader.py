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

from alpha_seed.workers.xperf_rollout.utils.weights_adapter import WeightsAdapter

from alpha_seed.workers.xperf_rollout.utils.bf16_convert_helper import (_reshard_fsdp_state_dict_to_xperf_deepseek_v3)

# megatron
from alpha_seed.workers.xperf_rollout.utils.bf16_convert_helper import (_reshard_fsdp_state_dict_to_xperf_m8_megatron)

from alpha_seed.workers.xperf_rollout.utils.fp8_convert_helper import (_reshard_fsdp_state_dict_to_xperf_m8_fp8,
                                                                       _reshard_fsdp_state_dict_to_xperf_deepseek_v3_fp8
                                                                      )


def get_xperf_gpt_weight_bind_fn(model_config: PretrainedConfig,
                                 quant_mode: str = "NO_QUANT",
                                 backend='fsdp',
                                 is_custom_xperf: bool = False,
                                 is_xperf_triton: bool = False,
                                 enable_actor_critic_spatial_mux: bool = False,
                                 bind_device_mesh=None):
    if is_custom_xperf:
        from alpha_seed.workers.xperf_rollout.utils.custom_xperf_convert_helper import _reshard_state_dict_to_xperf_custom
        return partial(_reshard_state_dict_to_xperf_custom, model_config=model_config, backend=backend)
    if is_xperf_triton:
        if model_config.model_type == 'seed_m8':
            from alpha_seed.workers.xperf_rollout.utils.xperf_gpt_triton_helper import _reshard_fsdp_state_dict_to_xperf_triton_m8
            return partial(_reshard_fsdp_state_dict_to_xperf_triton_m8, model_config=model_config, backend=backend)
        elif model_config.model_type == 'seed_vl':
            from alpha_seed.workers.xperf_rollout.utils.xperf_gpt_triton_helper import _reshard_fsdp_state_dict_to_xperf_triton_seed_vl
            return partial(_reshard_fsdp_state_dict_to_xperf_triton_seed_vl, model_config=model_config, backend=backend)
        else:
            raise NotImplementedError(f"xperf_triton does not support model_type: {model_config.model_type}")
    if backend in ('fsdp', 'vescale-fsdp2'):
        if model_config.model_type == 'deepseek_v3':
            if quant_mode == "WFP8":
                return partial(_reshard_fsdp_state_dict_to_xperf_deepseek_v3_fp8, model_config=model_config)
            else:
                return partial(_reshard_fsdp_state_dict_to_xperf_deepseek_v3, model_config=model_config)

        return WeightsAdapter(model_config, quant_mode, enable_actor_critic_spatial_mux, backend, bind_device_mesh)

    elif backend == 'megatron':
        if quant_mode == "WFP8":
            if model_config.model_type == 'seed_m8':
                return partial(_reshard_fsdp_state_dict_to_xperf_m8_fp8, model_config=model_config, backend=backend)
            else:
                raise NotImplementedError(f'Unsupported model type {model_config.model_type} in WFP8 quant mode')
        if model_config.model_type == 'seed_m8':
            return partial(_reshard_fsdp_state_dict_to_xperf_m8_megatron, model_config=model_config)
        else:
            raise NotImplementedError(f'Unsupported model type {model_config.model_type}')
