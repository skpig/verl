# TODO(lijiahao.plus): port this file back to verl

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
Apply monkey-patch function to models
"""

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from collections import defaultdict
from seed_models import M8Config


def get_ignore_modules_in_mixed_precision(model_type):
    from seed_models.models.m8.modeling_m8 import M8TopkCapGate
    from seed_models.models.p6.modeling_p6 import P6TopkCapGate
    from seed_models.models.p7.modeling_p7 import P7TopkCapGate
    _IGNORE_MODULES_IN_MIXED_PRECISION = defaultdict(tuple)

    _IGNORE_MODULES_IN_MIXED_PRECISION.update({
        "seed_p6": (P6TopkCapGate,),
        "seed_p7": (P7TopkCapGate,),
        "seed_m8": (M8TopkCapGate,)
    })

    return _IGNORE_MODULES_IN_MIXED_PRECISION[model_type]


def update_gate_ema(fsdp_module):
    config = fsdp_module.module.config
    if isinstance(config, M8Config):
        if dist.is_initialized() and dist.get_rank() == 0:
            print('Update gate ema for M8')
        update_gate_ema_m8(fsdp_module=fsdp_module)


def update_gate_ema_m8(fsdp_module):
    assert isinstance(fsdp_module.module.config, M8Config)
    with torch.inference_mode():
        for i in range(fsdp_module.module.config.num_hidden_layers):
            gate = fsdp_module.module.transformer.h[i].module.mlp.moe.gate
            with FSDP.summon_full_params(gate, rank0_only=False, offload_to_cpu=False):
                gate.module.update_gate_ema()


#### Open Source Models


#### Seed Models
def apply_monkey_patch_to_p6():
    from seed_models.models.p6.modeling_p6 import P6FlashAttention2, P6ForCausalLM
    from verl.models.transformers.seed_mlp import swiglu_mlp_forward
    from alpha_seed.models.transformers.modeling_p6 import flash_attn2_rmpad_forward, p6_model_forward
    P6FlashAttention2.forward = flash_attn2_rmpad_forward
    P6ForCausalLM.forward = p6_model_forward
    # P6ExpertMLP.forward = swiglu_mlp_forward
    from seed_models.integrations import apply_liger_kernel_to_p6
    apply_liger_kernel_to_p6()


def apply_monkey_patch_to_p6_dense():
    from seed_models.models.p6dense.modeling_p6d import P6DenseFlashAttention2, P6DenseForCausalLM
    from alpha_seed.models.transformers.modeling_p6d import flash_attn2_rmpad_forward, p6d_model_forward
    P6DenseFlashAttention2.forward = flash_attn2_rmpad_forward
    P6DenseForCausalLM.forward = p6d_model_forward
    from seed_models.integrations import apply_liger_kernel_to_p6d
    apply_liger_kernel_to_p6d()


def apply_monkey_patch_to_p7():
    from seed_models.models.p7.modeling_p7 import P7FlashAttention2, P7ForCausalLM
    from alpha_seed.models.transformers.modeling_p7 import flash_attn2_rmpad_forward, p7_model_forward
    P7FlashAttention2.forward = flash_attn2_rmpad_forward
    P7ForCausalLM.forward = p7_model_forward
    from seed_models.integrations import apply_liger_kernel_to_p7
    apply_liger_kernel_to_p7()


def apply_monkey_patch_to_m8():
    from seed_models.models.m8.modeling_m8 import M8FlashAttention2, M8FusedMoeBlock, M8PreTrainedModel, M8ForCausalLM
    from .modeling_m8 import flash_attn2_rmpad_forward, _fused_moe_ep_forward, release_m8_kv_mirror, m8_casual_lm_forward
    M8FlashAttention2.forward = flash_attn2_rmpad_forward
    M8FusedMoeBlock.forward = _fused_moe_ep_forward
    M8ForCausalLM.forward = m8_casual_lm_forward
    from seed_models.integrations import apply_liger_kernel_to_m8
    apply_liger_kernel_to_m8()
    M8PreTrainedModel.release_act_memory = release_m8_kv_mirror


_PATCH_NAME_TO_FUNC = {
    'seed_p6': apply_monkey_patch_to_p6,
    'seed_p6dense': apply_monkey_patch_to_p6_dense,
    'seed_p7': apply_monkey_patch_to_p7,
    'seed_m8': apply_monkey_patch_to_m8
}

from transformers import PretrainedConfig


def apply_monkey_patch(config: PretrainedConfig, verbose=True):
    success_apply_monkey_patch = False
    if config.model_type in _PATCH_NAME_TO_FUNC:
        _PATCH_NAME_TO_FUNC[config.model_type]()
        success_apply_monkey_patch = True

    if success_apply_monkey_patch and verbose:
        print(f'Applying monkey patch to model {config.model_type}')

    return success_apply_monkey_patch
