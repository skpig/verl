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
from typing import Dict
import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import DeviceMesh
from collections import defaultdict
from seed_models import M8Config
from alpha_seed.utils.functional import get_text_model_type


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
def apply_monkey_patch_to_p6(config):
    from seed_models.models.p6.modeling_p6 import P6FlashAttention2, P6ForCausalLM
    from verl.models.transformers.seed_mlp import swiglu_mlp_forward
    from alpha_seed.models.transformers.modeling_p6 import flash_attn2_rmpad_forward, p6_model_forward
    P6FlashAttention2.forward = flash_attn2_rmpad_forward
    P6ForCausalLM.forward = p6_model_forward
    # P6ExpertMLP.forward = swiglu_mlp_forward
    from seed_models.integrations import apply_liger_kernel_to_p6
    apply_liger_kernel_to_p6()


def apply_monkey_patch_to_p6_dense(config):
    from seed_models.models.p6dense.modeling_p6d import P6DenseFlashAttention2, P6DenseForCausalLM, P6DenseMLP
    from alpha_seed.models.transformers.modeling_p6d import flash_attn2_rmpad_forward, p6d_model_forward, mlp_tp_forward
    P6DenseFlashAttention2.forward = flash_attn2_rmpad_forward
    P6DenseMLP.forward = mlp_tp_forward
    P6DenseForCausalLM.forward = p6d_model_forward
    from seed_models.integrations import apply_liger_kernel_to_p6d
    apply_liger_kernel_to_p6d()


def apply_monkey_patch_to_p7(config):
    from seed_models.models.p7.modeling_p7 import P7FlashAttention2, P7ForCausalLM
    from alpha_seed.models.transformers.modeling_p7 import flash_attn2_rmpad_forward, p7_model_forward
    P7FlashAttention2.forward = flash_attn2_rmpad_forward
    P7ForCausalLM.forward = p7_model_forward
    from seed_models.integrations import apply_liger_kernel_to_p7
    apply_liger_kernel_to_p7()


def apply_monkey_patch_to_m8(config):
    from seed_models.models.m8.modeling_m8 import M8FlashAttention2, M8FusedMoeBlock, M8PreTrainedModel, M8ForCausalLM
    from .modeling_m8 import flash_attn2_rmpad_forward, _fused_moe_ep_forward, release_m8_kv_mirror, m8_casual_lm_forward
    M8FlashAttention2.forward = flash_attn2_rmpad_forward
    M8FusedMoeBlock.forward = _fused_moe_ep_forward
    M8ForCausalLM.forward = m8_casual_lm_forward
    from seed_models.integrations import apply_liger_kernel_to_m8
    apply_liger_kernel_to_m8()
    M8PreTrainedModel.release_act_memory = release_m8_kv_mirror


def apply_monkey_patch_to_ds3(config):
    from seed_models.models.deepseek_v3.modeling_deepseek import DeepseekV3FlashAttention2, DeepseekV3MLP, DeepseekV3FusedMoE, DeepseekV3ForCausalLM
    from seed_models.integrations import apply_liger_kernel_to_deepseek_v3
    from .modeling_ds import flash_attn2_forward, moe_ep_forward, mlp_tp_forward, deepseek_v3_casual_lm_forward
    DeepseekV3FlashAttention2.forward = flash_attn2_forward
    DeepseekV3FusedMoE.forward = moe_ep_forward
    DeepseekV3MLP.forward = mlp_tp_forward
    DeepseekV3ForCausalLM.forward = deepseek_v3_casual_lm_forward
    apply_liger_kernel_to_deepseek_v3()


def apply_monkey_patch_to_vlm(config):
    text_type = get_text_model_type(config)
    _PATCH_NAME_TO_FUNC[text_type](config)
    from seed_models.models.seed_vl.modeling_seed_vl import SeedVLForConditionalGeneration
    from .modeling_vlm import get_dummy_image_features, get_sp_input_embeds
    SeedVLForConditionalGeneration.get_image_features = get_dummy_image_features
    SeedVLForConditionalGeneration.get_input_embeds = get_sp_input_embeds


_PATCH_NAME_TO_FUNC = {
    'seed_p6': apply_monkey_patch_to_p6,
    'seed_p6dense': apply_monkey_patch_to_p6_dense,
    'seed_p7': apply_monkey_patch_to_p7,
    'seed_m8': apply_monkey_patch_to_m8,
    'deepseek_v3': apply_monkey_patch_to_ds3,
    'seed_vl': apply_monkey_patch_to_vlm
}

from transformers import PretrainedConfig


def apply_monkey_patch(config: PretrainedConfig, verbose=True):
    model_type = config.model_type
    success_apply_monkey_patch = False
    if model_type in _PATCH_NAME_TO_FUNC:
        _PATCH_NAME_TO_FUNC[model_type](config)
        success_apply_monkey_patch = True

    if success_apply_monkey_patch and verbose:
        print(f'Applying monkey patch to model {model_type}')

    return success_apply_monkey_patch


def get_parallel_plan(config, tp_mesh: DeviceMesh) -> Dict[str, Placement]:
    """
    Get tensor parallel plan for the model
    """
    make_plan_fn = None
    if config.model_type == 'seed_m8' or \
            (hasattr(config, "text_config") and config.text_config.model_type == 'seed_m8'):
        from .modeling_m8 import make_m8_plan
        make_plan_fn = make_m8_plan
    if config.model_type == "deepseek_v3":
        from .modeling_ds import make_dsv3_plan
        make_plan_fn = make_dsv3_plan
    if "P6Dense" in config.architectures[0]:
        from .modeling_p6d import make_p6d_plan
        make_plan_fn = make_p6d_plan

    if make_plan_fn is None:
        assert tp_mesh.size() == 1, f"tensor parallelism is not support for model: {config.model_type}"
        return {}
    plan: Dict = make_plan_fn()
    assert all(isinstance(p, Placement) for p in plan.values()), "Parallel plan must described as fqn:Placement"
    return plan
