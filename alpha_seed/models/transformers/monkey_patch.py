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

#### Open Source Models


def apply_monkey_patch_to_llama():
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2, LlamaModel
    from verl.models.transformers.llama import flash_attn2_rmpad_forward, llama_model_rmpad_forward
    LlamaFlashAttention2.forward = flash_attn2_rmpad_forward
    LlamaModel.forward = llama_model_rmpad_forward

    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_llama
    apply_liger_kernel_to_llama(rope=False,
                                cross_entropy=False,
                                fused_linear_cross_entropy=False,
                                rms_norm=True,
                                swiglu=True)


def apply_monkey_patch_to_qwen2():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2, Qwen2ForCausalLM
    from verl.models.transformers.qwen2 import flash_attn2_rmpad_forward, lce_forward
    Qwen2FlashAttention2.forward = flash_attn2_rmpad_forward
    Qwen2ForCausalLM.forward = lce_forward

    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen2
    apply_liger_kernel_to_qwen2(rope=False,
                                cross_entropy=False,
                                fused_linear_cross_entropy=False,
                                rms_norm=True,
                                swiglu=True)


#### Seed Models


def apply_monkey_patch_to_p4():
    from seed_models.models.p4.modeling_p4 import P4FlashAttention2
    from alpha_seed.models.transformers.seed_flash_attn_rmpad import flash_attn2_rmpad_forward
    P4FlashAttention2.forward = flash_attn2_rmpad_forward


def apply_monkey_patch_to_p5():
    from seed_models.models.p5.modeling_p5 import P5FlashAttention2
    from alpha_seed.models.transformers.seed_flash_attn_rmpad import flash_attn2_rmpad_forward
    P5FlashAttention2.forward = flash_attn2_rmpad_forward


def apply_monkey_patch_to_p6():
    from seed_models.models.p6.modeling_p6 import P6FlashAttention2, P6ExpertMLP
    from verl.models.transformers.seed_mlp import swiglu_mlp_forward
    from alpha_seed.models.transformers.seed_flash_attn_rmpad import flash_attn2_rmpad_forward
    # P6FlashAttention2.forward = flash_attn2_rmpad_forward
    # P6ExpertMLP.forward = swiglu_mlp_forward


_PATCH_NAME_TO_FUNC = {
    'llama': apply_monkey_patch_to_llama,
    'qwen2': apply_monkey_patch_to_qwen2,
    'seed_p4': apply_monkey_patch_to_p4,
    'seed_p5': apply_monkey_patch_to_p5,
    'seed_p6': apply_monkey_patch_to_p6
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
