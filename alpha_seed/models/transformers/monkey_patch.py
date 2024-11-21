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


#### Seed Models
def apply_monkey_patch_to_p6():
    from seed_models.models.p6.modeling_p6 import P6FlashAttention2
    from verl.models.transformers.seed_mlp import swiglu_mlp_forward
    from alpha_seed.models.transformers.modeling_p6 import flash_attn2_rmpad_forward
    P6FlashAttention2.forward = flash_attn2_rmpad_forward
    # P6ExpertMLP.forward = swiglu_mlp_forward
    from seed_models.integrations import apply_liger_kernel_to_p6
    apply_liger_kernel_to_p6()


def apply_monkey_patch_to_p6_dense():
    from seed_models.integrations import apply_liger_kernel_to_p6d
    apply_liger_kernel_to_p6d()


def apply_monkey_patch_to_p7():
    from seed_models.models.p7.modeling_p7 import P7FlashAttention2
    from alpha_seed.models.transformers.modeling_p7 import flash_attn2_rmpad_forward
    P7FlashAttention2.forward = flash_attn2_rmpad_forward
    from seed_models.integrations import apply_liger_kernel_to_p7
    apply_liger_kernel_to_p7()


_PATCH_NAME_TO_FUNC = {
    'seed_p6': apply_monkey_patch_to_p6,
    'seed_p6dense': apply_monkey_patch_to_p6_dense,
    'seed_p7': apply_monkey_patch_to_p7
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
