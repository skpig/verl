import os
import torch
import torch.nn as nn
from seed_models.models.seed_vl import SeedVLConfig
from transformers.modeling_utils import PreTrainedModel
from seed_models.models.seed_vl.modeling_seed_vl import SeedVisionTransformer
from transformers import AutoConfig


def gen_vit_cfg(vit_cfg_path):
    config = AutoConfig.from_pretrained(vit_cfg_path, trust_remote_code=True)
    return config.vision_config


# torch vit model from seed_models
class TorchVitInferencer(PreTrainedModel):

    def __init__(self, config: SeedVLConfig, *args, **kwargs) -> None:
        super().__init__(config)
        # model
        self.visual_encoder = SeedVisionTransformer(config)
        # for mlp pooling, compute input_dim
        input_dim = self.visual_encoder.num_features
        if hasattr(config, 'use_mlp_pooling'):
            use_mlp_pooling = config.use_mlp_pooling
        else:
            use_mlp_pooling = False
        if use_mlp_pooling:
            pooling_factor = self.visual_encoder.adapooling_factor**2
            input_dim = self.visual_encoder.num_features * pooling_factor
        # layer norm
        self.ln_vision = nn.LayerNorm(input_dim)
        self.vision_config = config
        bridge_activation_func = nn.ReLU if self.vision_config.bridge_activation_type == "ReLU" else nn.GELU
        # seed proj
        self.seed_proj = nn.Sequential(
            nn.Linear(input_dim, self.vision_config.projector_hidden_dim),
            bridge_activation_func(),
            nn.Linear(self.vision_config.projector_hidden_dim, self.vision_config.projector_embed_dim),
        )

        # default bf16 in xperf
        self.cuda().to(dtype=torch.bfloat16)
        print("init torch vit done...")

    @torch.no_grad()
    def get_image_features(self, pixel_values, grid_hw):
        device = next(self.visual_encoder.parameters()).device
        pixel_values = pixel_values.to(device)
        if isinstance(grid_hw, list):
            grid_hw = torch.tensor(grid_hw, dtype=torch.long)
        grid_hw = grid_hw.to(device)

        image_embeds = self.visual_encoder(pixel_values, grid_hw=grid_hw)
        if self.ln_vision is not None:
            image_embeds = self.ln_vision(image_embeds)
        if self.seed_proj is not None:
            image_embeds = self.seed_proj(image_embeds)
        return image_embeds

    def weights_update(self, state_dict):

        def assert_not_nan(tensor: torch.Tensor):
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

            if os.getenv('XPERF_CHECK_NAN', '1') == '1':
                assert not torch.any(torch.isnan(tensor)).item(), f'Got nan in parameter {tensor} on rank {rank}'

        def update_param(vit_model_param, key):
            if isinstance(key, str):
                param_in_state_dict = state_dict.pop(key).to(torch.bfloat16).full_tensor()
            else:
                param_in_state_dict = key
            assert vit_model_param.shape == param_in_state_dict.shape, f'{key=}, {vit_model_param.shape=}, {param_in_state_dict.shape=}'
            vit_model_param.data = param_in_state_dict.contiguous()
            assert_not_nan(vit_model_param.data)

        # for visual_encoder model
        for vision_key, param in self.visual_encoder.named_parameters():
            vision_key = "vision_encoder." + vision_key
            update_param(param, vision_key)

        for proj_key, proj_param in self.seed_proj.named_parameters():
            proj_key = "multi_modal_projector." + proj_key
            update_param(proj_param, proj_key)

        for ln_key, ln_param in self.ln_vision.named_parameters():
            ln_key = "ln_vision." + ln_key
            update_param(ln_param, ln_key)

        torch.cuda.empty_cache()
        print('====>> resharding VIT finished !!!')

    def update_standalone_weighs(self, comm_fn, comm_rank):

        def comm_and_assign(module, param_names=None):
            if param_names is None:
                param_names = ["weight", "bias"]
            for param_name in param_names:
                param = getattr(module, param_name)
                cu_param = param.cuda()
                comm_fn(cu_param, comm_rank)
                setattr(module, param_name, nn.Parameter(cu_param))

        for layer in self.visual_encoder.blocks:
            comm_and_assign(layer.norm1)
            comm_and_assign(layer.norm2)
            comm_and_assign(layer.mlp.fc1)
            comm_and_assign(layer.mlp.fc2)
            comm_and_assign(layer.attn.proj)
            comm_and_assign(layer.attn.qkv, ['weight'])
            comm_and_assign(layer.attn, param_names=['q_bias', 'v_bias'])
        comm_and_assign(self.visual_encoder.patch_embed.proj)
        comm_and_assign(self.ln_vision)
        comm_and_assign(self.seed_proj[0])
        comm_and_assign(self.seed_proj[2])
