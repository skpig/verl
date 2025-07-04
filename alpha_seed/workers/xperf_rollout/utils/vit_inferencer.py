import os
import torch
import torch.nn as nn
import xperf_gpt
from xperf_gpt.multi_models.visual.inferencer import VITInferencer as XPERF_VITInferencer
from xperf_gpt.multi_models.visual.token_builder import VisualBuilder
from xperf_gpt.multi_models.visual.eva_vit import EVA_VIT_CONFIGS
from xperf_gpt.model_implementations.config import XperfViTInferenceConfig
from torch.nn import LayerNorm as FusedLayerNorm
from xperf_gpt.model_implementations.vit_torch import Attention, DropPath, Mlp


# TODO: move it to xperf_gpt
class Block(nn.Module):

    def __init__(self, config: XperfViTInferenceConfig):
        super().__init__()
        dim = config.embed_dim
        self.norm1 = FusedLayerNorm(dim, eps=1e-6)
        self.attn = Attention(config)
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = FusedLayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * config.mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
        )

    def forward(self, x, rel_pos_bias=None, attn_mask=None, grid_hw=None, rotary_pos_emb=None):
        x = x + self.drop_path(
            self.attn(self.norm1(x),
                      rel_pos_bias=rel_pos_bias,
                      attn_mask=attn_mask,
                      grid_hw=grid_hw,
                      rotary_pos_emb=rotary_pos_emb))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VITInferencer(XPERF_VITInferencer):

    def __init__(self, *args, **kwargs) -> None:
        self.use_xperf_gpt = kwargs.get("use_xperf_gpt", False)
        super().__init__(*args, **kwargs)

    def build_encoder(self, generate_kwargs):
        if generate_kwargs is None or len(generate_kwargs) == 0:
            generate_kwargs = dict(max_new_tokens=128, do_sample=False, top_p=0.7, use_xperf_vit=self.use_xperf_gpt)

        detail_config = {}
        if self.vit_model_config['vit_model'] in EVA_VIT_CONFIGS.keys():
            detail_config = EVA_VIT_CONFIGS[self.vit_model_config['vit_model']]['transformer_config']

        ### begin ckpt policy ### default use xgpt infernece
        self.vit_config = {}
        self.vit_config['model_config'] = self.vit_model_config
        self.vit_config['model_config']['navit_anyres'] = self.vit_model_config.get(
            'use_navit', False) or self.vit_model_config.get('navit_anyres', False)
        self.vit_config['model_config'].update(detail_config)
        if 'transformer_config' in self.vit_config['model_config']:
            self.vit_config['model_config'].update(self.vit_config['model_config']['transformer_config'])
        if self.vit_model_ckpt_path:
            self.vit_config['vanilla_checkpoint_path'] = self.vit_model_ckpt_path
        self.vit_config['model_config']['model_name'] = 'EVAVisionTransformer'

        self.dp_vit = self.vit_config['model_config'].get('dp_vit', False) or self.vit_config.get('dp_vit', False)
        kwargs = {'rank0_split': True}
        # disable tp when using small vit or using dp vit
        if self.vit_config['model_config']['embed_dim'] < 2048 or self.dp_vit:
            kwargs['mp_size'] = 1
            kwargs['rank0_split'] = False

        from xperf_gpt.model_implementations import vit_torch
        vit_torch.Block = Block
        vit_model = xperf_gpt.init_inference(None,
                                             config=self.vit_config,
                                             dtype=torch.bfloat16,
                                             is_vit_model=True,
                                             use_xperf_gpt=self.use_xperf_gpt,
                                             **kwargs)
        if not self.use_xperf_gpt:
            assert vit_model.module.custom_decoder.layers[0].norm1.eps == 1e-6
            assert vit_model.module.custom_decoder.layers[0].norm2.eps == 1e-6

        if self.dp_vit:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.dp_group = dist.new_group(backend='nccl')
            self.vit_dp_infer_threshold = os.environ.get('XPERF_VIT_DP_INFER_THRESHOLD', 16)

        with xperf_gpt.OnDevice(dtype=torch.bfloat16, device="meta"):
            _, self.ln_vision, self.token_merger, self.seed_proj = VisualBuilder.build_vit(self.vit_model_config)

        def parameter_pack(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.anyres_image_newline = None
        self.multicrop_anyres = self.vit_model_config.get('multicrop_anyres', False)
        if self.multicrop_anyres:
            self.anyres_image_newline = parameter_pack(vit_model.module.weights['anyres_image_newline'][0])
        self.ln_vision.weight = parameter_pack(vit_model.module.weights['ln_vision'][0])
        self.ln_vision.bias = parameter_pack(vit_model.module.weights['ln_vision'][1])
        self.seed_proj[0].weight = parameter_pack(vit_model.module.weights['seed_proj'][0])
        self.seed_proj[0].bias = parameter_pack(vit_model.module.weights['seed_proj'][1])
        self.seed_proj[2].weight = parameter_pack(vit_model.module.weights['seed_proj'][2])
        self.seed_proj[2].bias = parameter_pack(vit_model.module.weights['seed_proj'][3])
        self.visual_encoder = vit_model
        self.llm_hidden_size = self.seed_proj[2].weight.shape[0]
        print('====>> Loading VIT Done')
