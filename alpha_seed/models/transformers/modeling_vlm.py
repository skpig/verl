import numpy as np
import torch
from seed_models.models.seed_vl.modeling_seed_vl import SeedVLForConditionalGeneration
from typing import Tuple
from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import slice_input_tensor
from typing import Tuple


def get_dummy_image_features(self, pixel_values, image_grid_hw=None):
    is_dummy = pixel_values is None
    if pixel_values is None:
        pixel_values = torch.ones([144, 14 * 14 * 3],
                                  dtype=torch.bfloat16,
                                  device=self.vision_encoder.patch_embed.proj.weight.device)
        image_grid_hw = torch.tensor([[12, 12]], device=self.vision_encoder.patch_embed.proj.weight.device)
    if self.use_navit:
        image_embeds = self.vision_encoder(pixel_values, grid_hw=image_grid_hw)
    else:
        image_embeds = self.vision_encoder(pixel_values)

    if self.ln_vision is not None:
        image_embeds = self.ln_vision(image_embeds)

    image_embeds = self.multi_modal_projector(image_embeds)
    if is_dummy:
        image_embeds = image_embeds[:0]
    return image_embeds


def get_image_inputs(non_tensor_batch):
    image_kwargs = {}
    image_keys = get_image_keys(non_tensor_batch)
    for key in image_keys:
        non_none_values = [_ for _ in non_tensor_batch[key] if _ is not None]
        if len(non_none_values) == 0:
            return {}
        if isinstance(non_none_values[0], np.ndarray):
            if key == 'image_grid_hw':
                non_none_values = [torch.from_numpy(value.astype(int)) for value in non_none_values]
            else:
                raise RuntimeError(f'tensor is expected, got {non_none_values}')
        image_kwargs[key] = torch.cat(non_none_values).cuda()
    return image_kwargs


def get_image_keys(non_tensor_batch):
    if 'pixel_values' in non_tensor_batch:
        return ['pixel_values', 'image_grid_hw']
    return []


def get_sp_input_embeds(
    self: SeedVLForConditionalGeneration,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_hw: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if inputs_embeds is None:
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.vision_encoder.get_dtype())
        image_embeds = self.get_image_features(pixel_values, image_grid_hw)
        n_image_features = image_embeds.shape[0]
        # get image mask and culculate n_image_tokens
        if image_mask is None:
            image_mask = input_ids == self.config.image_token_id
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        if pixel_values is not None and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: n_image_tokens: {n_image_tokens}, n_image_features {n_image_features}"
            )
        # fill image tokens to padding tokens, to avoid negative token_ids for text embedding
        input_ids[image_mask] = 1
        text_embeds = self.get_input_embeddings()(input_ids)
        image_mask = image_mask.unsqueeze(-1).expand_as(text_embeds).to(text_embeds.device)
        image_embeds = image_embeds.to(text_embeds.device)
        text_embeds = text_embeds.masked_scatter(image_mask, image_embeds)
        inputs_embeds = text_embeds
        input_ids = None
        sp_size = get_ulysses_sequence_parallel_world_size()
        if sp_size > 1:
            inputs_embeds = slice_input_tensor(inputs_embeds, dim=1, padding=False)
        return input_ids, inputs_embeds
