import numpy as np
import torch
from seed_models.models.seed_vl.modeling_seed_vl import SeedVLForConditionalGeneration
from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import slice_input_tensor
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
import ray


def add_pixel_values_to_inflight_query(queries, image_manager):
    image_refs = []
    for query in queries:
        if query.pixel_values_ref is not None:
            assert isinstance(query.pixel_values_ref,
                              str), f'pixel_values must be a str, but got {query.pixel_values_ref}'
            image_refs.append(query.pixel_values_ref)
    # convert str to object_ref
    image_refs = ray.get(image_manager.get_refs.remote(image_refs))
    pixel_values = ray.get(image_refs)
    index = 0
    for query in queries:
        if query.pixel_values_ref is not None:
            query.pixel_values = pixel_values[index]
            index += 1
    return queries


def convert_tensor_to_numpy(tensor):
    if tensor is None or isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, list):
        return [convert_tensor_to_numpy(t) for t in tensor]
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy()
    return tensor.numpy()


def convert_numpy_to_tensor(numpy_array, dtype=None):
    if numpy_array is None or isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if isinstance(numpy_array, list):
        return [convert_numpy_to_tensor(t, dtype) for t in numpy_array]
    if numpy_array.dtype == np.uint16:
        return torch.from_numpy(numpy_array).view(torch.bfloat16)
    if dtype is not None:
        return torch.from_numpy(numpy_array.astype(dtype))
    return torch.from_numpy(numpy_array)


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


def get_local_non_none_inputs(ref_list, image_manager):
    ref_list = [r for r in ref_list if r is not None]
    if len(ref_list) > 0:
        if isinstance(ref_list[0], str):
            for ref in ref_list:
                assert isinstance(ref, str), "all data are expected to be ref str"
            image_refs = ray.get(image_manager.get_refs.remote(ref_list))
            images = ray.get(image_refs)
        else:
            images = ref_list
        return images
    return ref_list


def get_image_inputs(non_tensor_batch, image_manager=None):
    image_kwargs = {}
    if 'pixel_values_ref' in non_tensor_batch:
        pixel_values_ref = non_tensor_batch['pixel_values_ref']
        assert image_manager is not None
        pixel_values = get_local_non_none_inputs(pixel_values_ref, image_manager)
    elif 'pixel_values' in non_tensor_batch:
        pixel_values = non_tensor_batch['pixel_values']
    else:
        return image_kwargs
    pixel_values = [convert_numpy_to_tensor(v, float) for v in pixel_values if v is not None]
    if len(pixel_values) == 0:
        return image_kwargs
    image_kwargs['pixel_values'] = torch.cat(pixel_values).cuda()
    image_grid_hw = [convert_numpy_to_tensor(v, int) for v in non_tensor_batch['image_grid_hw'] if v is not None]
    image_kwargs['image_grid_hw'] = torch.cat(image_grid_hw).cuda()
    return image_kwargs


def get_image_keys(non_tensor_batch):
    keys = ['pixel_values', 'image_grid_hw', 'pixel_values_ref']
    keys = [k for k in keys if k in non_tensor_batch]
    return keys


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


def vlm_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.Tensor = None,
    image_grid_hw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.IntTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    image_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    output_aux_losses: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    fuse_lm_head_ce_loss: Optional[bool] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> Union[Tuple, MoeCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_router_logits = (output_router_logits
                            if output_router_logits is not None else self.config.output_router_logits)
    output_aux_losses = output_aux_losses if output_aux_losses is not None else self.config.output_aux_losses

    output_hidden_states = (output_hidden_states
                            if output_hidden_states is not None else self.config.output_hidden_states)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    input_ids, inputs_embeds = self.get_input_embeds(input_ids, pixel_values, image_grid_hw, inputs_embeds, image_mask)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    return self.language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        output_aux_losses=output_aux_losses,
        return_dict=return_dict,
        fuse_lm_head_ce_loss=fuse_lm_head_ce_loss,
        temperature=temperature,
        labels=labels,
        compute_entropy=kwargs.get('compute_entropy', False),
    )
