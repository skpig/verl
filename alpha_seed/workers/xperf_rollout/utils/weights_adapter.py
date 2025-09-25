import torch
import logging
import torch.distributed as dist
from functools import partial
from transformers import PretrainedConfig
from typing import Tuple, Union, List, Dict, Optional, Protocol
from torch.distributed._tensor import DTensor, Shard, Replicate
from alpha_seed.workers.xperf_rollout.utils.vit_inferencer import TorchVitInferencer
from alpha_seed.workers.xperf_rollout.utils.quant_utils import quant_gemm_weight_w8a8, quant_group_gemm_weight_w4a8

logger = logging.getLogger(__name__)


class AdapterProtocol(Protocol):

    def get_model_info(self, xperf_model: torch.nn.Module) -> None:
        '''pass'''

    def load_from_state_dict(self, state_dict: Dict[str, Union[torch.Tensor, DTensor]], prefix: str) -> None:
        '''pass'''

    def process_and_assign_weights(self, xperf_model: torch.nn.Module) -> None:
        '''pass'''


class WeightsAdapter:

    def __init__(self,
                 model_config: PretrainedConfig,
                 quant_mode: str,
                 enable_actor_critic_spatial_mux: bool = False,
                 backend: str = 'fsdp',
                 bind_device_mesh: Optional[Dict[str, torch.distributed.ProcessGroup]] = None) -> None:
        self.adapter: AdapterProtocol = None
        self.vit_adapter: AdapterProtocol = None

        if model_config.model_type in FSDPLLMWeightsAdapter._support_model_type:
            self.adapter = FSDPLLMWeightsAdapter(model_config,
                                                 quant_mode,
                                                 enable_actor_critic_spatial_mux,
                                                 backend=backend,
                                                 bind_device_mesh=bind_device_mesh)
        elif model_config.model_type in FSDPVLMWeightsAdapter._support_model_type:
            self.adapter = FSDPLLMWeightsAdapter(model_config.text_config,
                                                 quant_mode,
                                                 enable_actor_critic_spatial_mux,
                                                 backend=backend,
                                                 bind_device_mesh=bind_device_mesh)
            self.vit_adapter = FSDPVLMWeightsAdapter(model_config.vision_config, quant_mode,
                                                     enable_actor_critic_spatial_mux)
        assert self.adapter is not None, f"Unsupported model type {model_config.model_type}"

    def __call__(self,
                 xperf_llm: torch.nn.Module,
                 xperf_vit: torch.nn.Module = None,
                 state_dict: Dict[str, Union[torch.Tensor, DTensor]] = None,
                 device_mesh: Optional[Dict[str, torch.distributed.ProcessGroup]] = None,
                 prefix: Optional[str] = None) -> None:

        # if xperf_vit params is used torch vit(based ):
        if xperf_vit is not None and isinstance(xperf_vit, TorchVitInferencer):
            xperf_vit.weights_update(state_dict)

        elif xperf_vit is not None:
            setattr(xperf_vit, "use_xperf_gpt", True)
            self.vit_adapter.setup_device_mesh(device_mesh)
            self.vit_adapter.get_model_info(xperf_vit)
            self.vit_adapter.load_from_state_dict(state_dict, prefix="")
            self.vit_adapter.process_and_assign_weights(xperf_vit)

        self.adapter.setup_device_mesh(device_mesh)
        self.adapter.get_model_info(xperf_llm)
        self.adapter.load_from_state_dict(state_dict,
                                          prefix=prefix or ("language_model." if xperf_vit is not None else ""))
        self.adapter.process_and_assign_weights(xperf_llm)
        torch.cuda.empty_cache()

    def setup_device_mesh(self, device_mesh: Optional[Dict[str, torch.distributed.ProcessGroup]]) -> None:

        self.device_mesh = device_mesh
        self.tp_size = device_mesh['tp'].size() if device_mesh else 1
        self.tp_rank = device_mesh['tp'].get_local_rank() if device_mesh else 0

    def _pop_with_fallback(self, state_dict: Dict[str, Union[torch.Tensor, DTensor]], prefix: str, *keys):

        for key in keys:
            full_key = f"{prefix}{key}"
            if full_key in state_dict:
                return state_dict.pop(full_key)
        return None

    def _get_full_tensor(self, tensor: DTensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:

        if dtype is not None and not self.enable_actor_critic_spatial_mux:
            tensor = self._cast_to(tensor, dtype)

        if isinstance(tensor, DTensor) or isinstance(tensor, torch.Tensor):
            tensor = tensor.cuda()

        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()

        if self.enable_actor_critic_spatial_mux:
            partial_world_size = self.device_mesh.size() // 2
            rank = self.device_mesh.get_rank()
            if rank < partial_world_size:
                dst = rank + partial_world_size
                ndim = torch.tensor([len(tensor.shape)], dtype=torch.long, device=tensor.device)
                shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
                dist.send(tensor=ndim, dst=dst)
                dist.send(tensor=shape_tensor, dst=dst)
                dist.send(tensor, dst=dst)
            else:
                src = rank - partial_world_size
                ndim = torch.empty(1, dtype=torch.long, device="cuda")
                dist.recv(tensor=ndim, src=src)
                shape = torch.empty(ndim.item(), dtype=torch.long, device="cuda")
                dist.recv(shape, src=src)
                tensor = torch.empty(shape.tolist(), dtype=torch.float32, device="cuda")
                dist.recv(tensor, src=src)

        return tensor

    def _get_partial_tensor(self, tensor: DTensor, dim: int, dtype=None) -> torch.Tensor:
        # used for train tp < gen tp
        if dtype is not None:
            tensor = self._cast_to(tensor, dtype)
        if isinstance(tensor, DTensor) or isinstance(tensor, torch.Tensor):
            tensor = tensor.cuda()
        if isinstance(tensor, DTensor) or isinstance(tensor, torch.Tensor):
            bind_size = self.bind_device_mesh['bind'].size()
            if 'tp' in self.bind_device_mesh.mesh_dim_names:
                # gen tp > train tp
                train_tp_size = self.tp_size // bind_size
                rank = self.bind_device_mesh.get_rank() % self.tp_size // train_tp_size
                if dim == 0:
                    per_rank_dim = tensor.shape[0] // bind_size
                    tensor_out = tensor[rank * per_rank_dim:(rank + 1) * per_rank_dim]
                elif dim == 1:
                    per_rank_dim = tensor.shape[1] // bind_size
                    tensor_out = tensor[:, rank * per_rank_dim:(rank + 1) * per_rank_dim]
                else:
                    assert False, f'{dim=} not support in _get_partial_tensor'
            elif 'gen_tp' in self.bind_device_mesh.mesh_dim_names:
                # gen tp < train tp
                assert dim < 2, f'{dim=} not support in _get_partial_tensor'
                if dim == 1:
                    tensor = tensor.transpose(0, 1).contiguous()
                shape_out = (x if i != 0 else x * bind_size for i, x in enumerate(tensor.shape))
                tensor_out = torch.zeros(*shape_out, dtype=tensor.dtype, device=tensor.device)
                bind_group = self.bind_device_mesh.get_group(mesh_dim="bind")
                dist.all_gather_into_tensor(tensor_out, tensor, group=bind_group)
                if dim == 1:
                    tensor_out = tensor_out.transpose(0, 1).contiguous()
        return tensor_out

    def _redistribute_dtensor(self,
                              tensor: DTensor,
                              placements: List[Union[Shard, Replicate]],
                              device_mesh: Optional[Dict[str, torch.distributed.ProcessGroup]] = None) -> torch.Tensor:

        device_mesh = device_mesh or self.device_mesh
        return tensor._local_tensor if not device_mesh else tensor.redistribute(device_mesh=device_mesh,
                                                                                placements=placements)._local_tensor

    def _cast_to(self, tensor: Union[torch.Tensor, DTensor], dtype: torch.dtype) -> Union[torch.Tensor, DTensor]:
        if tensor is None or tensor.dtype == dtype:
            return tensor
        return tensor.to(dtype)

    def _assign_and_validate(self,
                             src: Union[torch.Tensor, List[torch.Tensor]],
                             dst: Union[torch.Tensor, List[torch.Tensor]],
                             name: str,
                             is_int4: bool = False) -> None:

        if src is None:
            return
        if isinstance(src, list):
            for i, (src_item, dst_item) in enumerate(zip(src, dst)):
                if src_item is not None:
                    if is_int4:
                        assert src_item.numel() == dst_item.numel() or src_item.numel() * 2 == dst_item.numel(
                        ), f"Weight {name} shape mismatch: src {src_item.shape} vs dst {dst_item.shape}"
                    else:
                        assert src_item.shape == dst_item.shape or src_item.numel() == dst_item.numel(
                        ), f"Weight {name} shape mismatch: src {src_item.shape} vs dst {dst_item.shape}"
                    assert not torch.isnan(src_item).any(), f"Weight {name} contains NaN values"
                    torch.utils.swap_tensors(dst_item, src_item)
        else:
            if is_int4:
                assert src.numel() == dst.numel() or src.numel() * 2 == dst.numel(
                ), f"Weight {name} shape mismatch: src {src.shape} vs dst {dst.shape}"
            else:
                assert src.shape == dst.shape or src.numel() == dst.numel(
                ), f"Weight {name} shape mismatch: src {src.shape} vs dst {dst.shape}"
            assert not torch.isnan(src).any(), f"Weight {name} contains NaN values"
            if not src.is_contiguous():
                src = src.contiguous()
            torch.utils.swap_tensors(dst, src)


class FSDPLLMWeightsAdapter(WeightsAdapter, AdapterProtocol):
    _support_model_type = ["seed_p6", "seed_p6dense", "seed_p7", "seed_m8", "seed_m10", "seed_m11"]

    def __init__(self,
                 model_config: PretrainedConfig,
                 quant_mode: str,
                 enable_actor_critic_spatial_mux: bool,
                 backend='fsdp',
                 bind_device_mesh=None) -> None:

        self.model_config = model_config
        self.quant_mode = quant_mode
        self.enable_actor_critic_spatial_mux = enable_actor_critic_spatial_mux
        self.bind_device_mesh = bind_device_mesh
        self.source_weights: Dict[str, Union[torch.Tensor, DTensor]] = {}
        self.need_amax = "A8" in self.quant_mode
        self.amax_ready = False
        self.quant_ratio_ready = False
        self.backend = backend

    def get_model_info(self, xperf_model: torch.nn.Module) -> None:

        config = xperf_model.config
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_kv_heads = config.mqa_kv_heads
        self.kv_replicate = self.tp_size // self.num_kv_heads if self.tp_size % self.num_kv_heads == 0 else 1
        self.num_layers = config.num_layers
        self.attention_bias = getattr(config, "has_attn_bias", False)
        self.moe_num_expert = getattr(config, "moe_expert_num", 0)
        self.share_expert_num = getattr(config, "share_expert_num", 0)
        self.use_query_layernorm = getattr(config, "querynorm", False) or getattr(self.model_config, "use_qk_rmsnorm",
                                                                                  False)
        self.use_key_layernorm = getattr(config, "keynorm", False) or getattr(
            config, "has_k_layernorm", False) or getattr(self.model_config, "use_qk_rmsnorm", False)
        self.use_context_groupnorm = getattr(config, "contextnorm", False) or getattr(
            config, "has_context_layernorm", False)
        self.use_attention_output_layernorm = getattr(config, "attn_outputnorm", False)
        self.has_over_encoding = getattr(config, "has_over_encoding", False)
        self.over_enc_vocab_size = getattr(config, "over_enc_vocab_size", None)
        self.over_enc_embed_dim = getattr(config, "over_enc_embed_dim", None)
        self.over_enc_vocab_stride = getattr(config, "over_enc_vocab_stride", None)
        self.over_enc_m = getattr(config, "over_enc_m", None)
        self.over_enc_n_in = getattr(config, "over_enc_n_in", None)
        self.over_enc_n_out = getattr(config, "over_enc_n_out", None)
        self.use_ep = getattr(xperf_model, "use_ep", False)
        self.vocab_tp = getattr(xperf_model, "vocab_tp", False)
        self.use_mtp = getattr(xperf_model, "use_mtp", False)
        self.mtp_n_heads = getattr(config, "mtp_n_heads", 1)
        assert not self.use_mtp or self.mtp_n_heads > 1, "mtp_n_heads must be greater than 1 when use_mtp is True"

    def load_from_state_dict(self, state_dict: Dict[str, Union[torch.Tensor, DTensor]], prefix: str) -> None:

        loader = partial(self._pop_with_fallback, state_dict, prefix)

        self.source_weights['wte'] = loader("transformer.wte.weight", "model.embed_tokens.weight",
                                            "transformer.embed_tokens.weight")
        self.source_weights['lm_head'] = loader("lm_head.weight")
        ln_f = loader("transformer.model.mtp_ce_norms.0.head_ln.weight", "transformer.norm.weight", "model.norm.weight",
                      "transformer.ln_f.weight")
        ln_f_bias = loader("transformer.ln_f.bias")
        self.source_weights['ln_f'] = torch.cat((ln_f, ln_f_bias), dim=0) if ln_f_bias is not None else ln_f

        if self.has_over_encoding:
            self.source_weights["oe_emb"] = loader("transformer.over_encoded_embeddings.embedding_list.0.weight")
            self.source_weights["oe_proj"] = loader("transformer.over_encoded_embeddings.emb_proj.weight")

        for layer_idx in range(self.num_layers):
            if self.model_config.model_type == "seed_p6dense":
                layer_key = f"model.layers.{layer_idx}"
            elif self.model_config.model_type in ["seed_m10", "seed_m11"]:
                layer_key = f"transformer.model.layers.{layer_idx}"
            else:
                layer_key = f"transformer.h.{layer_idx}"

            ln_1 = loader(f"{layer_key}.input_layernorm.weight", f"{layer_key}.ln_1.weight")
            ln_1_bias = loader(f"{layer_key}.ln_1.bias")
            ln_1 = torch.cat((ln_1, ln_1_bias), dim=0) if ln_1_bias is not None else ln_1

            ln_2 = loader(f"{layer_key}.post_attention_layernorm.weight", f"{layer_key}.ln_2.weight")
            ln_2_bias = loader(f"{layer_key}.ln_2.bias")
            ln_2 = torch.cat((ln_2, ln_2_bias), dim=0) if ln_2_bias is not None else ln_2

            key_norm = loader(f"{layer_key}.attn.key_layernorm.weight", f"{layer_key}.self_attn.k_norm.weight",
                              f"{layer_key}.self_attention.k_norm.weight")
            key_norm_bias = loader(f"{layer_key}.attn.key_layernorm.bias")
            key_norm = torch.cat((key_norm, key_norm_bias), dim=0) if key_norm_bias is not None else key_norm

            context_norm = loader(f"{layer_key}.attn.context_norm.weight",
                                  f"{layer_key}.self_attention.context_groupnorm.weight")
            context_norm_bias = loader(f"{layer_key}.attn.context_norm.bias")
            context_norm = torch.cat(
                (context_norm, context_norm_bias), dim=0) if context_norm_bias is not None else context_norm

            self.source_weights[layer_idx] = {
                'ln_1':
                    ln_1,
                'query_norm':
                    loader(f"{layer_key}.self_attn.q_norm.weight", f"{layer_key}.self_attention.q_norm.weight"),
                'key_norm':
                    key_norm,
                'context_norm':
                    context_norm,
                'attn_output_norm':
                    loader(f"{layer_key}.self_attention.o_norm.weight"),
                'ffn_output_norm':
                    loader(f"{layer_key}.ffn_output_layernorm.weight"),
                'q_proj':
                    loader(f"{layer_key}.attn.q_proj.weight", f"{layer_key}.self_attn.q_proj.weight",
                           f"{layer_key}.self_attention.q_proj.weight"),
                'q_proj_b':
                    loader(f"{layer_key}.attn.q_proj.bias", f"{layer_key}.self_attn.q_proj.bias"),
                'k_proj':
                    loader(f"{layer_key}.attn.k_proj.weight", f"{layer_key}.self_attn.k_proj.weight",
                           f"{layer_key}.self_attention.k_proj.weight"),
                'k_proj_b':
                    loader(f"{layer_key}.attn.k_proj.bias", f"{layer_key}.self_attn.k_proj.bias"),
                'v_proj':
                    loader(f"{layer_key}.attn.v_proj.weight", f"{layer_key}.self_attn.v_proj.weight",
                           f"{layer_key}.self_attention.v_proj.weight"),
                'v_proj_b':
                    loader(f"{layer_key}.attn.v_proj.bias", f"{layer_key}.self_attn.v_proj.bias"),
                'o_proj':
                    loader(f"{layer_key}.attn.o_proj.weight", f"{layer_key}.self_attn.o_proj.weight",
                           f"{layer_key}.self_attention.o_proj.weight"),
                'o_proj_b':
                    loader(f"{layer_key}.attn.o_proj.bias", f"{layer_key}.self_attn.o_proj.bias"),
                'ln_2':
                    ln_2,
                'gate_wg':
                    loader(f"{layer_key}.mlp.moe.gate.wg"),
                'gate_wg_ema':
                    loader(f"{layer_key}.mlp.moe.gate.wg_ema"),
                'fc1_1':
                    loader(f"{layer_key}.mlp.moe.experts.fc1_1", f"{layer_key}.mlp.gate_proj.weight",
                           f"{layer_key}.mlp.moe.experts.gate_proj"),
                'fc1_2':
                    loader(f"{layer_key}.mlp.moe.experts.fc1_2", f"{layer_key}.mlp.up_proj.weight",
                           f"{layer_key}.mlp.moe.experts.up_proj"),
                'share_fc1_1':
                    loader(f"{layer_key}.mlp.moe.experts_share.fc1_1", f"{layer_key}.mlp.moe.shared_experts.gate_proj"),
                'share_fc1_2':
                    loader(f"{layer_key}.mlp.moe.experts_share.fc1_2", f"{layer_key}.mlp.moe.shared_experts.up_proj"),
                'fc2':
                    loader(f"{layer_key}.mlp.moe.experts.fc2", f"{layer_key}.mlp.down_proj.weight",
                           f"{layer_key}.mlp.moe.experts.down_proj"),
                'share_fc2':
                    loader(f"{layer_key}.mlp.moe.experts_share.fc2", f"{layer_key}.mlp.moe.shared_experts.down_proj"),
                'vwn0_static_alpha':
                    loader(f"{layer_key}.hc1.static_alpha"),
                'vwn0_static_beta':
                    loader(f"{layer_key}.hc1.static_beta"),
                'vwn0_dynamic_alpha':
                    loader(f"{layer_key}.hc1.dynamic_alpha_fn"),
                'vwn0_dynamic_alpha_scale':
                    loader(f"{layer_key}.hc1.dynamic_alpha_scale"),
                'vwn0_dynamic_beta':
                    loader(f"{layer_key}.hc1.dynamic_beta_fn"),
                'vwn0_dynamic_beta_scale':
                    loader(f"{layer_key}.hc1.dynamic_beta_scale"),
                'vwn0_layer_norm':
                    loader(f"{layer_key}.hc1.layer_norm.weight"),
                'vwn1_static_alpha':
                    loader(f"{layer_key}.hc2.static_alpha"),
                'vwn1_static_beta':
                    loader(f"{layer_key}.hc2.static_beta"),
                'vwn1_dynamic_alpha':
                    loader(f"{layer_key}.hc2.dynamic_alpha_fn"),
                'vwn1_dynamic_alpha_scale':
                    loader(f"{layer_key}.hc2.dynamic_alpha_scale"),
                'vwn1_dynamic_beta':
                    loader(f"{layer_key}.hc2.dynamic_beta_fn"),
                'vwn1_dynamic_beta_scale':
                    loader(f"{layer_key}.hc2.dynamic_beta_scale"),
                'vwn1_layer_norm':
                    loader(f"{layer_key}.hc2.layer_norm.weight"),
                'vwn_extra_layernorm':
                    loader(f"{layer_key}.extra_norm.weight"),
            }

            if self.need_amax:
                self.source_weights[layer_idx].update({
                    "qkv_proj_amax":
                        loader(f"{layer_key}.attn.q_proj.input_amax", f"{layer_key}.self_attn.q_proj.input_amax"),
                    "qkv_proj_ratio":
                        loader(f"{layer_key}.attn.q_proj.smooth_quant_ratio",
                               f"{layer_key}.self_attn.q_proj.smooth_quant_ratio"),
                    "o_proj_amax":
                        loader(f"{layer_key}.attn.o_proj.input_amax", f"{layer_key}.self_attn.o_proj.input_amax"),
                    "o_proj_ratio":
                        loader(f"{layer_key}.attn.o_proj.smooth_quant_ratio",
                               f"{layer_key}.self_attn.o_proj.smooth_quant_ratio"),
                    "fc1_amax":
                        loader(f"{layer_key}.mlp.moe.experts.input_amax_fc1", f"{layer_key}.mlp.input_amax_fc1"),
                    "fc1_ratio":
                        loader(f"{layer_key}.mlp.moe.experts.smooth_quant_ratio_fc1",
                               f"{layer_key}.mlp.smooth_quant_ratio_fc1"),
                    "fc2_amax":
                        loader(f"{layer_key}.mlp.moe.experts.input_amax_fc2", f"{layer_key}.mlp.input_amax_fc2"),
                    "fc2_ratio":
                        loader(f"{layer_key}.mlp.moe.experts.smooth_quant_ratio_fc2",
                               f"{layer_key}.mlp.smooth_quant_ratio_fc2"),
                    "share_fc1_amax":
                        loader(f"{layer_key}.mlp.moe.experts_share.input_amax_fc1"),
                    "share_fc1_ratio":
                        loader(f"{layer_key}.mlp.moe.experts_share.smooth_quant_ratio_fc1"),
                    "share_fc2_amax":
                        loader(f"{layer_key}.mlp.moe.experts_share.input_amax_fc2"),
                    "share_fc2_ratio":
                        loader(f"{layer_key}.mlp.moe.experts_share.smooth_quant_ratio_fc2")
                })
                if self.source_weights[layer_idx]["qkv_proj_amax"].max() > 0:
                    # 第一次rollout的时候没有amax信息，使用全1作为smoothQuant scale
                    self.amax_ready = True
                else:
                    self.amax_ready = False
                    if torch.distributed.get_rank() == 0:
                        print("amax is not ready, will use all ones as smoothQuant scale")
                if self.source_weights[layer_idx]["qkv_proj_ratio"].max() > 0:
                    self.quant_ratio_ready = True
                else:
                    self.quant_ratio_ready = False
                    if torch.distributed.get_rank() == 0:
                        print("quant ratio is not ready, will use all 0.5 as smoothQuant ratio")

        for mtp_idx in range(0, self.mtp_n_heads):
            layer_idx = self.num_layers - self.mtp_n_heads + mtp_idx

            if mtp_idx == 0:
                self.source_weights[mtp_idx]["static_reduce"] = loader(
                    f"transformer.model.layers.{layer_idx}.hc2.static_reduce")
                continue

            self.source_weights[mtp_idx].update({
                'draft_e_norm': loader(f"transformer.model.mtp_embs.{mtp_idx}.mtp_pre_emb_norm.weight"),
                'draft_h_norm': loader(f"transformer.model.mtp_embs.{mtp_idx}.mtp_pre_hidden_norm.weight"),
                'draft_ln_f': loader(f"transformer.model.mtp_ce_norms.{mtp_idx}.head_ln.weight"),
                'draft_token_proj_w': loader(f"transformer.model.mtp_embs.{mtp_idx}.embed_feat_proj.weight"),
                'draft_token_proj_b': loader(f"transformer.model.mtp_embs.{mtp_idx}.embed_feat_proj.bias"),
                'static_reduce': loader(f"transformer.model.layers.{layer_idx}.hc2.static_reduce"),
                'draft_vwn_static_alpha': loader(f"transformer.model.mtp_embs.{mtp_idx}.hc.static_alpha"),
                'draft_vwn_static_beta': loader(f"transformer.model.mtp_embs.{mtp_idx}.hc.static_beta"),
                'draft_vwn_dynamic_alpha': loader(f"transformer.model.mtp_embs.{mtp_idx}.hc.dynamic_alpha_fn"),
                'draft_vwn_dynamic_alpha_scale': loader(f"transformer.model.mtp_embs.{mtp_idx}.hc.dynamic_alpha_scale"),
                'draft_vwn_dynamic_beta': loader(f"transformer.model.mtp_embs.{mtp_idx}.hc.dynamic_beta_fn"),
                'draft_vwn_dynamic_beta_scale': loader(f"transformer.model.mtp_embs.{mtp_idx}.hc.dynamic_beta_scale"),
                'draft_vwn_layernorm_weight': loader(f"transformer.model.mtp_embs.{mtp_idx}.hc.layer_norm.weight"),
            })

    def process_and_assign_weights(self, xperf_model: torch.nn.Module) -> None:

        xperf_weights = xperf_model.weights

        def assign_weights(binding_weights, layer_idx=None, is_int4=False):
            for xperf_weight, weight, name in binding_weights:
                if weight is None:
                    continue
                dst = getattr(xperf_weight, name)
                if layer_idx is not None:
                    dst = dst[layer_idx]
                    name = f"{layer_idx}_{name}"
                self._assign_and_validate(src=weight, dst=dst, name=name, is_int4=is_int4)

        wte_weight, lm_head_weight, ln_f_weight, oe_weight, oe_proj = self._process_top_level_weights()
        binding_weights = [(xperf_weights.module_weight, wte_weight, "wte_weight"),
                           (xperf_weights.module_weight, lm_head_weight, "lm_head_weight"),
                           (xperf_weights.module_weight, ln_f_weight, "ln_f_weight"),
                           (xperf_weights.module_weight, oe_weight, "over_enc_emb_weight"),
                           (xperf_weights.module_weight, oe_proj, "over_enc_proj_weight")]
        assign_weights(binding_weights)

        for layer_idx in range(self.num_layers):
            ln_1_weight, ln_2_weight, query_norm_weight, key_norm_weight, context_norm_weight, attn_output_norm_weight, ffn_output_norm_weight = self._process_layernorm_weights(
                layer_idx)
            qkv_weight, qkv_bias, o_weight, o_bias = self._process_attention_weights(layer_idx)
            fc1_weight, share_fc1_weight, fc2_weight, share_fc2_weight = self._process_ffn_weights(layer_idx)
            wfp8_qscale = []
            w4_qscale = []
            a8_qscale = []
            if self.quant_mode == "WFP8":
                qkv_weight, o_weight, fc1_weight, fc2_weight, share_fc1_weight, share_fc2_weight, wfp8_qscale = self._process_quant_wfp8(
                    qkv_weight, o_weight, fc1_weight, fc2_weight, share_fc1_weight, share_fc2_weight)
            elif "W4A8" in self.quant_mode:
                qkv_proj_amax, o_proj_amax = self._process_attention_amax(layer_idx)
                fc1_amax, fc2_amax, share_fc1_amax, share_fc2_amax = self._process_ffn_amax(layer_idx)
                qkv_proj_ratio = self.source_weights[layer_idx]["qkv_proj_ratio"]
                o_proj_ratio = self.source_weights[layer_idx]["o_proj_ratio"]
                fc1_ratio = self.source_weights[layer_idx]["fc1_ratio"]
                fc2_ratio = self.source_weights[layer_idx]["fc2_ratio"]
                share_fc1_ratio = self.source_weights[layer_idx]["share_fc1_ratio"]
                share_fc2_ratio = self.source_weights[layer_idx]["share_fc2_ratio"]
                dense_gemm_w8a8_weights, dense_gemm_w8a8_smooth_quant_scale, dense_gemm_w8a8_weight_qscale, \
                    group_gemm_w4a8_weights, group_gemm_w4a8_smooth_quant_scale, group_gemm_w4a8_i8_weight_qscale, \
                    group_gemm_w4a8_i4_weight_qscale_zero = self._process_quant_w4a8(
                        [qkv_weight, o_weight, share_fc1_weight, share_fc2_weight],
                        [qkv_proj_amax, o_proj_amax, share_fc1_amax, share_fc2_amax],
                        [qkv_proj_ratio, o_proj_ratio, share_fc1_ratio, share_fc2_ratio],
                        [fc1_weight, fc2_weight],
                        [fc1_amax, fc2_amax],
                        [fc1_ratio, fc2_ratio])
                qkv_weight, o_weight, share_fc1_weight, share_fc2_weight = dense_gemm_w8a8_weights
                fc1_weight, fc2_weight = group_gemm_w4a8_weights
                w4_qscale = [
                    # attn_qkv
                    dense_gemm_w8a8_weight_qscale[0],
                    None,
                    # attn_proj
                    dense_gemm_w8a8_weight_qscale[1],
                    None,
                    # ff0
                    group_gemm_w4a8_i8_weight_qscale[0],
                    group_gemm_w4a8_i4_weight_qscale_zero[0],
                    # ffn1
                    group_gemm_w4a8_i8_weight_qscale[1],
                    group_gemm_w4a8_i4_weight_qscale_zero[1],
                ]
                if self.share_expert_num > 0:
                    w4_qscale.extend([
                        # shared_expert0
                        dense_gemm_w8a8_weight_qscale[2],
                        # shared_expert1
                        dense_gemm_w8a8_weight_qscale[3]
                    ])
                a8_qscale = [
                    # attn_qkv
                    dense_gemm_w8a8_smooth_quant_scale[0],
                    # attn_proj
                    dense_gemm_w8a8_smooth_quant_scale[1],
                    # ff0
                    group_gemm_w4a8_smooth_quant_scale[0],
                    # ff1
                    group_gemm_w4a8_smooth_quant_scale[1],
                    # shared_expert0
                    dense_gemm_w8a8_smooth_quant_scale[2],
                    # shared_expert1
                    dense_gemm_w8a8_smooth_quant_scale[3],
                ]
            gate_wg_weight = self._process_gate_weights(layer_idx)
            vwn0_static_alpha, vwn0_static_beta, vwn0_dynamic_alpha, vwn0_dynamic_alpha_scale, vwn0_dynamic_beta, vwn0_dynamic_beta_scale, vwn0_layer_norm, \
            vwn1_static_alpha, vwn1_static_beta, vwn1_dynamic_alpha, vwn1_dynamic_alpha_scale, vwn1_dynamic_beta, vwn1_dynamic_beta_scale, vwn1_layer_norm, \
            vwn_extra_layernorm_weight = self._process_vwn_weights(layer_idx)

            binding_weights = [
                (xperf_weights.layer_weight, ln_1_weight, "norm0_gamma_beta"),
                (xperf_weights.layer_weight, query_norm_weight, "query_gamma_beta"),
                (xperf_weights.layer_weight, key_norm_weight, "key_gamma_beta"),
                (xperf_weights.layer_weight, context_norm_weight, "context_gamma_beta"),
                (xperf_weights.layer_weight, attn_output_norm_weight, "attn_out_gamma_beta"),
                (xperf_weights.layer_weight, ffn_output_norm_weight, "ffn_out_gamma_beta"),
                (xperf_weights.layer_weight, ln_2_weight, "norm1_gamma_beta"),
                (xperf_weights.layer_weight, qkv_weight, "qkv_proj_weight"),
                (xperf_weights.layer_weight, qkv_bias, "qkv_proj_bias"),
                (xperf_weights.layer_weight, o_weight, "out_proj_weight"),
                (xperf_weights.layer_weight, o_bias, "out_proj_bias"),
                (xperf_weights.layer_weight, share_fc1_weight, "FFN0_share_weight"),
                (xperf_weights.layer_weight, share_fc2_weight, "FFN1_share_weight"),
                (xperf_weights.layer_weight, gate_wg_weight, "moe_gate_weight"),
                (xperf_weights.layer_weight, vwn0_static_alpha, "vwn0_static_alpha"),
                (xperf_weights.layer_weight, vwn0_static_beta, "vwn0_static_beta"),
                (xperf_weights.layer_weight, vwn0_dynamic_alpha, "vwn0_dynamic_alpha"),
                (xperf_weights.layer_weight, vwn0_dynamic_alpha_scale, "vwn0_dynamic_alpha_scale"),
                (xperf_weights.layer_weight, vwn0_dynamic_beta, "vwn0_dynamic_beta"),
                (xperf_weights.layer_weight, vwn0_dynamic_beta_scale, "vwn0_dynamic_beta_scale"),
                (xperf_weights.layer_weight, vwn0_layer_norm, "vwn0_layernorm_weight"),
                (xperf_weights.layer_weight, vwn1_static_alpha, "vwn1_static_alpha"),
                (xperf_weights.layer_weight, vwn1_static_beta, "vwn1_static_beta"),
                (xperf_weights.layer_weight, vwn1_dynamic_alpha, "vwn1_dynamic_alpha"),
                (xperf_weights.layer_weight, vwn1_dynamic_alpha_scale, "vwn1_dynamic_alpha_scale"),
                (xperf_weights.layer_weight, vwn1_dynamic_beta, "vwn1_dynamic_beta"),
                (xperf_weights.layer_weight, vwn1_dynamic_beta_scale, "vwn1_dynamic_beta_scale"),
                (xperf_weights.layer_weight, vwn1_layer_norm, "vwn1_layernorm_weight"),
                (xperf_weights.layer_weight, vwn_extra_layernorm_weight, "vwn_extra_layernorm_weight"),
            ]
            assign_weights(binding_weights, layer_idx)
            binding_weights = [
                (xperf_weights.layer_weight, fc1_weight, "FFN0_weight"),
                (xperf_weights.layer_weight, fc2_weight, "FFN1_weight"),
            ]
            assign_weights(binding_weights, layer_idx, "W4A8" in self.quant_mode)
            if "W4A8" in self.quant_mode:
                if hasattr(xperf_weights.layer_weight, "expert_size"):
                    xperf_weights.layer_weight.expert_size = [
                        xperf_model.config.moe_ffn_internal_dim // (self.tp_size if not self.use_ep else 1)
                        for _ in range(len(xperf_weights.layer_weight.expert_size))
                    ]
                binding_weights = [
                    (xperf_weights.quant_weight, w4_qscale, "w4_qscale"),
                    (xperf_weights.quant_weight, a8_qscale, "a8_qscale"),
                ]
                assign_weights(binding_weights, layer_idx)
            elif self.quant_mode == "WFP8":
                binding_weights = [
                    (xperf_weights.quant_weight, wfp8_qscale, "wfp8_qscale"),
                ]
                assign_weights(binding_weights, layer_idx)

        for mtp_idx in range(0, self.mtp_n_heads):

            if mtp_idx == 0:
                static_reduce = self._process_mtp_weights(mtp_idx)
                binding_weights = [(xperf_weights.module_weight, static_reduce, "reduce_static_weight")]
                assign_weights(binding_weights, mtp_idx)
                continue

            draft_enorm_weight, draft_hnorm_weight, draft_ln_f_weight, draft_token_proj_w, \
            draft_token_proj_b, static_reduce, draft_vwn_static_alpha, draft_vwn_static_beta, \
            draft_vwn_dynamic_alpha, draft_vwn_dynamic_alpha_scale, draft_vwn_dynamic_beta, \
            draft_vwn_dynamic_beta_scale, draft_vwn_layernorm_weight = self._process_mtp_weights(mtp_idx)

            binding_weights = [
                (xperf_weights.module_weight, draft_enorm_weight, "draft_enorm_weight"),
                (xperf_weights.module_weight, draft_hnorm_weight, "draft_hnorm_weight"),
                (xperf_weights.module_weight, draft_ln_f_weight, "draft_ln_f_weight"),
                (xperf_weights.module_weight, draft_vwn_static_alpha, "draft_vwn_static_alpha"),
                (xperf_weights.module_weight, draft_vwn_static_beta, "draft_vwn_static_beta"),
                (xperf_weights.module_weight, draft_vwn_dynamic_alpha, "draft_vwn_dynamic_alpha"),
                (xperf_weights.module_weight, draft_vwn_dynamic_alpha_scale, "draft_vwn_dynamic_alpha_scale"),
                (xperf_weights.module_weight, draft_vwn_dynamic_beta, "draft_vwn_dynamic_beta"),
                (xperf_weights.module_weight, draft_vwn_dynamic_beta_scale, "draft_vwn_dynamic_beta_scale"),
                (xperf_weights.module_weight, draft_vwn_layernorm_weight, "draft_vwn_layernorm_weight"),
            ]
            xperf_model.draft_token_proj[mtp_idx - 1].weight.data = draft_token_proj_w
            xperf_model.draft_token_proj[mtp_idx - 1].bias.data = draft_token_proj_b
            assign_weights(binding_weights, mtp_idx - 1)
            binding_weights = [(xperf_weights.module_weight, static_reduce, "reduce_static_weight")]
            assign_weights(binding_weights, mtp_idx)

        xperf_weights.prepare_infer_weights()
        xperf_model.layers_weight = xperf_weights.layers_weight
        xperf_model.layernorm_weight = xperf_weights.layernorm_weight
        xperf_model.lm_head_weight = xperf_weights.lm_head_weight
        xperf_model.wte_weight = xperf_weights.wte_weight
        self.source_weights.clear()

    def _process_top_level_weights(self) -> Tuple[torch.Tensor, ...]:

        wte = self._cast_to(self._get_full_tensor(self.source_weights['wte']), torch.bfloat16)
        ln_f = self._cast_to(self._get_full_tensor(self.source_weights['ln_f']), torch.bfloat16)
        lm_head = self._cast_to(self._get_full_tensor(self.source_weights['lm_head']), torch.bfloat16)

        ln_f_weight = ln_f.view(-1, self.hidden_size).contiguous()

        if self.has_over_encoding:
            assert self.device_mesh is not None and self.vocab_tp, "OE must have device mesh and vocab tp enabled"
            from vescale.initialize.mesh import create_mesh_with_names

            oe_emb: DTensor = self._cast_to(self.source_weights['oe_emb'], torch.bfloat16)
            # TODO: currently oe emb only supports sharding across all devices
            for mesh_size, placement in zip(oe_emb.device_mesh.shape, oe_emb.placements):
                if placement.is_replicate():
                    assert mesh_size == 1, "Over Encoding Embedding weight can only support shard across all devices"
            oe_emb: torch.Tensor = oe_emb._local_tensor

            oe_emb = DTensor.from_local(oe_emb, self.device_mesh['tp'],
                                        [Shard(0)]).redistribute(placements=[Shard(1)])._local_tensor
            out = DTensor.from_local(oe_emb, self.device_mesh['dp'],
                                     [Shard(0)]).redistribute(placements=[Replicate()])._local_tensor
            oe_emb_weight = out[:sum(self.over_enc_vocab_size), :].cpu()
            torch.cuda.empty_cache()

            oe_proj = self._cast_to(self._get_full_tensor(self.source_weights['oe_proj']), torch.bfloat16)
            oe_emb_dim = oe_emb_weight.shape[1] * self.tp_size
            oe_vocab_num = len(self.over_enc_vocab_size)
            total_oe_emb_dim = oe_emb_dim * oe_vocab_num
            wte_proj, oe_proj = torch.split(oe_proj, [oe_proj.shape[1] - total_oe_emb_dim, total_oe_emb_dim], dim=1)

            wte_proj = DTensor.from_local(wte_proj, self.device_mesh, [Replicate(), Replicate()])
            wte_proj = self._redistribute_dtensor(wte_proj, [Replicate(), Shard(1)])

            oe_proj = DTensor.from_local(oe_proj.reshape(-1, oe_vocab_num, oe_emb_dim), self.device_mesh,
                                         [Replicate(), Replicate()])
            oe_proj = self._redistribute_dtensor(oe_proj, [Replicate(), Shard(2)]).reshape(oe_proj.shape[0], -1)

            oe_proj_weight = torch.concat([wte_proj, oe_proj], dim=1)
        else:
            oe_emb_weight = oe_proj_weight = None

        if self.device_mesh is not None and self.vocab_tp:
            wte = DTensor.from_local(wte, self.device_mesh, [Replicate(), Replicate()])
            wte_weight = self._redistribute_dtensor(wte, [Replicate(), Shard(1)])
            lm_head = DTensor.from_local(lm_head, self.device_mesh, [Replicate(), Replicate()])
            lm_head_weight = self._redistribute_dtensor(lm_head, [Replicate(), Shard(0)])
        else:
            wte_weight = wte.contiguous()
            lm_head_weight = lm_head.contiguous()

        return wte_weight, lm_head_weight, ln_f_weight, oe_emb_weight, oe_proj_weight

    def _process_layernorm_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:

        ln_1_weight = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['ln_1']),
                                    torch.bfloat16).reshape(-1, self.hidden_size)

        ln_2_weight = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['ln_2']),
                                    torch.bfloat16).reshape(-1, self.hidden_size)

        if self.use_query_layernorm:
            query_norm_weight = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['query_norm']),
                                              torch.bfloat16).reshape(-1, self.head_dim)
        else:
            query_norm_weight = None

        if self.use_key_layernorm:
            key_norm_weight = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['key_norm']),
                                            torch.bfloat16).reshape(-1, self.head_dim)
        else:
            key_norm_weight = None

        if self.use_context_groupnorm:
            context_norm_weight = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['context_norm']),
                                                torch.bfloat16).reshape(-1, self.head_dim)
        else:
            context_norm_weight = None

        if self.use_attention_output_layernorm:
            attn_output_norm_weight = self._cast_to(
                self._get_full_tensor(self.source_weights[layer_idx]['attn_output_norm']),
                torch.bfloat16).reshape(-1, self.hidden_size)
            ffn_output_norm_weight = self._cast_to(
                self._get_full_tensor(self.source_weights[layer_idx]['ffn_output_norm']),
                torch.bfloat16).reshape(-1, self.hidden_size)
        else:
            attn_output_norm_weight = None
            ffn_output_norm_weight = None

        return ln_1_weight, ln_2_weight, query_norm_weight, key_norm_weight, context_norm_weight, attn_output_norm_weight, ffn_output_norm_weight

    def _process_attention_weights_interleave(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        q_proj = self._cast_to(self.source_weights[layer_idx]['q_proj'], torch.bfloat16)
        k_proj = self._cast_to(self.source_weights[layer_idx]['k_proj'], torch.bfloat16)
        v_proj = self._cast_to(self.source_weights[layer_idx]['v_proj'], torch.bfloat16)
        o_proj = self._cast_to(self.source_weights[layer_idx]['o_proj'], torch.bfloat16)
        q_proj = DTensor.from_local(local_tensor=q_proj._local_tensor,
                                    device_mesh=q_proj.device_mesh,
                                    placements=[q_proj.placements[0]]).full_tensor()
        k_proj = DTensor.from_local(local_tensor=k_proj._local_tensor,
                                    device_mesh=k_proj.device_mesh,
                                    placements=[k_proj.placements[0]]).full_tensor()
        v_proj = DTensor.from_local(local_tensor=v_proj._local_tensor,
                                    device_mesh=v_proj.device_mesh,
                                    placements=[v_proj.placements[0]]).full_tensor()
        o_proj = DTensor.from_local(local_tensor=o_proj._local_tensor,
                                    device_mesh=o_proj.device_mesh,
                                    placements=[o_proj.placements[0]]).full_tensor()
        kv_replicate = 1
        bind_size = self.bind_device_mesh['bind'].size()
        if 'tp' in self.bind_device_mesh.mesh_dim_names:
            # gen tp > train tp
            gen_num_kv_heads = max(1, k_proj.shape[0] // self.head_dim // bind_size)
            kv_replicate = gen_num_kv_heads * bind_size * self.head_dim // k_proj.shape[0]
            if kv_replicate > 1:
                k_proj = k_proj.view(-1, self.head_dim, self.hidden_size)
                v_proj = v_proj.view(-1, self.head_dim, self.hidden_size)
                k_proj = torch.tile(k_proj.unsqueeze(1), (1, kv_replicate, 1, 1)).reshape(-1, self.hidden_size)
                v_proj = torch.tile(v_proj.unsqueeze(1), (1, kv_replicate, 1, 1)).reshape(-1, self.hidden_size)

        q_proj = self._get_partial_tensor(q_proj, 0)
        k_proj = self._get_partial_tensor(k_proj, 0)
        v_proj = self._get_partial_tensor(v_proj, 0)
        o_proj = self._get_partial_tensor(o_proj, 1)

        if 'gen_tp' in self.bind_device_mesh.mesh_dim_names:
            # gen tp < train tp
            q_proj = q_proj.reshape(bind_size, -1, self.head_dim,
                                    self.hidden_size).permute(1, 0, 2, 3).reshape(-1, self.hidden_size)
            k_proj = k_proj.reshape(bind_size, -1, self.head_dim,
                                    self.hidden_size).permute(1, 0, 2, 3).reshape(-1, self.hidden_size)
            v_proj = v_proj.reshape(bind_size, -1, self.head_dim,
                                    self.hidden_size).permute(1, 0, 2, 3).reshape(-1, self.hidden_size)
            o_proj = o_proj.reshape(self.hidden_size, bind_size, -1,
                                    self.head_dim).permute(0, 2, 1, 3).reshape(-1, self.hidden_size)

        # torch.cuda.synchronize()
        if self.attention_bias:
            q_proj_b = self._cast_to(self.source_weights[layer_idx]['q_proj_b'], torch.bfloat16)
            k_proj_b = self._cast_to(self.source_weights[layer_idx]['k_proj_b'], torch.bfloat16)
            v_proj_b = self._cast_to(self.source_weights[layer_idx]['v_proj_b'], torch.bfloat16)
            o_proj_b = self._cast_to(self.source_weights[layer_idx]['o_proj_b'], torch.bfloat16)
            q_proj_b = DTensor.from_local(local_tensor=q_proj_b._local_tensor,
                                          device_mesh=q_proj_b.device_mesh,
                                          placements=[q_proj_b.placements[0]]).full_tensor()
            k_proj_b = DTensor.from_local(local_tensor=k_proj_b._local_tensor,
                                          device_mesh=k_proj_b.device_mesh,
                                          placements=[k_proj_b.placements[0]]).full_tensor()
            v_proj_b = DTensor.from_local(local_tensor=v_proj_b._local_tensor,
                                          device_mesh=v_proj_b.device_mesh,
                                          placements=[v_proj_b.placements[0]]).full_tensor()
            o_proj_b = DTensor.from_local(local_tensor=o_proj_b._local_tensor,
                                          device_mesh=o_proj_b.device_mesh,
                                          placements=[o_proj_b.placements[0]]).full_tensor()
            if kv_replicate > 1:
                k_proj_b = k_proj_b.view(-1, self.head_dim)
                v_proj_b = v_proj_b.view(-1, self.head_dim)
                k_proj_b = torch.tile(k_proj_b.unsqueeze(1), (1, kv_replicate, 1)).reshape(-1)
                v_proj_b = torch.tile(v_proj_b.unsqueeze(1), (1, kv_replicate, 1)).reshape(-1)
            q_proj_b = self._get_partial_tensor(q_proj_b, 0)
            k_proj_b = self._get_partial_tensor(k_proj_b, 0)
            v_proj_b = self._get_partial_tensor(v_proj_b, 0)

        qkv_proj = torch.cat((q_proj, k_proj, v_proj), dim=0).view(-1, self.hidden_size).contiguous()
        o_proj = o_proj.contiguous().view(self.hidden_size, -1).contiguous()
        if self.attention_bias:
            qkv_proj_b = torch.cat((q_proj_b, k_proj_b, v_proj_b), dim=0).view(-1).contiguous()
            o_proj_b = o_proj_b.contiguous()
        else:
            qkv_proj_b = None
            o_proj_b = None
        return qkv_proj, qkv_proj_b, o_proj, o_proj_b

    def _process_attention_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        if (self.device_mesh is not None) and (not self.enable_actor_critic_spatial_mux) and (
                self.bind_device_mesh is not None) and (self.backend == 'fsdp'):
            return self._process_attention_weights_interleave(layer_idx)
        q_proj = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['q_proj']), torch.bfloat16)
        k_proj = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['k_proj']), torch.bfloat16)
        v_proj = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['v_proj']), torch.bfloat16)
        o_proj = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['o_proj']), torch.bfloat16)

        if self.attention_bias:
            q_proj_b = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['q_proj_b']), torch.bfloat16)
            k_proj_b = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['k_proj_b']), torch.bfloat16)
            v_proj_b = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['v_proj_b']), torch.bfloat16)
            o_proj_b = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['o_proj_b']), torch.bfloat16)

        q_proj = q_proj.view(self.num_kv_heads, -1, self.head_dim, self.hidden_size).transpose(0, 1)
        k_proj = k_proj.view(1, self.num_kv_heads, self.head_dim, self.hidden_size)
        v_proj = v_proj.view(1, self.num_kv_heads, self.head_dim, self.hidden_size)
        o_proj = o_proj.view(self.hidden_size, self.num_kv_heads, -1, self.head_dim).transpose(1, 2)

        if self.attention_bias:
            q_proj_b = q_proj_b.view(self.num_kv_heads, -1, self.head_dim).transpose(0, 1)
            k_proj_b = k_proj_b.view(1, self.num_kv_heads, self.head_dim)
            v_proj_b = v_proj_b.view(1, self.num_kv_heads, self.head_dim)

        if self.kv_replicate > 1:
            q_proj = q_proj.reshape(-1, self.tp_size, self.head_dim, self.hidden_size)
            k_proj = torch.tile(k_proj, (1, self.kv_replicate, 1, 1))
            v_proj = torch.tile(v_proj, (1, self.kv_replicate, 1, 1))
            o_proj = o_proj.reshape(self.hidden_size, -1, self.tp_size, self.head_dim)

            if self.attention_bias:
                q_proj_b = q_proj_b.view(-1, self.tp_size, self.head_dim)
                k_proj_b = torch.tile(k_proj_b, (1, self.kv_replicate, 1))
                v_proj_b = torch.tile(v_proj_b, (1, self.kv_replicate, 1))

        if self.device_mesh is not None:
            q_proj = self._redistribute_dtensor(
                DTensor.from_local(q_proj, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(1)])
            k_proj = self._redistribute_dtensor(
                DTensor.from_local(k_proj, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(1)])
            v_proj = self._redistribute_dtensor(
                DTensor.from_local(v_proj, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(1)])
            o_proj = self._redistribute_dtensor(
                DTensor.from_local(o_proj, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(2)])

            if self.attention_bias:
                q_proj_b = self._redistribute_dtensor(
                    DTensor.from_local(q_proj_b, self.device_mesh, [Replicate(), Replicate()]),
                    [Replicate(), Shard(1)])
                k_proj_b = self._redistribute_dtensor(
                    DTensor.from_local(k_proj_b, self.device_mesh, [Replicate(), Replicate()]),
                    [Replicate(), Shard(1)])
                v_proj_b = self._redistribute_dtensor(
                    DTensor.from_local(v_proj_b, self.device_mesh, [Replicate(), Replicate()]),
                    [Replicate(), Shard(1)])

        qkv_proj = torch.cat((q_proj, k_proj, v_proj), dim=0).view(-1, self.hidden_size).contiguous()
        o_proj = o_proj.contiguous().view(self.hidden_size, -1).contiguous()

        if self.attention_bias:
            qkv_proj_b = torch.cat((q_proj_b, k_proj_b, v_proj_b), dim=0).view(-1).contiguous()
            o_proj_b = o_proj_b.contiguous()
        else:
            qkv_proj_b = None
            o_proj_b = None

        return qkv_proj, qkv_proj_b, o_proj, o_proj_b

    def _process_attention_amax(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        if self.amax_ready:
            qkv_proj_amax = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['qkv_proj_amax']),
                                          torch.bfloat16)
            o_proj_amax = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['o_proj_amax']),
                                        torch.bfloat16)
            if self.device_mesh is not None:
                o_proj_amax = self._redistribute_dtensor(
                    DTensor.from_local(o_proj_amax, self.device_mesh, [Replicate(), Replicate()]),
                    [Replicate(), Shard(0)])
            return qkv_proj_amax, o_proj_amax
        else:
            return None, None

    def _process_ffn_weights_local(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        dt1_1 = self._cast_to(self.source_weights[layer_idx]['fc1_1'], torch.bfloat16)
        dt1_2 = self._cast_to(self.source_weights[layer_idx]['fc1_2'], torch.bfloat16)
        dt2 = self._cast_to(self.source_weights[layer_idx]['fc2'], torch.bfloat16)
        # train/rollout ep_size equal, gather fsdp mesh only
        # same as below
        #   fc1_1 = dt1_1.redistribute(placements=[Replicate(), dt1_1.placements[1]]).to_local()
        #   fc1_2 = dt1_2.redistribute(placements=[Replicate(), dt1_2.placements[1]]).to_local()
        #   fc2 = dt2.redistribute(placements=[Replicate(), dt2.placements[1]]).to_local()
        fc1_1 = DTensor.from_local(local_tensor=dt1_1._local_tensor,
                                   device_mesh=dt1_1.device_mesh,
                                   placements=[dt1_1.placements[0]]).full_tensor()
        fc1_2 = DTensor.from_local(local_tensor=dt1_2._local_tensor,
                                   device_mesh=dt1_2.device_mesh,
                                   placements=[dt1_2.placements[0]]).full_tensor()
        fc2 = DTensor.from_local(local_tensor=dt2._local_tensor,
                                 device_mesh=dt2.device_mesh,
                                 placements=[dt2.placements[0]]).full_tensor()
        # still tp
        if self.share_expert_num > 0:
            share_fc1_1 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc1_1']),
                                        torch.bfloat16)
            share_fc1_2 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc1_2']),
                                        torch.bfloat16)
            share_fc2 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc2']),
                                      torch.bfloat16)
            share_fc1_1 = share_fc1_1.view(2, -1, share_fc1_1.shape[-1])
            share_fc1_2 = share_fc1_2.view(2, -1, share_fc1_2.shape[-1])
            share_fc2 = share_fc2.view(share_fc2.shape[-2], 2, -1).transpose(0, 1)

            share_fc1_1 = self._redistribute_dtensor(
                DTensor.from_local(share_fc1_1, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(1)])
            share_fc1_2 = self._redistribute_dtensor(
                DTensor.from_local(share_fc1_2, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(1)])
            share_fc2 = self._redistribute_dtensor(
                DTensor.from_local(share_fc2, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(2)])
        fc1 = torch.cat((fc1_1, fc1_2), dim=1)
        if self.share_expert_num > 0:
            share_fc1 = torch.cat((share_fc1_1, share_fc1_2), dim=1)
        fc1_weight = fc1.contiguous()
        fc2_weight = fc2.contiguous()
        if self.share_expert_num > 0:
            share_fc1_weight = share_fc1.reshape(share_fc1.shape[0], 2,
                                                 -1, share_fc1.shape[-1]).transpose(0, 1).reshape(
                                                     -1, share_fc1.shape[-1]).contiguous()
            share_fc2_weight = share_fc2.transpose(0, 1).reshape(share_fc2.shape[1], -1).contiguous()
        else:
            share_fc1_weight = None
            share_fc2_weight = None
        return fc1_weight, share_fc1_weight, fc2_weight, share_fc2_weight

    def _process_ffn_weights_interleave(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        dt1_1 = self._cast_to(self.source_weights[layer_idx]['fc1_1'], torch.bfloat16)
        dt1_2 = self._cast_to(self.source_weights[layer_idx]['fc1_2'], torch.bfloat16)
        dt2 = self._cast_to(self.source_weights[layer_idx]['fc2'], torch.bfloat16)
        fc1_1 = DTensor.from_local(local_tensor=dt1_1._local_tensor,
                                   device_mesh=dt1_1.device_mesh,
                                   placements=[dt1_1.placements[0]]).full_tensor()
        fc1_2 = DTensor.from_local(local_tensor=dt1_2._local_tensor,
                                   device_mesh=dt1_2.device_mesh,
                                   placements=[dt1_2.placements[0]]).full_tensor()
        fc2 = DTensor.from_local(local_tensor=dt2._local_tensor,
                                 device_mesh=dt2.device_mesh,
                                 placements=[dt2.placements[0]]).full_tensor()
        fc1_1 = self._get_partial_tensor(fc1_1, 0)
        fc1_2 = self._get_partial_tensor(fc1_2, 0)
        fc2 = self._get_partial_tensor(fc2, 0)
        fc1 = torch.cat((fc1_1, fc1_2), dim=1)
        if self.share_expert_num > 0:
            if self.backend == 'fsdp':
                share_fc1, share_fc2 = self._process_share_ffn_weights_fsdp(layer_idx)
            elif self.backend == 'vescale-fsdp2':
                share_fc1, share_fc2 = self._process_share_ffn_weights_vescale(layer_idx)
                share_fc1 = share_fc1.reshape(share_fc1.shape[0], 2, -1, share_fc1.shape[-1]).transpose(0, 1).reshape(
                    -1, share_fc1.shape[-1]).contiguous()
                share_fc2 = share_fc2.transpose(0, 1).reshape(share_fc2.shape[1], -1).contiguous()
        else:
            share_fc1 = None
            share_fc2 = None
        fc1_weight = fc1.contiguous()
        fc2_weight = fc2.contiguous()
        return fc1_weight, share_fc1, fc2_weight, share_fc2

    def _process_ffn_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        if self.device_mesh is not None and self.source_weights[layer_idx]['fc1_1'] is not None and self.source_weights[
                layer_idx]['fc1_1'].device_mesh.mesh.shape[1] == self.device_mesh.mesh.shape[
                    1] and self.use_ep and not self.enable_actor_critic_spatial_mux:
            return self._process_ffn_weights_local(layer_idx)
        elif self.device_mesh is not None and self.source_weights[layer_idx]['fc1_1'] is not None and \
            self.use_ep and not self.enable_actor_critic_spatial_mux and (self.bind_device_mesh is not None):
            return self._process_ffn_weights_interleave(layer_idx)
        dt1_1 = self.source_weights[layer_idx]['fc1_1']
        dt1_2 = self.source_weights[layer_idx]['fc1_2']
        dt2 = self.source_weights[layer_idx]['fc2']
        fc1_1 = self._cast_to(self._get_full_tensor(dt1_1), torch.bfloat16)
        fc1_2 = self._cast_to(self._get_full_tensor(dt1_2), torch.bfloat16)
        fc2 = self._cast_to(self._get_full_tensor(dt2), torch.bfloat16)
        if self.share_expert_num > 0:
            share_fc1_1 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc1_1']),
                                        torch.bfloat16)
            share_fc1_2 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc1_2']),
                                        torch.bfloat16)
            share_fc2 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc2']),
                                      torch.bfloat16)
            share_fc1_1 = share_fc1_1.view(2, -1, share_fc1_1.shape[-1])
            share_fc1_2 = share_fc1_2.view(2, -1, share_fc1_2.shape[-1])
            share_fc2 = share_fc2.view(share_fc2.shape[-2], 2, -1).transpose(0, 1)

        if self.device_mesh is not None:
            fc1_1 = self._redistribute_dtensor(
                DTensor.from_local(fc1_1, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(0 if self.use_ep or self.moe_num_expert == 0 else 1)])
            fc1_2 = self._redistribute_dtensor(
                DTensor.from_local(fc1_2, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(0 if self.use_ep or self.moe_num_expert == 0 else 1)])
            fc2 = self._redistribute_dtensor(
                DTensor.from_local(fc2, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(0 if self.use_ep else 1 if self.moe_num_expert == 0 else 2)])

            if self.share_expert_num > 0:
                share_fc1_1 = self._redistribute_dtensor(
                    DTensor.from_local(share_fc1_1, self.device_mesh, [Replicate(), Replicate()]),
                    [Replicate(), Shard(1)])
                share_fc1_2 = self._redistribute_dtensor(
                    DTensor.from_local(share_fc1_2, self.device_mesh, [Replicate(), Replicate()]),
                    [Replicate(), Shard(1)])
                share_fc2 = self._redistribute_dtensor(
                    DTensor.from_local(share_fc2, self.device_mesh, [Replicate(), Replicate()]),
                    [Replicate(), Shard(2)])

        fc1 = torch.cat((fc1_1, fc1_2), dim=0 if self.moe_num_expert == 0 else 1)
        if self.share_expert_num > 0:
            share_fc1 = torch.cat((share_fc1_1, share_fc1_2), dim=1)

        if self.moe_num_expert == 0:
            fc1_weight = fc1.contiguous()
            fc2_weight = fc2.contiguous()
            share_fc1_weight = None
            share_fc2_weight = None
        else:
            if self.use_ep:
                fc1_weight = fc1.contiguous()
                fc2_weight = fc2.contiguous()
                if self.share_expert_num > 0:
                    share_fc1_weight = share_fc1.reshape(share_fc1.shape[0], 2, -1, share_fc1.shape[-1]).transpose(
                        0, 1).reshape(-1, share_fc1.shape[-1]).contiguous()
                    share_fc2_weight = share_fc2.transpose(0, 1).reshape(share_fc2.shape[1], -1).contiguous()
                else:
                    share_fc1_weight = None
                    share_fc2_weight = None
            else:
                if self.share_expert_num > 0:
                    fc1_weight = torch.cat((fc1, share_fc1), dim=0)
                    fc2_weight = torch.cat((fc2, share_fc2), dim=0)
                else:
                    fc1_weight = fc1
                    fc2_weight = fc2
                share_fc1_weight = None
                share_fc2_weight = None

        return fc1_weight, share_fc1_weight, fc2_weight, share_fc2_weight

    def _process_share_ffn_weights_vescale(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        share_fc1_1 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc1_1']),
                                    torch.bfloat16)
        share_fc1_2 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc1_2']),
                                    torch.bfloat16)
        share_fc2 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc2']), torch.bfloat16)
        share_fc1_1 = share_fc1_1.view(2, -1, share_fc1_1.shape[-1])
        share_fc1_2 = share_fc1_2.view(2, -1, share_fc1_2.shape[-1])
        share_fc2 = share_fc2.view(share_fc2.shape[-2], 2, -1).transpose(0, 1)
        if self.device_mesh is not None:
            share_fc1_1 = self._redistribute_dtensor(
                DTensor.from_local(share_fc1_1, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(1)])
            share_fc1_2 = self._redistribute_dtensor(
                DTensor.from_local(share_fc1_2, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(1)])
            share_fc2 = self._redistribute_dtensor(
                DTensor.from_local(share_fc2, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(2)])
        share_fc1 = torch.cat((share_fc1_1, share_fc1_2), dim=1)
        return share_fc1, share_fc2

    def _process_share_ffn_weights_fsdp(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        share_fc1_1 = self._cast_to(self.source_weights[layer_idx]['share_fc1_1'], torch.bfloat16)
        share_fc1_2 = self._cast_to(self.source_weights[layer_idx]['share_fc1_2'], torch.bfloat16)
        share_fc2 = self._cast_to(self.source_weights[layer_idx]['share_fc2'], torch.bfloat16)
        share_fc1_1 = DTensor.from_local(local_tensor=share_fc1_1._local_tensor,
                                         device_mesh=share_fc1_1.device_mesh,
                                         placements=[share_fc1_1.placements[0]]).full_tensor()
        share_fc1_2 = DTensor.from_local(local_tensor=share_fc1_2._local_tensor,
                                         device_mesh=share_fc1_2.device_mesh,
                                         placements=[share_fc1_2.placements[0]]).full_tensor()
        share_fc2 = DTensor.from_local(local_tensor=share_fc2._local_tensor,
                                       device_mesh=share_fc2.device_mesh,
                                       placements=[share_fc2.placements[0]]).full_tensor()
        share_fc1_1 = self._get_partial_tensor(share_fc1_1, 0)
        share_fc1_2 = self._get_partial_tensor(share_fc1_2, 0)
        share_fc2 = self._get_partial_tensor(share_fc2, 1)
        share_fc1 = torch.cat((share_fc1_1, share_fc1_2), dim=0)
        return share_fc1, share_fc2

    def _process_ffn_amax(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        if self.amax_ready:
            fc1_amax = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['fc1_amax']), torch.bfloat16)
            fc2_amax = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['fc2_amax']), torch.bfloat16)

            if self.share_expert_num > 0:
                share_fc1_amax = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc1_amax']),
                                               torch.bfloat16)
                share_fc2_amax = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['share_fc2_amax']),
                                               torch.bfloat16)
            else:
                share_fc1_amax, share_fc2_amax = None, None

            if self.device_mesh is not None:
                if self.moe_num_expert > 0:
                    fc1_amax = self._redistribute_dtensor(
                        DTensor.from_local(fc1_amax, self.device_mesh, [Replicate(), Replicate()]),
                        [Replicate(), Shard(0)])
                    fc2_amax = self._redistribute_dtensor(
                        DTensor.from_local(fc2_amax, self.device_mesh, [Replicate(), Replicate()]),
                        [Replicate(), Shard(0)])
                    if self.share_expert_num > 0:
                        share_fc2_amax = self._redistribute_dtensor(
                            DTensor.from_local(share_fc2_amax, self.device_mesh, [Replicate(), Replicate()]),
                            [Replicate(), Shard(0)])
                else:
                    fc2_amax = self._redistribute_dtensor(
                        DTensor.from_local(fc2_amax, self.device_mesh, [Replicate(), Replicate()]),
                        [Replicate(), Shard(0)])

            return fc1_amax, fc2_amax, share_fc1_amax, share_fc2_amax
        else:
            return None, None, None, None

    def _process_gate_weights(self, layer_idx: int) -> torch.Tensor:

        if self.moe_num_expert > 0:
            gate_wg = self._cast_to(
                self._get_full_tensor(self.source_weights[layer_idx]['gate_wg'], None).T.contiguous(), torch.float)
            gate_wg_ema = self._cast_to(
                self._get_full_tensor(self.source_weights[layer_idx]['gate_wg_ema'], None).T.contiguous(), torch.float)
            gate_weight = ((gate_wg + gate_wg_ema) * 0.5).contiguous()
        else:
            gate_weight = None
        if (self.device_mesh is not None) and (not self.enable_actor_critic_spatial_mux) and (self.bind_device_mesh
                                                                                              is not None):
            bind_size = self.bind_device_mesh['bind'].size()
            if 'tp' in self.bind_device_mesh.mesh_dim_names:
                # gen tp > train tp
                train_tp_size = self.tp_size // bind_size
                partion_shape = gate_weight.shape[0] // self.tp_size
                index = torch.arange(gate_weight.shape[0]).reshape(train_tp_size, bind_size,
                                                                   partion_shape).permute(1, 0, 2).reshape(-1)
                gate_weight = gate_weight[index].contiguous()
            elif 'gen_tp' in self.bind_device_mesh.mesh_dim_names:
                # gen tp < train tp
                train_tp_size = self.tp_size * bind_size
                partion_shape = gate_weight.shape[0] // train_tp_size
                index = torch.arange(gate_weight.shape[0]).reshape(bind_size, self.tp_size,
                                                                   partion_shape).permute(1, 0, 2).reshape(-1)
                gate_weight = gate_weight[index].contiguous()
        return gate_weight

    def _process_vwn_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:

        vwn_weights = [None] * 15
        if self.has_over_encoding:
            vwn_weights = (self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn0_static_alpha']),
                                         torch.bfloat16),
                           self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn0_static_beta']),
                                         torch.bfloat16),
                           self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn0_dynamic_alpha']),
                                         torch.bfloat16).transpose(0, 1).contiguous(),
                           self._cast_to(
                               self._get_full_tensor(self.source_weights[layer_idx]['vwn0_dynamic_alpha_scale']),
                               torch.bfloat16),
                           self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn0_dynamic_beta']),
                                         torch.bfloat16).transpose(0, 1).contiguous(),
                           self._cast_to(
                               self._get_full_tensor(self.source_weights[layer_idx]['vwn0_dynamic_beta_scale']),
                               torch.bfloat16),
                           self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn0_layer_norm']),
                                         torch.bfloat16),
                           self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn1_static_alpha']),
                                         torch.bfloat16),
                           self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn1_static_beta']),
                                         torch.bfloat16),
                           self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn1_dynamic_alpha']),
                                         torch.bfloat16).transpose(0, 1).contiguous(),
                           self._cast_to(
                               self._get_full_tensor(self.source_weights[layer_idx]['vwn1_dynamic_alpha_scale']),
                               torch.bfloat16),
                           self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn1_dynamic_beta']),
                                         torch.bfloat16).transpose(0, 1).contiguous(),
                           self._cast_to(
                               self._get_full_tensor(self.source_weights[layer_idx]['vwn1_dynamic_beta_scale']),
                               torch.bfloat16),
                           self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn1_layer_norm']),
                                         torch.bfloat16),
                           self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['vwn_extra_layernorm']),
                                         torch.bfloat16))

        return vwn_weights

    def _process_mtp_weights(self, mtp_idx: int) -> Tuple[torch.Tensor, ...]:

        draft_enorm_weight, draft_hnorm_weight, draft_ln_f_weight, draft_token_proj_w, \
        draft_token_proj_b, static_reduce, draft_vwn_static_alpha, draft_vwn_static_beta, \
        draft_vwn_dynamic_alpha, draft_vwn_dynamic_alpha_scale, draft_vwn_dynamic_beta, \
        draft_vwn_dynamic_beta_scale, draft_vwn_layernorm_weight = [None] * 13

        if mtp_idx == 0:
            if self.has_over_encoding:
                static_reduce = self._cast_to(self._get_full_tensor(self.source_weights[mtp_idx]['static_reduce']),
                                              torch.bfloat16)
            return static_reduce

        draft_enorm_weight = self._cast_to(self._get_full_tensor(self.source_weights[mtp_idx]['draft_e_norm']),
                                           torch.bfloat16)[None, :]
        draft_hnorm_weight = self._cast_to(self._get_full_tensor(self.source_weights[mtp_idx]['draft_h_norm']),
                                           torch.bfloat16)[None, :]
        draft_ln_f_weight = self._cast_to(self._get_full_tensor(self.source_weights[mtp_idx]['draft_ln_f']),
                                          torch.bfloat16)[None, :]
        draft_token_proj_w = self._cast_to(self._get_full_tensor(self.source_weights[mtp_idx]['draft_token_proj_w']),
                                           torch.bfloat16)
        draft_token_proj_b = self._cast_to(self._get_full_tensor(self.source_weights[mtp_idx]['draft_token_proj_b']),
                                           torch.bfloat16)

        if self.has_over_encoding:
            static_reduce = self._cast_to(self._get_full_tensor(self.source_weights[mtp_idx]['static_reduce']),
                                          torch.bfloat16)
            draft_vwn_static_alpha = self._cast_to(
                self._get_full_tensor(self.source_weights[mtp_idx]['draft_vwn_static_alpha']), torch.bfloat16)
            draft_vwn_static_beta = self._cast_to(
                self._get_full_tensor(self.source_weights[mtp_idx]['draft_vwn_static_beta']), torch.bfloat16)
            draft_vwn_dynamic_alpha = self._cast_to(
                self._get_full_tensor(self.source_weights[mtp_idx]['draft_vwn_dynamic_alpha']),
                torch.bfloat16).transpose(0, 1).contiguous()
            draft_vwn_dynamic_alpha_scale = self._cast_to(
                self._get_full_tensor(self.source_weights[mtp_idx]['draft_vwn_dynamic_alpha_scale']), torch.bfloat16)
            draft_vwn_dynamic_beta = self._cast_to(
                self._get_full_tensor(self.source_weights[mtp_idx]['draft_vwn_dynamic_beta']),
                torch.bfloat16).transpose(0, 1).contiguous()
            draft_vwn_dynamic_beta_scale = self._cast_to(
                self._get_full_tensor(self.source_weights[mtp_idx]['draft_vwn_dynamic_beta_scale']), torch.bfloat16)
            draft_vwn_layernorm_weight = self._cast_to(
                self._get_full_tensor(self.source_weights[mtp_idx]['draft_vwn_layernorm_weight']), torch.bfloat16)

        return draft_enorm_weight, draft_hnorm_weight, draft_ln_f_weight, draft_token_proj_w, \
               draft_token_proj_b, static_reduce, draft_vwn_static_alpha, draft_vwn_static_beta, \
               draft_vwn_dynamic_alpha, draft_vwn_dynamic_alpha_scale, draft_vwn_dynamic_beta, \
               draft_vwn_dynamic_beta_scale, draft_vwn_layernorm_weight

    def _process_quant_wfp8(self, *args) -> Tuple[torch.Tensor, ...]:

        if self.quant_mode != "WFP8":
            return *args, None

        def wfp8_quantizer(weight: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            if weight.dim() == 2:
                fp8_scale = 1 / torch.classes.XGPT.Fp8GemmTestOp().GetPerTensorScale(weight, 0)
                fp8_weight = torch.classes.XGPT.Fp8GemmTestOp().Quant(weight, 1 / fp8_scale)
            elif weight.dim() == 3:
                fp8_scale, fp8_weight = zip(
                    *[wfp8_quantizer(local_weight) for local_weight in torch.unbind(weight, dim=0)])
                fp8_scale = torch.stack(fp8_scale)
                fp8_weight = torch.stack([weight.view(torch.int8) for weight in fp8_weight]).view(torch.float8_e4m3fn)
            return fp8_scale, fp8_weight

        fp8_weights = []
        fp8_qscale = []
        for bf16_weight in args:
            if bf16_weight is None:
                fp8_weights.append(None)
                continue
            fp8_scale, fp8_weight = wfp8_quantizer(bf16_weight)
            fp8_weights.append(fp8_weight)
            fp8_qscale.extend([None, None, fp8_scale])

        if self.share_expert_num > 0 and self.use_ep:
            fp8_qscale[12] = fp8_qscale[14]
            fp8_qscale[13] = fp8_qscale[17]
            fp8_qscale = fp8_qscale[:14]

        return *fp8_weights, fp8_qscale

    def _process_quant_w4a8(self, dense_gemm_weights: List[torch.Tensor], dense_gemm_amax: List[torch.Tensor],
                            dense_gemm_quant_ratio: List[torch.Tensor], group_gemm_weights: List[torch.Tensor],
                            group_gemm_amax: List[torch.Tensor], group_gemm_quant_ratio: List[torch.Tensor]):

        dense_gemm_w8a8_weights = []
        dense_gemm_w8a8_smooth_quant_scale = []
        dense_gemm_w8a8_weight_qscale = []
        group_gemm_w4a8_weights = []
        group_gemm_w4a8_smooth_quant_scale = []
        group_gemm_w4a8_i8_weight_qscale = []
        group_gemm_w4a8_i4_weight_qscale_zero = []

        for weight, amax, ratio in zip(dense_gemm_weights, dense_gemm_amax, dense_gemm_quant_ratio):
            ratio = ratio.item() if ratio is not None and self.quant_ratio_ready else 0.5
            if weight is None:
                smooth_quant_scale, weight, i8_weight_qscale = None, None, None
            else:
                smooth_quant_scale, weight, i8_weight_qscale = quant_gemm_weight_w8a8(weight, amax, ratio)
                smooth_quant_scale = (1.0 / smooth_quant_scale).to(smooth_quant_scale.dtype)
            dense_gemm_w8a8_weights.append(weight)
            dense_gemm_w8a8_smooth_quant_scale.append(smooth_quant_scale)
            dense_gemm_w8a8_weight_qscale.append(i8_weight_qscale)

        for weight, amax, ratio in zip(group_gemm_weights, group_gemm_amax, group_gemm_quant_ratio):
            is_dense = False
            ratio = ratio.item() if ratio is not None and self.quant_ratio_ready else 0.5
            if weight.dim() == 2:
                weight = weight.unsqueeze(0)
                is_dense = True
            smooth_quant_scale, i4_weight, i8_weight_qscale, i4_weight_qscale, i4_weight_qzero = \
                quant_group_gemm_weight_w4a8(weight, amax, ratio)
            smooth_quant_scale = (1.0 / smooth_quant_scale).to(smooth_quant_scale.dtype)
            i4_scale_zero = torch.stack([i4_weight_qscale, -i4_weight_qscale * i4_weight_qzero], dim=-1)

            # need process
            from xperf_gpt.utils.quant_utils import get_w4a8_weight_preprocessor
            i4_weight_out = []
            i8_weight_qscale_out = []
            i4_scale_zero_out = []
            e, n, k = i4_weight.shape
            group_size = 64
            for i in range(e):
                preprocessor = get_w4a8_weight_preprocessor(group_size, k, n)
                i4_weight_out.append(preprocessor.convert_weight(i4_weight[i].t()))
                i8_weight_qscale_out.append(preprocessor.convert_perchannel_scale(i8_weight_qscale[i]))
                i4_scale_zero_out.append(preprocessor.convert_group_scale(i4_scale_zero[i]))

            group_gemm_w4a8_weights.append(torch.stack(i4_weight_out, dim=0))
            group_gemm_w4a8_smooth_quant_scale.append(smooth_quant_scale)
            group_gemm_w4a8_i8_weight_qscale.append(torch.stack(i8_weight_qscale_out, dim=0))
            group_gemm_w4a8_i4_weight_qscale_zero.append(torch.stack(i4_scale_zero_out, dim=0))

            if is_dense:
                group_gemm_w4a8_weights[-1] = group_gemm_w4a8_weights[-1].squeeze(0)
                group_gemm_w4a8_smooth_quant_scale[-1] = group_gemm_w4a8_smooth_quant_scale[-1].squeeze(0)
                group_gemm_w4a8_i8_weight_qscale[-1] = group_gemm_w4a8_i8_weight_qscale[-1].squeeze(0)
                group_gemm_w4a8_i4_weight_qscale_zero[-1] = group_gemm_w4a8_i4_weight_qscale_zero[-1].squeeze(0)

        return dense_gemm_w8a8_weights, dense_gemm_w8a8_smooth_quant_scale, dense_gemm_w8a8_weight_qscale, \
            group_gemm_w4a8_weights, group_gemm_w4a8_smooth_quant_scale, group_gemm_w4a8_i8_weight_qscale, \
            group_gemm_w4a8_i4_weight_qscale_zero


class FSDPVLMWeightsAdapter(WeightsAdapter, AdapterProtocol):
    _support_model_type = ["seed_vl"]

    def __init__(self, model_config: PretrainedConfig, quant_mode: str, enable_actor_critic_spatial_mux: bool) -> None:

        self.model_config = model_config
        self.quant_mode = quant_mode
        self.enable_actor_critic_spatial_mux = enable_actor_critic_spatial_mux
        self.source_weights: Dict[str, Union[torch.Tensor, DTensor]] = {}

    def get_model_info(self, xperf_model: torch.nn.Module) -> None:

        config = self.model_config
        self.num_layers = config.depth
        self.hidden_size = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads
        self.num_kv_heads = config.num_heads
        self.ffn_intermediate_dim = (int(config.embed_dim * config.mlp_ratio) + 63) // 64 * 64
        self.padding_size = self.ffn_intermediate_dim - int(config.embed_dim * config.mlp_ratio)
        self.qkv_bias = config.qkv_bias
        self.use_xperf_gpt = xperf_model.use_xperf_gpt

    def load_from_state_dict(self, state_dict: Dict[str, Union[torch.Tensor, DTensor]], prefix: str) -> None:
        loader = partial(self._pop_with_fallback, state_dict, prefix)

        self.source_weights = {
            'patch_emb_proj_weight': loader("vision_encoder.patch_embed.proj.weight"),
            'patch_emb_proj_bias': loader("vision_encoder.patch_embed.proj.bias"),
            'ln_vision_weight': loader("ln_vision.weight"),
            'ln_vision_bias': loader("ln_vision.bias"),
            'seed_proj0_weight': loader("multi_modal_projector.0.weight"),
            'seed_proj0_bias': loader("multi_modal_projector.0.bias"),
            'seed_proj2_weight': loader("multi_modal_projector.2.weight"),
            'seed_proj2_bias': loader("multi_modal_projector.2.bias"),
        }

        for layer_idx in range(self.num_layers):
            layer_key = f"vision_encoder.blocks.{layer_idx}"

            ln_1 = loader(f"{layer_key}.norm1.weight")
            ln_1_bias = loader(f"{layer_key}.norm1.bias")
            ln_1 = torch.cat((ln_1, ln_1_bias), dim=0) if ln_1_bias is not None else ln_1

            ln_2 = loader(f"{layer_key}.norm2.weight")
            ln_2_bias = loader(f"{layer_key}.norm2.bias")
            ln_2 = torch.cat((ln_2, ln_2_bias), dim=0) if ln_2_bias is not None else ln_2

            self.source_weights[layer_idx] = {
                'ln_1': ln_1,
                'qkv_proj': loader(f"{layer_key}.attn.qkv.weight"),
                'q_proj_b': loader(f"{layer_key}.attn.q_bias"),
                'v_proj_b': loader(f"{layer_key}.attn.v_bias"),
                'o_proj': loader(f"{layer_key}.attn.proj.weight"),
                'o_proj_b': loader(f"{layer_key}.attn.proj.bias"),
                'ln_2': ln_2,
                'fc1': loader(f"{layer_key}.mlp.fc1.weight"),
                'fc1_b': loader(f"{layer_key}.mlp.fc1.bias"),
                'fc2': loader(f"{layer_key}.mlp.fc2.weight"),
                'fc2_b': loader(f"{layer_key}.mlp.fc2.bias"),
            }

    def process_and_assign_weights(self, xperf_model: torch.nn.Module) -> None:

        xperf_weights = xperf_model.visual_encoder.module.weights

        # disable small vit tp
        self.enable_tp = True
        if self.hidden_size < 2048:
            self.tp_size = 1
            self.tp_rank = 0
            self.enable_tp = False

        def assign_weights(binding_weights, layer_idx=None):
            for xperf_weight, weight, name in binding_weights:
                if weight is None:
                    continue
                dst = getattr(xperf_weight, name)
                if layer_idx is not None:
                    dst = dst[layer_idx]
                    name = f"{layer_idx}_{name}"
                self._assign_and_validate(src=weight, dst=dst, name=name)

        patch_emb_proj_weight, patch_emb_proj_bias, ln_vision_weight, ln_vision_bias, \
        seed_proj0_weight, seed_proj0_bias, seed_proj2_weight, seed_proj2_bias = self._process_top_level_weights()
        binding_weights = [(xperf_weights.module_weight, [patch_emb_proj_weight, patch_emb_proj_bias], "patch_embed"),
                           (xperf_weights.module_weight, [ln_vision_weight, ln_vision_bias], "ln_vision"),
                           (xperf_weights.module_weight,
                            [seed_proj0_weight, seed_proj0_bias, seed_proj2_weight, seed_proj2_bias], "seed_proj")]
        assign_weights(binding_weights)

        for layer_idx in range(self.num_layers):
            ln_1_weight, ln_2_weight = self._process_layernorm_weights(layer_idx)
            qkv_weight, qkv_bias, o_weight, o_bias = self._process_attention_weights(layer_idx)
            fc1_weight, fc1_bias, fc2_weight, fc2_bias = self._process_ffn_weights(layer_idx)

            binding_weights = [
                (xperf_weights.layer_weight, ln_1_weight, "norm0_gamma_beta"),
                (xperf_weights.layer_weight, ln_2_weight, "norm1_gamma_beta"),
                (xperf_weights.layer_weight, qkv_weight, "qkv_proj_weight"),
                (xperf_weights.layer_weight, qkv_bias, "qkv_proj_bias"),
                (xperf_weights.layer_weight, o_weight, "out_proj_weight"),
                (xperf_weights.layer_weight, o_bias, "out_proj_bias"),
                (xperf_weights.layer_weight, fc1_weight, "FFN0_weight"),
                (xperf_weights.layer_weight, fc1_bias, "FFN0_bias"),
                (xperf_weights.layer_weight, fc2_weight, "FFN1_weight"),
                (xperf_weights.layer_weight, fc2_bias, "FFN1_bias"),
            ]
            assign_weights(binding_weights, layer_idx)

        if self.use_xperf_gpt:
            xperf_weights.prepare_infer_weights()
            xperf_model.visual_encoder.module.layer_weight = xperf_weights.layers_weight
            xperf_model.visual_encoder.module.patch_embed.proj.weight.data = xperf_weights.module_weight.patch_embed[0]
            xperf_model.visual_encoder.module.patch_embed.proj.bias.data = xperf_weights.module_weight.patch_embed[1]
        else:
            xperf_model.visual_encoder.module.custom_decoder = None
            xperf_model.visual_encoder.module.build_decoder()
            xperf_model.visual_encoder.module.custom_decoder.cuda()
        xperf_model.ln_vision.weight.data = xperf_model.visual_encoder.module.weights.module_weight.ln_vision[0]
        xperf_model.ln_vision.bias.data = xperf_model.visual_encoder.module.weights.module_weight.ln_vision[1]
        xperf_model.seed_proj[0].weight.data = xperf_model.visual_encoder.module.weights.module_weight.seed_proj[0]
        xperf_model.seed_proj[0].bias.data = xperf_model.visual_encoder.module.weights.module_weight.seed_proj[1]
        xperf_model.seed_proj[2].weight.data = xperf_model.visual_encoder.module.weights.module_weight.seed_proj[2]
        xperf_model.seed_proj[2].bias.data = xperf_model.visual_encoder.module.weights.module_weight.seed_proj[3]
        self.source_weights.clear()

    def _process_top_level_weights(self) -> Tuple[torch.Tensor, ...]:

        patch_emb_proj_weight = self._cast_to(self._get_full_tensor(self.source_weights['patch_emb_proj_weight']),
                                              torch.bfloat16)
        patch_emb_proj_bias = self._cast_to(self._get_full_tensor(self.source_weights['patch_emb_proj_bias']),
                                            torch.bfloat16)

        ln_vision_weight = self._cast_to(self._get_full_tensor(self.source_weights['ln_vision_weight']), torch.bfloat16)
        ln_vision_bias = self._cast_to(self._get_full_tensor(self.source_weights['ln_vision_bias']), torch.bfloat16)

        seed_proj0_weight = self._cast_to(self._get_full_tensor(self.source_weights['seed_proj0_weight']),
                                          torch.bfloat16)
        seed_proj0_bias = self._cast_to(self._get_full_tensor(self.source_weights['seed_proj0_bias']), torch.bfloat16)

        seed_proj2_weight = self._cast_to(self._get_full_tensor(self.source_weights['seed_proj2_weight']),
                                          torch.bfloat16)
        seed_proj2_bias = self._cast_to(self._get_full_tensor(self.source_weights['seed_proj2_bias']), torch.bfloat16)

        return patch_emb_proj_weight, patch_emb_proj_bias, ln_vision_weight, ln_vision_bias, \
               seed_proj0_weight, seed_proj0_bias, seed_proj2_weight, seed_proj2_bias

    def _process_layernorm_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:

        ln_1_weight = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['ln_1']),
                                    torch.bfloat16).reshape(-1, self.hidden_size)

        ln_2_weight = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['ln_2']),
                                    torch.bfloat16).reshape(-1, self.hidden_size)

        return ln_1_weight, ln_2_weight

    def _process_attention_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:

        qkv_proj = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['qkv_proj']), torch.bfloat16)
        o_proj = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['o_proj']), torch.bfloat16)
        q_proj, k_proj, v_proj = torch.split(qkv_proj, qkv_proj.shape[0] // 3, dim=0)

        if self.qkv_bias:
            q_proj_b = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['q_proj_b']), torch.bfloat16)
            v_proj_b = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['v_proj_b']), torch.bfloat16)
            o_proj_b = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['o_proj_b']), torch.bfloat16)

        q_proj = q_proj.view(self.num_kv_heads, -1, self.head_dim, self.hidden_size).transpose(0, 1)
        k_proj = k_proj.view(1, self.num_kv_heads, self.head_dim, self.hidden_size)
        v_proj = v_proj.view(1, self.num_kv_heads, self.head_dim, self.hidden_size)
        o_proj = o_proj.view(self.hidden_size, self.num_kv_heads, -1, self.head_dim).transpose(1, 2)

        if self.qkv_bias:
            q_proj_b = q_proj_b.view(self.num_kv_heads, -1, self.head_dim).transpose(0, 1)
            v_proj_b = v_proj_b.view(1, self.num_kv_heads, self.head_dim)

        if self.device_mesh is not None and self.enable_tp:
            q_proj = self._redistribute_dtensor(
                DTensor.from_local(q_proj, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(1)])
            k_proj = self._redistribute_dtensor(
                DTensor.from_local(k_proj, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(1)])
            v_proj = self._redistribute_dtensor(
                DTensor.from_local(v_proj, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(1)])
            o_proj = self._redistribute_dtensor(
                DTensor.from_local(o_proj, self.device_mesh, [Replicate(), Replicate()]),
                [Replicate(), Shard(2)])

            if self.qkv_bias:
                q_proj_b = self._redistribute_dtensor(
                    DTensor.from_local(q_proj_b, self.device_mesh, [Replicate(), Replicate()]),
                    [Replicate(), Shard(1)])
                v_proj_b = self._redistribute_dtensor(
                    DTensor.from_local(v_proj_b, self.device_mesh, [Replicate(), Replicate()]),
                    [Replicate(), Shard(1)])

        qkv_proj = torch.cat((q_proj, k_proj, v_proj), dim=0).view(-1, self.hidden_size).contiguous()
        o_proj = o_proj.contiguous().view(self.hidden_size, -1).contiguous()

        if self.qkv_bias:
            qkv_proj_b = torch.cat((q_proj_b, torch.zeros_like(q_proj_b), v_proj_b), dim=0).view(-1).contiguous()
            o_proj_b = o_proj_b.contiguous()
        else:
            qkv_proj_b = None
            o_proj_b = None

        return qkv_proj, qkv_proj_b, o_proj, o_proj_b

    def _process_ffn_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:

        fc1 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['fc1']), torch.bfloat16)
        fc2 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['fc2']), torch.bfloat16)
        fc1_b = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['fc1_b']), torch.bfloat16)
        fc2_b = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['fc2_b']), torch.bfloat16)

        if self.padding_size > 0 and self.use_xperf_gpt:
            fc1 = torch.nn.functional.pad(fc1, (0, 0, 0, self.padding_size))
            fc2 = torch.nn.functional.pad(fc2, (0, self.padding_size))
            fc1_b = torch.nn.functional.pad(fc1_b, (0, self.padding_size))

        if self.device_mesh is not None and self.enable_tp:
            fc1 = self._redistribute_dtensor(DTensor.from_local(fc1, self.device_mesh,
                                                                [Replicate(), Replicate()]),
                                             [Replicate(), Shard(0)])
            fc2 = self._redistribute_dtensor(DTensor.from_local(fc2, self.device_mesh,
                                                                [Replicate(), Replicate()]),
                                             [Replicate(), Shard(1)])
            fc1_b = self._redistribute_dtensor(DTensor.from_local(fc1_b, self.device_mesh,
                                                                  [Replicate(), Replicate()]),
                                               [Replicate(), Shard(0)])
            fc2_b = self._redistribute_dtensor(DTensor.from_local(fc2_b, self.device_mesh,
                                                                  [Replicate(), Replicate()]),
                                               [Replicate(), Shard(0)])

        fc1_weight = fc1.contiguous()
        fc2_weight = fc2.contiguous()
        fc1_bias = fc1_b.contiguous()
        fc2_bias = fc2_b.contiguous()

        return fc1_weight, fc1_bias, fc2_weight, fc2_bias
