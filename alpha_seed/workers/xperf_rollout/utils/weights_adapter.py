import torch
import logging
import torch.distributed as dist
from functools import partial
from transformers import PretrainedConfig
from typing import Tuple, Union, List, Dict, Optional, Protocol
from torch.distributed._tensor import DTensor, Shard, Replicate

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
                 enable_actor_critic_spatial_mux: bool = False) -> None:
        self.adapter: AdapterProtocol = None
        self.vit_adapter: AdapterProtocol = None

        if model_config.model_type in FSDPLLMWeightsAdapter._support_model_type:
            self.adapter = FSDPLLMWeightsAdapter(model_config, quant_mode, enable_actor_critic_spatial_mux)
        elif model_config.model_type in FSDPVLMWeightsAdapter._support_model_type:
            self.adapter = FSDPLLMWeightsAdapter(model_config.text_config, quant_mode, enable_actor_critic_spatial_mux)
            self.vit_adapter = FSDPVLMWeightsAdapter(model_config.vision_config, quant_mode,
                                                     enable_actor_critic_spatial_mux)
        assert self.adapter is not None, f"Unsupported model type {model_config.model_type}"

    def __call__(self,
                 xperf_llm: torch.nn.Module,
                 xperf_vit: torch.nn.Module = None,
                 state_dict: Dict[str, Union[torch.Tensor, DTensor]] = None,
                 device_mesh: Optional[Dict[str, torch.distributed.ProcessGroup]] = None,
                 prefix: Optional[str] = None) -> None:

        if xperf_vit is not None:
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

    def _get_full_tensor(self, tensor: DTensor) -> torch.Tensor:

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

    def _redistribute_dtensor(self, tensor: DTensor, placements: List[Union[Shard, Replicate]]) -> torch.Tensor:

        return tensor._local_tensor if not self.device_mesh else tensor.redistribute(
            device_mesh=self.device_mesh, placements=placements)._local_tensor

    def _cast_to(self, tensor: Union[torch.Tensor, DTensor], dtype: torch.dtype) -> Union[torch.Tensor, DTensor]:

        return tensor.to(dtype)

    def _assign_and_validate(self, src: Union[torch.Tensor, List[torch.Tensor]],
                             dst: Union[torch.Tensor, List[torch.Tensor]], name: str) -> None:

        if src is None:
            return
        if isinstance(src, list):
            for i, (src_item, dst_item) in enumerate(zip(src, dst)):
                self._assign_and_validate(src_item, dst_item, f"{name}_{i}")
        else:
            assert src.shape == dst.shape or src.numel() == dst.numel(
            ), f"Weight {name} shape mismatch: src {src.shape} vs dst {dst.shape}"
            assert not torch.isnan(src).any(), f"Weight {name} contains NaN values"
            dst.data = src


class FSDPLLMWeightsAdapter(WeightsAdapter, AdapterProtocol):
    _support_model_type = ["seed_p6", "seed_p6dense", "seed_p7", "seed_m8", "seed_m10"]

    def __init__(self, model_config: PretrainedConfig, quant_mode: str, enable_actor_critic_spatial_mux: bool) -> None:

        self.model_config = model_config
        self.quant_mode = quant_mode
        self.enable_actor_critic_spatial_mux = enable_actor_critic_spatial_mux
        self.source_weights: Dict[str, Union[torch.Tensor, DTensor]] = {}

    def get_model_info(self, xperf_model: torch.nn.Module) -> None:

        config = self.model_config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.num_kv_heads = config.num_key_value_heads
        self.kv_replicate = self.tp_size // self.num_kv_heads if self.tp_size % self.num_kv_heads == 0 else 1
        self.num_layers = config.num_hidden_layers
        self.attention_bias = config.attention_bias
        self.moe_num_expert = getattr(config, "moe_num_expert", 0)
        self.share_expert_num = getattr(config, "share_expert_num", 0)
        self.use_query_layernorm = getattr(config, "use_query_layernorm", False)
        self.use_key_layernorm = getattr(config, "use_key_layernorm", False)
        self.use_qk_rmsnorm = getattr(config, "use_qk_rmsnorm", False)
        self.use_context_groupnorm = getattr(config, "use_context_groupnorm", False)
        self.use_attention_output_layernorm = getattr(config, "use_attention_output_layernorm", False)
        self.mtp_n_heads = getattr(config, "mtp_n_heads", 1)
        self.use_ep = getattr(xperf_model, "use_ep", False)
        self.use_mtp = getattr(xperf_model, "use_mtp", False)
        self.vocab_tp = getattr(xperf_model, "vocab_tp", False)
        if self.use_mtp:
            self.num_layers = self.num_layers + self.mtp_n_heads - 1

    def load_from_state_dict(self, state_dict: Dict[str, Union[torch.Tensor, DTensor]], prefix: str) -> None:

        loader = partial(self._pop_with_fallback, state_dict, prefix)

        self.source_weights['wte'] = loader("transformer.wte.weight", "model.embed_tokens.weight",
                                            "transformer.embed_tokens.weight")
        self.source_weights['lm_head'] = loader("lm_head.weight")
        ln_f = loader("transformer.model.mtp_ce_norms.0.head_ln.weight", "transformer.norm.weight", "model.norm.weight",
                      "transformer.ln_f.weight")
        ln_f_bias = loader("transformer.ln_f.bias")
        self.source_weights['ln_f'] = torch.cat((ln_f, ln_f_bias), dim=0) if ln_f_bias is not None else ln_f

        for layer_idx in range(self.num_layers):
            if self.model_config.model_type == "seed_p6dense":
                layer_key = f"model.layers.{layer_idx}"
            elif self.model_config.model_type == "seed_m10":
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

            context_norm = loader(f"{layer_key}.attn.context_norm.weight")
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
                    loader(f"{layer_key}.mlp.moe.experts_share.fc2", f"{layer_key}.mlp.moe.shared_experts.down_proj")
            }

    def process_and_assign_weights(self, xperf_model: torch.nn.Module) -> None:

        xperf_weights = xperf_model.weights

        def assign_weights(binding_weights, layer_idx=None):
            for xperf_weight, weight, name in binding_weights:
                if weight is None:
                    continue
                dst = getattr(xperf_weight, name)
                if layer_idx is not None:
                    dst = dst[layer_idx]
                    name = f"{layer_idx}_{name}"
                self._assign_and_validate(src=weight, dst=dst, name=name)

        wte_weight, lm_head_weight, ln_f_weight = self._process_top_level_weights()
        binding_weights = [(xperf_weights.module_weight, wte_weight, "wte_weight"),
                           (xperf_weights.module_weight, lm_head_weight, "lm_head_weight"),
                           (xperf_weights.module_weight, ln_f_weight, "ln_f_weight")]
        assign_weights(binding_weights)

        for layer_idx in range(self.num_layers):
            ln_1_weight, ln_2_weight, query_norm_weight, key_norm_weight, context_norm_weight, attn_output_norm_weight, ffn_output_norm_weight = self._process_layernorm_weights(
                layer_idx)
            qkv_weight, qkv_bias, o_weight, o_bias = self._process_attention_weights(layer_idx)
            fc1_weight, share_fc1_weight, fc2_weight, share_fc2_weight = self._process_ffn_weights(layer_idx)
            qkv_weight, o_weight, fc1_weight, fc2_weight, share_fc1_weight, share_fc2_weight, wfp8_qscale = self._process_quant_wfp8(
                qkv_weight, o_weight, fc1_weight, fc2_weight, share_fc1_weight, share_fc2_weight)
            gate_wg_weight = self._process_gate_weights(layer_idx)

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
                (xperf_weights.layer_weight, fc1_weight, "FFN0_weight"),
                (xperf_weights.layer_weight, share_fc1_weight, "FFN0_share_weight"),
                (xperf_weights.layer_weight, fc2_weight, "FFN1_weight"),
                (xperf_weights.layer_weight, share_fc2_weight, "FFN1_share_weight"),
                (xperf_weights.layer_weight, gate_wg_weight, "moe_gate_weight"),
                (xperf_weights.quant_weight, wfp8_qscale, "wfp8_qscale"),
            ]
            assign_weights(binding_weights, layer_idx)

        xperf_weights.prepare_infer_weights()
        xperf_model.layer_weight = xperf_weights.layers_weight
        xperf_model.layernorm_weight = xperf_weights.layernorm_weight
        xperf_model.lm_head_weight = xperf_weights.lm_head_weight
        xperf_model.wte_weight = xperf_weights.wte_weight
        self.source_weights.clear()

    def _process_top_level_weights(self) -> Tuple[torch.Tensor, ...]:

        wte = self._cast_to(self._get_full_tensor(self.source_weights['wte']), torch.bfloat16)
        ln_f = self._cast_to(self._get_full_tensor(self.source_weights['ln_f']), torch.bfloat16)

        ln_f_weight = ln_f.view(-1, self.hidden_size).contiguous()

        if self.device_mesh is not None and self.vocab_tp:
            wte = DTensor.from_local(wte, self.device_mesh, [Replicate(), Replicate()])
            wte_weight = self._redistribute_dtensor(wte, [Replicate(), Shard(1)])
            lm_head_weight = self._redistribute_dtensor(wte, [Replicate(), Shard(0)])
        else:
            wte_weight = wte.contiguous()
            lm_head_weight = wte.contiguous()

        return wte_weight, lm_head_weight, ln_f_weight

    def _process_layernorm_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:

        ln_1_weight = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['ln_1']),
                                    torch.bfloat16).reshape(-1, self.hidden_size)

        ln_2_weight = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['ln_2']),
                                    torch.bfloat16).reshape(-1, self.hidden_size)

        if self.use_query_layernorm or self.use_qk_rmsnorm:
            query_norm_weight = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['query_norm']),
                                              torch.bfloat16).reshape(-1, self.head_dim)
        else:
            query_norm_weight = None

        if self.use_key_layernorm or self.use_qk_rmsnorm:
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

    def _process_attention_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:

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

    def _process_ffn_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:

        fc1_1 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['fc1_1']), torch.bfloat16)
        fc1_2 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['fc1_2']), torch.bfloat16)
        fc2 = self._cast_to(self._get_full_tensor(self.source_weights[layer_idx]['fc2']), torch.bfloat16)

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

    def _process_gate_weights(self, layer_idx: int) -> torch.Tensor:

        if self.moe_num_expert > 0:
            gate_wg = self._cast_to(
                self._get_full_tensor(self.source_weights[layer_idx]['gate_wg']).T.contiguous(), torch.float)
            gate_wg_ema = self._cast_to(
                self._get_full_tensor(self.source_weights[layer_idx]['gate_wg_ema']).T.contiguous(), torch.float)
            gate_weight = ((gate_wg + gate_wg_ema) * 0.5).contiguous()
        else:
            gate_weight = None

        return gate_weight

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

        xperf_weights.prepare_infer_weights()
        xperf_model.visual_encoder.module.layer_weight = xperf_weights.layers_weight
        xperf_model.visual_encoder.module.patch_embed.proj.weight.data = xperf_weights.module_weight.patch_embed[0]
        xperf_model.visual_encoder.module.patch_embed.proj.bias.data = xperf_weights.module_weight.patch_embed[1]
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

        if self.padding_size > 0:
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
