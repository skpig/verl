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
Contains utilities to bind weights to XPerfGPT-Triton. It is model agnostic
"""
import os
import torch
import torch.distributed

from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
import logging

try:
    from xperf_gpt_triton.builder.model_builder import build_model
except Exception as e:
    logging.warning(f"Failed to import xperf_gpt_triton: {e}")
    build_model = None


def init_inference_triton(**kwargs):
    """Transform messy init_inference_kwargs to clean xperf_triton parameters.
    
    This function maps parameters from alpha-seed's init_inference call to XPerfTritonModelConfig.
    The xperf_triton_cfg dict has the highest priority and can override any field.
    
    Args:
        **kwargs: Arguments from init_inference including:
            - model_config_path: Path to model config JSON
            - dtype, slot_block_size, max_total_tokens, max_batch_size, max_length
            - vanilla_checkpoint_path, preshard_checkpoint_path  
            - enable_cuda_graph: Maps to use_cuda_graph
            - use_vllm: Maps to use_paged_attention
            - xperf_triton_cfg: Dict to override any config field
    """
    # Map fields from kwargs to XPerfTritonModelConfig names
    override_config_kwargs = {
        'dtype': kwargs.get('dtype', torch.bfloat16),
        'page_block_size': kwargs.get('slot_block_size', 1024),  # Default 1024
        'max_total_tokens': kwargs.get('max_total_tokens', 0),
        'max_batch_size': kwargs.get('max_batch_size', 16),
        'max_length': kwargs.get('max_length', 2048),
        'vanilla_checkpoint_path': kwargs.get('vanilla_checkpoint_path'),
        'preshard_checkpoint_path': kwargs.get('preshard_checkpoint_path'),
        'use_cuda_graph': kwargs.get('enable_cuda_graph', False),
        'use_paged_attention': kwargs.get('use_vllm', False),
        'world_size': int(os.getenv('WORLD_SIZE', 1)),
        'tp_size': kwargs.get('mp_size', int(os.getenv('WORLD_SIZE', 1))),
    }

    # xperf_triton_cfg has highest priority
    override_config_kwargs.update(kwargs.get('xperf_triton_cfg', {}))
    override_config_kwargs.pop('enable', None)

    return XPerfTritonInferenceEngine(
        model_config_path=override_config_kwargs.pop('model_config_path', kwargs.get('model_config_path')),
        override_config_kwargs=override_config_kwargs,
        max_ctx_batch_size=kwargs.get('max_ctx_batch_size', 8),
    )


class XPerfTritonInferenceEngine:

    def __init__(self, **kwargs):
        self.module = XPerfTritonInferenceModule(**kwargs)
        self.config = self.module.config
        self.is_xperf_triton = True

    @torch.inference_mode()
    def forward_orca(self, *args, **kwargs):
        return self.module.forward_orca(*args, **kwargs)

    @torch.inference_mode()
    def get_input_embeddings(self, input_ids):
        return self.module.model.word_embedding(input_ids)


class XPerfTritonInferenceModule:

    def __init__(
        self,
        model_config_path: str,
        override_config_kwargs: dict = None,
        max_ctx_batch_size: int = 8,
    ):
        """Initialize with clean interface.
        
        Args:
            model_config_path: Path to model configuration JSON
            override_config_kwargs: Dict of fields to override in XPerfTritonModelConfig
            max_ctx_batch_size: Maximum batch size for context phase. For CUDA graph capture.
        """
        override_config_kwargs = override_config_kwargs or {}
        override_config_kwargs['use_cuda_graph'] = False  # avoid capture cuda graph in init
        override_config_kwargs['mock_weights'] = not bool(override_config_kwargs.get('preshard_checkpoint_path'))
        self.model = build_model(model_config_path, **override_config_kwargs).eval()
        self.config = self.model.config
        # Correct config after model creation
        self.config.use_cuda_graph = override_config_kwargs.get('use_cuda_graph', False)
        self.config.mock_weights = False

        # Store runtime parameters
        self.max_ctx_batch_size = max_ctx_batch_size

        # Derived attributes
        self.num_layers = self.config.model_config.num_layers
        self.global_rank = self.config.global_rank
        self.world_size = self.config.world_size
        self.tp_size = self.config.tp_size
        self.graph_captured = False
        self.quant_mode = 'NO_QUANT'

    def _get_kv_cache(self, layer_idx, is_quant=False):
        return self.model.layers[layer_idx].self_attention._kv_cache

    def get_param_list(self, skip_meta=False):
        return [param.data for param in self.model.parameters() if not (param.is_meta and skip_meta)]

    def capture_cuda_graph(self):
        if self.config.use_cuda_graph and not self.graph_captured:
            from xperf_gpt_triton.model_implementations.m8.m8_model import GraphInstance
            self.release_cuda_graph()
            assert self.config.backend == "triton"

            def init_dummy_input_buffers(batch_size: int):
                max_pages = ((self.config.max_length // self.config.page_block_size +
                              1) if self.config.use_paged_attention else 1)
                input_buffers = {
                    "decode_input": torch.ones((batch_size,), dtype=torch.int64, device="cuda"),
                    "decode_kv_len": torch.ones((batch_size,), dtype=torch.int64, device="cuda"),
                    "kv_table": torch.zeros((batch_size, max_pages), dtype=torch.int64, device="cuda"),
                    "dp_rank_input_len": torch.ones((self.config.world_size,), dtype=torch.int64, device="cuda"),
                    "is_nccl_use_padding": True,
                }
                return input_buffers

            memory_pool = None

            for bs in range(self.max_ctx_batch_size, self.config.max_batch_size + 1, self.max_ctx_batch_size):
                print(f"capture cuda graph with batch_size={bs}", flush=True)
                inputs = init_dummy_input_buffers(bs)
                self.model.forward(**inputs)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=memory_pool):
                    output = self.model.forward(**inputs)
                if memory_pool is None:
                    memory_pool = g.pool()

                graph_instance = GraphInstance()
                graph_instance.batch_size = bs
                graph_instance.input_buffers = inputs
                graph_instance.output_buffer = output
                graph_instance.graph_obj = g
                self.model.cuda_graph_instances[bs] = graph_instance
            self.graph_captured = True

    def release_cuda_graph(self):
        if self.config.use_cuda_graph and self.graph_captured:
            self.model.release_cuda_graph()
            self.graph_captured = False
            torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        model = self.model
        for i in range(model.config.model_config.num_layers):
            if i + 1 not in getattr(model.config.model_config, "kv_mirror_layers", []):
                model.layers[i].create_kv_cache()
            else:
                src_layer_number = model.config.model_config.kv_mirror_imitated_layers[
                    model.config.model_config.kv_mirror_layers.index(i + 1)]
                model.layers[i]._kv_cache = model.layers[src_layer_number - 1]._kv_cache
            model.layers[i].self_attention._kv_cache = model.layers[i]._kv_cache

    def free_kv_cache(self):
        for i in range(self.num_layers):
            del self.model.layers[i]._kv_cache
            del self.model.layers[i].self_attention._kv_cache
            self.model.layers[i]._kv_cache = None
            self.model.layers[i].self_attention._kv_cache = None
        torch.cuda.empty_cache()

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def enter(self):
        self.allocate_kv_cache()
        self.capture_cuda_graph()

    def exit(self):
        self.release_cuda_graph()
        self.free_kv_cache()

    @torch.inference_mode()
    def forward_orca(
        self,
        context_input_ids,
        context_input_embeds,
        decode_input_ids,
        total_length,
        kv_cache_index,
        orca_updated,
        context_shifts,
        return_full_hidden_states,
        return_padding_tensor,
        last_token_only,
        # TODO(qingyuhao): not used by xperf_gpt_triton
        context_max_kv_len=None,
        context_total_kv_len=None,
        decode_max_kv_len=None,
        decode_total_kv_len=None,
    ):
        outputs = []
        ctx_bsz = 0

        def get_forward_params(input_len: int, **overrides):
            dp_rank_input_len = self._get_dp_rank_input_lens(input_len)
            params = {
                'context_input_ids': None,
                'context_input_embeds': None,
                'decode_input_ids': None,
                'total_length': None,
                'kv_cache_index': kv_cache_index,
                'orca_updated': orca_updated,
                'context_shifts': context_shifts,
                'return_full_hidden_states': return_full_hidden_states,
                'return_padding_tensor': return_padding_tensor,
                'last_token_only': last_token_only,
                'dp_rank_input_len': dp_rank_input_len,
                'is_nccl_use_padding': self.model.is_nccl_use_padding(dp_rank_input_len),
            }
            params.update(overrides)
            return params

        if context_input_embeds is not None:
            ctx_bsz = context_input_embeds.shape[0]
            ctx_params = get_forward_params(
                sum(total_length[:ctx_bsz]),
                context_input_embeds=context_input_embeds,
                total_length=total_length[:ctx_bsz],
                kv_cache_index=kv_cache_index[:ctx_bsz],
                context_shifts=context_shifts[:ctx_bsz] if context_shifts is not None else None,
            )
            outputs.append(self.model.forward_orca(**ctx_params))

        if decode_input_ids is not None:
            dec_len = decode_input_ids.shape[0] if hasattr(decode_input_ids, 'shape') and len(
                decode_input_ids.shape) > 0 else len(decode_input_ids)
            dec_params = get_forward_params(
                dec_len,
                decode_input_ids=decode_input_ids,
                total_length=total_length[ctx_bsz:],
                kv_cache_index=kv_cache_index[ctx_bsz:],
                context_shifts=context_shifts[ctx_bsz:] if context_shifts is not None else None,
            )
            outputs.append(self.model.forward_orca(**dec_params))

        if len(outputs) == 2:
            return torch.cat(outputs, dim=0)
        elif outputs:
            return outputs[0]
        return None

    def _get_dp_rank_input_lens(self, local_rank_input_len: int):
        import torch.distributed as dist
        if self.config.dp_size > 1:
            local_rank_input_len = torch.tensor(local_rank_input_len, dtype=torch.int64, device="cuda")
            all_rank_input_len = torch.empty(self.config.dp_size, dtype=torch.int64, device="cuda")
            dist.all_gather_into_tensor(all_rank_input_len, local_rank_input_len)
        else:
            all_rank_input_len = torch.tensor([local_rank_input_len], dtype=torch.int64, device="cuda")
        return all_rank_input_len


def _reshard_fsdp_state_dict_to_xperf_triton_m8(tp_model: XPerfTritonInferenceModule,
                                                state_dict,
                                                device_mesh: DeviceMesh,
                                                model_config,
                                                backend='fsdp',
                                                prefix=''):
    assert backend == 'fsdp', "Only support fsdp for xperf_triton"

    def split_with_dim(tensor: torch.Tensor, dim: int, rank: int = None, num_split: int = None):
        if tensor is None:
            return None
        rank = tp_model.global_rank % tp_model.world_size if rank is None else rank
        num_split = tp_model.world_size if num_split is None else num_split
        splits = torch.split(tensor, tensor.shape[dim] // num_split, dim=dim)
        return splits[rank]

    def split_qkv_project(c_attn: torch.Tensor):
        if tp_model.tp_size == 1:
            return c_attn.t().contiguous()
        else:
            q_dim = (tp_model.config.model_config.head_dim * tp_model.config.model_config.num_heads *
                     tp_model.config.model_config.q_head_times)
            kv_dim = (tp_model.config.model_config.head_dim * tp_model.config.model_config.num_kv_heads)
            tensor_q = c_attn[:, :q_dim]
            tensor_k = c_attn[:, q_dim:q_dim + kv_dim]
            tensor_v = c_attn[:, q_dim + kv_dim:]
            return torch.concat(
                [
                    split_with_dim(tensor_q, 1, tp_model.config.tp_rank, tp_model.config.tp_size),
                    split_with_dim(tensor_k, 1, tp_model.config.tp_rank, tp_model.config.tp_size),
                    split_with_dim(tensor_v, 1, tp_model.config.tp_rank, tp_model.config.tp_size),
                ],
                dim=1,
            ).t().contiguous()

    def split_out_project(c_proj: torch.Tensor):
        if tp_model.config.tp_size == 1:
            return c_proj.t().contiguous()
        else:
            return split_with_dim(c_proj, 0, tp_model.config.tp_rank, tp_model.config.tp_size).t().contiguous()

    def split_ffn_ep(tensor: torch.Tensor):
        return split_with_dim(tensor, 0)

    def split_ffn0_tp(tensor: torch.Tensor):
        # tensor: n, k
        return split_with_dim(tensor, 0)

    def split_ffn1_tp(tensor: torch.Tensor):
        # tensor: n, k
        return split_with_dim(tensor, 1)

    def split_swiglu_ffn0_shared_expert_tp(tensor: torch.Tensor):
        if tp_model.config.tp_size == 1:
            return tensor
        w1, w2 = torch.chunk(tensor, 2, dim=0)
        return torch.cat([split_ffn0_tp(w1), split_ffn0_tp(w2)])

    def split_swiglu_ffn1_shared_expert_tp(tensor: torch.Tensor):
        if tp_model.config.tp_size == 1:
            return tensor
        return split_ffn1_tp(tensor)

    def get_tensor(tensor):
        if isinstance(tensor, DTensor):
            return tensor.full_tensor()
        return tensor

    from seed_models import M8Config
    assert isinstance(model_config, M8Config)

    # checking
    model = tp_model.model
    hidden_size = model_config.hidden_size
    ln_f_weight = get_tensor(state_dict.pop(prefix + 'transformer.norm.weight')).to(torch.bfloat16).reshape(-1)
    model.final_layernorm.weight.data = ln_f_weight.contiguous()
    del ln_f_weight

    # TODO: use xperf vocab_tp
    wte = state_dict.pop(prefix + 'transformer.wte.weight').to(torch.bfloat16)
    state_dict.pop(prefix + 'lm_head.weight', None)

    wte_weight = get_tensor(wte)

    assert wte_weight.shape == model.word_embedding.weight.data.shape
    assert wte_weight.shape == model.lm_head.weight.data.shape

    model.word_embedding.weight.data = wte_weight.contiguous()
    model.lm_head.weight.data = wte_weight.contiguous()

    for layer_index, layer in enumerate(model.layers):
        k = prefix + f'transformer.h.{layer_index}.input_layernorm.weight'
        ln_1_weight = get_tensor(state_dict.pop(k))
        ln_1_weight = torch.stack((ln_1_weight,), dim=0).to(torch.bfloat16).reshape(-1)
        assert layer.rms_norm_1.weight.data.shape == ln_1_weight.shape, f"{layer.rms_norm_1.weight.data.shape} == {ln_1_weight.shape}"
        layer.rms_norm_1.weight.data = ln_1_weight.contiguous()

        self_attn = layer.self_attention
        k = prefix + f'transformer.h.{layer_index}.attn.key_layernorm.weight'
        key_norm_weight = get_tensor(state_dict[k])
        key_norm_weight = torch.stack((key_norm_weight,), dim=0).to(torch.bfloat16).reshape(-1)
        assert self_attn.key_norm.weight.data.shape == key_norm_weight.shape, f"{self_attn.key_norm.weight.data.shape} == {key_norm_weight.shape}"
        self_attn.key_norm.weight.data = key_norm_weight.contiguous()

        k = prefix + f'transformer.h.{layer_index}.attn.context_norm.weight'
        context_norm_weight = get_tensor(state_dict[k])
        context_norm_weight = torch.stack((context_norm_weight,), dim=0).to(torch.bfloat16).reshape(-1)
        assert self_attn.context_norm.weight.data.shape == context_norm_weight.shape, f"{self_attn.context_norm.weight.data.shape} == {context_norm_weight.shape}"
        self_attn.context_norm.weight.data = context_norm_weight.contiguous()

        k = prefix + f'transformer.h.{layer_index}.attn.q_proj.weight'
        q_proj_weight = get_tensor(state_dict.pop(k)).to(torch.bfloat16).view(-1, hidden_size)
        k = prefix + f'transformer.h.{layer_index}.attn.k_proj.weight'
        k_proj_weight = get_tensor(state_dict.pop(k)).to(torch.bfloat16).view(-1, hidden_size)
        k = prefix + f'transformer.h.{layer_index}.attn.v_proj.weight'
        v_proj_weight = get_tensor(state_dict.pop(k)).to(torch.bfloat16).view(-1, hidden_size)

        qkv_weight = split_qkv_project(torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0).transpose(0, 1))

        assert self_attn.qkv_proj.weight.data.shape == qkv_weight.shape, f"{self_attn.qkv_proj.weight.data.shape} == {qkv_weight.shape}"
        self_attn.qkv_proj.weight.data = qkv_weight.contiguous()

        k = prefix + f'transformer.h.{layer_index}.attn.o_proj.weight'
        o_proj_weight = get_tensor(state_dict.pop(k)).to(torch.bfloat16)
        o_proj_weight = split_out_project(o_proj_weight.transpose(0, 1))

        assert self_attn.out_proj.weight.data.shape == o_proj_weight.shape, f"{self_attn.out_proj.weight.data.shape} == {o_proj_weight.shape}"
        self_attn.out_proj.weight.data = o_proj_weight.contiguous()

        k = prefix + f'transformer.h.{layer_index}.post_attention_layernorm.weight'
        ln_2_weight = get_tensor(state_dict.pop(k)).to(torch.bfloat16)
        ln_2_weight = torch.stack((ln_2_weight,), dim=0).reshape(-1)
        assert layer.rms_norm_2.weight.data.shape == ln_2_weight.shape, f"{layer.rms_norm_2.weight.data.shape} == {ln_2_weight.shape}"
        layer.rms_norm_2.weight.data = ln_2_weight.contiguous()

        k = prefix + f'transformer.h.{layer_index}.mlp.moe.gate.wg'
        gate_wg = get_tensor(state_dict.pop(k)).float()
        k = prefix + f'transformer.h.{layer_index}.mlp.moe.gate.wg_ema'
        gate_wg_ema = state_dict.pop(k).float()

        gate_wg = (gate_wg + gate_wg_ema) * 0.5

        ffn = layer.ffn
        assert ffn.gate.weight.shape == gate_wg.shape, f"{ffn.gate.weight.shape} == {gate_wg.shape}"
        ffn.gate.weight.data = gate_wg.contiguous()

        k = prefix + f'transformer.h.{layer_index}.mlp.moe.experts.fc1_1'
        fc1_1_weight = get_tensor(state_dict.pop(k).to(torch.bfloat16))
        k = prefix + f'transformer.h.{layer_index}.mlp.moe.experts.fc1_2'
        fc1_2_weight = get_tensor(state_dict.pop(k).to(torch.bfloat16))
        fc1_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)

        k = prefix + f'transformer.h.{layer_index}.mlp.moe.experts.fc2'
        fc2_weight = get_tensor(state_dict.pop(k).to(torch.bfloat16))

        fc1_weight = split_ffn_ep(fc1_weight)
        fc2_weight = split_ffn_ep(fc2_weight)
        assert ffn.fc1.shape == fc1_weight.shape, f"{ffn.fc1.shape} == {fc1_weight.shape}"
        assert ffn.fc2.shape == fc2_weight.shape, f"{ffn.fc2.shape} == {fc2_weight.shape}"
        ffn.fc1.data = fc1_weight.contiguous()
        ffn.fc2.data = fc2_weight.contiguous()

        experts_share = layer.experts_share
        k = prefix + f'transformer.h.{layer_index}.mlp.moe.experts_share.fc1_1'
        share_fc1_1_weight = get_tensor(state_dict.pop(k).to(torch.bfloat16))
        k = prefix + f'transformer.h.{layer_index}.mlp.moe.experts_share.fc1_2'
        share_fc1_2_weight = get_tensor(state_dict.pop(k).to(torch.bfloat16))
        share_fc1_weight = torch.cat([share_fc1_1_weight, share_fc1_2_weight], dim=0)

        k = prefix + f'transformer.h.{layer_index}.mlp.moe.experts_share.fc2'
        share_fc2_weight = get_tensor(state_dict.pop(k).to(torch.bfloat16))

        share_fc1_weight = split_swiglu_ffn0_shared_expert_tp(share_fc1_weight)
        share_fc2_weight = split_swiglu_ffn1_shared_expert_tp(share_fc2_weight)
        assert experts_share.fc1.weight.shape == share_fc1_weight.shape, f"{experts_share.fc1.weight.shape} == {share_fc1_weight.shape}"
        assert experts_share.fc2.weight.shape == share_fc2_weight.shape, f"{experts_share.fc2.weight.shape} == {share_fc2_weight.shape}"
        experts_share.fc1.weight.data = share_fc1_weight.contiguous()
        experts_share.fc2.weight.data = share_fc2_weight.contiguous()

    # enforce check nan
    # for name, param in model.named_parameters():
    #     assert_not_nan(param.data), f"{name=} has nan weight"

    tp_model = tp_model.to("cuda")
    torch.cuda.empty_cache()


def _reshard_fsdp_state_dict_to_xperf_triton_seed_vl(
        tp_model: XPerfTritonInferenceModule,
        vit_model,  # TorchVitInferencer or other VIT model
        state_dict,
        device_mesh: DeviceMesh,
        model_config,
        backend='fsdp',
        prefix=''):
    """
    Reshard FSDP state dict to XPerf Triton for seed_vl models.
    
    This function handles both the LLM and VIT components of VL models.
    
    Args:
        tp_model: XPerf Triton inference module (LLM part)
        vit_model: VIT inference module (TorchVitInferencer)
        state_dict: FSDP state dict containing both LLM and VIT weights
        device_mesh: Device mesh for distributed training
        model_config: Model configuration (seed_vl config)
        backend: Backend type (default: 'fsdp')
        prefix: Prefix for state dict keys (default: '')
    """
    assert backend == 'fsdp', "Only support fsdp for xperf_triton"

    # Import here to avoid circular dependency
    from alpha_seed.workers.xperf_rollout.utils.vit_inferencer import TorchVitInferencer

    # Ensure we're using TorchVitInferencer
    if not isinstance(vit_model, TorchVitInferencer):
        raise NotImplementedError(f"XPerf Triton with seed_vl models requires TorchVitInferencer. "
                                  f"Set vit_use_xperf_gpt=False in rollout config. Got: {type(vit_model).__name__}")

    # Define VIT prefixes for efficient checking
    VIT_PREFIXES = ('vision_encoder.', 'ln_vision.', 'multi_modal_projector.')

    # Separate VIT and LLM weights in a single pass
    vit_state_dict = {}
    llm_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith(VIT_PREFIXES):
            vit_state_dict[key] = value
        else:
            # Remove language_model prefix for LLM weights if present
            llm_key = key[len('language_model.'):] if key.startswith('language_model.') else key
            llm_state_dict[llm_key] = value

    # Update VIT model weights
    vit_model.weights_update(vit_state_dict)

    # Update LLM weights using the existing M8 function
    llm_model_config = getattr(model_config, 'text_config', model_config)

    _reshard_fsdp_state_dict_to_xperf_triton_m8(tp_model=tp_model,
                                                state_dict=llm_state_dict,
                                                device_mesh=device_mesh,
                                                model_config=llm_model_config,
                                                backend=backend,
                                                prefix=prefix)

    torch.cuda.empty_cache()
