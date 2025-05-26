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

import torch
import json
import torch.distributed

from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
import logging

try:
    from xperf_gpt_triton.builder.model_builder import build_model_from_config
    from xperf_gpt_triton.builder.config import XPerfTritonModelConfig
except Exception as e:
    logging.warning(f"Failed to import xperf_gpt_triton: {e}")
    build_model_from_config = None
    XPerfTritonModelConfig = None
import xperf_gpt.utils.comm as comm


def init_inference_triton(**kwargs):
    return XPerfTritonInferenceEngine(**kwargs)


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

    def __init__(self, **kwargs):

        config_dict = {}
        config_dict["world_size"] = comm.get_world_size()
        config_dict["global_rank"] = comm.get_rank()
        config_dict["local_rank"] = comm.get_local_rank()
        config_dict["dtype"] = kwargs.pop("dtype", torch.bfloat16)
        config_dict["page_block_size"] = kwargs.pop("slot_block_size", 1024)
        config_dict["max_total_tokens"] = kwargs.pop("max_total_tokens", 0)
        config_dict["num_pages"] = config_dict["max_total_tokens"] // config_dict["page_block_size"]
        config_dict["max_batch_size"] = kwargs.pop("max_batch_size", 16)
        config_dict["max_length"] = kwargs.pop("max_length", 2048)
        config_dict["vanilla_checkpoint_path"] = kwargs.pop("vanilla_checkpoint_path", None)
        config_dict["preshard_checkpoint_path"] = kwargs.pop("reshard_checkpoint_path", None)
        config_dict["use_cuda_graph"] = False  # avoid capture cuda graph in init
        config_dict['use_paged_attention'] = kwargs.pop("use_pated_attn", False)

        model_config_path = kwargs.pop('model_config_path')
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
            if "head_dim" not in model_config.keys():
                assert ("hidden_size" in model_config.keys() and "num_heads" in model_config.keys())
                assert model_config["hidden_size"] % model_config["num_heads"] == 0
                model_config["head_dim"] = (model_config["hidden_size"] // model_config["num_heads"])
            if "window_size" in model_config and model_config["window_size"] is None:
                model_config.pop("window_size")
            comm.logging_rank_only(logging.warning, 0, f"xgpt_triton config: {model_config}")

        for k in XPerfTritonModelConfig.__fields__.keys():
            if k in model_config:
                config_dict[k] = model_config[k]
        config_dict["model_name"] = "M8"
        config_dict['mock_weights'] = True
        config_dict["model_config"] = XPerfTritonModelConfig(**model_config)

        config = XPerfTritonModelConfig(**config_dict)
        self.model = build_model_from_config(config).eval()
        self.config = self.model.config

        # correct the config
        self.config.use_cuda_graph = kwargs.pop("enable_cuda_graph", False)
        self.config.mock_weights = False
        self.num_layers = self.config.model_config.num_layers
        self.global_rank = self.config.global_rank
        self.world_size = self.config.world_size
        self.quant_mode = self.config.model_config.quant_mode
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.tp_size = 1
        self.graph_captured = False
        self.xperf_triton_cfg = kwargs.pop('xperf_triton_cfg')

    def _get_kv_cache(self, layer_idx, is_quant=False):
        return self.model.layers[layer_idx].self_attnetion._kv_cache

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

            for bs in range(self.xperf_triton_cfg.max_ctx_batch_size, self.config.max_batch_size + 1,
                            self.xperf_triton_cfg.max_ctx_batch_size):
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
    ):
        ctx_bsz = 0

        ctx_output = None
        if context_input_embeds is not None:
            ctx_bsz = context_input_embeds.shape[0]
            ctx_total_length = total_length[:ctx_bsz]
            ctx_kv_cache_index = kv_cache_index[:ctx_bsz]
            ctx_context_shifts = context_shifts[:ctx_bsz]
            ctx_output = self.forward_orca_context_input(
                context_input_ids=None,
                context_input_embeds=context_input_embeds,
                decode_input_ids=None,
                total_length=ctx_total_length,
                kv_cache_index=ctx_kv_cache_index,
                orca_updated=orca_updated,
                context_shifts=ctx_context_shifts,
                return_full_hidden_states=return_full_hidden_states,
                return_padding_tensor=return_padding_tensor,
                last_token_only=last_token_only,
            )

        dec_output = None
        if decode_input_ids is not None:
            dec_total_length = total_length[ctx_bsz:]
            dec_kv_cache_index = kv_cache_index[ctx_bsz:]
            dec_context_shifts = context_shifts[ctx_bsz:]
            dec_output = self.forward_orca_decode_input(
                context_input_ids=None,
                context_input_embeds=None,
                decode_input_ids=decode_input_ids,
                total_length=dec_total_length,
                kv_cache_index=dec_kv_cache_index,
                orca_updated=orca_updated,
                context_shifts=dec_context_shifts,
                return_full_hidden_states=return_full_hidden_states,
                return_padding_tensor=return_padding_tensor,
                last_token_only=last_token_only,
            )
        if ctx_output is not None and dec_output is not None:
            return torch.cat([ctx_output, dec_output], dim=0)
        elif ctx_output is not None:
            return ctx_output
        elif dec_output is not None:
            return dec_output

    def forward_orca_decode_input(
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
    ):
        assert context_input_ids is None and context_input_embeds is None
        local_rank_input_len = len(decode_input_ids)
        all_rank_input_len = self._get_all_rank_input_len(local_rank_input_len)
        return self.model.forward_orca(
            context_input_ids=context_input_ids,
            context_input_embeds=context_input_embeds,
            decode_input_ids=decode_input_ids,
            total_length=total_length,
            kv_cache_index=kv_cache_index,
            orca_updated=orca_updated,
            context_shifts=context_shifts,
            return_full_hidden_states=return_full_hidden_states,
            return_padding_tensor=return_padding_tensor,
            last_token_only=last_token_only,
            dp_input_lens=all_rank_input_len,
            is_nccl_use_padding=self.model.is_nccl_use_padding(all_rank_input_len),
        )

    def forward_orca_context_input(
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
    ):
        assert decode_input_ids is None
        local_rank_input_len = sum(total_length)
        all_rank_input_len = self._get_all_rank_input_len(local_rank_input_len)
        return self.model.forward_orca(
            context_input_ids=context_input_ids,
            context_input_embeds=context_input_embeds,
            decode_input_ids=None,
            total_length=total_length,
            kv_cache_index=kv_cache_index,
            orca_updated=orca_updated,
            context_shifts=context_shifts,
            return_full_hidden_states=return_full_hidden_states,
            return_padding_tensor=return_padding_tensor,
            last_token_only=last_token_only,
            dp_input_lens=all_rank_input_len,
            is_nccl_use_padding=self.model.is_nccl_use_padding(all_rank_input_len),
        )

    def _get_all_rank_input_len(self, local_rank_input_len: int):
        import torch.distributed as dist
        if self.config.model_config.world_size > 1:
            local_rank_input_len = torch.tensor(local_rank_input_len, dtype=torch.int64, device="cuda")
            all_rank_input_len = torch.empty(self.config.model_config.world_size, dtype=torch.int64, device="cuda")
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

    def split_with_dim(tensor: torch.Tensor, dim: int):
        if tensor is None:
            return None
        splits = torch.split(tensor, tensor.shape[dim] // tp_model.world_size, dim=dim)
        return splits[tp_model.global_rank]

    def split_ffn_ep(tensor: torch.Tensor):
        return split_with_dim(tensor, 0)

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
        q_proj_weight = get_tensor(state_dict.pop(k)).to(torch.bfloat16).view(-1, hidden_size).contiguous()
        k = prefix + f'transformer.h.{layer_index}.attn.k_proj.weight'
        k_proj_weight = get_tensor(state_dict.pop(k)).to(torch.bfloat16).view(-1, hidden_size)
        k = prefix + f'transformer.h.{layer_index}.attn.v_proj.weight'
        v_proj_weight = get_tensor(state_dict.pop(k)).to(torch.bfloat16).view(-1, hidden_size)

        qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0).contiguous()

        assert self_attn.qkv_proj.weight.data.shape == qkv_weight.shape, f"{self_attn.qkv_proj.weight.data.shape} == {qkv_weight.shape}"
        self_attn.qkv_proj.weight.data = qkv_weight.contiguous()

        k = prefix + f'transformer.h.{layer_index}.attn.o_proj.weight'
        o_proj_weight = get_tensor(state_dict.pop(k)).to(torch.bfloat16)

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

        assert experts_share.fc1.weight.shape == share_fc1_weight.shape, f"{experts_share.fc1.weight.shape} == {share_fc1_weight.shape}"
        assert experts_share.fc2.weight.shape == share_fc2_weight.shape, f"{experts_share.fc2.weight.shape} == {share_fc2_weight.shape}"
        experts_share.fc1.weight.data = share_fc1_weight.contiguous()
        experts_share.fc2.weight.data = share_fc2_weight.contiguous()

    # enforce check nan
    # for name, param in model.named_parameters():
    #     assert_not_nan(param.data), f"{name=} has nan weight"

    tp_model = tp_model.to("cuda")
    torch.cuda.empty_cache()
