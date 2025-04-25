import torch
from torch.distributed._tensor import DTensor
from typing import *


class PluginProtocol(Protocol):
    name: str
    layer_id: Union[int, None]
    weight_key: Union[str, List[str]]
    weight: Union[torch.Tensor, List[torch.Tensor]]


class XCustomInferenceModuleAdapter:
    """An adapter class, with this we can use 
        XperfCustom's engine like xperfgpt engine.module
    """

    def __init__(self, engine):
        from xperf_gpt.inference.config import XPerfGPTConfig
        from xperf_gpt.utils import get_tp_rank

        inference_config = XPerfGPTConfig(num_return_sequences=engine.num_return_sequences,
                                          context_only=engine.context_only,
                                          config=engine.config)
        for key, val in inference_config.__dict__.items():
            setattr(self, key, val)
        for key, val in engine.config.__dict__.items():
            setattr(self, key, val)
        self.tp_rank = get_tp_rank()
        self._engine = engine
        self._initial_weights = dict()
        self._backup_weights()
        self._init_layer_id()  # prevent attention plugin layer_id not set

    def get_param_list(self, skip_meta=False):
        param_list = []
        for _, node_dict in self._engine.infer_graph.graph.nodes(data=True):
            plugin = node_dict.get('plugin', None)
            if plugin is not None:
                if isinstance(plugin.weight, list):
                    for weight in plugin.weight:
                        if isinstance(weight, torch.Tensor):
                            if (not skip_meta) or (not weight.is_meta):
                                param_list.append(weight)
                elif isinstance(plugin.weight, torch.Tensor):
                    if (not skip_meta) or (not plugin.weight.is_meta):
                        param_list.append(plugin.weight)
        return param_list

    def _get_backup_weight(self, weight):
        if isinstance(weight, torch.Tensor):
            assert weight.is_meta
            return weight.clone().detach()
        else:
            assert weight is None
            return None

    def _init_layer_id(self):
        for node_name, node_dict in self._engine.infer_graph.graph.nodes(data=True):
            plugin = node_dict.get('plugin', None)
            if plugin is None:
                continue
            plugin.layer_id = get_layer_id(plugin)

    def _backup_weights(self):
        """backup the initial meta weights"""
        for node_name, node_dict in self._engine.infer_graph.graph.nodes(data=True):
            plugin = node_dict.get('plugin', None)
            if plugin is None:
                continue
            if isinstance(plugin.weight, list):
                backup_weights = [self._get_backup_weight(weight) for weight in plugin.weight]
                self._initial_weights[node_name] = (plugin, backup_weights)
            else:
                self._initial_weights[node_name] = (plugin, self._get_backup_weight(plugin.weight))

    def reset_weights(self):
        """reset weights to initial states"""
        for _, (plugin, backup_weights) in self._initial_weights.items():
            if isinstance(backup_weights, list):
                plugin.weight = [self._get_backup_weight(weight) for weight in backup_weights]
            else:
                plugin.weight = self._get_backup_weight(backup_weights)


def get_full_tensor(tensor: Union[torch.Tensor, DTensor]):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    else:
        return tensor


def get_layer_id(plugin: PluginProtocol):
    layer_id = None if "." not in plugin.name else int(plugin.name.split(".")[0])
    return layer_id


def _reshard_state_dict_to_OrcaWordEmbeddingPlugin(model: XCustomInferenceModuleAdapter,
                                                   plugin: PluginProtocol,
                                                   state_dict,
                                                   model_config,
                                                   backend='fsdp'):
    assert backend == 'fsdp', "only fsdp supported now"
    full_state_dict = dict()
    # assert plugin.name == 'wte'
    full_wte = get_full_tensor(state_dict['transformer.wte.weight'].to(model.dtype))
    full_state_dict[plugin.weight_key] = full_wte
    plugin.set_parameters(full_state_dict)


def _reshard_state_dict_to_RMSNormPlugin(model: XCustomInferenceModuleAdapter,
                                         plugin: PluginProtocol,
                                         state_dict,
                                         model_config,
                                         backend='fsdp'):
    assert backend == 'fsdp', "only fsdp supported now"
    full_state_dict = dict()
    layer_id = get_layer_id(plugin)
    get_weight = lambda name: get_full_tensor(state_dict[name].to(model.dtype))
    # transform_weight = lambda weight:  torch.stack((weight,), dim=0).reshape(1, weight.shape[-1]).contiguous()

    if plugin.name == f"ln_f":
        ln_f_weight = get_weight("transformer.norm.weight")
        full_state_dict[plugin.weight_key] = ln_f_weight
    elif plugin.name == f"{layer_id}.ln_1":
        ln_1_weight = get_weight(f'transformer.h.{layer_id}.input_layernorm.weight')
        full_state_dict[plugin.weight_key] = ln_1_weight
    elif plugin.name == f"{layer_id}.ln_2":
        ln_2_weight = get_weight(f'transformer.h.{layer_id}.post_attention_layernorm.weight')
        full_state_dict[plugin.weight_key] = ln_2_weight
    else:
        raise ValueError(f"unsupported plugin_name={plugin.name}")

    plugin.set_parameters(full_state_dict, layer_id=layer_id)


def _reshard_state_dict_to_QKVProjectionPlugin(model: XCustomInferenceModuleAdapter,
                                               plugin: PluginProtocol,
                                               state_dict,
                                               model_config,
                                               backend='fsdp'):
    assert backend == 'fsdp', "only fsdp supported now"
    full_state_dict = dict()
    layer_id = get_layer_id(plugin)
    # assert plugin.name == f"{layer_id}.c_attn"
    # assert plugin.weight_key == [f"gpt.transformer.h.{layer_id}.attn.c_attn.weight", None]

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    num_kv_heads = model_config.num_key_value_heads

    # [NK * NH/NK*Q_Scale * HD,H] + [NK * HD, H] + [NH * HD, H]
    # => [H, (NH/NK*Q_Scale + 2) * NK * HD]
    q_proj_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.attn.q_proj.weight'].to(model.dtype))
    q_proj_weight = q_proj_weight.view(-1, head_dim, hidden_size)
    k_proj_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.attn.k_proj.weight'].to(model.dtype))
    k_proj_weight = k_proj_weight.view(-1, num_kv_heads, head_dim, hidden_size)
    v_proj_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.attn.v_proj.weight'].to(model.dtype))
    v_proj_weight = v_proj_weight.view(-1, num_kv_heads, head_dim, hidden_size)

    kv_replicate = model.tp_size // model_config.num_key_value_heads
    # duplicate kv
    if kv_replicate > 1:
        k_proj_weight = torch.tile(k_proj_weight, (1, kv_replicate, 1, 1))
        v_proj_weight = torch.tile(v_proj_weight, (1, kv_replicate, 1, 1))

    k_proj_weight = k_proj_weight.view(-1, head_dim, hidden_size)
    v_proj_weight = v_proj_weight.view(-1, head_dim, hidden_size)

    qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight),
                           dim=0).view(-1, hidden_size).transpose(0, 1).contiguous()
    full_state_dict[plugin.weight_key[0]] = qkv_weight

    plugin.set_parameters(full_state_dict, layer_id)


def _reshard_state_dict_to_KeyRMSNormPlugin(model: XCustomInferenceModuleAdapter,
                                            plugin: PluginProtocol,
                                            state_dict,
                                            model_config,
                                            backend='fsdp'):
    assert backend == 'fsdp', "only fsdp supported now"
    full_state_dict = dict()
    layer_id = get_layer_id(plugin)
    # assert plugin.name == f"{layer_id}.k_norm"
    # assert plugin.weight_key == f"gpt.transformer.h.{layer_id}.attn.layernorm.weight"
    key_norm_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.attn.key_layernorm.weight'].to(model.dtype))
    full_state_dict[plugin.weight_key] = key_norm_weight
    plugin.set_parameters(full_state_dict, layer_id)


def _reshard_state_dict_to_AttentionProjectionPlugin(model: XCustomInferenceModuleAdapter,
                                                     plugin: PluginProtocol,
                                                     state_dict,
                                                     model_config,
                                                     backend='fsdp'):
    assert backend == 'fsdp', "only fsdp supported now"
    full_state_dict = dict()
    layer_id = get_layer_id(plugin)
    # assert plugin.name == f"{layer_id}.c_proj"
    # assert plugin.weight_key == [f'gpt.transformer.h.{layer_id}.attn.c_proj.weight', None]

    head_dim = model_config.hidden_size // model_config.num_attention_heads
    hidden_size = model_config.hidden_size
    num_kv_heads = model_config.num_key_value_heads

    # [H,NK * NH/NK*Q_Scale * HD] => [NH/NK*Q_Scale * NK * HD,H]
    o_proj_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.attn.o_proj.weight'].to(model.dtype))
    o_proj_weight = o_proj_weight.view(hidden_size, num_kv_heads, -1, head_dim)

    kv_replicate = model.tp_size // model_config.num_key_value_heads
    # duplicate kv
    if kv_replicate > 1:
        o_proj_weight = o_proj_weight.view(hidden_size, -1, model.tp_size, head_dim)

    o_proj_weight = o_proj_weight.view(hidden_size, -1).transpose(0, 1).contiguous()

    full_state_dict[plugin.weight_key[0]] = o_proj_weight
    plugin.set_parameters(full_state_dict, layer_id)


def _reshard_state_dict_to_ContextRMSNormPlugin(model: XCustomInferenceModuleAdapter,
                                                plugin: PluginProtocol,
                                                state_dict,
                                                model_config,
                                                backend='fsdp'):
    assert backend == 'fsdp', "only fsdp supported now"
    full_state_dict = dict()
    layer_id = get_layer_id(plugin)
    # assert plugin.name == f"{layer_id}.context_norm"
    # assert plugin.weight_key == f"gpt.transformer.h.{layer_id}.attn.context_norm.weight"
    context_norm_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.attn.context_norm.weight'].to(
        model.dtype))
    full_state_dict[plugin.weight_key] = context_norm_weight
    plugin.set_parameters(full_state_dict, layer_id)


def _reshard_state_dict_to_MoESwiGLUSharePlugin(model: XCustomInferenceModuleAdapter,
                                                plugin: PluginProtocol,
                                                state_dict,
                                                model_config,
                                                backend='fsdp'):
    assert backend == 'fsdp', "only fsdp supported now"
    full_state_dict = dict()
    layer_id = get_layer_id(plugin)
    # assert plugin.name == f"{layer_id}.ffn"
    # assert plugin.weight_key == [f'gpt.transformer.h.{layer_id}.mlp.moe.gate.wg',
    #                              f'gpt.transformer.h.{layer_id}.mlp.moe.gate.wg_ema',
    #                              f'gpt.transformer.h.{layer_id}.mlp.moe.experts.fc1',
    #                              f'gpt.transformer.h.{layer_id}.mlp.moe.experts.fc2',
    #                              f'gpt.transformer.h.{layer_id}.mlp.moe.experts_share.fc1',
    #                              f'gpt.transformer.h.{layer_id}.mlp.moe.experts_share.fc2']
    assert isinstance(plugin.weight_key, list) and len(plugin.weight_key) == 6
    gate_wg_key, wg_ema_key, fc1_key, fc2_key, share_fc1_key, share_fc2_key = plugin.weight_key

    # [H,E]
    gate_wg = get_full_tensor(state_dict[f"transformer.h.{layer_id}.mlp.moe.gate.wg"]).contiguous().float()
    # [E]
    gate_wg_ema = get_full_tensor(state_dict[f'transformer.h.{layer_id}.mlp.moe.gate.wg_ema']).contiguous().float()

    # [E,FFN,H] + [E,FFN,H] => [E,FFN*2,H]
    fc1_1_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.mlp.moe.experts.fc1_1'].to(model.dtype))
    fc1_2_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.mlp.moe.experts.fc1_2'].to(model.dtype))
    fc1_weight = torch.cat((fc1_1_weight, fc1_2_weight), dim=1).contiguous()
    del fc1_1_weight, fc1_2_weight

    # num_share_experts = model_config.share_expert_num
    # [SE*FFN,H] + [SE*FFN,H] => [2*SE*FFN,H]
    share_fc1_1_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.mlp.moe.experts_share.fc1_1'].to(
        model.dtype))
    share_fc1_2_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.mlp.moe.experts_share.fc1_2'].to(
        model.dtype))
    share_fc1_weight = torch.cat((share_fc1_1_weight, share_fc1_2_weight), dim=0)
    del share_fc1_1_weight, share_fc1_2_weight

    # [E,H,FFN]
    fc2_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.mlp.moe.experts.fc2'].to(model.dtype))
    # [H,SE*FFN]
    share_fc2_weight = get_full_tensor(state_dict[f'transformer.h.{layer_id}.mlp.moe.experts_share.fc2'].to(
        model.dtype))

    full_state_dict[gate_wg_key] = gate_wg
    full_state_dict[wg_ema_key] = gate_wg_ema
    full_state_dict[fc1_key] = fc1_weight
    full_state_dict[fc2_key] = fc2_weight
    full_state_dict[share_fc1_key] = share_fc1_weight
    full_state_dict[share_fc2_key] = share_fc2_weight

    plugin.set_parameters(full_state_dict, layer_id)


def _reshard_state_dict_to_LMHeadPlugin(model: XCustomInferenceModuleAdapter,
                                        plugin: PluginProtocol,
                                        state_dict,
                                        model_config,
                                        backend='fsdp'):
    assert backend == 'fsdp', "only fsdp supported now"
    # assert plugin.name == "lm_head"
    # assert plugin.weight_key == "gpt.transformer.wte.weight"
    full_state_dict = dict()

    lm_head = get_full_tensor(state_dict["transformer.wte.weight"].to(model.dtype))
    full_state_dict[plugin.weight_key] = lm_head
    plugin.set_parameters(full_state_dict)


def _get_plugin_type_to_shard_fn_map():
    import xperf_gpt_custom.plugins.xperf_plugins
    PLUGIN_TYPE_TO_SHARD_FN = {
        xperf_gpt_custom.plugins.xperf_plugins.embedding_plugin.OrcaWordEmbeddingPlugin:
            _reshard_state_dict_to_OrcaWordEmbeddingPlugin,
        xperf_gpt_custom.plugins.xperf_plugins.layernorm_plugin.RMSNormPlugin:
            _reshard_state_dict_to_RMSNormPlugin,
        xperf_gpt_custom.plugins.xperf_plugins.gemm_plugin.QKVProjectionPlugin:
            _reshard_state_dict_to_QKVProjectionPlugin,
        xperf_gpt_custom.plugins.xperf_plugins.layernorm_plugin.KeyRMSNormPlugin:
            _reshard_state_dict_to_KeyRMSNormPlugin,
        xperf_gpt_custom.plugins.xperf_plugins.layernorm_plugin.ContextRMSNormPlugin:
            _reshard_state_dict_to_ContextRMSNormPlugin,
        xperf_gpt_custom.plugins.xperf_plugins.gemm_plugin.AttentionProjectionPlugin:
            _reshard_state_dict_to_AttentionProjectionPlugin,
        xperf_gpt_custom.plugins.xperf_plugins.ffn_plugin.MoESwiGLUSharePlugin:
            _reshard_state_dict_to_MoESwiGLUSharePlugin,
        xperf_gpt_custom.plugins.xperf_plugins.gemm_plugin.LMHeadPlugin:
            _reshard_state_dict_to_LMHeadPlugin,
    }
    return PLUGIN_TYPE_TO_SHARD_FN


def _reshard_state_dict_to_xperf_custom(tp_model, state_dict, model_config, device_mesh, backend='fsdp'):
    assert isinstance(tp_model, XCustomInferenceModuleAdapter)
    tp_model.reset_weights()
    engine = tp_model._engine
    for _, node_dict in engine.infer_graph.graph.nodes(data=True):
        plugin = node_dict.get('plugin', None)
        if plugin is not None:
            reshard_fn = _get_plugin_type_to_shard_fn_map().get(type(plugin), None)
            if reshard_fn is not None:
                reshard_fn(tp_model, plugin, state_dict, model_config=model_config, backend=backend)
            else:
                has_weight = False
                if isinstance(plugin.weight, list):
                    has_weight = any(lambda x: isinstance(x, torch.Tensor) for x in plugin.weight)
                else:
                    has_weight = isinstance(plugin.weight, torch.Tensor)
                if has_weight:
                    raise RuntimeError(
                        f"plugin {plugin.name}:{plugin} has weights:[{plugin.weight_key}] and does not have a reshard function"
                    )
