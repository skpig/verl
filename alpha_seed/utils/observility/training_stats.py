from typing import ContextManager, Literal, Optional
import contextlib
from itertools import chain

from transformers import PretrainedConfig

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class MetricsTorchDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):

    def __init__(self):
        self.activation_stats = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        args_list, args_spec = torch.utils._pytree.tree_flatten(args)
        kwargs_list, kwargs_spec = torch.utils._pytree.tree_flatten(kwargs)
        for x in chain(args_list, kwargs_list):
            if hasattr(x, "stat_meta"):
                stat_meta = x.stat_meta
                amin, amax = x.data.aminmax()
                mean = x.data.abs().mean()
                name = stat_meta["name"]
                if name not in self.activation_stats:
                    self.activation_stats[name] = {
                        "min": amin.item(),
                        "max": amax.item(),
                        "mean": mean.item(),
                        "count": 1
                    }
                else:  # gradient accumulation
                    stats = self.activation_stats[name]
                    stats["min"] = min(stats["min"], amin.item())
                    stats["max"] = max(stats["max"], amax.item())
                    stats["mean"] += mean.item()
                    stats["count"] += 1

        return func(*args, **kwargs)

    def sync_and_clear_activation_stats(self):
        collected_stats = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(collected_stats, self.activation_stats)
        global_stats = {}
        for k in self.activation_stats.keys():
            global_stats["actor/training_stats/activations/min/" + k] = min(
                [stats[k]["min"] for stats in collected_stats])
            global_stats["actor/training_stats/activations/max/" + k] = max(
                [stats[k]["max"] for stats in collected_stats])
            global_stats["actor/training_stats/activations/abs.mean/" + k] = sum(
                [stats[k]["mean"] for stats in collected_stats]) / sum([stats[k]["count"] for stats in collected_stats])

        self.activation_stats.clear()
        return global_stats


def metrics_context_fn(metrics_context: ContextManager):
    return metrics_context, contextlib.nullcontext()


def all_reduce(
    data,
    op: Literal["mean", "sum", "max", "min"] = "mean",
    group: Optional["ProcessGroup"] = None,
):
    data = torch.tensor(data, dtype=torch.float, device="cuda")

    reduce_ops = {
        "mean": dist.ReduceOp.SUM,
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN
    }
    dist.all_reduce(data, op=reduce_ops[op], group=group)
    if op == "mean":  # ReduceOp.AVG is not supported by the NPU backend
        data /= dist.get_world_size(group=group)

    if data.numel() == 1:
        return data.item()
    else:
        return data.tolist()


def sync_tensor_stats(tensor: torch.Tensor, unshard_param_size: int, fsdp_size: int):
    if tensor is not None and tensor.numel() > 0:
        tmin, tmax = tensor.aminmax()
        tmin, tmax = tmin.item(), tmax.item()
        tmean = tensor.abs().sum().item() / unshard_param_size
    else:
        tmin, tmax = float("inf"), float("-inf")
        tmean = 0.0

    tensor_min = all_reduce(tmin, op="min")
    tensor_max = all_reduce(tmax, op="max")
    tensor_mean = all_reduce(tmean, op="sum") * fsdp_size / dist.get_world_size()

    return tensor_min, tensor_max, tensor_mean


def sync_params_and_grads_stats(param: torch.Tensor, param_name: str, unshard_param_size: int, fsdp_size: int):
    weight_min, weight_max, weight_mean = sync_tensor_stats(param, unshard_param_size, fsdp_size)
    grad_min, grad_max, grad_mean = sync_tensor_stats(param.grad, unshard_param_size, fsdp_size)

    return {
        "actor/training_stats/params/min/" + param_name: weight_min,
        "actor/training_stats/params/max/" + param_name: weight_max,
        "actor/training_stats/params/abs.mean/" + param_name: weight_mean,
        "actor/training_stats/grads/min/" + param_name: grad_min,
        "actor/training_stats/grads/max/" + param_name: grad_max,
        "actor/training_stats/grads/abs.mean/" + param_name: grad_mean,
    }


def sync_training_stats(
    metrics_context: ContextManager,
    model_config: PretrainedConfig,
    actor_module: torch.nn.Module,
    fsdp_size: int,
):
    training_stats = metrics_context.sync_and_clear_activation_stats()

    if model_config.model_type == "seed_p6dense":
        for layer_idx in range(model_config.num_hidden_layers):
            attn = actor_module.module.model.layers[layer_idx].module.self_attn

            head_dim = model_config.hidden_size / model_config.num_attention_heads
            unshard_q_proj_size = model_config.hidden_size * model_config.hidden_size
            training_stats.update(
                sync_params_and_grads_stats(attn.q_proj.weight, f"layer_{layer_idx}.q_proj", unshard_q_proj_size,
                                            fsdp_size))

            unshard_k_proj_size = model_config.hidden_size * model_config.num_key_value_heads * head_dim
            training_stats.update(
                sync_params_and_grads_stats(attn.k_proj.weight, f"layer_{layer_idx}.k_proj", unshard_k_proj_size,
                                            fsdp_size))

            unshard_v_proj_size = model_config.hidden_size * model_config.num_key_value_heads * head_dim
            training_stats.update(
                sync_params_and_grads_stats(attn.v_proj.weight, f"layer_{layer_idx}.v_proj", unshard_v_proj_size,
                                            fsdp_size))

        training_stats.update(
            sync_params_and_grads_stats(actor_module.module.model.embed_tokens.weight, "embed_tokens",
                                        model_config.hidden_size * model_config.vocab_size, fsdp_size))
        training_stats.update(
            sync_params_and_grads_stats(actor_module.module.lm_head.weight, "lm_head",
                                        model_config.hidden_size * model_config.vocab_size, fsdp_size))
    elif model_config.model_type == "seed_m8":
        for layer_idx in range(model_config.num_hidden_layers):
            attn = actor_module.module.transformer.h[layer_idx].module.attn

            head_dim = model_config.hidden_size // model_config.num_attention_heads
            unshard_q_proj_size = model_config.hidden_size * model_config.query_head_scale_factor * model_config.hidden_size
            training_stats.update(
                sync_params_and_grads_stats(attn.q_proj.weight, f"layer_{layer_idx}.q_proj", unshard_q_proj_size,
                                            fsdp_size))

            unshard_k_proj_size = model_config.hidden_size * model_config.num_key_value_heads * head_dim
            training_stats.update(
                sync_params_and_grads_stats(attn.k_proj.weight, f"layer_{layer_idx}.k_proj", unshard_k_proj_size,
                                            fsdp_size))

            unshard_v_proj_size = model_config.hidden_size * model_config.num_key_value_heads * head_dim
            training_stats.update(
                sync_params_and_grads_stats(attn.v_proj.weight, f"layer_{layer_idx}.v_proj", unshard_v_proj_size,
                                            fsdp_size))

        training_stats.update(
            sync_params_and_grads_stats(actor_module.module.transformer.wte.weight, "wte",
                                        model_config.hidden_size * model_config.vocab_size, fsdp_size))
        training_stats.update(
            sync_params_and_grads_stats(actor_module.module.lm_head.weight, "lm_head",
                                        model_config.hidden_size * model_config.vocab_size, fsdp_size))

    return training_stats
