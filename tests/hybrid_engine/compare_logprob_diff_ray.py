#!/usr/bin/env python3
"""
Ray-based GPU resource management for logprob comparison between rollout and FSDP.
This version uses AsyncActorRolloutRefWorker directly with ray worker groups.
"""

import os
import ray
import torch
import copy
import json
import time
from omegaconf import OmegaConf
from typing import Dict, List, Optional

from mono_rl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from tests.test_utils import get_config, get_tokenizer
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from mono_rl.single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
from mono_rl.single_controller.ray import create_colocated_worker_cls
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions


def get_data():
    """Load test data from HDFS."""
    local_path = copy_local_path_from_hdfs(
        'hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/qiying.01/projects/alphaseed2/experiments/0811q2/batch_data/global_step_20_batch.pickle'
    )
    dataproto = DataProto.load_from_disk(local_path)
    dataproto = dataproto.chunk(len(dataproto) // 8)[0]
    return dataproto


def get_gen_batch(batch: DataProto):
    """
    Convert DataProto data to rollout input batch format.
    
    Args:
        batch: DataProto containing the exported data
    
    Returns:
        DataProto: formatted for rollout input
    """

    input_ids = batch.batch['prompts']
    attention_mask = batch.batch['attention_mask'][:, :input_ids.shape[1]]
    gen_batch = DataProto.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    })
    gen_batch.batch[key] = torch.empty_like(batch.batch[(key := 'rollout_behavior_log_probs')]).fill_(1.0)
    gen_batch.batch[key] = torch.empty_like(batch.batch[(key := 'off_policy_steps')]).fill_(-1)

    gen_batch.non_tensor_batch = copy.deepcopy(batch.non_tensor_batch)
    gen_batch.meta_info = copy.deepcopy(batch.meta_info)
    return gen_batch


def get_common_config():
    """Get common configuration for ray workers."""
    override_config = OmegaConf.create({
        "data": {
            "train_batch_size": 8,
            "max_prompt_length": 2048,
            "max_response_length": 16384,
        },
        "actor_rollout_ref": {
            "model": {
                "path":
                    "hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/qiying.01/projects/alphaseed2/experiments/0811q2/checkpoints/global_step_20/actor/huggingface"
            },
            "actor": {
                "ppo_max_token_len": 128 * 1024,
                "ppo_micro_batch_size": 8,
            },
            "rollout": {
                "tensor_model_parallel_size": 8,
                "mode": "batch",
                "use_vllm": True,
                "weights_communicator": "nccl",
                "complete_ratio": 1.0,
                "rollout_pool": {
                    "warmup_step": 0,
                },
                "gpu_memory_utilization": 0.7,
                "train_generate_kwargs": {
                    "max_new_tokens": 16384,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "do_sample": True,
                }
            },
            "fsdp": {
                "param_offload": False,
                "grad_offload": False,
                "optimizer_offload": False,
                "fsdp_size": 8,
            }
        },
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 8,
            "project_name": "alpha_seed_test",
            "experiment_name": "compare_logprob_diff_ray",
            "logger": ['console'],
        },
        "misc": {
            "aiomonitor": {
                "enable": True,
            },
        },
    })
    return get_config(override_config)


def create_ray_worker_group(config, role: str, ngpus: int):
    """
    Create ray worker group using AsyncActorRolloutRefWorker.
    
    Args:
        config: configuration object
        role: role name for the worker group
        ngpus: number of GPUs per node
    
    Returns:
        RayWorkerGroup: configured worker group
    """
    resource_pool = RayResourcePool(process_on_nodes=[ngpus], use_gpu=True, name_prefix=role)

    worker_cls_with_init = RayClassWithInitArgs(cls=AsyncActorRolloutRefWorker, config=config, role=role)

    class_dict = {role: worker_cls_with_init}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
    wg_dict = wg.spawn(prefix_set=class_dict.keys())

    worker_group = wg_dict[role]

    # Initialize model
    ray.get(worker_group.init_model())

    return worker_group


def create_hybrid_wg(config):
    return create_ray_worker_group(
        config.actor_rollout_ref,
        'actor_rollout',
        config.trainer.n_gpus_per_node,
    )


def print_large_diff_token(tokenizer, batch: DataProto, threshold=0.5):
    response_length = batch.batch['responses'].shape[1]
    for idx in range(batch.batch['responses'].shape[0]):
        print(f"============={idx}=================")
        responses = batch.batch['responses'][idx, :]
        off_policy = batch.batch['off_policy_steps'][idx, :]
        rollout_logprobs = batch.batch['rollout_behavior_log_probs'][idx, :]
        fsdp_logprobs = batch.batch['old_log_probs'][idx, :]
        masks = batch.batch['attention_mask'][idx, -response_length:]

        for i in range(response_length):
            x = rollout_logprobs[i]
            y = fsdp_logprobs[i]
            diff = x - y
            if masks[i] and ((x == -1) or (y == -1) or (abs(diff) >= threshold)):
                tokens = responses[max(0, i - 1):i + 1]
                print(f"#{i} [{tokenizer.decode(tokens)}], diff={x}-{y}={diff}, step={off_policy[i]}")


def compare_logprobs(rollout_logprobs: torch.Tensor,
                     fsdp_logprobs: torch.Tensor,
                     response_mask: Optional[torch.Tensor] = None,
                     off_policy_steps: Optional[torch.Tensor] = None) -> Dict:
    """
    Compare log probabilities between rollout and FSDP computations.
    
    Args:
        rollout_logprobs: log probabilities from rollout computation
        fsdp_logprobs: log probabilities from FSDP computation
        response_mask: mask tensor to apply to differences (multiply diff by mask)
    
    Returns:
        dict: comparison metrics including mean absolute difference, max difference, etc.
    """
    # Ensure both are on the same device and dtype
    fsdp_logprobs = fsdp_logprobs.to(rollout_logprobs.device, rollout_logprobs.dtype)

    # Handle potential shape mismatches
    if rollout_logprobs.shape != fsdp_logprobs.shape:
        assert rollout_logprobs.shape[0] == fsdp_logprobs.shape[0]
        min_len = min(rollout_logprobs.shape[0], fsdp_logprobs.shape[0])
        rollout_logprobs = rollout_logprobs[:min_len]
        fsdp_logprobs = fsdp_logprobs[:min_len]
        if response_mask is not None:
            response_mask = response_mask[:min_len]

    # Compute differences
    diff = rollout_logprobs - fsdp_logprobs

    # Apply response mask if provided
    if response_mask is not None:
        response_mask = response_mask.to(diff.device, diff.dtype)
        diff = diff * response_mask

        # Create masked tensors for metrics computation
        mask_bool = response_mask.bool()
        masked_diff = diff[mask_bool]
        masked_rollout = rollout_logprobs[mask_bool]
        masked_fsdp = fsdp_logprobs[mask_bool]
    else:
        masked_diff = diff
        masked_rollout = rollout_logprobs
        masked_fsdp = fsdp_logprobs

    abs_diff = torch.abs(diff)
    masked_abs_diff = torch.abs(masked_diff)

    # Compute metrics
    metrics = {
        'mean_abs_diff':
            masked_abs_diff.mean().item() if len(abs_diff) > 0 else 0.0,
        'max_abs_diff':
            masked_abs_diff.max().item() if len(abs_diff) > 0 else 0.0,
        'min_abs_diff':
            masked_abs_diff.min().item() if len(abs_diff) > 0 else 0.0,
        'std_abs_diff':
            masked_abs_diff.std().item() if len(abs_diff) > 0 else 0.0,
        'mean_diff':
            masked_diff.mean().item() if len(masked_diff) > 0 else 0.0,
        'relative_error':
            (masked_abs_diff / (torch.abs(masked_rollout) + 1e-8)).mean().item() if len(abs_diff) > 0 else 0.0,
        'rollout_mean':
            masked_rollout.mean().item() if len(masked_rollout) > 0 else 0.0,
        'fsdp_mean':
            masked_fsdp.mean().item() if len(masked_fsdp) > 0 else 0.0,
        'sample_count':
            len(masked_diff),
        'total_samples':
            len(diff.view(-1)),
        'masked_samples':
            len(masked_diff.view(-1))
    }
    if off_policy_steps is not None:
        max_off_policy_step = off_policy_steps.max().item()
        for step in range(max_off_policy_step + 1):
            masked_diff = abs_diff[off_policy_steps == step]
            metrics[f'max_abs_diff_{step}'] = masked_diff.max().item() if len(masked_diff) > 0 else 0.0

    return metrics


def balance_batch_(batch, hybrid_wg):
    attention_mask = batch.batch['attention_mask']
    assert len(attention_mask.shape) == 2 or len(attention_mask.shape) == 3
    print(f'Perform seqlen balancing with shape {attention_mask.shape}')
    batch_size = attention_mask.shape[0]
    global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
    # note that this may be problematic when the world_size differs
    world_size = hybrid_wg.world_size
    global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
    # reorder based on index. The data will be automatically equally partitioned by dispatch function
    global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
    batch.reorder(global_idx)


def print_metrics(name: str, metrics):
    print(f"\n=== {name} Logprob Comparison Results ===")
    print(f"Mean absolute difference: {metrics['mean_abs_diff']:.8f}")
    print(f"Max absolute difference: {metrics['max_abs_diff']:.8f}")
    print(f"Std absolute difference: {metrics['std_abs_diff']:.8f}")
    print(f"Mean difference: {metrics['mean_diff']:.8f}")
    print(f"Relative error: {metrics['relative_error']:.8f}")
    print(f"Rollout mean logprob: {metrics['rollout_mean']:.4f}")
    print(f"FSDP mean logprob: {metrics['fsdp_mean']:.4f}")
    print(f"Sample count: {metrics['sample_count']}")
    for i in range(10):
        key = f'max_abs_diff_{i}'
        if key in metrics:
            print(f"Max absolute difference for off_policy_step={i}: {metrics[key]:.8f}")
        else:
            break


def run_comparison():
    """Main function to run the logprob comparison."""
    print("Starting ray-based logprob comparison...")

    # Initialize ray
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "false",
                "NCCL_DEBUG": "WARN",
                "XPERF_DUMP_NAN": "0",
            }})

    try:
        # Get configuration
        config = get_common_config()

        # Load test data
        print("Loading test data...")
        origin_batch = get_data()

        response_length = origin_batch.batch['responses'].shape[-1]
        response_mask = origin_batch.batch['attention_mask'][:, -response_length:]
        origin_metrics = compare_logprobs(origin_batch.batch['rollout_behavior_log_probs'],
                                          origin_batch.batch['old_log_probs'],
                                          response_mask=response_mask,
                                          off_policy_steps=origin_batch.batch['off_policy_steps'])

        print_metrics("Origin Rollout vs FSDP", origin_metrics)

        batch = get_gen_batch(origin_batch)

        tokenizer = get_tokenizer(config)
        print_large_diff_token(tokenizer, origin_batch)

        # Create worker group
        hybrid_wg = create_hybrid_wg(config)

        print("Computing rollout logprobs...")
        start_time = time.time()
        batch = hybrid_wg.generate_sequences(batch)
        batch.batch["prompts"] = batch.batch["input_ids"][:, config.data.max_prompt_length]
        batch.batch["responses"] = batch.batch["input_ids"][:, config.data.max_prompt_length:]
        rollout_time = time.time() - start_time
        print(f"Rollout computation completed in {rollout_time:.2f}s")

        print("Computing FSDP logprobs...")
        start_time = time.time()
        # balance_batch_(batch, hybrid_wg)

        batch = hybrid_wg.old_log_probs(batch).get()
        fsdp_time = time.time() - start_time

        print(f"FSDP computation completed in {fsdp_time:.2f}s")

        # Compare results
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        rollout_metrics = compare_logprobs(batch.batch['rollout_behavior_log_probs'],
                                           batch.batch['old_log_probs'],
                                           response_mask=response_mask)
        # rollout_vs_origin_metrics = compare_logprobs(batch.batch['rollout_behavior_log_probs'],
        #                                              origin_batch.batch['rollout_behavior_log_probs'],
        #                                              response_mask=response_mask)
        # fsdp_vs_origin_metrics = compare_logprobs(batch.batch['old_log_probs'],
        #                                           origin_batch.batch['old_log_probs'],
        #                                           response_mask=response_mask)

        # Print results
        print_metrics("Rollout vs FSDP", rollout_metrics)
        # print_metrics("Rollout recomputed vs origin", rollout_vs_origin_metrics)
        # print_metrics("FSDP recomputed vs origin", fsdp_vs_origin_metrics)

        print(f"\nRollout time: {rollout_time:.2f}s")
        print(f"FSDP time: {fsdp_time:.2f}s")

        # Save results
        results = {
            'origin_rollout_vs_fsdp_metrics': origin_metrics,
            'rollout_vs_fsdp_metrics': rollout_metrics,
            # 'rollout_recomputed_vs_origin_metrics': None,
            # 'fsdp_recomputed_vs_origin_metrics': None,
            'timing': {
                'rollout_time': rollout_time,
                'fsdp_time': fsdp_time
            },
            'config': OmegaConf.to_container(config, resolve=True)
        }

        with open('logprob_comparison_ray_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\nResults saved to logprob_comparison_ray_results.json")
        print("\Batch saved to recomputed_batch.pickle")
        batch.save_to_disk(f'recomputed_batch.pickle')

        # return {'rollout_vs_fsdp': rollout_metrics, 'original_vs_recomputed': origin_metrics}

    finally:
        # Clean up ray
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    run_comparison()
