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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import itertools

from ray import ObjectRef

from alpha_seed.logging import refine_log
from alpha_seed.workers.streaming_service.rollout_query_timeline import RolloutQueryTimeline
from alpha_seed.workers.streaming_service.rollout_request_manager import get_all_request_manager_actors

refine_log()

import contextlib
import random
import os
import copy
import json
import queue
from functools import partial
from alpha_seed.trainer.utils.lineage import (report_job_config, report_data_loaded, report_trial_ckpts_load,
                                              report_rl_ckpts_load, safely_do)
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Type, List, Union, Tuple
import hdfs_io

import pandas as pd
import numpy as np
from codetiming import Timer

from alpha_seed.utils.select_strategy.bon_strategy import *
from alpha_seed.utils.select_strategy.league_training_strategy import *
from alpha_seed.utils.validator.validation_manager import *
from alpha_seed.workers.streaming_service.streaming_utils import pad, process_output
from alpha_seed.workers.actors.checkpoint import CkptGlobalUploader
from alpha_seed.workers.actors.rollout_pool import RolloutPool
from alpha_seed.workers.ppo_actor import make_mini_step_dataloader
from alpha_seed.utils.observility.pretty_print import pprint
from alpha_seed.utils import ndtimeline
from alpha_seed.utils.functional import print_dataproto_size
from alpha_seed.utils.tracking_utils import async_process_batch_samples_to_wandb
from alpha_seed.utils.multithreads import ThreadPoolManager
from alpha_seed.utils.dataset.vlm_rl_dataset import load_image_data_dist
from alpha_seed.workers.actors.checkpoint.utils import find_latest_ckpt_path_
from alpha_seed.trainer.utils.dataloader_mgr import DataLoaderMgr
from alpha_seed.workers.actors.sample_pool import SamplePool

from mono_rl.single_controller import Worker
from mono_rl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from mono_rl.single_controller.ray import create_colocated_worker_cls
from mono_rl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from hdfs_io import makedirs, hput, hcopy, hexists
from omnistore.utilities.io.bfile import is_local_path

try:
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.protocol import unfold_batch_dim, fold_batch_dim
except ImportError:
    print('Cannot find pad_dataproto_to_divisor. Please use latest verl master')
    raise
from alpha_seed import core_algos
import pickle as pkl

try:
    from bytedance.trainingmetrics.rl_metrics_client_context_manager import RLMetricsClientContextManager as MegavisionMetricsCtx
except ImportError:
    MegavisionMetricsCtx = None

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Validator = 7
    RolloutServer = 8


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, Tuple[list[int], str]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
    server_client_split: bool = False

    def create_resource_pool(self):
        for resource_pool_name, (process_on_nodes, pool_name) in self.resource_pool_spec.items():
            additional_res = []
            if pool_name:
                additional_res = [pool_name]
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                name_prefix=resource_pool_name,
                max_colocate_count=1,  # alphaseed里通过fused worker合并了之后，只占用1个gpu，因此不需要计算colocate count
                additional_resources=additional_res)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
import torch.nn.functional as F
from tensordict import TensorDict
from verl.utils.torch_functional import masked_mean


def calculate_score_in_length_ranges(raw_scores_log, response_length, ranges):
    scores = {}
    for min_length, max_length in ranges:
        if min_length is not None and max_length is not None:
            mask = torch.logical_and(response_length >= min_length, response_length < max_length)
        elif min_length is not None:
            mask = response_length >= min_length
        elif max_length is not None:
            mask = response_length < max_length
        else:
            continue
        scores[f'score/raw_score_in_{min_length}_{max_length}'] = (raw_scores_log[mask].sum() / (mask.sum() + 1)).item()
    return scores


def apply_kl_penalty(data: DataProto,
                     kl_ctrl: core_algos.AdaptiveKLController,
                     kl_penalty='kl',
                     use_model_output_mask=False):
    rollout_log_probs = data.batch['rollout_log_probs']
    old_log_probs = data.batch['old_log_probs']
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]

    if use_model_output_mask:
        loss_mask = data.batch['model_output_mask']
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(old_log_probs, data.batch['ref_log_prob'],
                                    kl_penalty_type=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)
    kld = torch.clamp(kld, max=10.0, min=-10.0)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    current_kl_sum = torch.mean(torch.sum(kld * response_mask, dim=-1), dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta, 'critic/kl_sum': current_kl_sum}

    # track KL divergence changes in model outputs, xperf logprobs versus seedmodels old logprobs
    kl_diff = (rollout_log_probs - old_log_probs) * response_mask
    kl_diff[:, -1] = 0
    kl_diff_mean = (kl_diff.sum() / response_mask.sum()).item()
    metrics.update({'rollout/kl_diff_mean': kl_diff_mean})

    kl_diff_max = kl_diff.max().item()
    idx = torch.nonzero(kl_diff_max == kl_diff)
    rollout_log_probs_max = rollout_log_probs[idx[0][0], idx[0][1]].item()
    old_log_probs_max = old_log_probs[idx[0][0], idx[0][1]].item()
    metrics.update({
        'rollout/kl_diff_max': kl_diff_max,
        'rollout/kl_diff_max_rollout_log_probs': rollout_log_probs_max,
        'rollout/kl_diff_max_old_log_probs': old_log_probs_max
    })

    kl_diff_min = kl_diff.min().item()
    idx = torch.nonzero(kl_diff_min == kl_diff)
    rollout_log_probs_min = rollout_log_probs[idx[0][0], idx[0][1]].item()
    old_log_probs_min = old_log_probs[idx[0][0], idx[0][1]].item()
    metrics.update({
        'rollout/kl_diff_min': kl_diff_min,
        'rollout/kl_diff_min_rollout_log_probs': rollout_log_probs_min,
        'rollout/kl_diff_min_old_log_probs': old_log_probs_min
    })

    # kl_diff_p90 = torch.quantile(kl_diff.view(-1), 0.9, dim=-1).item()
    # kl_diff_p99 = torch.quantile(kl_diff.view(-1), 0.99, dim=-1).item()
    # metrics.update({'rollout/kl_diff_p90': kl_diff_p90, 'rollout/kl_diff_p99': kl_diff_p99})

    kl_diff_sum = torch.mean(torch.sum(kl_diff, dim=-1), dim=0).item()
    metrics.update({'rollout/kl_diff_sum': kl_diff_sum})
    return data, metrics


def compute_advantage(data: DataProto, gamma, lam, use_variable_lambda, variable_lambda_scalar, adv_estimator,
                      upgo_loss_version, num_bon, adv_whiten, use_async_gen, use_separate_critic_lam, critic_lam,
                      group_mode, use_model_output_mask):
    # TODO: add other ways to estimate advantages
    token_level_rewards = data.batch['token_level_rewards']
    responses = data.batch['responses']
    response_length = responses.size(1)

    if use_model_output_mask:
        loss_mask = data.batch['model_output_mask']
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

    if adv_estimator == 'gae':
        values = data.batch['values']
        origin_advantages, advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam,
            use_variable_lambda=use_variable_lambda,
            variable_lambda_scalar=variable_lambda_scalar,
            adv_whiten=adv_whiten,
            use_separate_critic_lam=use_separate_critic_lam,
            critic_lam=critic_lam)
        data.batch['advantages'] = advantages
        data.batch['origin_advantages'] = origin_advantages
        data.batch['returns'] = returns
        upgo_advantages = core_algos.compute_upgo_advantage(token_level_rewards=token_level_rewards,
                                                            values=values,
                                                            eos_mask=response_mask,
                                                            upgo_loss_version=upgo_loss_version)
        data.batch['upgo_advantages'] = upgo_advantages
        adv_metrics = {}
    elif adv_estimator == 'grpo':
        token_level_scores = data.batch['token_level_scores']
        index = data.non_tensor_batch['index']
        advantages, returns, adv_metrics = core_algos.compute_grpo_advantage_return(
            token_level_scores=token_level_scores,
            eos_mask=response_mask,
            index=index,
            num_bon=num_bon,
            use_async_gen=use_async_gen,
            group_mode=group_mode)
        data.batch['advantages'] = advantages
        data.batch['origin_advantages'] = advantages
        data.batch['returns'] = returns
        data.batch['upgo_advantages'] = torch.zeros_like(advantages)
    else:
        raise NotImplementedError
    return data, adv_metrics


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def calculate_batch_bon_metrics(batch: DataProto, prefix):
    id2acc = defaultdict(list)
    cur_bsz = batch.batch.batch_size[0]
    for i in range(cur_bsz):
        index = batch.non_tensor_batch['index'][i]
        score = batch.batch['token_level_scores'][i].sum().item()
        id2acc[index].append(score)
    for key, val_lst in id2acc.items():
        acc = (np.mean(val_lst).item() + 1) / 2
        id2acc[key] = acc
    metrics = {
        f"{prefix}/acc_100": len(list(filter(lambda x: x == 1.0, id2acc.values()))) / len(id2acc),
        f"{prefix}/acc_80+": len(list(filter(lambda x: x > 0.8, id2acc.values()))) / len(id2acc),
        f"{prefix}/acc_50+": len(list(filter(lambda x: x > 0.5, id2acc.values()))) / len(id2acc),
        f"{prefix}/acc_20+": len(list(filter(lambda x: x > 0.2, id2acc.values()))) / len(id2acc),
        f"{prefix}/acc_10+": len(list(filter(lambda x: x > 0.1, id2acc.values()))) / len(id2acc),
        f"{prefix}/acc_0": len(list(filter(lambda x: x == 0, id2acc.values()))) / len(id2acc),
    }
    return metrics


def merge_metrics(merged_metrics, metrics2):
    for key, val in metrics2.items():
        if key not in merged_metrics:
            merged_metrics[key] = val
        else:
            merged_metrics[key] += val
    return merge_metrics


def merge_ministeps_metrics(ministeps_metrics: List[dict]):
    merged_metrics = {}
    for idx, metrics in enumerate(ministeps_metrics):
        for key, val in metrics.items():
            merged_metrics.setdefault(key, [])
            if isinstance(val, list):
                merged_metrics[key].extend(val)
            elif isinstance(val, (int, float)):
                merged_metrics[key].append(val)
            else:
                assert False, f"ministep{idx}[{key}]: {val} is not list/int/float"
            merged_metrics[f"ministep{idx}/{key}"] = val

    merged_metrics = reduce_metrics(merged_metrics)
    return merged_metrics


def compute_data_metrics(self, batch: DataProto):
    """
    Important: Note that this function will be executed in distributed.
    Note that you can't delete self in this function.
    """
    from verl.utils.torch_functional import distributed_mean_max_min_std
    import torch.distributed as dist

    use_critic = batch.meta_info['use_critic']
    mean = batch.meta_info['mean']
    std = batch.meta_info['std']

    batch = batch.to('cuda')

    sequence_score = batch.batch['token_level_scores'].sum(-1)
    raw_score = batch.batch['raw_scores'].sum(-1) * std + mean
    origin_sequence_score = sequence_score * std + mean  # 打原始的分数
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    response_length = batch.batch['responses'].shape[-1]

    advantages = batch.batch['advantages']
    origin_advantages = batch.batch['origin_advantages']
    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]
    if batch.meta_info['use_model_output_mask']:
        model_output_mask = batch.batch['model_output_mask'][:, -response_length:]
    else:
        model_output_mask = response_mask

    old_log_probs = batch.batch['old_log_probs']
    old_entropy = batch.batch['old_entropy']

    prompt_length = prompt_mask.sum(-1).float()
    model_output_length = model_output_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    max_prompt_length = float(prompt_mask.size(-1))
    max_response_length = float(response_mask.size(-1))

    returns = batch.batch['returns']

    model_output_mask_bool = model_output_mask.bool()

    valid_entropy = torch.masked_select(old_entropy, model_output_mask_bool)

    valid_adv = torch.masked_select(advantages, model_output_mask_bool)
    valid_origin_adv = torch.masked_select(origin_advantages, model_output_mask_bool)
    valid_returns = torch.masked_select(returns, model_output_mask_bool)
    valid_old_logprob = torch.masked_select(old_log_probs, model_output_mask_bool)

    eos_adv = torch.gather(advantages, dim=1, index=response_length.unsqueeze(dim=1).long() - 1).reshape(-1)
    eos_original_adv = torch.gather(origin_advantages, dim=1,
                                    index=response_length.unsqueeze(dim=1).long() - 1).reshape(-1)

    mean_entropy = distributed_mean_max_min_std(valid_entropy, compute_max=False, compute_min=False,
                                                compute_std=False)[0]

    score_mean, score_max, score_min, score_std = distributed_mean_max_min_std(sequence_score)
    raw_score_mean, raw_score_max, raw_score_min, raw_score_std = distributed_mean_max_min_std(raw_score)
    original_score_mean, original_score_max, original_score_min, original_score_std = distributed_mean_max_min_std(
        origin_sequence_score)
    sequence_reward_mean, sequence_reward_max, sequence_reward_min, sequence_reward_std = distributed_mean_max_min_std(
        sequence_reward)
    valid_adv_mean, valid_adv_max, valid_adv_min, valid_adv_std = distributed_mean_max_min_std(valid_adv)
    eos_adv_mean = distributed_mean_max_min_std(eos_adv, compute_max=False, compute_min=False, compute_std=False)[0]
    valid_origin_adv_mean, valid_origin_adv_max, valid_origin_adv_min, valid_origin_adv_std = distributed_mean_max_min_std(
        valid_origin_adv)
    eos_original_adv_mean = distributed_mean_max_min_std(eos_original_adv,
                                                         compute_max=False,
                                                         compute_min=False,
                                                         compute_std=False)[0]
    valid_returns_mean, valid_returns_max, valid_returns_min, valid_returns_std = distributed_mean_max_min_std(
        valid_returns)
    response_length_mean, response_length_max, response_length_min, response_length_std = distributed_mean_max_min_std(
        response_length, compute_std=True)
    model_output_length_mean, model_output_length_max, model_output_length_min, model_output_length_std = distributed_mean_max_min_std(
        model_output_length, compute_std=True)
    response_clip_ratio = distributed_mean_max_min_std(torch.eq(response_length, max_response_length).float(),
                                                       compute_max=False,
                                                       compute_min=False,
                                                       compute_std=False)[0]
    prompt_length_mean, prompt_length_max, prompt_length_min, prompt_length_std = distributed_mean_max_min_std(
        prompt_length, compute_std=False)
    prompt_length_clip_ratio = distributed_mean_max_min_std(torch.eq(prompt_length, max_prompt_length).float(),
                                                            compute_max=False,
                                                            compute_min=False,
                                                            compute_std=False)[0]

    prob_mean = distributed_mean_max_min_std(torch.exp(valid_old_logprob))[0]

    metrics = {
        # actor
        'actor/entropy': mean_entropy.detach().item(),
        # score
        'critic/score/mean': score_mean.detach().item(),
        'critic/score/max': score_max.detach().item(),
        'critic/score/min': score_min.detach().item(),
        'critic/score/std': score_std.detach().item(),
        # raw_score
        'critic/raw_score/mean': raw_score_mean.detach().item(),
        'critic/raw_score/max': raw_score_max.detach().item(),
        'critic/raw_score/min': raw_score_min.detach().item(),
        'critic/raw_score/std': raw_score_std.detach().item(),
        # original score
        'critic/original_score/mean': original_score_mean.detach().item(),
        'critic/original_score/max': original_score_max.detach().item(),
        'critic/original_score/min': original_score_min.detach().item(),
        'critic/original_score/std': original_score_std.detach().item(),
        # reward
        'critic/rewards/mean': sequence_reward_mean.detach().item(),
        'critic/rewards/max': sequence_reward_max.detach().item(),
        'critic/rewards/min': sequence_reward_min.detach().item(),
        'critic/rewards/std': sequence_reward_std.detach().item(),
        # adv
        'critic/advantages/mean': valid_adv_mean.detach().item(),
        'critic/advantages/eos_adv_mean': eos_adv_mean.detach().item(),
        'critic/advantages/max': valid_adv_max.detach().item(),
        'critic/advantages/min': valid_adv_min.detach().item(),
        'critic/advantages/std': valid_adv_std.detach().item(),
        # original adv
        'critic/original_advantages/mean': valid_origin_adv_mean.detach().item(),
        'critic/original_advantages/eos_adv_mean': eos_original_adv_mean.detach().item(),
        'critic/original_advantages/max': valid_origin_adv_max.detach().item(),
        'critic/original_advantages/min': valid_origin_adv_min.detach().item(),
        'critic/original_advantages/std': valid_origin_adv_std.detach().item(),
        # returns
        'critic/returns/mean': valid_returns_mean.detach().item(),
        'critic/returns/max': valid_returns_max.detach().item(),
        'critic/returns/min': valid_returns_min.detach().item(),
        'critic/returns/std': valid_returns_std.detach().item(),
        # response length
        'response_length/mean': response_length_mean.detach().item(),
        'response_length/max': response_length_max.detach().item(),
        'response_length/min': response_length_min.detach().item(),
        'response_length/std': response_length_std.detach().item(),
        'model_output_length/mean': model_output_length_mean.detach().item(),
        'model_output_length/max': model_output_length_max.detach().item(),
        'model_output_length/min': model_output_length_min.detach().item(),
        'model_output_length/std': model_output_length_std.detach().item(),
        ## response clip ratio
        'response_length/clip_ratio': response_clip_ratio.detach().item(),
        # prompt length
        'prompt_length/mean': prompt_length_mean.detach().item(),
        'prompt_length/max': prompt_length_max.detach().item(),
        'prompt_length/min': prompt_length_min.detach().item(),
        ## prompt clip ratio
        'prompt_length/clip_ratio': prompt_length_clip_ratio.detach().item(),
        # prob
        'prob/mean': prob_mean.detach().item(),
    }
    for threshold in [1e-6, 1e-5, 1e-4, 1e-3]:
        small_prob_mask = torch.logical_and(model_output_mask_bool, old_log_probs.exp() < threshold)
        small_prob_mask_sum = small_prob_mask.float().sum()
        local_small_prob_mask_sum = small_prob_mask_sum.clone()
        model_output_mask_bool_sum = model_output_mask_bool.float().sum()
        dist.all_reduce(small_prob_mask_sum, op=dist.ReduceOp.SUM, group=None, async_op=False)
        dist.all_reduce(model_output_mask_bool_sum, op=dist.ReduceOp.SUM, group=None, async_op=False)
        small_prob_ratio = small_prob_mask_sum / model_output_mask_bool_sum

        if local_small_prob_mask_sum == 0:
            small_prob_adv = torch.tensor(0, device=advantages.device)
        else:
            small_prob_adv = torch.masked_select(advantages, small_prob_mask)
            small_prob_adv = torch.sum(small_prob_adv)

        dist.all_reduce(small_prob_adv, op=dist.ReduceOp.SUM, group=None, async_op=False)
        small_prob_adv = small_prob_adv / small_prob_mask_sum

        metrics.update({
            f'prob/prob_lt_{threshold}_ratio': small_prob_ratio.detach().item(),
            f'prob/prob_lt_{threshold}_adv': small_prob_adv.detach().item()
        })
    if use_critic:
        values = batch.batch['values']
        upgo_advantages = batch.batch['upgo_advantages']
        valid_values = torch.masked_select(values, model_output_mask_bool)
        valid_upgo_adv = torch.masked_select(upgo_advantages, model_output_mask_bool)

        valid_values_mean, valid_values_max, valid_values_min, valid_values_std = distributed_mean_max_min_std(
            valid_values)
        valid_upgo_adv_mean, valid_upgo_adv_max, valid_upgo_adv_min, valid_upgo_adv_std = distributed_mean_max_min_std(
            valid_upgo_adv)

        return_diff_std = distributed_mean_max_min_std(torch.masked_select(returns - values, model_output_mask_bool),
                                                       compute_max=False,
                                                       compute_min=False,
                                                       compute_std=True)[-1]
        return_std = distributed_mean_max_min_std(torch.masked_select(returns, model_output_mask_bool),
                                                  compute_max=False,
                                                  compute_min=False,
                                                  compute_std=True)[-1]
        return_diff_var = return_diff_std**2
        return_var = return_std**2

        values_metrics = {
            # values
            'critic/values/mean': valid_values_mean.detach().item(),
            'critic/values/max': valid_values_max.detach().item(),
            'critic/values/min': valid_values_min.detach().item(),
            'critic/values/std': valid_values_std.detach().item(),
            # upgo adv
            'critic/upgo_advantages/mean': valid_upgo_adv_mean.detach().item(),
            'critic/upgo_advantages/max': valid_upgo_adv_max.detach().item(),
            'critic/upgo_advantages/min': valid_upgo_adv_min.detach().item(),
            'critic/upgo_advantages/std': valid_upgo_adv_std.detach().item(),
            # vf explained var
            'critic/vf/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        }
        metrics.update(values_metrics)
    return DataProto.from_dict({'dummy': torch.ones(size=(1,))}, meta_info={'metrics': metrics})


def _deduplicate_shared_tensors(obj, visited_tensors=None):
    """
    Recursively deduplicate shared tensors to avoid RuntimeError during torch.save.
    This function creates clones of tensors that share storage to ensure they can be saved.
    Similar to how safetensors automatically handles tensor deduplication.
    
    Args:
        obj: The object to process (can be tensor, dict, list, tuple, or custom object)
        visited_tensors: Dict to track tensors by their storage pointer to avoid infinite recursion
    
    Returns:
        Object with deduplicated tensors
    """
    if visited_tensors is None:
        visited_tensors = {}

    if isinstance(obj, torch.Tensor):
        # Check if we've already processed a tensor with the same storage
        storage_ptr = obj.storage().data_ptr() if obj.storage().size() > 0 else None
        if storage_ptr is not None and storage_ptr in visited_tensors:
            # Return a clone to avoid shared storage issues
            return obj.detach().clone()

        # Mark this storage as visited
        if storage_ptr is not None:
            visited_tensors[storage_ptr] = True

        # For the first occurrence, we can return the original tensor
        # but we'll clone it anyway to be safe
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {key: _deduplicate_shared_tensors(value, visited_tensors) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        deduplicated = [_deduplicate_shared_tensors(item, visited_tensors) for item in obj]
        return type(obj)(deduplicated)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by processing their attributes
        try:
            new_obj = copy.copy(obj)
            for attr_name, attr_value in obj.__dict__.items():
                setattr(new_obj, attr_name, _deduplicate_shared_tensors(attr_value, visited_tensors))
            return new_obj
        except (TypeError, AttributeError, pkl.PicklingError):
            # If the object can't be copied (e.g., contains staticmethod, lambda, etc.)
            # just return the original object since our main goal is tensor deduplication
            return obj
    else:
        # For primitive types and other objects, return as-is
        return obj


def save_dataproto(data: DataProto, path, prefix=''):
    # Deduplicate shared tensors before saving to avoid RuntimeError
    torch.save(_deduplicate_shared_tensors(data.batch), f"{prefix}.batch.pt")
    torch.save(data.non_tensor_batch, f"{prefix}.non_tensor_batch.pt")
    torch.save(data.meta_info, f"{prefix}.meta_info.pt")
    hcopy(f"{prefix}.batch.pt", path)
    hcopy(f"{prefix}.non_tensor_batch.pt", path)
    hcopy(f"{prefix}.meta_info.pt", path)


def load_dataproto(path, prefix=''):
    batch = f"{path}/{prefix}.batch.pt"
    non_tensor_batch = f"{path}/{prefix}.non_tensor_batch.pt"
    meta_info = f"{path}/{prefix}.meta_info.pt"

    batch = copy_local_path_from_hdfs(batch)
    batch = torch.load(batch)
    non_tensor_batch = copy_local_path_from_hdfs(non_tensor_batch)
    non_tensor_batch = torch.load(non_tensor_batch)
    load_image_data_dist(non_tensor_batch)
    meta_info = copy_local_path_from_hdfs(meta_info)
    meta_info = torch.load(meta_info)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver on a single GPU
    """

    # TODO: support each role have individual ray_worker_group_cls
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 logger=None,
                 processor=None,
                 remote_client=None):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.standalone_rollout_wg = None  # worker group
        self.standalone_validator_wg = None
        self.critic_wg = None
        self.ref_policy_wg = None
        self.rm_wg = None
        self.all_wg = {}
        self.internal_wgs: List[RayWorkerGroup] = []
        self.internal_wg_roles = []
        self.all_meta = {}
        self.remote_client = remote_client
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.logger = logger
        self.workers = []

        # ckpt global uploader will be instantiated in init_workers func
        self.ckpt_global_uploader = None

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, \
                  f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self._global_step = 0
        self._timeline_futures = []

        self.use_standalone_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_colocate_reference_policy = Role.ActorRolloutRef in role_worker_mapping
        self.use_standalone_rollout = self.config.streaming_rollout.nnodes > 0
        self.use_standalone_validator = self.config.streaming_validator.nnodes > 0
        self.use_elastic_streaming_rollout = self.config.streaming_rollout.elastic.enable and self.use_standalone_rollout
        self.use_reference_policy = self.use_standalone_reference_policy or self.use_colocate_reference_policy
        self.use_rm = Role.RewardModel in role_worker_mapping

        self.ray_worker_group_cls = ray_worker_group_cls
        self.num_bon = self.config.actor_rollout_ref.rollout.get("num_bon", 1)

        self.phasic_critic_buffer = None

        self.standalone_gen_batch_output_resume = None
        self.standalone_batch_resume = None
        self.megavision_metrics_collector = MegavisionMetricsCtx() if MegavisionMetricsCtx else None

        self.data_len_per_query = None
        self.acc_per_query = {}  # moving avg acc
        self.sample_acc_dir = config.trainer.default_hdfs_dir + "/sample_acc"
        self.is_vlm = config.data['image_key'] is not None
        self.save_batch_dir = ""

        # tracking logging rl samples takes quite a long time, put it in a background processes
        self.async_tracking_pool = ProcessPoolExecutor(max_workers=8)
        # use this to track how many running tasks in the background processes
        self.async_tracking_running_tasks = set()
        if config.trainer.default_hdfs_dir and config.trainer.save_cases_to_hdfs:
            self.save_batch_dir = os.path.join(config.trainer.default_hdfs_dir, "batch_data")

        self._rollout_query_tl = RolloutQueryTimeline(self.config.streaming_rollout.query_trace)
        self.request_managers = []
        if self.config.actor_rollout_ref.rollout.mode == "server":
            self.request_managers = get_all_request_manager_actors()

        self.enable_actor_critic_spatial_mux = self.config.trainer.get("enable_actor_critic_spatial_mux", False)

        safely_do(lambda: report_job_config(config), rank=0)()

    def _create_dataloader(self):
        self.dataloader_mgr = DataLoaderMgr(self.config, self.tokenizer, self.is_vlm, self.processor)
        self.train_dataloader = self.dataloader_mgr.train_dataloader
        self.val_dataloader = self.dataloader_mgr.val_dataloader
        self.total_training_steps = self.dataloader_mgr.total_training_steps
        safely_do(
            lambda: report_data_loaded(train_files=self.config.data.train_files, val_files=self.config.data.val_files),
            rank=0)()

    def _create_kl_control(self):
        # define KL control
        if self.use_reference_policy:
            if self.config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=self.config.algorithm.kl_ctrl.kl_coef)
            elif self.config.algorithm.kl_ctrl.type == 'adaptive':
                assert self.config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {self.config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=self.config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=self.config.algorithm.kl_ctrl.target_kl,
                                                               horizon=self.config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

    def _create_rollout_manager(self):
        from alpha_seed.workers.streaming_service.rollout_manager import RolloutManager
        self.rollout_manager = RolloutManager(config=self.config, logger=self.logger, tokenizer=self.tokenizer)
        hybrid_wg = self.actor_rollout_wg
        self.rollout_manager.initialize(hybrid_wg=hybrid_wg,
                                        rollout_pool=self.rollout_pool,
                                        train_standalone_wg=self.standalone_rollout_wg,
                                        val_standalone_wg=self.standalone_validator_wg)

    def _create_validation_manager(self):
        self.validation_manager = ValidateManager(self.config, self.logger, self.val_dataloader, self.tokenizer,
                                                  self.use_rm, self.val_reward_fn, self.rollout_manager)

    def init_workers(self, kv_store=None, ckpt_global_uploader=None, from_step=0, resume_folder=None):
        """Init resource pool and worker group"""
        from_scratch = from_step == 0
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        worker_configs = {}
        # create actor and rollout
        if self.hybrid_engine:
            if self.use_standalone_reference_policy or not self.use_reference_policy:
                role = 'rollout' if self.config.trainer.val_only else 'actor_rollout'
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
                actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                         config=self.config.actor_rollout_ref,
                                                         role=role)
                self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
                worker_configs['actor_rollout'] = self.config.actor_rollout_ref
            elif self.use_colocate_reference_policy:
                role = 'rollout' if self.config.trainer.val_only else 'actor_rollout_ref'
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
                actor_rollout_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[Role.ActorRolloutRef],
                    config=self.config.actor_rollout_ref,
                    role=role,
                    enable_actor_critic_spatial_mux=self.enable_actor_critic_spatial_mux)
                self.resource_pool_to_cls[resource_pool]['actor_rollout_ref'] = actor_rollout_cls
                worker_configs['actor_rollout_ref'] = self.config.actor_rollout_ref
            else:
                raise NotImplementedError('Must instantiate actor and rollout')

            # elastic rollout下不用提前创建好，所以不注册resource pool to class
            if self.use_standalone_rollout and not self.use_elastic_streaming_rollout:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
                rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Rollout],
                                                   config=self.config.actor_rollout_ref,
                                                   role='standalone_rollout')
                self.resource_pool_to_cls[resource_pool]['standalone_rollout'] = rollout_cls
                worker_configs['standalone_rollout'] = self.config.actor_rollout_ref

            if self.use_standalone_validator:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.Validator)
                validator_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Validator],
                                                     config=self.config.actor_rollout_ref,
                                                     role='standalone_validator')
                self.resource_pool_to_cls[resource_pool]['standalone_validator'] = validator_cls
                worker_configs['standalone_validator'] = self.config.actor_rollout_ref
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic],
                                              config=self.config.critic,
                                              enable_actor_critic_spatial_mux=self.enable_actor_critic_spatial_mux)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            worker_configs['critic'] = self.config.critic
        else:
            # support GRPO and ReMax
            if self.config.algorithm.adv_estimator == 'grpo':
                # grpo需要用所有的bon来算adv
                assert self.config.actor_rollout_ref.rollout.bon_strategy == "all"
                # grpo下不算UPGO loss
                assert self.config.actor_rollout_ref.actor.upgo_loss_weight <= 1e-10
                # grpo时要使用kl loss
                assert self.config.actor_rollout_ref.actor.kl_loss_weight >= 0.0
            self.use_critic = False

        # create reference policy if needed
        if self.use_standalone_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls
            worker_configs['ref'] = self.config.actor_rollout_ref

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls
            worker_configs['rm'] = self.config.reward_model

        self.rollout_pool = RolloutPool.get_or_create_actor(self.config)
        self.rollout_pool_warmup_step = self.config.actor_rollout_ref.rollout.rollout_pool.get("warmup_step", 0)

        server_client_split = self.config.server_client.role in ["server", "client"]
        self.sample_pool = SamplePool(self.config)

        # initialize WorkerGroup
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # @type class_dict: {Str(name) -> RayClassWithInitArgs}
            # no role allocated to this resource pool
            if len(class_dict) == 0:
                continue

            role_names = list(class_dict.keys())
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

            if self.config.server_client.role == "client":
                # call this function to initialize resource_pool's placement_groups
                # by attaching to existing ones
                try:
                    resource_pool.get_placement_groups(attach_existing=True)
                except Exception as e:
                    print("[WARN]: attach to existing placement group has error: {e}")

                assert kv_store is not None, f"client script must have kv_store"
                worker_names = ray.get(kv_store.get_by_key.remote(resource_pool.name_prefix))
                assert isinstance(worker_names, list) and len(worker_names) > 0
                wg_dict = self.ray_worker_group_cls(ray_cls_with_init=worker_dict_cls, worker_names=worker_names)

                if self.config.server_client.reset_server:
                    # already existing actors, stop and recreate
                    try:
                        print("Killing existing actors...")
                        for worker in wg_dict.workers:
                            ray.kill(worker)
                    except Exception as e:
                        print(f"[WARN] lookup and kill existing actors fail with: [{e}]")

                    print("Recreating actors...")
                    with Timer(name="recreate_actors"):
                        wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool,
                                                            ray_cls_with_init=worker_dict_cls)
                    # update worker_names
                    kv_store.set_key_val.remote(resource_pool.name_prefix, wg_dict.worker_names)

                self.workers += wg_dict.workers
            else:
                # create workers
                # RayWorkerGroup初始化时，worker_dict_cls会被在resource_pool中进行实例化，
                # 创建对应的actor，而actor的数量（也是world_size）由这个resource_pool最开始的spec所决定，因此这里的resource_pool是被切分过的
                wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
                if self.config.server_client.role == "server":
                    kv_store.set_key_val.remote(resource_pool.name_prefix, wg_dict.worker_names)

            # 重新把fused worker里的不同的原始worker的方法分离出来到spawn_wg，然后all_wg里还是像原来访问当个worker那样访问
            spawn_wg = wg_dict.spawn(prefix_set=role_names)
            self.all_wg.update(spawn_wg)
            self.internal_wgs.append(wg_dict)
            self.internal_wg_roles.append(role_names)

        # init ckpt global uploader
        uploader_tracker_role = 'actor'
        if self.use_critic:
            uploader_tracker_role = 'critic'
        if self.config.actor_rollout_ref.ref.ema < 1:
            # will save ref ckpt
            uploader_tracker_role = 'ref'
        self.ckpt_global_uploader = CkptGlobalUploader.options(name=CkptGlobalUploader.name).remote(
            tracker_role=uploader_tracker_role,
            ckpt_version=self.config.trainer.ckpt_version,
            default_local_dir=self.config.trainer.default_local_dir
            if not is_local_path(self.config.trainer.default_hdfs_dir) else self.config.trainer.default_hdfs_dir,
            default_remote_dir=self.config.trainer.default_hdfs_dir,
            upload_retry_count=int(self.config.trainer.ckpt_upload_retry_count)) if (
                ckpt_global_uploader is None and not server_client_split) else ckpt_global_uploader
        for wg_name in self.all_wg:
            self.all_meta[wg_name] = self.all_wg[wg_name].get_meta()
        ndtimeline.report_topo(self.all_meta)
        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        if self.use_standalone_reference_policy or not self.use_reference_policy:
            self.actor_rollout_wg = self.all_wg['actor_rollout']
        elif self.use_colocate_reference_policy:
            self.actor_rollout_wg = self.all_wg['actor_rollout_ref']
        else:
            raise NotImplementedError

        # (方法名，要wait的futures), 方法名就是一个标注的名称，方便报错的时候可以找回哪个报错的对象
        init_futures: List[Tuple[str, List[ObjectRef]]] = []

        actor_rollout_init_fut = self.actor_rollout_wg.init_model(
            remove_safetensors_after_init=self.config.trainer.remove_safetensors_after_init,
            from_scratch=self.build_model_from_scratch(from_step, 'actor'))
        init_futures.append(('actor_rollout_init_model', actor_rollout_init_fut))

        if self.use_standalone_rollout and not self.use_elastic_streaming_rollout:
            # 使用固定副本数的standalone rollout，跟着actor_rollout_ref的逻辑一起走fusedworker在resource_pool定义好的资源上创建
            self.standalone_rollout_wg = self.all_wg['standalone_rollout']
            init_futures.append(("standalone_rollout_wg_init_model",
                                 self.standalone_rollout_wg.init_model(
                                     remove_safetensors_after_init=self.config.trainer.remove_safetensors_after_init,
                                     from_scratch=from_scratch)))

        if self.use_standalone_validator:
            self.standalone_validator_wg = self.all_wg['standalone_validator']
            init_futures.append(('standalone_val_init_model',
                                 self.standalone_validator_wg.init_model(
                                     remove_safetensors_after_init=self.config.trainer.remove_safetensors_after_init,
                                     from_scratch=from_scratch)))

        # ensure errors in model_init will be raised
        # wait for MODEL INITIALIZATION
        for name, fut in init_futures:
            try:
                ray.get(fut)
            except:
                print(name)
                raise
        init_futures.clear()

        if self.config.actor_rollout_ref.actor.kl_loss_weight >= 1e-10:
            # 两种情况下使用kl loss，一种是grpo，另一种是在rewards里不加kl惩罚
            assert self.config.algorithm.adv_estimator == 'grpo' or self.config.algorithm.kl_ctrl.kl_coef <= 1e-10

        if self.use_critic:
            self.critic_wg = self.all_wg['critic']
            # 因为actor和critic是跑在同一个ray actor里，但他们的model各自调用init(也包括其他需要全局同步的nccl调用)，
            # init时需要全局同步初始化，这里如果并发会出现不知道谁先走到nccl 同步调用，如果有的rank先跑了actor，有的先跑了critic，
            # 而nccl又不兼容python async，不会yield，就会互相等
            # 因此这里必须先等actor初始化完了再跑critic 初始化
            ray.get(actor_rollout_init_fut)
            init_futures.append(('critic_wg_init_model',
                                 self.critic_wg.init_model(
                                     remove_safetensors_after_init=self.config.trainer.remove_safetensors_after_init,
                                     from_scratch=from_scratch)))

        if self.use_standalone_reference_policy:
            if self.config.actor_rollout_ref.ref.ema == 1:
                from_scratch_ref = True
            else:
                from_scratch_ref = from_scratch
            self.ref_policy_wg = self.all_wg['ref']
            init_futures.append(('ref_policy_init_model',
                                 self.ref_policy_wg.init_model(
                                     remove_safetensors_after_init=self.config.trainer.remove_safetensors_after_init,
                                     from_scratch=from_scratch_ref)))
        elif self.use_colocate_reference_policy:
            self.ref_policy_wg = self.all_wg['actor_rollout_ref']

        if self.use_rm:
            self.rm_wg = self.all_wg['rm']
            self.rm_wg.init_model(remove_safetensors_after_init=self.config.trainer.remove_safetensors_after_init,
                                  from_scratch=True)  # blocking

        safely_do(lambda: report_rl_ckpts_load(worker_configs=worker_configs), rank=0)()
        self.global_step = from_step
        self.resume_folder = resume_folder

        # wait for model communication setup and weight sync
        # note(lixiang): make this separate from init_model to avoid deadlock
        for name, fut in init_futures:
            try:
                ray.get(fut)
            except:
                print(name)
                raise
        init_futures.clear()

    def save_checkpoint(self, specified_ckpt_version=None):
        """Save checkpoint to hdfs.
        Checkpoint structure
        default_local_dir:
            - checkpoints
                - latest_checkpointed_iteration.txt
                - global_step_xxx
                    - loader.pt
                    - actor
                        - model
                        - optimizer
                        - extra_state
                    - critic
                        - model
                        - optimizer
                        - extra_state
            - config.yaml (TODO)
            - step2token.json (TODO)
            - training_trajectory.yaml (TODO)

        """
        # Attention!!! note that the latest_checkpointed_iteration.txt will be overriden if you resume from a previous checkpoint
        if self.config.trainer.default_hdfs_dir and is_local_path(self.config.trainer.default_hdfs_dir):
            print(f"save_checkpoint: default_hdfs_dir={self.config.trainer.default_hdfs_dir} is a local or fuse dir, "
                  f"set default_local_dir={self.config.trainer.default_hdfs_dir}")
            self.config.trainer.default_local_dir = self.config.trainer.default_hdfs_dir

        local_checkpoint_folder = os.path.join(self.config.trainer.default_local_dir, 'checkpoints')
        local_global_step_folder = os.path.join(local_checkpoint_folder, f'global_step_{self.global_step}')
        os.makedirs(local_global_step_folder, exist_ok=True)

        actor_local_path = os.path.join(local_global_step_folder, 'actor')
        critic_local_path = os.path.join(local_global_step_folder, 'critic')
        ref_local_path = os.path.join(local_global_step_folder, 'ref')

        remote_checkpoint_folder = os.path.join(self.config.trainer.default_hdfs_dir, 'checkpoints')
        remote_global_step_folder = os.path.join(remote_checkpoint_folder, f'global_step_{self.global_step}')

        makedirs(remote_global_step_folder, exist_ok=True)

        actor_remote_path = os.path.join(remote_global_step_folder, 'actor')
        critic_remote_path = os.path.join(remote_global_step_folder, 'critic')
        ref_remote_path = os.path.join(remote_global_step_folder, 'ref')

        # save data len per query
        data_len_per_query_local_path = os.path.join(local_global_step_folder, 'data_len_per_query.pkl')
        if self.data_len_per_query is not None:
            with open(data_len_per_query_local_path, 'wb') as fout:
                pkl.dump(self.data_len_per_query, fout)
            # hcopy(data_len_per_query_local_path, f"{remote_global_step_folder}/data_len_per_query.pkl")
            ray.get(
                self.ckpt_global_uploader.register_upload_task.remote("default", self.global_step,
                                                                      ray.get_runtime_context().get_node_id(),
                                                                      data_len_per_query_local_path,
                                                                      remote_global_step_folder))

        # save len_ema
        len_ema_local_path = os.path.join(local_global_step_folder, 'len_ema.json')
        if self.data_len_per_query is not None:
            with open(len_ema_local_path, 'wb') as fout:
                pkl.dump(self.reward_fn.len_ema, fout)
            ray.get(
                self.ckpt_global_uploader.register_upload_task.remote("default", self.global_step,
                                                                      ray.get_runtime_context().get_node_id(),
                                                                      len_ema_local_path, remote_global_step_folder))

        # save acc_per_query
        acc_per_query_local_path = os.path.join(local_global_step_folder, 'acc_per_query.pkl')
        if len(self.acc_per_query) != 0:
            with open(acc_per_query_local_path, 'wb') as fout:
                pkl.dump(self.acc_per_query, fout)
            ray.get(
                self.ckpt_global_uploader.register_upload_task.remote("default", self.global_step,
                                                                      ray.get_runtime_context().get_node_id(),
                                                                      acc_per_query_local_path,
                                                                      remote_global_step_folder))

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)
        # save repaly buffer
        if self.config.algorithm.get('replay', False):
            replay_buffer_path = os.path.join(local_global_step_folder, 'replay_buffer.pt')
            replay_buffer_count_path = os.path.join(local_global_step_folder, 'replay_buffer_count.pt')
            torch.save(self.replay_buffer, replay_buffer_path)
            torch.save(self.replay_buffer_count, replay_buffer_count_path)
            # upload to hdfs
            ray.get(
                self.ckpt_global_uploader.register_upload_task.remote("default", self.global_step,
                                                                      ray.get_runtime_context().get_node_id(),
                                                                      replay_buffer_path, remote_global_step_folder))
            ray.get(
                self.ckpt_global_uploader.register_upload_task.remote("default", self.global_step,
                                                                      ray.get_runtime_context().get_node_id(),
                                                                      replay_buffer_count_path,
                                                                      remote_global_step_folder))
        if self.config.algorithm.priority_sample:
            sample_pool_path = os.path.join(local_global_step_folder, 'sample_pool.pickle')
            with open(sample_pool_path, 'wb') as f:
                pkl.dump(self.sample_pool, f)
            ray.get(
                self.ckpt_global_uploader.register_upload_task.remote("default", self.global_step,
                                                                      ray.get_runtime_context().get_node_id(),
                                                                      sample_pool_path, remote_global_step_folder))
        # upload to hdfs
        ray.get(
            self.ckpt_global_uploader.register_upload_task.remote("default", self.global_step,
                                                                  ray.get_runtime_context().get_node_id(),
                                                                  dataloader_local_path, remote_global_step_folder))
        self.ckpt_global_uploader.start_uploading.remote("default", self.global_step)

        use_ref_ema = self.config.actor_rollout_ref.ref.ema < 1

        actor_upload_future = None
        if self.config.trainer.critic_warmup > self.global_step:
            # critic warmup ckpt
            if not os.path.exists(actor_local_path):
                os.makedirs(actor_local_path)
            ignore_marker = os.path.join(actor_local_path, self._ckpt_ignore_marker_name('actor'))
            print(f'Actor is not optimized so its checkpoint will not be saved. A marker file {ignore_marker} '
                  f'will be uploaded to {remote_global_step_folder}.')
            with open(ignore_marker, 'wb') as fout:
                pkl.dump('actor', fout)
            ray.get(
                self.ckpt_global_uploader.register_upload_task.remote('actor', self.global_step,
                                                                      ray.get_runtime_context().get_node_id(),
                                                                      ignore_marker, remote_global_step_folder))
            self.ckpt_global_uploader.start_uploading.remote('actor', self.global_step)
        else:
            actor_upload_future = self.actor_rollout_wg.save_checkpoint(
                actor_local_path, actor_remote_path,
                specified_ckpt_version if specified_ckpt_version is not None else self.config.trainer.ckpt_version,
                self.global_step, self.ckpt_global_uploader, self.config.trainer.ckpt_enable_shm, 'actor')

        if actor_upload_future is not None:
            ray.get(actor_upload_future)

        if self.use_critic:
            critic_upload_future = self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path,
                specified_ckpt_version if specified_ckpt_version is not None else self.config.trainer.ckpt_version,
                self.global_step, self.ckpt_global_uploader, self.config.trainer.ckpt_enable_shm)
        else:
            critic_upload_future = None

        if use_ref_ema:
            ref_uploader_future = self.actor_rollout_wg.save_checkpoint(
                ref_local_path, ref_remote_path,
                specified_ckpt_version if specified_ckpt_version is not None else self.config.trainer.ckpt_version,
                self.global_step, self.ckpt_global_uploader, self.config.trainer.ckpt_enable_shm, 'ref')
        else:
            ref_uploader_future = None

        if critic_upload_future is not None:
            ray.get(critic_upload_future)

        if ref_uploader_future is not None:
            ray.get(ref_uploader_future)

    def get_resume_checkpoint_info(self):
        """
        Get checkpoint info from checkpoint, return 0 if no checkpoint exists or train from scratch.
        Otherwise, resume training.
        """
        if self.config.trainer.resume_steps == 'disable':
            return 0, None

        remote_checkpoint_folder = os.path.join(self.config.trainer.default_hdfs_dir, 'checkpoints')
        remote_global_step_folder = find_latest_ckpt_path_(
            remote_checkpoint_folder, self.use_standalone_rollout,
            self.config.actor_rollout_ref.rollout.mode)  # None if no latest
        # find remote_global_step_folder
        if self.config.trainer.resume_steps == 'auto':
            if remote_global_step_folder is None:
                print('Training from scratch')
                return 0, None
        else:
            if not (self.config.trainer.auto_over_others and remote_global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_steps, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_steps, "resume ckpt must specify the global_step"
                remote_global_step_folder = self.config.trainer.resume_steps

        # set global step
        global_step = int(remote_global_step_folder.split('global_step_')[-1])
        return global_step, remote_global_step_folder

    def build_model_from_scratch(self, resume_from_step, role=''):
        from_scratch = resume_from_step == 0
        if from_scratch:
            return from_scratch

        # if resuming ckpt
        if self.config.trainer.resume_steps == 'auto':
            remote_checkpoint_folder = os.path.join(self.config.trainer.default_hdfs_dir, 'checkpoints')
            remote_global_step_folder = find_latest_ckpt_path_(remote_checkpoint_folder, self.use_standalone_rollout,
                                                               self.config.actor_rollout_ref.rollout.mode)
        else:
            remote_global_step_folder = self.config.trainer.resume_steps
        if hdfs_io.hexists(os.path.join(remote_global_step_folder,
                                        self._ckpt_ignore_marker_name(role))) and not hdfs_io.hexists(
                                            os.path.join(remote_global_step_folder, role)):
            print(f'Build {role} from scratch though resuming ckpt step is not 0 because {role} ckpt for step was not '
                  'saved due to the model was not optimized.')
            from_scratch = True
        return from_scratch

    @staticmethod
    def _ckpt_ignore_marker_name(role):
        return f'{role}_not_optimized_ignore.txt'

    def load_checkpoint(self, is_self_load=False):
        """is_self_load True if the checkpoint is from current job (for example convert to omnistore),
                        False if the checkpoint is from other job
        """
        global_step = self.global_step
        remote_global_step_folder = self.resume_folder  # None if no latest

        print(f'Setting global step to {global_step}')
        print(f'Resuming from {remote_global_step_folder}')

        actor_remote_path = os.path.join(remote_global_step_folder, 'actor')
        critic_remote_path = os.path.join(remote_global_step_folder, 'critic')
        ref_remote_path = os.path.join(remote_global_step_folder, 'ref')
        checkpoint_infos = {}
        # load actor
        if hdfs_io.hexists(
                os.path.join(remote_global_step_folder,
                             self._ckpt_ignore_marker_name('actor'))) and not hdfs_io.hexists(actor_remote_path):
            # critic warmup ckpt
            print('Ignore to load actor checkpoint which does not exist because it has not been optimized and thus '
                  'not been saved.')
        else:
            self.actor_rollout_wg.load_checkpoint(actor_remote_path, self.config.trainer.ckpt_version,
                                                  self.config.trainer.ckpt_enable_shm, 'actor')
        checkpoint_infos[actor_remote_path] = {"tag": "actor", "step": global_step}
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_remote_path, self.config.trainer.ckpt_version,
                                           self.config.trainer.ckpt_enable_shm)
            checkpoint_infos[critic_remote_path] = {"tag": "critic", "step": global_step}

        # load ref
        use_ref_ema = self.config.actor_rollout_ref.ref.ema < 1
        if use_ref_ema:
            self.actor_rollout_wg.load_checkpoint(ref_remote_path, self.config.trainer.ckpt_version,
                                                  self.config.trainer.ckpt_enable_shm, 'ref')
            checkpoint_infos[ref_remote_path] = {"tag": "ref", "step": global_step}

        if not is_self_load:
            safely_do(lambda: report_trial_ckpts_load(checkpoint_infos=checkpoint_infos), rank=0)()

        # load dataloader
        self.train_dataloader = self.dataloader_mgr._load_dataloader(remote_global_step_folder,
                                                                     self.config.trainer.donot_resume_data)

        # load replay buffer
        if self.config.algorithm.get('replay', False):
            replay_buffer_remote_path = os.path.join(remote_global_step_folder, 'replay_buffer.pt')
            replay_buffer_count_remote_path = os.path.join(remote_global_step_folder, 'replay_buffer_count.pt')
            replay_buffer_local_path = copy_local_path_from_hdfs(replay_buffer_remote_path)
            replay_buffer_count_local_path = copy_local_path_from_hdfs(replay_buffer_count_remote_path)
            self.replay_buffer = torch.load(replay_buffer_local_path)
            self.replay_buffer_count = torch.load(replay_buffer_count_local_path)
        if self.config.algorithm.priority_sample:
            sample_pool_remote_path = os.path.join(remote_global_step_folder, 'sample_pool.pickle')
            sample_pool_local_path = copy_local_path_from_hdfs(sample_pool_remote_path)
            with open(sample_pool_local_path, 'rb') as f:
                self.sample_pool = pkl.load(f)

        # resume data_len info
        data_len_per_query_remote_path = os.path.join(remote_global_step_folder, 'data_len_per_query.pkl')
        if hexists(data_len_per_query_remote_path):
            data_len_per_query_local_path = copy_local_path_from_hdfs(data_len_per_query_remote_path)
            with open(data_len_per_query_local_path, 'rb') as fin:
                self.data_len_per_query = pkl.load(fin)
                print("DATA_LEN INFO RESUMED!!!!!!")
        else:
            self.data_len_per_query = None

        # resume len_ema
        len_ema_remote_path = os.path.join(remote_global_step_folder, 'len_ema.json')
        if hexists(len_ema_remote_path):
            len_ema_local_path = copy_local_path_from_hdfs(len_ema_remote_path)
            with open(len_ema_local_path, 'rb') as fin:
                self.reward_fn.len_ema = pkl.load(fin)
                print("LEN_EMA RESUMED!!!!!!")

        # resume acc_per_query
        acc_per_query_remote_path = os.path.join(remote_global_step_folder, 'acc_per_query.pkl')
        if hexists(acc_per_query_remote_path):
            acc_per_query_local_path = copy_local_path_from_hdfs(acc_per_query_remote_path)
            with open(acc_per_query_local_path, 'rb') as fin:
                self.acc_per_query = pkl.load(fin)
                print("acc_per_query RESUMED!!!!!!")

        self.rollout_manager.resume(remote_global_step_folder, load_dataproto_fn=load_dataproto)

    def _balance_batch(self, batch, metrics, logging_prefix='global_seqlen'):
        # Note that the reorder is in place
        # perform global sequence balancing here
        # attention_mask can be [bsz * bon, seqlen] or [bsz, bon, seqlen]
        attention_mask = batch.batch['attention_mask']
        assert len(attention_mask.shape) == 2 or len(attention_mask.shape) == 3
        print(f'Perform seqlen balancing with shape {attention_mask.shape}')
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        # note that this may be problematic when the world_size differs
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)
        print_dataproto_size(batch, head='After Sequence Balancing')

    def _preprocess_batch_before_gen(self, batch, metrics, start_step):
        # print the size of each data proto before training
        print_dataproto_size(batch, head='Before generation')

        if self.config.data.num_prompts_per_data > 1:
            batch = batch.unfold_column_chunks(self.config.data.num_prompts_per_data,
                                               split_keys=['input_ids', 'attention_mask', 'prompt_names'])

        if self.config.algorithm.prior_sampling.enable:
            repeat_num = self.compute_repeat_num_per_query(batch)
            save_repeat_num = pd.DataFrame({
                'index': batch.non_tensor_batch['index'],
                'repeat_num': repeat_num,
                'acc': [self.acc_per_query.get(idx, None) for idx in batch.non_tensor_batch['index']]
            })
            if self.global_step == 1:
                makedirs(self.sample_acc_dir)

            save_repeat_num.to_parquet(f"sample_acc.{self.global_step}.parquet")
            hcopy(f"sample_acc.{self.global_step}.parquet", self.sample_acc_dir)

            repeat_num = repeat_num if repeat_num != None else [self.num_bon] * len(batch)
            batch = batch.sample_level_repeat(repeat_num)

            metrics.update({
                'repeat_num/max': max(repeat_num),
                'repeat_num/min': min(repeat_num),
                'repeat_num/std': np.std(repeat_num),
                'repeat_num/median': np.median(repeat_num),
            })
            batch.non_tensor_batch['rollout_id'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))],
                                                            dtype=object)
        else:
            batch.non_tensor_batch['rollout_id'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))],
                                                            dtype=object)
            batch = batch.repeat(self.num_bon)

        # create a uid for each data inside the batch
        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
        batch.check_consistency()
        return batch

    def get_mean_max_len_per_query(self, batch, metrics):
        if self.config.trainer.per_query_max_length is True:
            if self.data_len_per_query is None:
                batch.non_tensor_batch['max_new_tokens'] = np.array([self.config.trainer.per_query_max_length_init] *
                                                                    batch.batch['input_ids'].shape[0],
                                                                    dtype=object)
                batch.non_tensor_batch['mean_new_tokens'] = np.array(
                    [self.config.trainer.per_query_max_length_init // 2] * batch.batch['input_ids'].shape[0],
                    dtype=object)
            else:
                per_query_max_length = []
                per_query_mean_length = []
                for i in range(batch.batch['input_ids'].shape[0]):
                    len_lst = torch.tensor(
                        self.data_len_per_query.get(batch.non_tensor_batch['index'][i],
                                                    self.data_len_per_query.get('overall_mean')))  # TODO
                    count = 16
                    mean_len = torch.mean(len_lst[-count:])
                    std_len = torch.std(len_lst[-count:]) if len(len_lst) > 2 else mean_len
                    max_len = 3 * mean_len + std_len
                    per_query_max_length.append(max_len)
                    per_query_mean_length.append(mean_len)
                batch.non_tensor_batch['max_new_tokens'] = np.array(per_query_max_length, dtype=object)
                batch.non_tensor_batch['mean_new_tokens'] = np.array(per_query_mean_length, dtype=object)

            metrics.update({
                "max_len_per_query/mean": batch.non_tensor_batch['max_new_tokens'].mean(),
                "max_len_per_query/std": batch.non_tensor_batch['max_new_tokens'].std(),
                "max_len_per_query/min": batch.non_tensor_batch['max_new_tokens'].min(),
                "max_len_per_query/max": batch.non_tensor_batch['max_new_tokens'].max(),
            })

    def update_len_per_query(self, batch, metrics):
        if self.data_len_per_query is None:
            self.data_len_per_query = defaultdict(list)

        cur_batch_len = []
        prompt_length = self.config.data.max_prompt_length
        max_response_length = self.config.data.max_response_length
        overlonged_count = 0
        for i in range(batch.batch['input_ids'].shape[0]):
            valid_response_length = batch.batch['attention_mask'][i, prompt_length:].sum().item()  # TODO
            self.data_len_per_query[batch.non_tensor_batch['index'][i]].append(valid_response_length)
            cur_batch_len.append(valid_response_length)

            if 'max_new_tokens' in batch.non_tensor_batch:
                query_max_len = batch.non_tensor_batch['max_new_tokens'][i]
            else:
                query_max_len = max_response_length
            if valid_response_length >= query_max_len:
                overlonged_count += 1

        cur_batch_mean = sum(cur_batch_len) // len(cur_batch_len)
        self.data_len_per_query['overall_mean'].append(cur_batch_mean)
        overlonged_frac = overlonged_count / len(batch.batch['input_ids'])

        metrics.update({
            "max_len_per_query/cur_batch_mean": sum(cur_batch_len) // len(cur_batch_len),
            "max_len_per_query/cur_batch_std": np.std(cur_batch_len),
            "max_len_per_query/cur_batch_min": min(cur_batch_len),
            "max_len_per_query/cur_batch_max": max(cur_batch_len),
            "max_len_per_query/overlonged_frac": overlonged_frac,
        })

    def update_acc_per_query(self, id2acc):
        for k, v in id2acc.items():
            if k not in self.acc_per_query:
                self.acc_per_query[k] = v
            else:
                self.acc_per_query[k] = (
                    1 - self.config.algorithm.prior_sampling.prior_sampling_ema
                ) * self.acc_per_query[k] + self.config.algorithm.prior_sampling.prior_sampling_ema * v

    def compute_repeat_num_per_query(self, batch):
        """
        return: list of allocations (length == len(weights))
        """
        N = len(batch) * self.num_bon  # total sample num
        M = self.num_bon  # mean sample num
        acc_list = [self.acc_per_query.get(index, None) for index in batch.non_tensor_batch['index']
                   ]  # [0, 1], None for no weights
        weights = []
        weight_clip = self.config.algorithm.prior_sampling.min_weight_clip
        temperature = self.config.algorithm.prior_sampling.temperature
        for acc in acc_list:
            if acc is None:
                weights.append(None)
            elif acc > self.config.algorithm.prior_sampling.no_sample_max_threshold:  # no weight for samples with acc > threshold
                weights.append(0)
            elif acc < self.config.algorithm.prior_sampling.no_sample_min_threshold:  # no weight for samples with acc < threshold
                weights.append(weight_clip)
            else:
                if self.config.algorithm.prior_sampling.only_filtering:
                    weights.append(1)
                else:  # weight reverse to acc
                    weights.append(1 - acc)

        # add temperature for weights
        weights = [v**temperature if v is not None else None for v in weights]
        B = len(weights)

        weighted_indices = []
        unweighted_indices = []
        for i, w in enumerate(weights):
            if w is None:
                unweighted_indices.append(i)
            else:
                weighted_indices.append(i)

        # Allocate M repeats to each unweighted sample (samples with no acc)
        allocation = [0] * B
        for i in unweighted_indices:
            allocation[i] = M

        # Compute quota remained after unweighted students
        allocated_unweighted = len(unweighted_indices) * M
        R0 = N - allocated_unweighted
        if R0 < 0:
            raise ValueError(f"Not enough samples (N={N}) to give M={M} each "
                             f"to {len(unweighted_indices)} unweighted samples!")

        # Allocate at least 0 repeats to each weighted sample
        for i in weighted_indices:
            allocation[i] = 0

        # Distribute the final remainder R0 among the weighted samples proportionally based on their weights.
        # Sum of all valid weights
        sum_weights = sum(weights[i] for i in weighted_indices if weights[i] is not None)
        # Calculate fractional allocations and integer floors
        fractional_allocations = []
        for i in weighted_indices:
            w = weights[i]
            frac = (w / sum_weights) * R0 if sum_weights > 0 else 0
            fractional_allocations.append((i, frac))

        base_allocations = []
        for i, frac in fractional_allocations:
            base_allocations.append((i, int(frac)))  # floor

        # Sum of base allocations
        base_sum = sum(x[1] for x in base_allocations)
        leftover = R0 - base_sum

        # Sort by fractional remainder, descending
        # remainder_i = frac - floor(frac)
        remainders = [(i, frac - int(frac)) for (i, frac) in fractional_allocations]
        remainders.sort(key=lambda x: x[1], reverse=True)

        # Assign leftover repeats, one by one, to the top fractional remainders
        for k in range(leftover):
            idx = remainders[k][0]
            # Increase that base allocation by 1
            for bi in range(len(base_allocations)):
                if base_allocations[bi][0] == idx:
                    base_allocations[bi] = (idx, base_allocations[bi][1] + 1)
                    break

        for i, val in base_allocations:
            allocation[i] += val

        # Sanity-check: The total must be exactly N
        if sum(allocation) != N:
            raise RuntimeError("Allocation does not sum to N; check logic.")

        return allocation

    def cal_corr(self, x, y, mask, epsilon=1e-5):
        x_mean = (x * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        y_mean = (y * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        xy_mean = (x * y * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        x_square_mean = (x * x * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        y_square_mean = (y * y * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        return (xy_mean - x_mean * y_mean) / (
            torch.sqrt(torch.clamp(
                (x_square_mean - x_mean * x_mean) * (y_square_mean - y_mean * y_mean), min=0.0)) + epsilon)

    def fit(self):
        self._create_kl_control()
        self._create_dataloader()
        self._create_rollout_manager()
        self._create_validation_manager()

        if self.save_batch_dir:
            makedirs(self.save_batch_dir, exist_ok=True)

        metric_collection_context = self.megavision_metrics_collector.collect_resume_from_checkpoint_duration() \
            if MegavisionMetricsCtx else contextlib.nullcontext()
        with metric_collection_context:
            if self.global_step != 0:
                # load checkpoint before doing anything
                self.load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and (self.config.trainer.eval_before_training or
                                               self.config.trainer.val_only):
            self.validation_manager.validate(val_epoch=self.config.trainer.val_epoch,
                                             need_log=self.config.trainer.need_log,
                                             log_file=self.config.trainer.log_file,
                                             is_async=False,
                                             global_step=self.global_step)
        if self.config.trainer.val_only:
            if self.config.trainer.save_train_batch_dir is not None and self.config.trainer.need_log:
                hput(self.config.trainer.log_file, self.config.trainer.save_train_batch_dir)
            return

        # Note that we start from step 1. After resume, we increment step by 1 to start next step
        self.global_step += 1
        start_step = self.global_step
        rollout_counter = 0
        rollout_pool_metrics = {}
        while True:
            start_data_time = time.time()
            for batch_dict in self.train_dataloader:
                metrics = {}
                metrics['timing/dataloader'] = time.time() - start_data_time
                with Timer(name='step', logger=None) as step_timer:
                    # hybrid generate (on policy)
                    if self.config.trainer.load_train_batch_path is None:
                        batch: DataProto = DataProto.from_single_dict(batch_dict)

                        if self.config.algorithm.priority_sample:
                            self.sample_pool.fill_sample_pool(batch)
                            self.sample_pool.rearrange_sample_pool()
                            if self.config.algorithm.TD_priority_ratio > 0:
                                self.sample_pool.rearrange_TD_sample_pool(
                                    int(self.config.data.train_batch_size * self.config.algorithm.TD_priority_ratio))
                            batch = self.sample_pool.get_gen_batch(self.config.data.train_batch_size)
                        self.get_mean_max_len_per_query(batch, metrics)

                        batch = self._preprocess_batch_before_gen(batch, metrics, start_step)
                        # generate
                        is_warmup_step = self.global_step < self.rollout_pool_warmup_step + start_step
                        with Timer(name='generate', logger=None) as timer:
                            save_path = f"{self.config.trainer.default_hdfs_dir}/checkpoints/global_step_{self.global_step - 1}/"
                            save_dataproto_fn = partial(save_dataproto, path=save_path)
                            batch = self.rollout_manager.train_generate(batch,
                                                                        step=self.global_step,
                                                                        save_dataproto_fn=save_dataproto_fn,
                                                                        is_warmup_step=is_warmup_step,
                                                                        metrics=metrics)
                        metrics['timing/generate'] = timer.last
                        if batch is None or len(batch) == 0 or is_warmup_step:
                            self.logger.log(data=metrics, step=self.global_step)
                            self.global_step += 1
                            continue
                        if self.config.trainer.save_train_batch_dir is not None:
                            makedirs(self.config.trainer.save_train_batch_dir, exist_ok=True)
                            local_path = f'train_batch_{self.global_step}.pt'
                            batch.save_to_disk(local_path)
                            hput(local_path, self.config.trainer.save_train_batch_dir)
                            print(f'Saving train batch from {local_path} to {self.config.trainer.save_train_batch_dir}')
                    else:
                        print(f'Using loaded train batch {self.config.trainer.load_train_batch_path} for training')
                        batch_local_filepath = copy_local_path_from_hdfs(self.config.trainer.load_train_batch_path)
                        batch = DataProto.load_from_disk(batch_local_filepath)

                    batch.meta_info['global_step'] = self.global_step
                    self.update_len_per_query(batch, metrics)

                    # training
                    with Timer(name='rm_score', logger=None) as timer:
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                            metrics['memory/rm_max_allocated'] = reward_tensor.meta_info['memory/rm_max_allocated']
                            metrics['memory/rm_max_reserved'] = reward_tensor.meta_info['memory/rm_max_reserved']
                    metrics['timing/rm_score'] = timer.last

                    print_dataproto_size(batch, head='After Reward Model')

                    with Timer(name='reward_fn', logger=None) as timer:
                        # we combine with rule-based rm
                        reward_tensor, raw_scores, length_scores, eos_ids = self.reward_fn(batch,
                                                                                           global_step=self.global_step)
                        batch.batch['token_level_scores'] = reward_tensor
                        batch.batch['raw_scores'] = raw_scores
                        batch.batch['eos_ids'] = eos_ids
                        response_length = batch.batch['attention_mask'][:, -batch.batch['responses'].shape[1]:].sum(-1)
                        raw_scores_log = raw_scores.sum(-1)
                        length_ranges = [(None, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192),
                                         (8192, 16384), (16384, 32768), (32768, 65536)]
                        scores = calculate_score_in_length_ranges(raw_scores_log, response_length, length_ranges)
                        metrics.update(scores)
                        self.logger.log(data={"score/raw_score": wandb.Histogram(raw_scores_log)},
                                        step=self.global_step)
                        if self.config.algorithm.inference_scaling != 'v0':
                            length_scores = length_scores.sum(-1)
                            self.logger.log(data={"score/length_score": wandb.Histogram(length_scores)},
                                            step=self.global_step)
                    metrics['timing/reward_fn'] = timer.last

                    if self.config.algorithm.priority_sample:
                        self.sample_pool.update_priority_dict(batch)

                    with Timer(name='dynamic_sampling', logger=None) as timer:
                        if self.config.algorithm.dynamic_sampling.enable:
                            batch_metrics_before_fill = calculate_batch_bon_metrics(batch, "rollout_pool")
                            merge_metrics(rollout_pool_metrics, batch_metrics_before_fill)

                            # fill rollout out pool with grad
                            fill_size, pool_size = RolloutPool.dynamic_call(self.rollout_pool,
                                                                            "fill_rollout_pool_dynamic_sampling", batch)
                            return_batch_size = self.config.data.actor_training_batch_size * self.num_bon
                            rollout_counter += 1
                            if RolloutPool.dynamic_call(self.rollout_pool,
                                                        "get_dynamic_sampling_pool_size") < return_batch_size:
                                print(
                                    f'[RolloutPool] pool_with_grad_size: {RolloutPool.dynamic_call(self.rollout_pool, "get_dynamic_sampling_pool_size")}'
                                )
                                metrics[f'rollout_pool/fill_size_{rollout_counter}'] = fill_size
                                metrics[f'rollout_pool/pool_size_{rollout_counter}'] = pool_size
                                if rollout_counter > 10:
                                    assert False, 'Do not make sense. Check Your DATA!!!'
                                continue
                            else:
                                train_batch = RolloutPool.dynamic_call(self.rollout_pool, "get_train_batch_grad",
                                                                       return_batch_size)
                                batch = DataProto.concat(train_batch)
                                batch.meta_info[
                                    'generation_kwargs'] = self.config.actor_rollout_ref.rollout.train_generate_kwargs
                                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'],
                                                                                dim=-1).tolist()
                                if 'pixel_values' in batch.non_tensor_batch:
                                    batch.meta_info['global_img_token_num'] = [
                                        t.shape[0] if t is not None else 0
                                        for t in batch.non_tensor_batch['pixel_values']
                                    ]
                                metrics['rollout/training_batch'] = len(batch)
                                for key in rollout_pool_metrics:  # fix: mean acc for each rollout batch
                                    if '/acc_' in key and type(rollout_pool_metrics[key]) in [float, int]:
                                        rollout_pool_metrics[key] /= rollout_counter
                                print(
                                    f'[RolloutPool] BeginTraining pool_with_grad_size: {RolloutPool.dynamic_call(self.rollout_pool, "get_dynamic_sampling_pool_size")}'
                                )
                            if self.config.algorithm.dynamic_sampling.sync:
                                pool_with_grad_size = RolloutPool.dynamic_call(self.rollout_pool,
                                                                               "pool_with_grad_clear")
                                print(f'[RolloutPool] AfterClear pool_with_grad_size: {pool_with_grad_size}')
                            rollout_pool_metrics['rollout_pool/fill_counter'] = rollout_counter
                            metrics.update(rollout_pool_metrics)
                            rollout_pool_metrics = {}
                            rollout_counter = 0
                    metrics['timing/dynamic_sampling'] = timer.last

                    if self.config.algorithm.mask_overlong:
                        prompt_length = self.config.data.max_prompt_length
                        if 'max_new_tokens' in batch.non_tensor_batch:
                            response_length = batch.non_tensor_batch['max_new_tokens']
                        else:
                            response_length = self.config.data.max_response_length
                        valid_response_length = batch.batch['attention_mask'][:, prompt_length:].sum(-1)
                        is_vlm = self.config.data['image_key'] is not None
                        if is_vlm:
                            is_overlong = (response_length == valid_response_length) & (raw_scores_log == 0)
                        else:
                            is_overlong = (response_length
                                           == valid_response_length) & (batch.batch['raw_scores'].sum(-1) < 0)
                        # batch.batch['attention_mask'][is_overlong] = 0
                        # batch.batch['answer_attention_mask'][is_overlong] = 0
                        batch.batch['overlong_mask'] = (~is_overlong).int()
                        metrics.update({'max_len_per_query/overlong_masked': is_overlong.to(torch.int64).sum().item()})

                    print_dataproto_size(batch, head='After Reward function')

                    # league training，筛选平均通过率低的prompt
                    use_async_gen = self.config.streaming_rollout.nnodes > 0
                    if self.config.trainer.league_training_config.enable:
                        with Timer(name='select_league_training_prompts', logger=None) as timer:
                            if use_async_gen:
                                batch, league_training_metrics = league_training_filter_prompt_v2(
                                    batch=batch,
                                    strategy=self.config.trainer.league_training_config.strategy,
                                    config=self.config)
                                metrics.update(league_training_metrics)
                            else:
                                batch = league_training_filter_prompt(
                                    batch=batch,
                                    strategy=self.config.trainer.league_training_config.strategy,
                                    config=self.config)
                        metrics['timing/select_league_training_prompts'] = timer.last

                    id2acc = defaultdict(list)
                    # bon策略，筛选prompt内部的response，有不同策略，all、best、best_mix_random、best_worst
                    if self.num_bon > 1:
                        with Timer(name='select_bon_samples', logger=None) as timer:
                            batch, bon_metrics, id2acc = select_training_samples_v2(
                                batch=batch,
                                strategy=self.config.actor_rollout_ref.rollout.bon_strategy,
                                config=self.config)
                            if use_async_gen:
                                metrics.update(bon_metrics)
                        metrics['timing/select_bon_samples'] = timer.last

                    # update acc_per_query
                    if len(id2acc) == 0:
                        for idx, score in zip(batch.non_tensor_batch['index'],
                                              batch.batch['token_level_scores'].sum(-1)):
                            score = score.item()
                            id2acc[idx].append(score)
                        for k, v in id2acc.items():
                            id2acc[k] = sum([1 for i in v if i == 1]) / len(v)
                    self.update_acc_per_query(id2acc)

                    if self.config.algorithm.shuffle_sample_batch:
                        idx_lst = list(range(batch.batch.batch_size[0]))
                        random.shuffle(idx_lst)
                        batch.reorder(torch.tensor(idx_lst))

                    # perform sequence balancing.
                    # Very important: Note that this reorders data globally.
                    # So anything that requires ordering below this line will cause incorrect results
                    self._balance_batch(batch=batch, metrics=metrics, logging_prefix='global_seqlen')

                    metrics.setdefault('timing/train_mem_offload', 0)

                    # compute reference
                    if self.use_reference_policy:
                        if not (self.config.actor_rollout_ref.actor.kl_loss_weight == 0 and
                                self.config.algorithm.kl_ctrl.kl_coef == 0):
                            # skip ref log prob if no kl loss
                            # compute reference log_prob
                            with Timer(name='ref', logger=None) as timer:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)
                            metrics['timing/ref'] = timer.last
                            metrics['memory/ref_max_allocated'] = ref_log_prob.meta_info['memory/ref_max_allocated']
                            metrics['memory/ref_max_reserved'] = ref_log_prob.meta_info['memory/ref_max_reserved']

                    print_dataproto_size(batch, head='After reference policy')

                    input_batch = batch
                    if self.enable_actor_critic_spatial_mux:
                        input_batch = input_batch.repeat(2, interleave=False)

                    # compute actor
                    actor_future = self.actor_rollout_wg.old_log_probs(input_batch)

                    # compute values
                    if self.use_critic and self.enable_actor_critic_spatial_mux:
                        critic_future = self.critic_wg.compute_values(input_batch)

                    # get old_log_probs
                    with Timer(name='old_log_probs', logger=None) as timer:
                        output_batch = actor_future.get()
                        batch = output_batch.chunk(2)[0] if self.enable_actor_critic_spatial_mux else output_batch
                    metrics['timing/old_log_probs'] = timer.last

                    print_dataproto_size(batch, head='After old log probs')

                    # get values
                    if self.use_critic:
                        if not self.enable_actor_critic_spatial_mux:
                            critic_future = self.critic_wg.compute_values(input_batch)
                        with Timer(name='values', logger=None) as timer:
                            values = critic_future.get()
                            values = values.chunk(2)[1] if self.enable_actor_critic_spatial_mux else values
                            batch = batch.union(values)
                        metrics['timing/values'] = timer.last

                    print_dataproto_size(batch, head='After compute values')

                    with Timer(name='adv', logger=None) as timer:
                        # compute rewards. apply_kl_penalty if available
                        batch, kl_metrics = apply_kl_penalty(
                            batch,
                            kl_ctrl=self.kl_ctrl,
                            kl_penalty=self.config.algorithm.kl_penalty,
                            use_model_output_mask=self.config.algorithm.use_model_output_mask)
                        metrics.update(kl_metrics)

                        # compute advantages
                        batch, adv_metrics = compute_advantage(
                            batch,
                            self.config.algorithm.gamma,
                            self.config.algorithm.lam,
                            self.config.algorithm.use_variable_lambda,
                            self.config.algorithm.variable_lambda_scalar,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            upgo_loss_version=self.config.actor_rollout_ref.actor.upgo_loss_version,
                            num_bon=self.config.actor_rollout_ref.rollout.num_bon,
                            adv_whiten=self.config.algorithm.adv_whiten,
                            use_async_gen=use_async_gen,
                            group_mode=self.config.algorithm.group_mode,
                            use_separate_critic_lam=self.config.algorithm.use_separate_critic_lam,
                            critic_lam=self.config.algorithm.critic_lam,
                            use_model_output_mask=self.config.algorithm.use_model_output_mask,
                        )
                        metrics.update(adv_metrics)
                    metrics['timing/adv'] = timer.last

                    print_dataproto_size(batch, head='After compute adv')

                    if self.global_step == 1:
                        print('Debugging', batch.batch)

                    input_batch = batch
                    if self.enable_actor_critic_spatial_mux:
                        input_batch = input_batch.repeat(2, interleave=False)

                    # update actor
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_step and self.global_step % self.config.trainer.actor_update_freq == 0:
                        actor_future = self.actor_rollout_wg.update_actor(input_batch)

                    # update critic
                    if self.use_critic:
                        critic_future = self.critic_wg.update_critic(input_batch)

                    if self.config.trainer.critic_warmup <= self.global_step and self.global_step % self.config.trainer.actor_update_freq == 0:
                        with Timer(name='update_actor', logger=None) as timer:
                            if os.environ.get("MINISTEPS_ON_DRIVER", "0") == "1":
                                dataloader = make_mini_step_dataloader(
                                    batch, self.config.actor_rollout_ref.actor.ppo_mini_batch_size, True)
                                ministeps_metrics = []
                                for batch_idx, mini_batch in enumerate(dataloader):
                                    if batch_idx == (len(dataloader) - 1):
                                        mini_batch.meta_info["lr_scheduler_step"] = True
                                    actor_output_mini = self.actor_rollout_wg.train_actor(mini_batch)
                                    ministeps_metrics.append(actor_output_mini.meta_info['metrics'])

                                actor_output = actor_output_mini
                                actor_output_metrics = merge_ministeps_metrics(ministeps_metrics)
                            else:
                                actor_output = actor_future.get()
                                actor_output = actor_output.chunk(
                                    2)[0] if self.enable_actor_critic_spatial_mux else actor_output
                                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])

                        metrics.update(actor_output_metrics)
                        metrics['memory/actor_max_allocated'] = actor_output.meta_info['memory/actor_max_allocated']
                        metrics['memory/actor_max_reserved'] = actor_output.meta_info['memory/actor_max_reserved']
                        metrics['timing/update_actor'] = timer.last
                        print(f"After update_actor")

                    # update critic
                    if self.use_critic:
                        with Timer(name='update_critic', logger=None) as timer:
                            critic_output = critic_future.get()
                            critic_output = critic_output.chunk(
                                2)[1] if self.enable_actor_critic_spatial_mux else critic_output
                        batch.batch['seq_vf'] = critic_output.batch['seq_vf']
                        metrics['timing/update_critic'] = timer.last
                        metrics['memory/critic_max_allocated'] = critic_output.meta_info['memory/critic_max_allocated']
                        metrics['memory/critic_max_reserved'] = critic_output.meta_info['memory/critic_max_reserved']
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                        print(f"After update_critic")
                    if self.config.algorithm.phasic_critic_interval > 0:
                        select_keys = ['input_ids', 'responses', 'attention_mask', 'values', 'returns']
                        buffer_batch = batch.select(batch_keys=select_keys)
                        if self.phasic_critic_buffer is None:
                            self.phasic_critic_buffer = buffer_batch
                        else:
                            self.phasic_critic_buffer = DataProto.concat([self.phasic_critic_buffer, buffer_batch])

                    # priority use TD-error
                    if self.config.algorithm.priority_sample and self.config.algorithm.TD_priority_ratio > 0:
                        self.sample_pool.update_TD_priority_dict(batch)

                    # phasic critic update
                    phasic_critic_update = self.config.algorithm.phasic_critic_interval > 0 and self.global_step % self.config.algorithm.phasic_critic_interval == 0
                    if phasic_critic_update:
                        self.phasic_critic_buffer.meta_info['phasic_update'] = True
                        with Timer(name='phasic_critic_update', logger=None) as timer:
                            critic_output = self.critic_wg.update_critic(self.phasic_critic_buffer)
                        metrics['timing/phasic_critic_update'] = timer.last
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                        self.phasic_critic_buffer = None

                    # update ref ema
                    with Timer(name='update_ref_ema', logger=None) as timer:
                        self.ref_policy_wg.update_ref_ema()
                    metrics['timing/update_ref_ema'] = timer.last

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_step % self.config.trainer.test_freq == 0:
                        with Timer(name='testing', logger=None) as timer:
                            self.validation_manager.validate(is_async=self.use_standalone_validator,
                                                             global_step=self.global_step)
                        metrics['timing/testing'] = timer.last

                    metric_collection_context = self.megavision_metrics_collector.collect_compute_metrics_duration() \
                        if MegavisionMetricsCtx else contextlib.nullcontext()
                    # collect metrics
                    with metric_collection_context:
                        with Timer(name='compute_metrics', logger=None) as timer:
                            # Note that we can use any worker groups here
                            batch.meta_info['use_critic'] = self.use_critic
                            batch.meta_info['mean'] = self.config.reward_model.mean
                            batch.meta_info['std'] = self.config.reward_model.std
                            batch.meta_info['use_model_output_mask'] = self.config.algorithm.use_model_output_mask
                            data_metrics: DataProto = self.actor_rollout_wg.execute_with_func_generator(
                                compute_data_metrics, batch)
                            data_metrics = data_metrics.meta_info['metrics']
                            metrics.update(data_metrics)
                            sequence_score = batch.batch['token_level_scores'].sum(-1)
                            score_metrics = {}
                            data_sources = batch.non_tensor_batch.get('data_source',
                                                                      ['unknown'] * sequence_score.shape[0])
                            # evaluate test_score based on data source
                            data_source_reward_2nd = defaultdict(list)
                            data_source_reward_1st = defaultdict(list)
                            for i in range(sequence_score.shape[0]):
                                data_source = data_sources[i]
                                data_source_reward_2nd[data_source].append(sequence_score[i])
                                # 一级分类
                                data_source = data_source.split('##')[0]
                                data_source_reward_1st[data_source].append(sequence_score[i])
                            for data_source, rewards in data_source_reward_2nd.items():
                                rewards_tensor_data_source = torch.stack(rewards, dim=0)
                                score_mean = torch.mean(rewards_tensor_data_source)
                                score_max = torch.max(rewards_tensor_data_source)
                                score_min = torch.min(rewards_tensor_data_source)
                                score_std = torch.std(rewards_tensor_data_source)
                                score_metrics.update({
                                    f'critic/score_per_source_mean_2nd/{data_source}':
                                        score_mean.detach().item(),
                                    f'critic/score_per_source_max_2nd/{data_source}':
                                        score_max.detach().item(),
                                    f'critic/score_per_source_min_2nd/{data_source}':
                                        score_min.detach().item(),
                                    f'critic/score_per_source_std_2nd/{data_source}':
                                        score_std.detach().item(),
                                    f'critic/score_per_source_num_2nd/{data_source}':
                                        rewards_tensor_data_source.shape[0],
                                })
                            for data_source, rewards in data_source_reward_1st.items():
                                rewards_tensor_data_source = torch.stack(rewards, dim=0)
                                score_mean = torch.mean(rewards_tensor_data_source)
                                score_max = torch.max(rewards_tensor_data_source)
                                score_min = torch.min(rewards_tensor_data_source)
                                score_std = torch.std(rewards_tensor_data_source)
                                score_metrics.update({
                                    f'critic/score_per_source_mean_1st/{data_source}':
                                        score_mean.detach().item(),
                                    f'critic/score_per_source_max_1st/{data_source}':
                                        score_max.detach().item(),
                                    f'critic/score_per_source_min_1st/{data_source}':
                                        score_min.detach().item(),
                                    f'critic/score_per_source_std_1st/{data_source}':
                                        score_std.detach().item(),
                                    f'critic/score_per_source_num_1st/{data_source}':
                                        rewards_tensor_data_source.shape[0],
                                })
                            metrics.update(score_metrics)

                            # save batch to hdfs
                            if self.save_batch_dir:
                                # show diagnose info for background tracking
                                finished = set()
                                for t in self.async_tracking_running_tasks:
                                    if t.done():
                                        t.result()  # call this to collect the result(including error traceback)
                                        finished.add(t)
                                for t in finished:
                                    self.async_tracking_running_tasks.remove(t)
                                print(f"remaining async tracking tasks {len(self.async_tracking_running_tasks)}")

                                batch_fname = f"global_step_{self.global_step}_batch.pickle"
                                batch.save_to_disk(batch_fname)

                                async_tracking_args = (batch_fname, self.save_batch_dir, self.tokenizer,
                                                       self.global_step)
                                task = self.async_tracking_pool.submit(async_process_batch_samples_to_wandb,
                                                                       *async_tracking_args)
                                self.async_tracking_running_tasks.add(task)

                            advantages = batch.batch['advantages']
                            response_length = batch.batch['responses'].shape[-1]
                            if self.config.algorithm.use_model_output_mask:
                                loss_mask = batch.batch['model_output_mask']
                                response_mask = loss_mask[:, -response_length:]
                            else:
                                attention_mask = batch.batch['attention_mask']
                                response_mask = attention_mask[:, -response_length:]
                            idx = torch.arange(advantages.shape[1]).unsqueeze(dim=0).tile(advantages.shape[0], 1)
                            adv_idx_corr = self.cal_corr(advantages, idx, response_mask)
                            metrics['critic/advantages/adv_idx_corr'] = adv_idx_corr.mean()
                        metrics['timing/compute_metrics'] = timer.last

                    metric_collection_context = self.megavision_metrics_collector.collect_save_checkpoint_duration() \
                        if MegavisionMetricsCtx else contextlib.nullcontext()

                    with Timer(name='save_checkpoint', logger=None) as timer:
                        if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                            with metric_collection_context:
                                self.save_checkpoint()
                    metrics['timing/save_checkpoint'] = timer.last

                    # collect sandbox client remaining results
                    if self.config.trainer.use_remote_sandbox:
                        remote_client = ray.get_actor('remote_client')
                        num_remaining_results = ray.get(remote_client.get_num_pending_outputs.remote())
                        metrics['remote_client/remaining_results'] = num_remaining_results

                metrics['timing/step'] = step_timer.last
                # TODO: make a canonical logger that supports various backend
                self.logger.log(data=metrics, step=self.global_step)
                start_data_time = time.time()

                self.global_step += 1
                if self.global_step >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self.validation_manager.validate(is_async=False, global_step=self.global_step)
                        pprint(f'Final validation metrics: {val_metrics}')

                    # wait for the last ckpt to finish uploading if there are any
                    ray.get(self.ckpt_global_uploader.final_wait_all_steps.remote())

                    # wait for async tracking
                    for t in self.async_tracking_running_tasks:
                        t.result()  # call this to collect the result(including error traceback)
                    wandb.finish()
                    return

    def do_ndtimeline_action(self, *args, **kwargs):
        """Call a function on each actor.
        Args:
            *args: arguments
            **kwargs: keyword arguments
        """
        results = []
        # because worker dict may be used, we only want the real worker to execute do_ndtimeline_action once
        executed_wg_name_prefix = set()
        for wg_name, wg in self.all_wg.items():
            if wg.name_prefix not in executed_wg_name_prefix:
                results.append(wg.do_ndtimeline_action(*args, **kwargs))
                executed_wg_name_prefix.add(wg.name_prefix)
        return results

    def convert_ckpt_to_omnistore(self):
        self.global_step = 0

        # load checkpoint before doing anything
        _ = self.load_checkpoint(is_self_load=True)
        # save omnistore ckpt
        self.save_checkpoint(specified_ckpt_version='omnistore')
        ray.get(self.ckpt_global_uploader.wait_all.remote(self.global_step, False))

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, step: int):
        if self._global_step == step:
            return
        if self._global_step + 1 == step:
            if ndtimeline.use_cuda_timer():
                for fut in self._timeline_futures:
                    ray.get(fut)
                futs = self.do_ndtimeline_action("flush_set_upload", global_step=step, ts=int(time.time()))
                self._timeline_futures = futs
            if self.config.streaming_rollout.query_trace.enable:
                self._rollout_query_tl.step(step)
        for req_mgr in self.request_managers:
            ray.get(req_mgr.set_global_step.remote(self.global_step))
        self._global_step = step
