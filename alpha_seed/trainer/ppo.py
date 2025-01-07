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

import time
import uuid
import contextlib

import ray
import random
import os
import copy
import json
import wandb
import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Type, Tuple, Union

from omegaconf import OmegaConf, open_dict
import numpy as np
from codetiming import Timer

from alpha_seed.trainer.tensorcore_collect import tensorcore_collection
from alpha_seed.utils.select_strategy.bon_strategy import *
from alpha_seed.utils.select_strategy.league_training_strategy import *
from alpha_seed.utils.validator.validation_manager import *
from alpha_seed.workers.streaming_service.streaming_utils import pad, process_output
from alpha_seed.workers.actors.checkpoint import CkptGlobalUploader
from alpha_seed.utils.observility.pretty_print import pprint
from alpha_seed.utils import ndtimeline
from single_controller.base import Worker
from single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from single_controller.ray.base import create_colocated_worker_cls
from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from alpha_seed.workers.streaming_service.streaming_utils import record_xperf_metrics
from hdfs_io import makedirs, hput, hcopy, hexists

try:
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.protocol import unfold_batch_dim, fold_batch_dim
except ImportError:
    print('Cannot find pad_dataproto_to_divisor. Please use latest verl master')
    raise
from alpha_seed import core_algos

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


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
import torch.nn.functional as F
from tensordict import TensorDict
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
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

    return data, metrics


def compute_advantage(data: DataProto, gamma, lam, adv_estimator, upgo_loss_version, num_bon, adv_whiten,
                      use_async_gen):
    # TODO: add other ways to estimate advantages
    token_level_rewards = data.batch['token_level_rewards']
    responses = data.batch['responses']
    response_length = responses.size(1)
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
            adv_whiten=adv_whiten)
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
            use_async_gen=use_async_gen)
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

    old_log_probs = batch.batch['old_log_probs']
    old_entropy = batch.batch['old_entropy']

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    max_prompt_length = float(prompt_mask.size(-1))
    max_response_length = float(response_mask.size(-1))

    returns = batch.batch['returns']

    response_mask_bool = response_mask.bool()

    valid_entropy = torch.masked_select(old_entropy, response_mask_bool)

    valid_adv = torch.masked_select(advantages, response_mask_bool)
    valid_origin_adv = torch.masked_select(origin_advantages, response_mask_bool)
    valid_returns = torch.masked_select(returns, response_mask_bool)
    valid_old_logprob = torch.masked_select(old_log_probs, response_mask_bool)

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
        response_length, compute_std=False)
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
        small_prob_mask = torch.logical_and(response_mask_bool, old_log_probs.exp() < threshold)
        small_prob_mask_sum = small_prob_mask.float().sum()
        local_small_prob_mask_sum = small_prob_mask_sum.clone()
        response_mask_bool_sum = response_mask_bool.float().sum()
        dist.all_reduce(small_prob_mask_sum, op=dist.ReduceOp.SUM, group=None, async_op=False)
        dist.all_reduce(response_mask_bool_sum, op=dist.ReduceOp.SUM, group=None, async_op=False)
        small_prob_ratio = small_prob_mask_sum / response_mask_bool_sum

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
        valid_values = torch.masked_select(values, response_mask_bool)
        valid_upgo_adv = torch.masked_select(upgo_advantages, response_mask_bool)

        valid_values_mean, valid_values_max, valid_values_min, valid_values_std = distributed_mean_max_min_std(
            valid_values)
        valid_upgo_adv_mean, valid_upgo_adv_max, valid_upgo_adv_min, valid_upgo_adv_std = distributed_mean_max_min_std(
            valid_upgo_adv)

        return_diff_std = distributed_mean_max_min_std(torch.masked_select(returns - values, response_mask_bool),
                                                       compute_max=False,
                                                       compute_min=False,
                                                       compute_std=True)[-1]
        return_std = distributed_mean_max_min_std(torch.masked_select(returns, response_mask_bool),
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


def print_dataproto_size(data: DataProto, head):
    size_of_tensordict = 0
    for key, tensor in data.batch.items():
        size_of_tensordict += tensor.element_size() * tensor.numel()
    size_of_numpy_array = 0
    for key, numpy_array in data.non_tensor_batch.items():
        size_of_numpy_array += numpy_array.nbytes

    size_of_numpy_array /= 1024**3
    size_of_tensordict /= 1024**3
    print(f'{head}, Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB')


def save_dataproto(data: DataProto, path, prefix=''):
    torch.save(data.batch, f"{prefix}.batch.pt")
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
                 sandbox_client=None):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.all_wg = {}
        self.internal_wgs = []
        self.internal_wg_roles = []
        self.all_meta = {}
        self.sandbox_client = sandbox_client
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.logger = logger

        # ckpt global uploader will be instantiated in init_workers func
        self.ckpt_global_uploader = None

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, \
                  f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager

        self.use_standalone_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_colocate_reference_policy = Role.ActorRolloutRef in role_worker_mapping
        self.use_standalone_rollout = self.config.streaming_rollout.nnodes > 0
        self.use_standalone_validator = self.config.streaming_validator.nnodes > 0
        self.use_reference_policy = self.use_standalone_reference_policy or self.use_colocate_reference_policy
        self.use_rm = Role.RewardModel in role_worker_mapping

        self.ray_worker_group_cls = ray_worker_group_cls
        self.num_bon = self.config.actor_rollout_ref.rollout.get("num_bon", 1)

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self.validation_manager = ValidateManager(self.config, self.logger, self.val_dataloader, self.tokenizer,
                                                  self.use_rm, self.val_reward_fn)

        self.phasic_critic_buffer = None

        self.standalone_gen_batch_output_resume = None
        self.standalone_batch_resume = None
        self.megavision_metrics_collector = MegavisionMetricsCtx() if MegavisionMetricsCtx else None

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        version = self.config.data.get('version', 'v1')
        # TODO: we have to make sure the batch size is divisible by the dp size
        from alpha_seed.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        train_batch_size = self.config.data.train_batch_size
        if self.config.trainer.league_training_config.enable:
            train_batch_size = train_batch_size * self.config.trainer.league_training_config.buffer_size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         answer_key=self.config.data.answer_key,
                                         use_ref_answer=self.config.data.use_ref_answer,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         multi_prompts=self.config.data.get("multi_prompts", "none"),
                                         num_prompts_per_data=self.config.data.get("num_prompts_per_data", 1))

        if self.config.data.BITWISE_RESUME:
            from alpha_seed.utils.dataset.sampler import RandomSampler, SequentialSampler
        else:
            from torch.utils.data import RandomSampler, SequentialSampler

        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=train_batch_size,
                                           shuffle=None,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       answer_key=self.config.data.answer_key,
                                       use_ref_answer=self.config.data.use_ref_answer,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=True,
                                       truncation=self.config.data.get('truncation', 'error'),
                                       multi_prompts=self.config.data.get("multi_prompts", "none"),
                                       num_prompts_per_data=1,
                                       is_eval=True)

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=self.config.data.shuffle,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_steps is not None:
            total_training_steps = self.config.trainer.total_steps

        self.total_training_steps = total_training_steps

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def init_workers(self, kv_store_name="kv_store", ckpt_global_uploader=None):
        """Init resource pool and worker group"""

        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            if self.use_standalone_reference_policy or not self.use_reference_policy:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
                actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                         config=self.config.actor_rollout_ref,
                                                         role='actor_rollout')
                self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
            elif self.use_colocate_reference_policy:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
                actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRolloutRef],
                                                         config=self.config.actor_rollout_ref,
                                                         role='actor_rollout_ref')
                self.resource_pool_to_cls[resource_pool]['actor_rollout_ref'] = actor_rollout_cls
            else:
                raise NotImplementedError('Must instantiate actor and rollout')

            if self.use_standalone_rollout:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
                rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Rollout],
                                                   config=self.config.actor_rollout_ref,
                                                   role='standalone_rollout')
                self.resource_pool_to_cls[resource_pool]['standalone_rollout'] = rollout_cls

            if self.use_standalone_validator:
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.Validator)
                validator_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Validator],
                                                     config=self.config.actor_rollout_ref,
                                                     role='standalone_validator')
                self.resource_pool_to_cls[resource_pool]['standalone_validator'] = validator_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
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

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        kv_store = ray.get_actor(name=kv_store_name)
        server_only = ray.get(kv_store.get_by_key.remote("server_only"))
        # initialize WorkerGroup
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # no role allocated to this resource pool
            if len(class_dict) == 0:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            worker_names = ray.get(kv_store.get_by_key.remote(resource_pool.name_prefix)) if server_only else None
            wg_dict = self.ray_worker_group_cls(
                ray_cls_with_init=worker_dict_cls, worker_names=worker_names) if isinstance(
                    worker_names, list) and len(worker_names) > 0 else self.ray_worker_group_cls(
                        resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            if worker_names is None:
                kv_store.set_key_val.remote(resource_pool.name_prefix, wg_dict.worker_names)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            self.all_wg.update(spawn_wg)
            self.internal_wgs.append(wg_dict)
            self.internal_wg_roles.append(list(class_dict.keys()))

        # init ckpt global uploader
        self.ckpt_global_uploader = CkptGlobalUploader.options(
            name=CkptGlobalUploader.name).remote(use_critic=self.use_critic,
                                                 ckpt_version=self.config.trainer.ckpt_version,
                                                 default_local_dir=self.config.trainer.default_local_dir,
                                                 default_remote_dir=self.config.trainer.default_hdfs_dir
                                                ) if ckpt_global_uploader is None else ckpt_global_uploader

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

        self.actor_rollout_wg.init_model()
        hybrid_master_address = self.actor_rollout_wg.get_master_addr()
        self.validation_manager.actor_rollout_wg = self.actor_rollout_wg

        if self.use_standalone_rollout:
            self.standalone_rollout_wg = self.all_wg['standalone_rollout']
            self.standalone_rollout_wg.init_model()
            standalone_rollout_address = self.standalone_rollout_wg.get_master_addr()
            self.actor_rollout_wg.setup_standalone_worker_comm(hybrid_master_address, standalone_rollout_address,
                                                               "12345", "standalone_rollout")
            self.standalone_rollout_wg.setup_standalone_worker_comm(hybrid_master_address, standalone_rollout_address,
                                                                    "12345", "standalone_rollout")
            # offload standalone_rollout_wg FSDP GPU memory
            self.standalone_rollout_wg.to('cpu')
        else:
            self.standalone_rollout_wg = None

        if self.use_standalone_validator:
            self.standalone_validator_wg = self.all_wg['standalone_validator']
            self.standalone_validator_wg.init_model()
            standalone_validator_address = self.standalone_validator_wg.get_master_addr()
            self.actor_rollout_wg.setup_standalone_worker_comm(hybrid_master_address, standalone_validator_address,
                                                               "14567", "standalone_validator")
            self.standalone_validator_wg.setup_standalone_worker_comm(hybrid_master_address,
                                                                      standalone_validator_address, "14567",
                                                                      "standalone_validator")
            self.validation_manager.standalone_validator_wg = self.standalone_validator_wg
            # offload standalone_validator_wg FSDP GPU memory
            self.standalone_validator_wg.to('cpu')

        if self.config.actor_rollout_ref.actor.kl_loss_weight >= 1e-10:
            # 两种情况下使用kl loss，一种是grpo，另一种是在rewards里不加kl惩罚
            assert self.config.algorithm.adv_estimator == 'grpo' or self.config.algorithm.kl_ctrl.kl_coef <= 1e-10

        if self.use_critic:
            self.critic_wg = self.all_wg['critic']
            self.critic_wg.init_model()

        if self.use_standalone_reference_policy:
            self.ref_policy_wg = self.all_wg['ref']
            self.ref_policy_wg.init_model()
        elif self.use_colocate_reference_policy:
            self.ref_policy_wg = self.all_wg['actor_rollout_ref']

        if self.use_rm:
            self.rm_wg = self.all_wg['rm']
            self.rm_wg.init_model()

        if self.config.trainer.use_remote_sandbox:
            # set the eos_callback_fn of actor_rollout
            from xperf_gpt.inference.session import Query

            def sandbox_callback_fn(query: Query):
                input_ids = query.input_ids + query.new_token_ids
                req_id = query.meta_info['uid']
                reward_model = query.meta_info['reward_model']
                reward_style = reward_model['style']
                ground_truth = reward_model['ground_truth']

                # note that the uid of padding dataproto should be None
                if reward_style == 'code-sandbox' and req_id is not None:
                    # get the sandbox ray handler
                    handler = ray.get_actor('sandbox_client')
                    # this is non-blocking
                    handler.add_requests.remote(req_id=req_id, input_ids=input_ids, ground_truth=ground_truth)

            self.actor_rollout_wg.set_eos_callback_fn(sandbox_callback_fn)
            if self.standalone_rollout_wg is not None:
                self.standalone_rollout_wg.set_eos_callback_fn(sandbox_callback_fn)
            if self.use_standalone_validator:
                self.standalone_validator_wg.set_eos_callback_fn(sandbox_callback_fn)

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

        local_checkpoint_folder = os.path.join(self.config.trainer.default_local_dir, 'checkpoints')
        local_global_step_folder = os.path.join(local_checkpoint_folder, f'global_step_{self.global_step}')
        os.makedirs(local_global_step_folder, exist_ok=True)

        actor_local_path = os.path.join(local_global_step_folder, 'actor')
        critic_local_path = os.path.join(local_global_step_folder, 'critic')

        remote_checkpoint_folder = os.path.join(self.config.trainer.default_hdfs_dir, 'checkpoints')
        remote_global_step_folder = os.path.join(remote_checkpoint_folder, f'global_step_{self.global_step}')

        makedirs(remote_global_step_folder)

        actor_remote_path = os.path.join(remote_global_step_folder, 'actor')
        critic_remote_path = os.path.join(remote_global_step_folder, 'critic')

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)
        # upload to hdfs
        ray.get(
            self.ckpt_global_uploader.register_upload_task.remote("default", self.global_step,
                                                                  ray.get_runtime_context().get_node_id(),
                                                                  dataloader_local_path, remote_global_step_folder))
        self.ckpt_global_uploader.start_uploading.remote("default", self.global_step)

        actor_upload_future = self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path,
            specified_ckpt_version if specified_ckpt_version is not None else self.config.trainer.ckpt_version,
            self.global_step, self.ckpt_global_uploader, self.config.trainer.ckpt_enable_flatten)

        if self.use_critic:
            critic_upload_future = self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path,
                specified_ckpt_version if specified_ckpt_version is not None else self.config.trainer.ckpt_version,
                self.global_step, self.ckpt_global_uploader, self.config.trainer.ckpt_enable_flatten)
        else:
            critic_upload_future = None

        ray.get(actor_upload_future)

        if critic_upload_future is not None:
            ray.get(critic_upload_future)

    def load_checkpoint(self):
        if self.config.trainer.resume_steps == 'disable':
            return 0

        # find the latest global step
        if self.config.trainer.resume_steps == 'auto':
            try:
                from omnistore.utilities.ckpt_format_tool import find_latest_ckpt_path
            except ImportError:
                from omnistore.utilities.ckpt_format.common_utils import find_latest_ckpt_path

            remote_checkpoint_folder = os.path.join(self.config.trainer.default_hdfs_dir, 'checkpoints')
            remote_global_step_folder = find_latest_ckpt_path(remote_checkpoint_folder)

            if remote_global_step_folder is None:
                print('Training from scratch')
                return 0

            # set global step
            self.global_step = int(remote_global_step_folder.split('global_step_')[-1])

        else:
            assert isinstance(self.config.trainer.resume_steps, str), "resume ckpt must be str type"
            assert 'global_step_' in self.config.trainer.resume_steps, "resume ckpt must specify the global_step"
            remote_global_step_folder = self.config.trainer.resume_steps
            self.global_step = int(remote_global_step_folder.split('global_step_')[-1])

        # note that we start from the next global_step
        # self.global_step += 1

        print(f'Setting global step to {self.global_step}')
        print(f'Resuming from {remote_global_step_folder}')

        actor_remote_path = os.path.join(remote_global_step_folder, 'actor')
        critic_remote_path = os.path.join(remote_global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_remote_path, self.config.trainer.ckpt_version,
                                              self.config.trainer.ckpt_enable_flatten)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_remote_path, self.config.trainer.ckpt_version,
                                           self.config.trainer.ckpt_enable_flatten)
        # load dataloader
        dataloader_remote_path = os.path.join(remote_global_step_folder, 'data.pt')
        dataloader_local_path = copy_local_path_from_hdfs(dataloader_remote_path)
        self.train_dataloader = torch.load(dataloader_local_path)

        # async resume
        if hexists(f"{remote_global_step_folder}/standalone_gen_batch_output.batch.pt"):
            pprint('resume standalone rollout.')
            self.standalone_gen_batch_output_resume = load_dataproto(path=remote_global_step_folder,
                                                                     prefix='standalone_gen_batch_output')
            self.standalone_batch_resume = load_dataproto(path=remote_global_step_folder, prefix='standalone_batch')

        return self.global_step

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

    def _generate(self, batch, resume_step, metrics, standalone_batch, pending_batch_queue, ready_batch_queue):
        # print the size of each data proto before training
        print_dataproto_size(batch, head='Before generation')

        if self.config.data.num_prompts_per_data > 1:
            batch = batch.unfold_column_chunks(self.config.data.num_prompts_per_data,
                                               split_keys=['input_ids', 'attention_mask', 'off_policy_steps'])

        # hybrid rollout
        batch = batch.repeat(self.num_bon)
        # create a uid for each data inside the batch
        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
        batch.check_consistency()

        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'off_policy_steps'])
        # assign the non_tensor_batch uid to the generator as well.
        non_tensor_infos = ['uid', 'reward_model']
        for key in non_tensor_infos:
            gen_batch.non_tensor_batch[key] = batch.non_tensor_batch[key]

        gen_batch.meta_info.update({
            'generation_kwargs':
                self.config.actor_rollout_ref.rollout.train_generate_kwargs,
            'complete_ratio':
                1.0 if self.global_step < resume_step + self.config.streaming_rollout.warmup_step else
                self.config.actor_rollout_ref.rollout.get("complete_ratio", 1.0)
        })
        pprint(f'start hybrid rollout, input batches {len(gen_batch)}.')

        with Timer(name='gen', logger=None) as timer:
            with tensorcore_collection():
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                # TODO: The following two lines should be memory view. However it's not. Let's remove it by removing all its dependency
                gen_batch_output.batch['prompts'] = gen_batch_output.batch['input_ids'][:, :self.config.data.
                                                                                        max_prompt_length]
                gen_batch_output.batch['responses'] = gen_batch_output.batch['input_ids'][:, self.config.data.
                                                                                          max_prompt_length:]

            print_dataproto_size(gen_batch_output, head='After generation')

        metrics['timing/gen'] = timer.last
        metrics['rollout/hybrid_input_batch'] = len(batch)

        # for debugging purpose only. we manually set all the attention_mask to 1 to
        # test the training performance under maximum workload.
        if self.config.trainer.set_fake_attention_mask:
            with Timer(name='fake_mask', logger=None) as timer:
                from verl.utils.model import create_random_mask
                total_length = self.config.data.max_prompt_length + self.config.data.max_response_length

                min_ratio_of_valid_token = self.config.trainer.fake_seqlen_ratio
                max_ratio_of_valid_token = self.config.trainer.fake_seqlen_ratio

                assert self.config.trainer.fake_seqlen_ratio > (self.config.data.max_prompt_length + 1) / total_length

                max_ratio_of_left_padding = 0
                attention_mask = create_random_mask(gen_batch_output.batch['input_ids'],
                                                    max_ratio_of_valid_token=max_ratio_of_valid_token,
                                                    max_ratio_of_left_padding=max_ratio_of_left_padding,
                                                    min_ratio_of_valid_token=min_ratio_of_valid_token)

                gen_batch_output.batch['attention_mask'] = attention_mask

                # force actor and critic stop updating weights because the data is fake
                self.config.actor_rollout_ref.actor.optim.lr = 0
                self.config.critic.optim.lr = 0

            metrics['timing/fake_mask'] = timer.last
            pprint(f'set fake attention mask')

        # only report metrics from one generation replica
        record_xperf_metrics(gen_batch_output, metrics, self.logger, self.global_step, prefix='hybrid')

        # stop hybrid rollout
        finished_num, ready_batch_queue, pending_batch_queue = process_output(batch, gen_batch_output, self.tokenizer,
                                                                              ready_batch_queue, pending_batch_queue,
                                                                              self.config)
        pprint(f'stop hybrid rollout, completed_batch {finished_num}, incompleted_batch {len(batch) - finished_num} ' +
               f'ready_queue {ready_batch_queue.qsize()}, pending_queue {pending_batch_queue.qsize()}.')
        metrics['rollout/hybrid_completed_batch'] = finished_num
        metrics['rollout/hybrid_incompleted_batch'] = len(batch) - finished_num

        # stop standalone rollout to update model
        if self.standalone_batch_resume is not None:
            standalone_batch = self.standalone_batch_resume
            self.standalone_batch_resume = None

        finished_num = 0
        if len(standalone_batch) > 0:
            if not self.standalone_gen_batch_output_resume:
                gen_batch_output = self.standalone_rollout_wg.generate_sequences_get()
            else:
                gen_batch_output = self.standalone_gen_batch_output_resume
                self.standalone_gen_batch_output_resume = None
            if self.config.trainer.save_freq > 0 and (self.global_step -
                                                      1) % self.config.trainer.save_freq == 0 and self.global_step != 1:
                print(f"step {self.global_step}, saving... standalone_gen_batch")
                # save standalone_batch and gen_batch_output
                save_path = f"{self.config.trainer.default_hdfs_dir}/checkpoints/global_step_{self.global_step - 1}/"
                save_dataproto(gen_batch_output, path=save_path, prefix='standalone_gen_batch_output')
                save_dataproto(standalone_batch, path=save_path, prefix='standalone_batch')
            # only report metrics from one generation replica
            record_xperf_metrics(gen_batch_output, metrics, self.logger, self.global_step, prefix='standalone')
            finished_num, ready_batch_queue, pending_batch_queue = process_output(standalone_batch,
                                                                                  gen_batch_output,
                                                                                  self.tokenizer,
                                                                                  ready_batch_queue,
                                                                                  pending_batch_queue,
                                                                                  self.config,
                                                                                  standalone=True)
            pprint(
                f'stop standalone rollout, completed_batch {finished_num}, incompleted_batch {len(standalone_batch) - finished_num}'
                + f'ready_queue {ready_batch_queue.qsize()}, pending_queue {pending_batch_queue.qsize()}.')
        metrics['rollout/standalone_completed_batch'] = finished_num
        metrics['rollout/standalone_incompleted_batch'] = len(standalone_batch) - finished_num

        # update standalone rollout weights
        with Timer(name='update_standalone', logger=None) as timer:
            if self.standalone_rollout_wg is not None:
                self.actor_rollout_wg.update_standalone_worker("standalone_rollout")
                self.standalone_rollout_wg.update_standalone_worker("standalone_rollout")
        self.actor_rollout_wg.release_param_and_cache()
        metrics['timing/update_standalone'] = timer.last

        # standalone generate (off policy)
        standalone_batch = []
        max_standalone_len = 0
        while self.standalone_rollout_wg is not None and pending_batch_queue.qsize(
        ) >= self.standalone_rollout_wg.world_size:
            for _ in range(self.standalone_rollout_wg.world_size):
                standalone_batch.append(pending_batch_queue.get())
                max_standalone_len = max(max_standalone_len, standalone_batch[-1].batch['attention_mask'].sum(-1))
        for i in range(len(standalone_batch)):
            standalone_batch[i] = pad(standalone_batch[i], max_standalone_len, self.tokenizer)
        if len(standalone_batch) > 0:
            standalone_batch = DataProto.concat(standalone_batch)
            standalone_gen_batch = standalone_batch.pop(batch_keys=['input_ids', 'attention_mask', 'off_policy_steps'])
            non_tensor_infos = ['uid', 'reward_model']
            for key in non_tensor_infos:
                standalone_gen_batch.non_tensor_batch[key] = standalone_batch.non_tensor_batch[key]

            standalone_gen_batch.meta_info[
                'generation_kwargs'] = self.config.actor_rollout_ref.rollout.train_generate_kwargs
            standalone_gen_batch.meta_info['complete_ratio'] = 1
            self.standalone_rollout_wg.generate_sequences_put(standalone_gen_batch)
            pprint(f'start standalone rollout, input batches {len(standalone_gen_batch)}.')
        metrics['rollout/standalone_input_batch'] = len(standalone_batch)
        # get training batch from ready queue, make it stable by random pick
        return_batch_size = self.config.data.train_batch_size * \
            self.config.trainer.league_training_config.buffer_size * \
            self.num_bon
        ready_batch = [ready_batch_queue.get() for _ in range(min(return_batch_size, ready_batch_queue.qsize()))]
        real_bsz = len(ready_batch)
        random_choise_batch = []
        if len(ready_batch) < return_batch_size:
            random_choise_batch.extend(
                [random.choice(ready_batch) for _ in range(return_batch_size - len(ready_batch))])
        fake_bsz = len(random_choise_batch)
        ready_batch.extend(random_choise_batch)
        metrics.update({"rollout/real_bsz": real_bsz, "rollout/fake_bsz": fake_bsz})

        batch = DataProto.concat(ready_batch)
        if self.config.algorithm.force_append_eos:
            batch.batch["input_ids"][:, -1] = self.tokenizer.eos_token_id
            batch.batch["responses"][:, -1] = self.tokenizer.eos_token_id
        batch.meta_info['generation_kwargs'] = self.config.actor_rollout_ref.rollout.train_generate_kwargs
        batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
        metrics['rollout/training_batch'] = len(batch)
        pprint(f'training batches {len(batch)}.')

        return batch, standalone_batch

    def fit(self):
        self.global_step = 0

        metric_collection_context = self.megavision_metrics_collector.collect_resume_from_checkpoint_duration() \
            if MegavisionMetricsCtx else contextlib.nullcontext()
        with metric_collection_context:
            # load checkpoint before doing anything
            resume_step = self.load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.eval_before_training:
            self.validation_manager.validate(val_epoch=self.config.trainer.val_epoch,
                                             need_log=self.config.trainer.need_log,
                                             log_file=self.config.trainer.log_file,
                                             is_async=False,
                                             global_step=self.global_step)
        if self.config.trainer.val_only:
            return

        # Note that we start from step 1. After resume, we increment step by 1 to start next step
        self.global_step += 1
        start_step = self.global_step

        if ndtimeline.use_cuda_timer():
            self.call_once_on_each_ray_actor("do_ndtimeline_action", "set_global_step", global_step=self.global_step)

        # before training, move out the training resource as rollout begins first
        if self.config.trainer.offload_train_memory:
            self.actor_rollout_wg.to("cpu", model=False, optimizer=True)
            if self.use_critic:
                self.critic_wg.to("cpu")

        # TODO: add staleness
        standalone_batch = []
        pending_batch_queue = queue.Queue()
        ready_batch_queue = queue.Queue()
        while True:
            for batch_dict in self.train_dataloader:
                metrics = {}
                with Timer(name='step', logger=None) as step_timer:
                    # hybrid generate (on policy)
                    if self.config.trainer.load_train_batch_path is None:
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                        batch, standalone_batch = self._generate(batch=batch,
                                                                 resume_step=resume_step,
                                                                 metrics=metrics,
                                                                 standalone_batch=standalone_batch,
                                                                 pending_batch_queue=pending_batch_queue,
                                                                 ready_batch_queue=ready_batch_queue)
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

                    # training
                    with Timer(name='rm_score', logger=None) as timer:
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                    metrics['timing/rm_score'] = timer.last

                    print_dataproto_size(batch, head='After Reward Model')

                    with Timer(name='reward_fn', logger=None) as timer:
                        # we combine with rule-based rm
                        reward_tensor, raw_scores = self.reward_fn(batch, global_step=self.global_step)
                        batch.batch['token_level_scores'] = reward_tensor
                        batch.batch['raw_scores'] = raw_scores
                    metrics['timing/reward_fn'] = timer.last

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

                    # perform sequence balancing.
                    # Very important: Note that this reorders data globally.
                    # So anything that requires ordering below this line will cause incorrect results
                    self._balance_batch(batch=batch, metrics=metrics, logging_prefix='global_seqlen')

                    with Timer(name='old_log_probs', logger=None) as timer:
                        batch = self.actor_rollout_wg.old_log_probs(batch)
                    metrics['timing/old_log_probs'] = timer.last

                    print_dataproto_size(batch, head='After old log probs')

                    if self.use_reference_policy:
                        if not (self.config.actor_rollout_ref.actor.kl_loss_weight == 0 and
                                self.config.algorithm.kl_ctrl.kl_coef == 0):
                            # skip ref log prob if no kl loss
                            # compute reference log_prob
                            with Timer(name='ref', logger=None) as timer:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)
                            metrics['timing/ref'] = timer.last

                    print_dataproto_size(batch, head='After reference policy')

                    metrics.setdefault('timing/train_mem_offload', 0)

                    # compute values
                    if self.use_critic:
                        # load critic
                        with Timer(name='train_mem_offload', logger=None) as timer:
                            if self.config.trainer.offload_train_memory:
                                self.critic_wg.to("cuda")
                        metrics['timing/train_mem_offload'] += timer.last

                        with Timer(name='values', logger=None) as timer:
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                        metrics['timing/values'] = timer.last

                    print_dataproto_size(batch, head='After compute values')

                    with Timer(name='adv', logger=None) as timer:
                        # compute rewards. apply_kl_penalty if available
                        batch, kl_metrics = apply_kl_penalty(batch,
                                                             kl_ctrl=self.kl_ctrl,
                                                             kl_penalty=self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)

                        # compute advantages
                        batch, adv_metrics = compute_advantage(
                            batch,
                            self.config.algorithm.gamma,
                            self.config.algorithm.lam,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            upgo_loss_version=self.config.actor_rollout_ref.actor.upgo_loss_version,
                            num_bon=self.config.actor_rollout_ref.rollout.num_bon,
                            adv_whiten=self.config.algorithm.adv_whiten,
                            use_async_gen=use_async_gen)
                        metrics.update(adv_metrics)
                    metrics['timing/adv'] = timer.last

                    print_dataproto_size(batch, head='After compute adv')

                    if self.global_step == 1:
                        print('Debugging', batch.batch)

                    # update critic
                    phasic_critic_update = self.config.algorithm.phasic_critic_interval > 0 and self.global_step % self.config.algorithm.phasic_critic_interval == 0
                    if self.use_critic:
                        with Timer(name='update_critic', logger=None) as timer:
                            critic_output = self.critic_wg.update_critic(batch)
                        metrics['timing/update_critic'] = timer.last
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                        with Timer(name='train_mem_offload', logger=None) as timer:
                            if self.config.trainer.offload_train_memory and not phasic_critic_update:
                                self.critic_wg.to("cpu")
                        metrics['timing/train_mem_offload'] += timer.last
                    if self.config.algorithm.phasic_critic_interval > 0:
                        select_keys = ['input_ids', 'responses', 'attention_mask', 'values', 'returns']
                        buffer_batch = batch.select(batch_keys=select_keys)
                        if self.phasic_critic_buffer is None:
                            self.phasic_critic_buffer = buffer_batch
                        else:
                            self.phasic_critic_buffer = DataProto.concat([self.phasic_critic_buffer, buffer_batch])

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_step and self.global_step % self.config.trainer.actor_update_freq == 0:

                        with Timer(name='train_mem_offload', logger=None) as timer:
                            if self.config.trainer.offload_train_memory:
                                self.actor_rollout_wg.to("cuda", model=False, optimizer=True)
                        metrics['timing/train_mem_offload'] += timer.last

                        # update actor
                        with Timer(name='update_actor', logger=None) as timer:
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics['timing/update_actor'] = timer.last
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                        with Timer(name='train_mem_offload', logger=None) as timer:
                            if self.config.trainer.offload_train_memory:
                                self.actor_rollout_wg.to("cpu", model=False, optimizer=True)
                        metrics['timing/train_mem_offload'] += timer.last

                    # phasic critic update
                    if phasic_critic_update:
                        self.phasic_critic_buffer.meta_info['phasic_update'] = True
                        with Timer(name='phasic_critic_update', logger=None) as timer:
                            critic_output = self.critic_wg.update_critic(self.phasic_critic_buffer)
                        metrics['timing/phasic_critic_update'] = timer.last
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                        with Timer(name='train_mem_offload', logger=None) as timer:
                            if self.config.trainer.offload_train_memory:
                                self.critic_wg.to("cpu")
                        metrics['timing/train_mem_offload'] += timer.last

                        self.phasic_critic_buffer = None

                    # update ref ema
                    with Timer(name='update_ref_ema', logger=None) as timer:
                        self.ref_policy_wg.update_ref_ema()
                    metrics['timing/update_ref_ema'] = timer.last

                    # validate
                    if self.val_reward_fn is not None and self.global_step % self.config.trainer.test_freq == 0:
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
                        metrics['timing/compute_metrics'] = timer.last

                    metric_collection_context = self.megavision_metrics_collector.collect_save_checkpoint_duration() \
                        if MegavisionMetricsCtx else contextlib.nullcontext()

                    with Timer(name='save_checkpoint', logger=None) as timer:
                        if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                            with metric_collection_context:
                                if self.config.trainer.offload_train_memory:
                                    self.actor_rollout_wg.to("cuda", model=False, optimizer=True)
                                    if self.use_critic:
                                        self.critic_wg.to("cuda")

                                self.save_checkpoint()

                                if self.config.trainer.offload_train_memory:
                                    self.actor_rollout_wg.to("cpu", model=False, optimizer=True)
                                    if self.use_critic:
                                        self.critic_wg.to("cpu")

                        metrics['timing/save_checkpoint'] = timer.last

                    # collect sandbox client remaining results
                    if self.config.trainer.use_remote_sandbox:
                        sandbox_client = ray.get_actor('sandbox_client')
                        num_remaining_results = ray.get(sandbox_client.get_num_pending_outputs.remote())
                        metrics['sandbox/remaining_results'] = num_remaining_results

                metrics['timing/step'] = step_timer.last
                # TODO: make a canonical logger that supports various backend
                self.logger.log(data=metrics, step=self.global_step)

                if ndtimeline.use_cuda_timer():
                    # Always be the last op before increasing global_step by 1
                    print(f"flush ndtimeline at {self.global_step}")
                    self.call_once_on_each_ray_actor("do_ndtimeline_action", "flush_and_inc")

                self.global_step += 1
                if start_step + 1 == self.global_step:
                    now = int(time.time())
                    self.ref_policy_wg.upload_process_group(now)
                    if self.config.streaming_rollout.nnodes > 0:
                        self.standalone_rollout_wg.upload_process_group(now)
                    pprint(f'start upload process group at {now}')

                if self.global_step >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self.validation_manager.validate(is_async=False, global_step=self.global_step)
                        pprint(f'Final validation metrics: {val_metrics}')

                    return

    def call_once_on_each_ray_actor(self, func_name: str, *args, **kwargs):
        """Call a function on each actor.
        Args:
            func_name (str): func_name must be registered as each WorkGroup's user_defined_cls's method
            *args: arguments
            **kwargs: keyword arguments
        """
        assert len(self.internal_wgs) == len(self.internal_wg_roles)
        assert len(self.internal_wgs) > 0
        for i, wg in enumerate(self.internal_wgs):
            prefix = self.internal_wg_roles[i][0]  # first role name
            f = getattr(wg, f"{prefix}_{func_name}")
            f(*args, **kwargs)

    def convert_ckpt_to_omnistore(self):
        self.global_step = 0

        # load checkpoint before doing anything
        _ = self.load_checkpoint()
        # save omnistore ckpt
        self.save_checkpoint(specified_ckpt_version='omnistore')
        ray.get(self.ckpt_global_uploader.wait_all.remote(self.global_step, False))
