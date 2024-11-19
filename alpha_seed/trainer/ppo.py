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

import ray
import os
import copy
import json
import wandb
import queue
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Callable, Type, Tuple, Union

from omegaconf import OmegaConf, open_dict
import numpy as np
from codetiming import Timer

from alpha_seed.utils.select_strategy.bon_strategy import *
from alpha_seed.utils.select_strategy.league_training_strategy import *
from single_controller.base import Worker
from single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from single_controller.ray.base import create_colocated_worker_cls
from verl import DataProto

from hdfs_io import makedirs, hput, hcopy

try:
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
except ImportError:
    print('Cannot find pad_dataproto_to_divisor. Please use latest verl master')
    raise

from alpha_seed import core_algos

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


def compute_advantage(data: DataProto, gamma, lam, adv_estimator, upgo_loss_version, num_bon, adv_whiten):
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
    elif adv_estimator == 'grpo':
        token_level_scores = data.batch['token_level_scores']
        advantages, returns = core_algos.compute_grpo_advantage_return(token_level_scores=token_level_scores,
                                                                       eos_mask=response_mask,
                                                                       num_bon=num_bon)
        data.batch['advantages'] = advantages
        data.batch['origin_advantages'] = advantages
        data.batch['returns'] = returns
        data.batch['upgo_advantages'] = torch.zeros_like(advantages)
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch, use_critic, mean, std):
    # TODO: add response length
    if torch.cuda.is_available():
        print('Using GPU to compute_data_metrics')
        batch = batch.to('cuda')
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    origin_sequence_score = sequence_score * std + mean  # 打原始的分数
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    response_length = batch.batch['responses'].shape[-1]

    advantages = batch.batch['advantages']
    origin_advantages = batch.batch['origin_advantages']
    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    old_log_probs = batch.batch['old_log_probs']

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    max_prompt_length = float(prompt_mask.size(-1))
    max_response_length = float(response_mask.size(-1))

    returns = batch.batch['returns']

    reflection_nums = batch.batch.get('reflection_nums', torch.Tensor([0.0]))

    response_mask_bool = response_mask.bool()
    valid_adv = torch.masked_select(advantages, response_mask_bool)
    valid_origin_adv = torch.masked_select(origin_advantages, response_mask_bool)
    valid_returns = torch.masked_select(returns, response_mask_bool)
    valid_old_logprob = torch.masked_select(old_log_probs, response_mask_bool)

    eos_adv = torch.gather(advantages, dim=1, index=response_length.unsqueeze(dim=1).long() - 1).reshape(-1)
    eos_original_adv = torch.gather(origin_advantages, dim=1,
                                    index=response_length.unsqueeze(dim=1).long() - 1).reshape(-1)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        'critic/score/std':
            torch.std(sequence_score).detach().item(),
        # original score
        'critic/original_score/mean':
            torch.mean(origin_sequence_score).detach().item(),
        'critic/original_score/max':
            torch.max(origin_sequence_score).detach().item(),
        'critic/original_score/min':
            torch.min(origin_sequence_score).detach().item(),
        'critic/original_score/std':
            torch.std(origin_sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        'critic/rewards/std':
            torch.std(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            masked_mean(advantages, response_mask).detach().item(),
        'critic/advantages/eos_adv_mean':
            torch.mean(eos_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        'critic/advantages/std':
            torch.std(valid_adv).detach().item(),
        # original adv
        'critic/original_advantages/mean':
            masked_mean(origin_advantages, response_mask).detach().item(),
        'critic/original_advantages/eos_adv_mean':
            torch.mean(eos_original_adv).detach().item(),
        'critic/original_advantages/max':
            torch.max(valid_origin_adv).detach().item(),
        'critic/original_advantages/min':
            torch.min(valid_origin_adv).detach().item(),
        'critic/original_advantages/std':
            torch.std(valid_origin_adv).detach().item(),
        # returns
        'critic/returns/mean':
            masked_mean(returns, response_mask).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        'critic/returns/std':
            torch.std(valid_returns).detach().item(),
        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/mean_reflection_num':
            torch.mean(reflection_nums.float()).detach().item(),
        'response_length/reflection_ratio':
            torch.mean(torch.gt(reflection_nums, 0.0).float()).detach().item(),
        ## response clip ratio
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        ## prompt clip ratio
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # prob
        'prob/mean':
            torch.mean(torch.exp(valid_old_logprob)).detach().item(),
    }
    for threshold in [1e-6, 1e-5, 1e-4, 1e-3]:
        small_prob_mask = torch.logical_and(response_mask_bool, old_log_probs.exp() < threshold)
        small_prob_ratio = small_prob_mask.float().sum() / response_mask_bool.float().sum()
        small_prob_adv = torch.masked_select(advantages, small_prob_mask)
        metrics.update({
            f'prob/prob_lt_{threshold}_ratio': small_prob_ratio.detach().item(),
            f'prob/prob_lt_{threshold}_adv': torch.mean(small_prob_adv).detach().item()
        })
    if use_critic:
        values = batch.batch['values']
        upgo_advantages = batch.batch['upgo_advantages']
        valid_values = torch.masked_select(values, response_mask_bool)
        valid_upgo_adv = torch.masked_select(upgo_advantages, response_mask_bool)
        values_metrics = {
            # values
            'critic/values/mean':
                masked_mean(values, response_mask).detach().item(),
            'critic/values/max':
                torch.max(valid_values).detach().item(),
            'critic/values/min':
                torch.min(valid_values).detach().item(),
            'critic/values/std':
                torch.std(valid_values).detach().item(),
            # upgo adv
            'critic/upgo_advantages/mean':
                masked_mean(upgo_advantages, response_mask).detach().item(),
            'critic/upgo_advantages/max':
                torch.max(valid_upgo_adv).detach().item(),
            'critic/upgo_advantages/min':
                torch.min(valid_upgo_adv).detach().item(),
            'critic/upgo_advantages/std':
                torch.std(valid_upgo_adv).detach().item(),
            # vf explained var
            'critic/vf/vf_explained_var':
                (1.0 - torch.var(torch.masked_select(returns - values, response_mask_bool)) /
                 (torch.var(torch.masked_select(returns, response_mask_bool)) + 1e-5)).detach().item(),
        }
        metrics.update(values_metrics)
    return metrics


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
                 logger=None):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.logger = logger

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
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

        self.valid_hdfs_global_step = None

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

        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=train_batch_size,
                                           shuffle=self.config.data.shuffle,
                                           drop_last=True,
                                           generator=train_dataloader_generator,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       answer_key=self.config.data.answer_key,
                                       use_ref_answer=self.config.data.use_ref_answer,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.get('truncation', 'error'))
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=self.config.data.shuffle,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.total_training_steps = total_training_steps

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self, val_epoch=1, need_log=False, log_file="/opt/tiger/alpha-seed/log.jsonl"):
        metric_dict = {}
        reward_tensor_lst = []
        data_source_lst = []
        if need_log:
            f = open(log_file, "w")
        for val_epoch_idx in range(val_epoch):
            for val_idx, test_data in enumerate(self.val_dataloader):
                test_batch = DataProto.from_single_dict(test_data)

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'validate': True,
                    'complete_ratio': 1,  # validation does not need timeout
                }
                test_gen_batch.meta_info[
                    'generation_kwargs'] = self.config.actor_rollout_ref.rollout.val_generate_kwargs

                # pad test_gen_batch to divisible by world_size. TODO(zhangchi.usc1992): shall we move this logic to dispatch?
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch,
                                                                           self.actor_rollout_wg.world_size)

                eval_bon = self.config.actor_rollout_ref.rollout.get("eval_bon", 1)
                test_gen_batch_padded.meta_info["num_bon"] = eval_bon

                test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * eval_bon)

                print(
                    f'{val_epoch_idx + 1}-th/{val_epoch} {val_idx + 1}-th/{len(self.val_dataloader)} validation generation end'
                )
                if eval_bon > 1:
                    test_batch = test_batch.repeat(eval_bon)

                test_batch = test_batch.union(test_output_gen_batch)

                if self.use_rm:
                    # we first compute reward model score
                    test_batch_padded, pad_size = pad_dataproto_to_divisor(test_batch,
                                                                           size_divisor=self.rm_wg.world_size)
                    reward_tensor = self.rm_wg.compute_rm_score(test_batch_padded)
                    reward_tensor = unpad_dataproto(reward_tensor, pad_size=pad_size)

                    test_batch = test_batch.union(reward_tensor)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch, global_step=self.global_step, need_norm=False)

                reward_tensor_before_select = reward_tensor.clone()  # (B x bon, seqlen)
                if eval_bon > 1 and self.global_step % self.config.actor_rollout_ref.rollout.get("eval_bon_every",
                                                                                                 20) == 0:
                    from alpha_seed.utils.reward_score.boostrap_bon import bootstrap_bon_metric
                    nxm_mat = reward_tensor_before_select.sum(-1).reshape(-1, eval_bon)
                    bon_matrix, bon_metric = bootstrap_bon_metric(nxm_mat)  #  nxm
                    print("Bon matrix: {}".format(bon_matrix.mean(0).tolist()))
                    metric_dict.update({f"diversity/eval_bo{k}": v for k, v in bon_metric.items()})
                    metric_dict['diversity/eval_bon_hist'] = wandb.Histogram(np_histogram=np.histogram(
                        np.arange(0, eval_bon) + 0.5, bins=eval_bon, weights=bon_matrix.mean(0)))
                    reward_tensor = bon_matrix[:, 0]  # bo1 as reward
                else:
                    reward_tensor = reward_tensor.sum(-1)  # sum over seqlen

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(
                    test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                if need_log:
                    input_ids = test_output_gen_batch.batch['input_ids'].cpu().numpy()
                    prompt_ids = input_ids[:, :self.config.data.max_prompt_length]
                    response_ids = input_ids[:, self.config.data.max_prompt_length:]
                    prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                    reward_tensor_before_select = reward_tensor_before_select.sum(-1).cpu()
                    for reward, prompt, response in zip(reward_tensor_before_select, prompts, responses):
                        data = {"reward": reward.item(), "prompt": prompt, "response": response}
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        f.flush()

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        for data_source, rewards in data_source_reward.items():
            metric_dict[f'test_score/{data_source}'] = np.mean(rewards)

        if need_log:
            f.close()
        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls

            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
            rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Rollout],
                                               config=self.config.actor_rollout_ref,
                                               role='standalone_rollout')
            self.resource_pool_to_cls[resource_pool]['standalone_rollout'] = rollout_cls
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
        if self.use_reference_policy:
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

        # initialize WorkerGroup
        all_wg = {}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()
        # breakpoint()
        if self.config.streaming_rollout.nnodes > 0:
            # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
            self.actor_rollout_wg = all_wg['actor_rollout']
            self.standalone_rollout_wg = all_wg['standalone_rollout']
            hybrid_master_address = self.actor_rollout_wg.get_master_addr()
            standalone_master_address = self.standalone_rollout_wg.get_master_addr()

            self.actor_rollout_wg.init_model(hybrid_master_address, standalone_master_address)
            self.standalone_rollout_wg.init_model(hybrid_master_address, standalone_master_address)
        else:
            self.actor_rollout_wg = all_wg['actor_rollout']
            self.actor_rollout_wg.init_model()

        if self.config.actor_rollout_ref.actor.kl_loss_weight >= 1e-10:
            # 两种情况下使用kl loss，一种是grpo，另一种是在rewards里不加kl惩罚
            assert self.config.algorithm.adv_estimator == 'grpo' or self.config.algorithm.kl_ctrl.kl_coef <= 1e-10

    def save_checkpoint(self):
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

        actor_upload_future = self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_upload_future = self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)
        else:
            critic_upload_future = None

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)
        # upload to hdfs
        hput(dataloader_local_path, remote_global_step_folder)

        # TODO(zhangchi.usc1992). Actually, we should postpone writing latest_checkpointed_iteration when all the hdfs upload finishes
        # save latest_checkpointed_iteration.txt
        if self.valid_hdfs_global_step is not None:
            local_latest_checkpointed_iteration = os.path.join(local_checkpoint_folder,
                                                               'latest_checkpointed_iteration.txt')
            with open(local_latest_checkpointed_iteration, 'w') as f:
                f.write(str(self.valid_hdfs_global_step))
            hput(local_latest_checkpointed_iteration, remote_checkpoint_folder)

        self.valid_hdfs_global_step = self.global_step

        # mark a checkpoint version for future checkpoint format change and compatibility
        local_ckpt_version = os.path.join(local_checkpoint_folder, 'checkpoint_version.txt')
        with open(local_ckpt_version, 'w') as f:
            f.write('v1')
        hput(local_ckpt_version, remote_checkpoint_folder)

        ray.get(actor_upload_future)

        if critic_upload_future is not None:
            ray.get(critic_upload_future)

    def load_checkpoint(self):
        if self.config.trainer.resume_steps == 'disable':
            return 0

        from verl.utils.fs import copy_local_path_from_hdfs
        # find the latest global step
        if self.config.trainer.resume_steps == 'auto':
            from omnistore.utilities.ckpt_format_tool import find_latest_ckpt_path
            remote_checkpoint_folder = os.path.join(self.config.trainer.default_hdfs_dir, 'checkpoints')
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
        self.global_step += 1

        print(f'Setting global step to {self.global_step}')
        print(f'Resuming from {remote_global_step_folder}')

        actor_remote_path = os.path.join(remote_global_step_folder, 'actor')
        critic_remote_path = os.path.join(remote_global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_remote_path)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_remote_path)
        # load dataloader
        dataloader_remote_path = os.path.join(remote_global_step_folder, 'data.pt')
        dataloader_local_path = copy_local_path_from_hdfs(dataloader_remote_path)
        self.train_dataloader = torch.load(dataloader_local_path)
        return self.global_step

    def fit(self):
        self.global_step = 0

        # load checkpoint before doing anything
        resume_step = self.load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.eval_before_training:
            val_metrics = self._validate(val_epoch=self.config.trainer.val_epoch,
                                         need_log=self.config.trainer.need_log,
                                         log_file=self.config.trainer.log_file)
            pprint(f'Initial validation metrics: {val_metrics}')
            self.logger.log(data=val_metrics, step=self.global_step)
        if self.config.trainer.val_only:
            return

        self.global_step = 1

        # TODO: add staleness
        standalone_batch = []
        pending_batch_queue = queue.Queue()
        ready_batch_queue = queue.Queue()
        while True:
            for batch_dict in self.train_dataloader:
                metrics = {}
                with Timer(name='step', logger=None) as step_timer:
                    # hybrid generate (on policy)
                    batch: DataProto = DataProto.from_single_dict(batch_dict)

                    if self.config.data.num_prompts_per_data > 1:
                        batch = batch.unfold_column_chunks(self.config.data.num_prompts_per_data,
                                                           split_keys=['input_ids', 'attention_mask', 'position_ids'])

                    batch = batch.repeat(self.num_bon)
                    tmp_batch = copy.deepcopy(batch)
                    gen_batch = tmp_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                    gen_batch.meta_info[
                        'generation_kwargs'] = self.config.actor_rollout_ref.rollout.train_generate_kwargs
                    if self.global_step < resume_step + self.config.streaming_rollout.warmup_step:
                        complete_ratio = 1.0
                    else:
                        complete_ratio = self.config.actor_rollout_ref.rollout.get("complete_ratio", 1)
                    gen_batch.meta_info['complete_ratio'] = complete_ratio
                    pprint(f'start rollout, batches {len(gen_batch)}.')
                    with Timer(name='gen', logger=None) as timer:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    metrics['timing/gen'] = timer.last
                    metrics['rollout/hybrid_input_batch'] = len(batch)
                    # only report metrics from one generation replica
                    if 'xperf_metrics' in gen_batch_output.meta_info:
                        for name, x_metric in gen_batch_output.meta_info['xperf_metrics'].items():
                            self.logger.log(data={"rollout/gen/hybrid_{}".format(name): wandb.Histogram(x_metric)},
                                            step=self.global_step)

                    # prepare for standalone generation
                    gen_batch_output = tmp_batch.union(gen_batch_output)
                    is_finished = gen_batch_output.pop(batch_keys=['is_finished']).batch['is_finished']
                    for i in range(len(gen_batch_output)):
                        item = gen_batch_output[i] if is_finished[i] else batch[i]
                        item.batch = item.batch.unsqueeze(0)
                        for key, value in item.non_tensor_batch.items():
                            item.non_tensor_batch[key] = np.atleast_1d(np.array(value, dtype=object))
                        if is_finished[i]:
                            ready_batch_queue.put(item)
                        else:
                            pending_batch_queue.put(item)
                    pprint(
                        f'stop rollout, ready batches {ready_batch_queue.qsize()}, pending batches {pending_batch_queue.qsize()}.'
                    )
                    finished_num = is_finished.sum().int().item()
                    metrics['rollout/hybrid_completed_batch'] = finished_num
                    metrics['rollout/hybrid_incompleted_batch'] = len(batch) - finished_num

                    # stop standalone rollout to update model
                    finished_num = 0
                    if len(standalone_batch) > 0:
                        gen_batch_output = self.standalone_rollout_wg.generate_sequences_get(standalone_gen_batch)
                        gen_batch_output = standalone_tmp_batch.union(gen_batch_output)
                        is_finished = gen_batch_output.pop(batch_keys=['is_finished']).batch['is_finished']
                        for i in range(len(gen_batch_output)):
                            item = gen_batch_output[i] if is_finished[i] else standalone_batch[i]
                            item.batch = item.batch.unsqueeze(0)
                            for key, value in item.non_tensor_batch.items():
                                item.non_tensor_batch[key] = np.atleast_1d(np.array(value, dtype=object))
                            if is_finished[i]:
                                ready_batch_queue.put(item)
                            else:
                                pending_batch_queue.put(item)
                        pprint(
                            f'stop standalone rollout, ready batches {ready_batch_queue.qsize()}, pending batches {pending_batch_queue.qsize()}.'
                        )
                        finished_num = is_finished.sum().int().item()
                        # only report metrics from one generation replica
                        if 'xperf_metrics' in gen_batch_output.meta_info:
                            for name, x_metric in gen_batch_output.meta_info['xperf_metrics'].items():
                                self.logger.log(
                                    data={"rollout/gen/standalone_{}".format(name): wandb.Histogram(x_metric)},
                                    step=self.global_step)
                    metrics['rollout/standalone_completed_batch'] = finished_num
                    metrics['rollout/standalone_incompleted_batch'] = len(standalone_batch) - finished_num

                    # update standalone rollout weights
                    with Timer(name='update_standalone', logger=None) as timer:
                        if hasattr(self, "standalone_rollout_wg"):
                            self.actor_rollout_wg.update_standalone_rollout()
                            self.standalone_rollout_wg.update_standalone_rollout()
                    metrics['timing/update_standalone'] = timer.last

                    # standalone generate (off policy)
                    standalone_batch = []
                    while hasattr(self, "standalone_rollout_wg") and pending_batch_queue.qsize(
                    ) >= self.standalone_rollout_wg.world_size:
                        for _ in range(self.standalone_rollout_wg.world_size):
                            standalone_batch.append(pending_batch_queue.get())
                    if len(standalone_batch) > 0:
                        standalone_batch = DataProto.concat(standalone_batch)
                        standalone_tmp_batch = copy.deepcopy(standalone_batch)
                        standalone_gen_batch = standalone_tmp_batch.pop(
                            batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                        standalone_gen_batch.meta_info[
                            'generation_kwargs'] = self.config.actor_rollout_ref.rollout.train_generate_kwargs
                        standalone_gen_batch.meta_info['complete_ratio'] = 1
                        self.standalone_rollout_wg.generate_sequences_put(standalone_gen_batch)
                        pprint(f'start standalone rollout, batches {len(standalone_batch)}.')
                    metrics['rollout/standalone_input_batch'] = len(standalone_batch)

                    # get training batch from ready queue
                    ready_batch = []
                    while ready_batch_queue.qsize() >= self.config.actor_rollout_ref.actor.ppo_mini_batch_size:
                        for _ in range(self.config.actor_rollout_ref.actor.ppo_mini_batch_size):
                            ready_batch.append(ready_batch_queue.get())
                    batch = DataProto.concat(ready_batch)
                    if self.config.algorithm.force_append_eos:
                        batch.batch["input_ids"][:, -1] = self.tokenizer.eos_token_id
                        batch.batch["responses"][:, -1] = self.tokenizer.eos_token_id
                    batch.meta_info['generation_kwargs'] = self.config.actor_rollout_ref.rollout.train_generate_kwargs
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    metrics['rollout/training_batch'] = len(batch)

                    # training
                    with Timer(name='old_log_probs', logger=None) as timer:
                        batch = self.actor_rollout_wg.old_log_probs(batch)
                    metrics['timing/old_log_probs'] = timer.last

                    with Timer(name='rm_score', logger=None) as timer:
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                    metrics['timing/rm_score'] = timer.last

                    with Timer(name='reward_fn', logger=None) as timer:
                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch, global_step=self.global_step)
                        batch.batch['token_level_scores'] = reward_tensor
                    metrics['timing/reward_fn'] = timer.last

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
                            if use_async_gen:
                                batch, bon_metrics = select_training_samples_v2(
                                    batch=batch,
                                    strategy=self.config.actor_rollout_ref.rollout.bon_strategy,
                                    config=self.config)
                                metrics.update(bon_metrics)
                            else:
                                batch = select_training_samples(
                                    batch=batch,
                                    strategy=self.config.actor_rollout_ref.rollout.bon_strategy,
                                    config=self.config)
                        metrics['timing/select_bon_samples'] = timer.last

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with Timer(name='ref', logger=None) as timer:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                        metrics['timing/ref'] = timer.last

                    # compute values
                    if self.use_critic:
                        with Timer(name='values', logger=None) as timer:
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                        metrics['timing/values'] = timer.last

                    with Timer(name='adv', logger=None) as timer:
                        # compute rewards. apply_kl_penalty if available
                        batch, kl_metrics = apply_kl_penalty(batch,
                                                             kl_ctrl=self.kl_ctrl,
                                                             kl_penalty=self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)

                        # compute advantages
                        batch = compute_advantage(
                            batch,
                            self.config.algorithm.gamma,
                            self.config.algorithm.lam,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            upgo_loss_version=self.config.actor_rollout_ref.actor.upgo_loss_version,
                            num_bon=self.config.actor_rollout_ref.rollout.num_bon,
                            adv_whiten=self.config.algorithm.adv_whiten)
                    metrics['timing/adv'] = timer.last

                    # update critic
                    if self.use_critic:
                        with Timer(name='update_critic', logger=None) as timer:
                            critic_output = self.critic_wg.update_critic(batch)
                        metrics['timing/update_critic'] = timer.last
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_step and self.global_step % self.config.trainer.actor_update_freq == 0:
                        # update actor
                        with Timer(name='update_actor', logger=None) as timer:
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics['timing/update_actor'] = timer.last
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.global_step % self.config.trainer.test_freq == 0:
                        with Timer(name='testing', logger=None) as timer:
                            val_metrics: dict = self._validate()
                            val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
                        metrics['timing/testing'] = timer.last
                        metrics.update(val_metrics)

                    # collect metrics
                    with Timer(name='compute_metrics', logger=None) as timer:
                        # Note that we can use any worker groups here
                        data_metrics = self.actor_rollout_wg.execute_func_rank_zero(compute_data_metrics, batch,
                                                                                    self.use_critic,
                                                                                    self.config.reward_model.mean,
                                                                                    self.config.reward_model.std)
                    metrics['timing/compute_metrics'] = timer.last
                    metrics.update(data_metrics)

                    with Timer(name='save_checkpoint', logger=None) as timer:
                        if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                            self.save_checkpoint()
                    metrics['timing/save_checkpoint'] = timer.last

                metrics['timing/step'] = step_timer.last
                # TODO: make a canonical logger that supports various backend
                self.logger.log(data=metrics, step=self.global_step)

                self.global_step += 1

                if self.global_step >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')

                    return
