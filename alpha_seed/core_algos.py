# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor, use_variable_lambda: torch.Tensor,
                                 variable_lambda_scalar: torch.Tensor, adv_whiten: bool, use_separate_critic_lam: bool,
                                 critic_lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    token_level_rewards = token_level_rewards * eos_mask
    values = values * eos_mask
    if use_variable_lambda:
        seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
        lam = torch.clamp(1 - 1 / (variable_lambda_scalar * seq_len_per_sample), min=lam)
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        if use_separate_critic_lam:
            critic_lastgaelam = 0
            critic_advantages_reversed = []

        gen_len = token_level_rewards.shape[-1]
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            if use_separate_critic_lam:
                critic_lastgaelam = delta + gamma * critic_lam * critic_lastgaelam
                critic_advantages_reversed.append(critic_lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        if use_separate_critic_lam:
            critic_advantages = torch.stack(critic_advantages_reversed[::-1], dim=1)
            returns = critic_advantages + values
        else:
            returns = advantages + values
        origin_advantages = advantages
        if adv_whiten:
            advantages = verl_F.masked_whiten(origin_advantages, eos_mask)
        else:
            advantages = torch.clone(origin_advantages)
    return origin_advantages, advantages, returns


def compute_upgo_advantage(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                           upgo_loss_version: int):
    token_level_rewards = token_level_rewards * eos_mask
    values = values * eos_mask
    # upgo的return、adv计算
    upgo_returns = torch.zeros_like(token_level_rewards)
    upgo_returns[:, -1] = token_level_rewards[:, -1]
    if upgo_loss_version == 0:
        upgo_indicator = torch.ge(token_level_rewards[:, :-1] + values[:, 1:] - values[:, :-1], 0)
    elif upgo_loss_version == 1:
        upgo_indicator = torch.ge(token_level_rewards[:, 1:-1] + values[:, 2:] - values[:, 1:-1], 0)
        last_upgo_indicator = torch.ge(token_level_rewards[:, -1:] - values[:, -1], 0)
        upgo_indicator = torch.concat([upgo_indicator, last_upgo_indicator], dim=1)
    else:
        raise NotImplemented
    gen_len = token_level_rewards.shape[-1]
    for i in reversed(range(gen_len - 1)):
        upgo_returns[:, i] = torch.where(upgo_indicator[:, i], token_level_rewards[:, i] + upgo_returns[:, i + 1],
                                         token_level_rewards[:, i] + values[:, i + 1])
    upgo_advantages = upgo_returns - values
    return upgo_advantages


def compute_grpo_advantage_return(token_level_scores: torch.Tensor,
                                  num_bon: torch.Tensor,
                                  eos_mask: torch.Tensor,
                                  index: torch.Tensor,
                                  epsilon: float = 1e-6,
                                  use_async_gen: bool = False,
                                  group_mode: str = "normal"):  # normal, no_std, clamp, trinary
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_scores: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        num_bon: `(float)`
            response num per prompt

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    response_length = token_level_scores.shape[-1]
    scores = token_level_scores.sum(-1)
    metrics = {}
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if group_mode not in ["clamp", "trinary"]:
                id2score[index[i]].append(scores[i])
            elif group_mode == "clamp":
                id2score[index[i]].append(min(max(scores[i], -1.0), 1.0))
            elif group_mode == "trinary":
                id2score[index[i]].append(1.0 if scores[i] > 0 else 0 if scores[i] >= 0 else -1.0)

        lens = list(map(lambda x: len(x), id2score.values()))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
        for i in range(bsz):
            if group_mode != "no_std":
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(dim=1).tile([1, response_length]) * eos_mask

    if use_async_gen:
        metrics.update({
            "GRPO_aysnc/max_response_num": max(lens),
            "GRPO_aysnc/min_response_num": min(lens),
            "GRPO_aysnc/mean_response_num": np.mean(lens),
        })
    return scores, scores, metrics


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_lm_loss(log_prob, raw_scores, eos_ids):
    eos_ids = eos_ids.unsqueeze(1)
    scores = torch.gather(raw_scores, 1, eos_ids)
    ids = torch.arange(log_prob.shape[1], device=eos_ids.device).unsqueeze(0).repeat(log_prob.shape[0], 1)
    mask0 = ids <= eos_ids
    mask1 = (scores > 0).repeat(1, log_prob.shape[1])
    mask = mask0 & mask1
    lm_loss = torch.masked_select(log_prob, mask)
    lm_loss = -torch.sum(lm_loss) / max(lm_loss.numel(), 1)
    return lm_loss


def compute_policy_loss(old_log_prob, ref_log_prob, log_prob, advantages, upgo_advantages, eos_mask, cliprange_low,
                        cliprange_high, cliprange2, scale_pg_by_kl, scale_pg_by_local_kl, upgo_loss_weight,
                        use_ewma_loss, kl_penalty_type, overlong_mask, loss_average_method):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
    if not use_ewma_loss:
        ratio = torch.exp(log_prob - old_log_prob)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)
        pg_losses2_hi = -advantages * torch.clamp(ratio, max=1.0 + cliprange_high)
        pg_losses2_lo = -advantages * torch.clamp(ratio, min=1.0 - cliprange_low)
        pg_losses3 = torch.abs(-advantages * cliprange2)
        pg_losses_clip = torch.maximum(pg_losses1, pg_losses2)
        pg_losses = torch.minimum(pg_losses_clip, pg_losses3)  # 这个应该对advantage为正的情况不影响
    else:
        # ref: https://github.com/openai/ppo-ewma/blob/master/ppo_ewma/ppo.py#L93
        # log space importance sampling
        log_ratio = log_prob - ref_log_prob  # old
        # clip by 10.0
        logp_adj = torch.max(old_log_prob, log_prob.detach() - np.log(10.))
        # log space importance sampling again
        pg_losses1 = -advantages * torch.exp(log_prob - logp_adj)
        clipped_logratio = torch.clamp(log_ratio, np.log(1.0 - cliprange_low), np.log(1.0 + cliprange_high))
        pg_losses2 = -advantages * torch.exp(clipped_logratio + ref_log_prob - logp_adj)
        clipped_logratio_hi = torch.clamp(log_ratio, max=np.log(1.0 + cliprange_high))
        clipped_logratio_lo = torch.clamp(log_ratio, min=np.log(1.0 - cliprange_low))
        pg_losses2_hi = -advantages * torch.exp(clipped_logratio_hi + ref_log_prob - logp_adj)
        pg_losses2_lo = -advantages * torch.exp(clipped_logratio_lo + ref_log_prob - logp_adj)

        pg_losses3 = torch.abs(-advantages * cliprange2)
        pg_losses_clip = torch.maximum(pg_losses1, pg_losses2)
        pg_losses = torch.minimum(pg_losses_clip, pg_losses3)  # 这个应该对advantage为正的情况不影响

    assert loss_average_method in ['sample', 'token'
                                  ], f"loss_average_method must be 'sample' or 'token', but got {loss_average_method}"

    if loss_average_method == 'sample':
        pg_loss = torch.sum(pg_losses * eos_mask, dim=1) / seq_len_per_sample  # batch
    else:
        pg_loss = pg_losses  # batch x seq_len

    negative_approx_kl = kl_penalty(log_prob, old_log_prob, kl_penalty_type=kl_penalty_type)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)
    ppo_kl_sum = torch.mean(torch.sum(-negative_approx_kl * eos_mask, dim=1))

    if scale_pg_by_kl:
        sqrt_kl = torch.sqrt(
            torch.clamp(torch.sum(kl_penalty(old_log_prob, ref_log_prob, kl_penalty_type=kl_penalty_type) * eos_mask,
                                  dim=1),
                        min=1.0))
        normed_sqrt_kl = (1 / sqrt_kl) / (torch.sum(1 / sqrt_kl)) * torch.clamp(torch.sum(eos_mask[:, 0]), min=1.0)
        if loss_average_method == 'sample':
            pg_loss = pg_loss * normed_sqrt_kl
        else:
            pg_loss = pg_loss * normed_sqrt_kl.unsqueeze(-1)

    if scale_pg_by_local_kl:
        sqrt_kl = torch.sqrt(torch.clamp(torch.sum(negative_approx_kl * eos_mask, dim=1), min=1.0))
        normed_sqrt_kl = (1 / sqrt_kl) / (torch.sum(1 / sqrt_kl)) * torch.clamp(torch.sum(eos_mask[:, 0]), min=1.0)
        if loss_average_method == 'sample':
            pg_loss = pg_loss * normed_sqrt_kl
        else:
            pg_loss = pg_loss * normed_sqrt_kl.unsqueeze(-1)

    pg_loss_mask = eos_mask
    if overlong_mask is not None:
        if loss_average_method == 'sample':
            pg_loss = pg_loss * overlong_mask
        else:
            pg_loss_mask = pg_loss_mask * overlong_mask.unsqueeze(-1)

    if loss_average_method == 'sample':
        pg_loss = torch.mean(pg_loss)
    else:
        pg_loss = verl_F.masked_mean(pg_loss, pg_loss_mask)

    if upgo_loss_weight > 0.0:
        rho = torch.minimum(ratio, torch.ones_like(ratio)).detach()
        upgo_losses = -rho * upgo_advantages * log_prob
        upgo_losses = torch.sum(upgo_losses * eos_mask, dim=1) / seq_len_per_sample
        upgo_loss = torch.mean(upgo_losses)
    else:
        upgo_loss = torch.zeros(()).to(pg_loss.device)
    total_loss = pg_loss + upgo_loss_weight * upgo_loss

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)
    pg_clipfrac_hi = verl_F.masked_mean(torch.gt(pg_losses2_hi, pg_losses1).float(), eos_mask)
    pg_clipfrac_lo = verl_F.masked_mean(torch.gt(pg_losses2_lo, pg_losses1).float(), eos_mask)
    pg_clipfrac2 = verl_F.masked_mean(torch.gt(pg_losses1, pg_losses3).float(), eos_mask)

    if total_loss.isnan().any():
        print("find nan in total_loss, tracing...")
        variables = {"pg_losses1": pg_losses1, "pg_losses2": pg_losses2, "pg_losses3": pg_losses3}
        variables.update({"log_prob": log_prob, "old_log_prob": old_log_prob, "advantages": advantages, "ratio": ratio})
        for k, v in variables.items():
            if v.isnan().any():
                print("find nan in ", k, v)
            if v.isinf().any():
                print("find inf in ", k, v)
        raise ValueError("find nan in total_loss")

    return total_loss, pg_loss, upgo_loss, pg_clipfrac, pg_clipfrac_hi, pg_clipfrac_lo, pg_clipfrac2, ppo_kl, ppo_kl_sum


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value_low, cliprange_value_high, overlong_mask,
                       loss_average_method):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value_low, values + cliprange_value_high)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
    if loss_average_method == 'token':
        if overlong_mask is not None:
            vf_loss = verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask * overlong_mask.unsqueeze(1))
        else:
            vf_loss = verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    elif loss_average_method == 'sample':
        if overlong_mask is not None:
            vf_loss = 0.5 * torch.mean(
                torch.sum(torch.max(vf_losses1, vf_losses2) * eos_mask, dim=1) / seq_len_per_sample * overlong_mask)
        else:
            vf_loss = 0.5 * torch.mean(
                torch.sum(torch.max(vf_losses1, vf_losses2) * eos_mask, dim=1) / seq_len_per_sample)
    else:
        raise NotImplementedError
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    vf_loss = vf_loss
    seq_level_vf_loss = torch.sum(torch.max(vf_losses1, vf_losses2) * eos_mask, dim=1) / seq_len_per_sample
    return vf_loss, vf_clipfrac, seq_level_vf_loss


def compute_kl_loss(log_prob, ref_log_prob, eos_mask, kl_penalty_):
    if kl_penalty_ in ("abs", "mse", "low_var_kl"):
        kl = kl_penalty(log_prob, ref_log_prob, kl_penalty_)
    elif kl_penalty_ in ("kl"):
        kl = kl_penalty(log_prob, ref_log_prob, kl_penalty_).square()
    else:
        raise NotImplementedError
    seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
    kl_loss = torch.mean(torch.sum(kl * eos_mask, dim=1) / seq_len_per_sample)
    return kl_loss


from alpha_seed.utils.functional import clip_by_value_preserve_gradient


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty_type) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob: (bs, response_len)
        ref_logprob: (bs, response_len)

    Returns:
        per_token_kl: (bs, response_len)

    """
    if kl_penalty_type == "kl":
        return logprob - ref_logprob

    if kl_penalty_type == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty_type == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty_type == "low_var_kl":
        ratio = ref_logprob - logprob
        return torch.clamp(torch.exp(ratio) - ratio - 1, min=-10, max=10)

    if kl_penalty_type == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError
        # total_logprob = torch.softmax(logits, dim=-1)
        # total_ref_logprob = torch.softmax(ref_logits, dim=-1)
        # return torch.sum(total_logprob * torch.log(total_logprob / (total_ref_logprob + 1e-10)), dim=-1)

    raise NotImplementedError
