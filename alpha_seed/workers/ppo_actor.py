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
Single Process Actor
"""
from typing import Dict
import math

import torch
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from mono_rl import DataProto
from verl.workers.actor import BasePPOActor
import verl.utils.torch_functional as verl_F
from alpha_seed.utils.functional import append_dict_items_to_dict

from mono_rl.models.seed_models.modeling_vlm import get_image_keys
from alpha_seed import core_algos
from mono_rl.worker.engine.fsdp.models.model import FSDPModel
from omegaconf import DictConfig, OmegaConf

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(self, as_config: DictConfig, model_engine: FSDPModel):
        super().__init__(as_config)
        self.engine = model_engine

    def compute_log_prob(self, data: DataProto, reuse_old_experts=False):
        select_keys = ['responses', 'input_ids', 'attention_mask']
        image_keys = get_image_keys(data.non_tensor_batch)
        selected_data = data.select(batch_keys=select_keys, non_tensor_batch_keys=image_keys)

        mtp_n_heads = getattr(self.engine.model_config, 'mtp_n_heads', 1)

        entropy_lst = []
        log_prob_lst = []
        acceptance_matrix_lst = [[] for _ in range(mtp_n_heads - 1)]
        selected_experts_lst = []
        # Note: mismatched data order (here vs. upldate policy) can lead to
        # mismatched log probs. In order to match them, we need to split
        # batch into mini batches (same with training).
        chunk_size = math.ceil(selected_data.batch.batch_size[0] / self.config.ppo_mini_batch_size)
        for _, mini_batch in enumerate(selected_data.chunk(chunk_size)):
            output_proto = self.engine.forward_backward_step(data=mini_batch,
                                                             forward_only=True,
                                                             reuse_old_experts=reuse_old_experts)
            if isinstance(self.engine.model_module, FSDP):
                self.engine.model_module._handle.reshard(True)  # release memory
            log_prob_lst.append(output_proto.batch['logprobs'])
            entropy_lst.append(output_proto.batch['entropy'])
            if reuse_old_experts:
                selected_experts_lst.append(output_proto.batch['old_experts'])
            for j in range(mtp_n_heads - 1):
                if output_proto.batch.get(f"acceptance_matrix_{j}", None) is not None:
                    acceptance_matrix_lst[j].append(output_proto.batch[f'acceptance_matrix_{j}'])
        log_probs = torch.concat(log_prob_lst, dim=0)
        entropy = torch.concat(entropy_lst, dim=0)
        if reuse_old_experts:
            selected_experts = torch.concat(selected_experts_lst, dim=0)
        else:
            selected_experts = None
        if len(acceptance_matrix_lst) and len(acceptance_matrix_lst[0]):
            acceptance_matrix = [torch.concat(acceptance_matrix_lst[j], dim=0) for j in range(mtp_n_heads - 1)]
        else:
            acceptance_matrix = []
        return entropy, log_probs, tuple(acceptance_matrix), selected_experts

    def train_one_step(self, data: DataProto):
        self.engine.set_loss(pg_loss_fn, OmegaConf.to_container(self.config, resolve=True))
        output_proto = self.engine.forward_backward_step(data=data, forward_only=False)
        metrics_opt = self.engine.optimizer_step()
        self.engine.optimizer_zero_grad()

        metrics = {
            'actor/grad_norm': metrics_opt['grad_norm'],  # NOTE: grad_norm is a float from monorl
            'actor/#micro_batch_update': output_proto.meta_info["metrics"].pop('#micro_batch_update'),
            **output_proto.meta_info["metrics"],
        }
        return metrics

    def update_policy(self, data: DataProto):
        config_dict = OmegaConf.to_container(self.config, resolve=True)

        # compute batch full token count
        response_length = data.batch['responses'].size(1)
        batch_full_token_count_mask = data.batch['attention_mask'][:, -response_length:]
        if 'overlong_mask' in data.batch.keys():
            batch_full_token_count_mask *= data.batch['overlong_mask'].unsqueeze(-1)
        batch_full_token_count = max(1, batch_full_token_count_mask.sum().item())
        config_dict["response_length"] = response_length
        config_dict["batch_full_token_count"] = batch_full_token_count

        dataloader = make_mini_step_dataloader(data, self.config.ppo_mini_batch_size, return_dataproto=True)
        config_dict["dataloader_length"] = len(dataloader)
        metrics = {}

        for batch_idx, mini_batch in enumerate(dataloader):
            self.engine.optimizer_zero_grad()

            # compute minibatch full token count
            mini_batch_full_token_count_mask = mini_batch.batch['attention_mask'][:, -response_length:]
            if 'overlong_mask' in mini_batch.batch.keys():
                mini_batch_full_token_count_mask *= mini_batch.batch['overlong_mask'].unsqueeze(-1)
            mini_batch_full_token_count = max(1, mini_batch_full_token_count_mask.sum().item())
            config_dict["mini_batch_full_token_count"] = mini_batch_full_token_count

            # Set the loss function of update actor function for every mini batch
            self.engine.set_loss(pg_loss_fn, config_dict)
            output_proto = self.engine.forward_backward_step(data=mini_batch, forward_only=False)

            # TODO: support early stop by kl in monorl

            metrics_opt = self.engine.optimizer_step()
            self.engine.optimizer_zero_grad()
            data_metric = {
                'actor/grad_norm': metrics_opt['grad_norm'],  # NOTE: grad_norm is a float from monorl
                'actor/#micro_batch_update': output_proto.meta_info["metrics"].pop('#micro_batch_update'),
                **output_proto.meta_info["metrics"],
            }
            append_dict_items_to_dict(metrics, data_metric)

        # TODO: support first_mini_ppo_kl_sum in monorl
        # append_to_dict(metrics, {'first_mini_ppo_kl_sum': first_mini_ppo_kl_sum})

        self.engine.optimizer_zero_grad()
        self.engine.clear_memory_cache()
        return metrics


def pg_loss_fn(config: Dict, output: TensorDict, micro_data: TensorDict):
    loss_average_method = config.get("loss_average_method", "sample")
    use_dynamic_bsz = config.get("use_dynamic_bsz", True)
    ppo_mini_batch_size = config.get("ppo_mini_batch_size", -1)
    ppo_micro_batch_size = config.get("ppo_micro_batch_size", -1)
    mini_batch_full_token_count = config.get("mini_batch_full_token_count", -1)
    batch_full_token_count = config.get("batch_full_token_count", -1)
    dataloader_length = config.get("dataloader_length", -1)
    full_entropy = output['entropy']
    log_prob = output['logprobs']

    policy_loss, micro_data_metric = default_pg_loss_fn(config, micro_data, full_entropy, log_prob)

    if loss_average_method in ['token', 'sample', 'constant']:
        if use_dynamic_bsz:
            loss = policy_loss * (len(micro_data) / ppo_mini_batch_size)
        else:
            gradient_accumulation = ppo_mini_batch_size // ppo_micro_batch_size
            loss = policy_loss / gradient_accumulation
    elif loss_average_method == 'minibatch':
        loss = policy_loss / mini_batch_full_token_count
    elif loss_average_method == 'batch':
        loss = policy_loss / batch_full_token_count * dataloader_length
    else:
        raise NotImplementedError(f'loss_average_method {loss_average_method} not implemented.')

    return loss, micro_data_metric


def default_pg_loss_fn(config, micro_data, full_entropy, log_prob):
    """
    Note that the loss function used by monorl should return a tuple as below:
    1. policy_loss: a single float tensor that will be used for backward function.
    2. metrics: a dict in the format of {key: val (float/int/list/tensor)} that records key metrics.
    """
    response_length = config.get("response_length", 2048)
    attention_mask = micro_data["attention_mask"]
    if config.get("use_model_output_mask", False):
        loss_mask = micro_data['model_output_mask']
        response_mask = loss_mask[:, -response_length:]
    else:
        response_mask = attention_mask[:, -response_length:]

    use_rollout_behavior_log_probs = config.get("use_rollout_behavior_log_probs", False)
    if use_rollout_behavior_log_probs:
        # use ewma if use_rollout_behavior_log_probs: importance sampling by rollout_logprob, clip by old_log_prob
        use_ewma_loss = True
        old_log_prob = micro_data['rollout_behavior_log_probs']
        ref_log_prob = micro_data['old_log_probs']
    else:
        use_ewma_loss = config.get("use_ewma_loss", False)
        old_log_prob = micro_data["old_log_probs"]
        ref_log_prob = micro_data.get("ref_log_prob", None)

    advantages = micro_data["advantages"]
    upgo_advantages = micro_data["upgo_advantages"]
    overlong_mask = micro_data.get("overlong_mask", None)

    clip_ratio = config.get("clip_ratio", 0.2)
    clip_ratio_low = clip_ratio
    clip_ratio_high = clip_ratio
    if config.get("clip_ratio_low", None):
        clip_ratio_low = config.get("clip_ratio_low", None)
    if config.get("clip_ratio_high", None):
        clip_ratio_high = config.get("clip_ratio_high", None)
    clip_ratio2 = config.get("clip_ratio2", 10.0)
    scale_pg_by_kl = config.get("scale_pg_by_kl", False)
    scale_pg_by_local_kl = config.get("scale_pg_by_local_kl", False)
    entropy_coeff = config.get("entropy_coeff", 0.001)
    upgo_loss_weight = config.get("upgo_loss_weight", 0.0)
    kl_loss_weight = config.get("kl_loss_weight", 0.0)
    offpolicy_kl_loss_weight = config.get("offpolicy_kl_loss_weight", 0.0)
    lm_loss_weight = config.get("lm_loss_weight", 0.0)
    kl_penalty_type = config.get("kl_penalty", "low_var_kl")
    loss_average_method = config.get("loss_average_method", "sample")
    loss_average_constant = config.get("loss_average_constant", 0)
    total_loss, pg_loss, upgo_loss, pg_clipfrac, pg_clipfrac_hi, pg_clipfrac_lo, pg_clipfrac2, ppo_kl, ppo_kl_sum = (
        core_algos.compute_policy_loss(
            old_log_prob=old_log_prob,
            ref_log_prob=ref_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            upgo_advantages=upgo_advantages,
            eos_mask=response_mask,
            cliprange_low=clip_ratio_low,
            cliprange_high=clip_ratio_high,
            cliprange2=clip_ratio2,
            scale_pg_by_kl=scale_pg_by_kl,
            scale_pg_by_local_kl=scale_pg_by_local_kl,
            upgo_loss_weight=upgo_loss_weight,
            use_ewma_loss=use_ewma_loss,
            kl_penalty_type=kl_penalty_type,
            overlong_mask=overlong_mask,
            loss_average_method=loss_average_method,
            loss_average_constant=loss_average_constant,
        ))

    if kl_loss_weight > 0.0:
        kl_loss = core_algos.compute_kl_loss(log_prob,
                                             ref_log_prob,
                                             response_mask,
                                             kl_penalty_type,
                                             loss_average_method=loss_average_method)
    else:
        kl_loss = torch.zeros((), device=pg_loss.device)

    if offpolicy_kl_loss_weight > 0.0:
        offpolicy_kl_loss = core_algos.compute_kl_loss(log_prob,
                                                       old_log_prob,
                                                       response_mask,
                                                       kl_penalty_type,
                                                       loss_average_method=loss_average_method)
    else:
        offpolicy_kl_loss = torch.zeros((), device=pg_loss.device)

    if lm_loss_weight > 0.0:
        eos_ids = micro_data["eos_ids"]
        raw_scores = micro_data["token_level_scores"]
        lm_loss = core_algos.compute_lm_loss(log_prob, raw_scores, eos_ids, loss_average_method=loss_average_method)
    else:
        lm_loss = torch.zeros((), device=pg_loss.device)

    if entropy_coeff <= 0.0:
        compute_entropy = False
    else:
        compute_entropy = True

    if compute_entropy:
        entropy_loss = verl_F.masked_mean(full_entropy, response_mask)
    else:
        entropy_loss = torch.zeros((), device=pg_loss.device)

    policy_loss = total_loss - entropy_loss * entropy_coeff + kl_loss_weight * kl_loss + lm_loss_weight * lm_loss + offpolicy_kl_loss_weight * offpolicy_kl_loss

    logprob_eq_zero = torch.lt(torch.exp(old_log_prob), 1e-10).float().mean()

    metrics = {
        # 'actor/entropy': entropy_loss.detach().item(),
        "actor/pg_loss": pg_loss.detach().item(),
        "actor/upgo_loss": upgo_loss.detach().item(),
        "actor/kl_loss": kl_loss.detach().item(),
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/pg_clipfrac_hi": pg_clipfrac_hi.detach().item(),
        "actor/pg_clipfrac_lo": pg_clipfrac_lo.detach().item(),
        "actor/pg_clipfrac2": pg_clipfrac2.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/ppo_kl_sum": ppo_kl_sum.detach().item(),
        "actor/tokens_per_micro_batch_update": attention_mask.sum().detach().item(),
        "actor/lm_loss": lm_loss.detach().item(),
        "actor/logprob_eq_zero": logprob_eq_zero.detach().item(),
    }
    return policy_loss, metrics


def make_mini_step_dataloader(data, ppo_mini_batch_size, return_dataproto=False):
    select_keys = [
        'responses', 'input_ids', 'attention_mask', 'old_log_probs', 'advantages', 'upgo_advantages', 'off_policy_steps'
    ]
    for opt_key in [
            'ref_log_prob', 'rollout_behavior_log_probs', 'overlong_mask', 'eos_ids', 'token_level_scores',
            'model_output_mask', 'old_experts'
    ]:
        if opt_key in data.batch.keys():
            select_keys.append(opt_key)
    non_tensor_keys = get_image_keys(data.non_tensor_batch)
    if non_tensor_keys:
        assert return_dataproto
    if return_dataproto:
        mini_steps = data.batch.batch_size[0] // ppo_mini_batch_size
        dataloader = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_keys).chunk(mini_steps)
    else:
        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(ppo_mini_batch_size)
    return dataloader
