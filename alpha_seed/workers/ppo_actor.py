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
import ray
import torch
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from collections import defaultdict
from mono_rl import DataProto
from verl.workers.actor import BasePPOActor
import verl.utils.torch_functional as verl_F
from alpha_seed.utils.functional import append_dict_items_to_dict

from mono_rl.models.seed_models.modeling_vlm import get_image_keys
from mono_rl.utils.dataset.dist_data_util import get_dist_data_manager
from alpha_seed import core_algos
from mono_rl.worker.engine.fsdp.models.model import FSDPModel
from omegaconf import DictConfig, OmegaConf

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(self, as_config: DictConfig, model_engine: FSDPModel):
        super().__init__(as_config)
        self.engine = model_engine
        self.dist_data_manager = get_dist_data_manager()
        # Cache for old_experts to avoid communication overhead

    def compute_log_prob(self, data: DataProto, reuse_old_experts: bool = False):
        select_keys = ['responses', 'input_ids', 'attention_mask']
        image_keys = get_image_keys(data.non_tensor_batch)
        non_tensor_keys = image_keys
        selected_data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_keys)

        mtp_n_heads = getattr(self.engine.model_config, 'mtp_n_heads', 1)

        entropy_lst = []
        log_prob_lst = []
        acceptance_matrix_lst = [[] for _ in range(mtp_n_heads - 1)]

        if reuse_old_experts:
            self.old_experts = self.get_old_experts(data)
            selected_data.batch['old_experts'] = self.old_experts
        # Note: mismatched data order (here vs. upldate policy) can lead to
        # mismatched log probs. In order to match them, we need to split
        # batch into mini batches (same with training).
        if self.config.infer_num_mini_batch < 0:
            chunk_size = math.ceil(selected_data.batch.batch_size[0] / self.config.ppo_mini_batch_size)
        else:
            chunk_size = self.config.infer_num_mini_batch
        for chunk_idx, mini_batch in enumerate(selected_data.chunk(chunk_size)):
            output_proto = self.engine.forward_backward_step(data=mini_batch, forward_only=True)
            if isinstance(self.engine.model_module, FSDP):
                self.engine.model_module._handle.reshard(True)  # release memory
            log_prob_lst.append(output_proto.batch['logprobs'])
            entropy_lst.append(output_proto.batch['entropy'])

            for j in range(mtp_n_heads - 1):
                if output_proto.batch.get(f"acceptance_matrix_{j}", None) is not None:
                    acceptance_matrix_lst[j].append(output_proto.batch[f'acceptance_matrix_{j}'])

        log_probs = torch.concat(log_prob_lst, dim=0)
        entropy = torch.concat(entropy_lst, dim=0)

        if len(acceptance_matrix_lst) and len(acceptance_matrix_lst[0]):
            acceptance_matrix = [torch.concat(acceptance_matrix_lst[j], dim=0) for j in range(mtp_n_heads - 1)]
        else:
            acceptance_matrix = []

        role = data.meta_info['role']
        metrics = {
            f'{role}/#infer_micro_batch': output_proto.meta_info["metrics"].pop('#micro_batch_update'),
        }
        # Don't return selected_experts since they are already cached
        return entropy, log_probs, tuple(acceptance_matrix), metrics

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

    def get_old_experts(self, data: DataProto):
        refs = data.non_tensor_batch['old_experts_ref'].tolist()
        old_experts_ref = ray.get(self.dist_data_manager.get_refs.remote(refs))
        return torch.cat(ray.get(old_experts_ref), dim=0)

    def update_policy(self, data: DataProto):
        config_dict = OmegaConf.to_container(self.config, resolve=True)

        if self.config.reuse_old_experts:
            data.batch['old_experts'] = self.old_experts
            self.old_experts = None
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

            # Retrieve cached old_experts if reuse_old_experts is enabled
            if self.config.get('reuse_old_experts', False):
                device = mini_batch.batch['responses'].device  # Use responses tensor to get device
                mini_batch.batch['old_experts'] = mini_batch.batch['old_experts'].to(device)

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
        if self.config.reuse_old_experts:
            data.batch.pop('old_experts')

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

    if loss_average_method in ['token', 'sample', 'constant', 'direct_mean']:
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
    if config.get("force_append_eos", False):
        response_mask[:, -1] = 0  # mask force_append_eos

    use_rollout_behavior_log_probs = config['use_rollout_behavior_log_probs']
    use_old_as_ema = config['use_old_as_ema']
    use_ewma_loss = config['use_ewma_loss']
    ref_log_prob = micro_data.get("ref_log_prob", None)
    if use_rollout_behavior_log_probs:
        old_log_prob = micro_data['rollout_behavior_log_probs']
    else:
        old_log_prob = micro_data["old_log_probs"]
    if use_ewma_loss:
        assert ref_log_prob is not None, "ref_log_probs is needed if ewma enabled"
    if use_old_as_ema:
        ref_log_prob = micro_data['old_log_probs']

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
    use_experimental = config.get('experimental_algorithm', False)
    clip_mode = config.get("clip_mode", "token")
    dynamic_clip = config.get("dynamic_clip", False)

    compute_loss_fn = core_algos.compute_policy_loss_experimental if use_experimental else core_algos.compute_policy_loss

    total_loss, pg_loss, upgo_loss, pg_clipfrac, pg_clipfrac_hi, pg_clipfrac_lo, pg_clipfrac2, ppo_kl, ppo_kl_sum = (
        compute_loss_fn(
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
            clip_mode=clip_mode,
            dynamic_clip=dynamic_clip,
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
    # Only include training_uid for proper routing in update_policy
    non_tensor_keys = get_image_keys(data.non_tensor_batch)
    if 'training_uid' in data.non_tensor_batch:
        non_tensor_keys = non_tensor_keys + ['training_uid']
    if non_tensor_keys:
        assert return_dataproto
    if return_dataproto:
        mini_steps = data.batch.batch_size[0] // ppo_mini_batch_size
        dataloader = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_keys).chunk(mini_steps)
    else:
        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(ppo_mini_batch_size)
    return dataloader
