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
from typing import Iterable, ContextManager
import itertools
import math

import gc
import torch
from tensordict import TensorDict
from transformers import PretrainedConfig

from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

from verl import DataProto
from verl.trainer.ppo.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, log_probs_from_logits_response_rmpad, get_unpad_data
import verl.utils.torch_functional as verl_F

from verl.utils.model import compute_position_id_with_mask

from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import gather_outputs

from alpha_seed.utils.functional import rearrange_micro_data_proto
from alpha_seed.workers.hybrid_engine.fsdp_gather import ulysses_pad_and_slice_inputs, ulysses_pad
from alpha_seed.models.transformers.modeling_vlm import get_image_inputs, get_image_keys
from alpha_seed.utils.observility.training_stats import sync_training_stats
from alpha_seed.utils.observility import get_profiler_context_wrapped, profile_step
from alpha_seed import core_algos
from alpha_seed.models.transformers.monkey_patch import update_gate_ema
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx

from alpha_seed.workers.fsdp.offload import offload_fsdp_optimizer, load_fsdp_optimizer

from contextlib import nullcontext
import ray

__all__ = ['DataParallelPPOActor']

try:
    from verl.utils.debug import MemoryProfiler
except:
    print('Cannot find profile utilities. Please use latest verl master')
    raise


class DataParallelPPOActor(BasePPOActor):

    def __init__(
            self,
            config,
            actor_module: nn.Module,
            actor_optimizer: torch.optim.Optimizer = None,
            actor_model_config: PretrainedConfig = None,
            enable_non_reentrant_recompute: bool = False,
            metrics_context: ContextManager = nullcontext(),
    ):
        """When optimizer is None, it is Reference Policy.

        Note that the class is instantiated more than once in PPO training, whose config is
        created with a dedicated struct instead of using users' config directly.
        """
        super().__init__(config)
        self.actor_module: FSDP = actor_module
        self.actor_optimizer = actor_optimizer
        self.actor_model_config = actor_model_config
        self.metrics_context = metrics_context
        self.enable_non_reentrant_recompute = enable_non_reentrant_recompute
        self.use_rmpad = self.config.get('use_rmpad', False)
        if torch.distributed.get_rank() == 0:
            print(f'Actor use_rmpad={self.use_rmpad}')
        self.use_ce_loss_fusion = config.get('use_ce_loss_fusion', False)

        if hasattr(self.config, 'profile'):
            # refernce doesn't need debug
            self.profiler_context = get_profiler_context_wrapped(filename=self.config.profile.filename,
                                                                 profile_on_ranks=self.config.profile.profile_on_ranks,
                                                                 upload_to_mlx=self.config.profile.upload_to_mlx,
                                                                 enable=self.config.profile.enable,
                                                                 warmup=self.config.profile.warmup,
                                                                 wait=self.config.profile.wait,
                                                                 active=self.config.profile.active)

            self.memory_profiler = MemoryProfiler(filename=self.config.profile.filename + 'memory',
                                                  enable=torch.distributed.get_rank() == 0 and
                                                  self.config.profile.mem_enable,
                                                  upload_to_mlx=self.config.profile.upload_to_mlx,
                                                  wait=10)

        self.compute_entropy_loss = core_algos.compute_entropy_loss
        self.entropy_from_logits = verl_F.entropy_from_logits
        self.loss_fn = default_loss_fn

    def _forward_micro_batch(self, micro_batch: TensorDict, temperature, compute_entropy, non_tensor_batch):
        from flash_attn.bert_padding import index_first_axis, rearrange

        response_length = micro_batch['responses'].size(-1)
        image_kwargs = get_image_inputs(non_tensor_batch)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if not self.use_rmpad:
                raise NotImplementedError('only support rmpad mode')

            # TODO(zhangchi.usc1992): we can actually remove padding for the whole batch and perform balancing to reduce peak memory
            input_ids = micro_batch['input_ids'].to(torch.int64)
            attention_mask = micro_batch['attention_mask'].to(torch.int64)
            position_ids = compute_position_id_with_mask(attention_mask)
            input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                input_ids.unsqueeze(-1), attention_mask=attention_mask)  # (totol_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                  indices).transpose(0, 1)

            # handle ulysses sequence parallelism
            sp_size = get_ulysses_sequence_parallel_world_size()
            total_nnz = input_ids_rmpad.size(1)
            if 'vision_config' in self.actor_model_config:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(input_ids_rmpad, position_ids_rmpad,
                                                                            sp_size)
            else:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size)
            input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, sp_size)
            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            batch_size, seqlen = input_ids.shape
            seqlen_rmpad = input_ids_rmpad.size(1)

            # forward
            if self.use_ce_loss_fusion and not compute_entropy:
                # forward with lm_head CE fusion
                kwargs = {
                    'input_ids': input_ids_rmpad,
                    'position_ids': position_ids_rmpad,
                    'labels': input_ids_rmpad_rolled,
                    'temperature': temperature,
                    'fuse_lm_head_ce_loss': True,
                }

                # merge two dict
                kwargs.update(image_kwargs)
                output = self.actor_module(
                    **kwargs,
                    use_cache=False,
                    output_hidden_states=False,
                )
                full_log_probs_rmpad = output.loss * (-1.0)
            else:
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           position_ids=position_ids_rmpad,
                                           labels=input_ids_rmpad_rolled,
                                           use_cache=False,
                                           compute_entropy=compute_entropy,
                                           **image_kwargs)
                full_entropy_rmpad, full_log_probs_rmpad = output.entropy, output.log_probs

            if sp_size > 1:
                full_log_probs_rmpad = gather_outputs(full_log_probs_rmpad,
                                                      gather_dim=0,
                                                      padding_dim=0,
                                                      unpad_dim_size=total_nnz)
            full_log_probs = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1),
                                       indices=indices,
                                       batch=batch_size,
                                       seqlen=seqlen)
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # [batch_size, response_length]
            if compute_entropy:
                if sp_size > 1:
                    full_entropy_rmpad = gather_outputs(full_entropy_rmpad,
                                                        gather_dim=0,
                                                        padding_dim=0,
                                                        unpad_dim_size=total_nnz)  # (total_nnz,)
                # pad it back
                entropy = pad_input(hidden_states=full_entropy_rmpad.unsqueeze(-1),
                                    indices=indices,
                                    batch=batch_size,
                                    seqlen=seqlen).squeeze(-1)[:,
                                                               -response_length - 1:-1]  # (batch_size, response_length)
            else:
                entropy = None
            return entropy, log_probs, seqlen_rmpad

    def _make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        select_keys = ['responses', 'input_ids', 'attention_mask', 'old_log_probs', 'advantages', 'upgo_advantages']
        for opt_key in ['ref_log_prob', 'model_output_mask']:
            if opt_key in data.batch.keys():
                select_keys.append(opt_key)
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs,
                                  dataloader_kwargs={'shuffle': self.config.shuffle})

    def _optimizer_step(self):

        # release kv mirror memory for m8
        if hasattr(self.actor_module, 'release_act_memory'):
            self.actor_module.release_act_memory()
        assert self.config.grad_clip is not None
        grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def _optimizer_zero_grad(self):
        # NOTE(zhiqi.0): when use_orig_params=True, parameters in optimizer are nn.Parameter instead of
        # FlatParam. The param.grad is a view of FlatParam.grad. Therefore, optimizer.zero_grad()
        # only removes tensor views of gradients, but cannot remove the FlatParam.grad.
        self.actor_optimizer.zero_grad()
        if isinstance(self.actor_module, FSDP):
            for module in FSDP.fsdp_modules(self.actor_module):
                module._flat_param.grad = None

    def compute_log_prob(self, data: DataProto) -> DataProto:
        # set to eval
        self.actor_module.eval()

        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len']
        else:
            micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask']
        image_keys = get_image_keys(data.non_tensor_batch)
        selected_data = data.select(batch_keys=select_keys, non_tensor_batch_keys=image_keys)
        entropy_lst = []
        log_prob_lst = []
        # Note: mismatched data order (here vs. upldate policy) can lead to
        # mismatched log probs. In order to match them, we need to split
        # batch into mini batches (same with training).
        chunk_size = math.ceil(selected_data.batch.batch_size[0] / self.config.ppo_mini_batch_size)
        for batch_idx, mini_batch in enumerate(selected_data.chunk(chunk_size)):
            if use_dynamic_bsz:
                indices, micro_batches, non_tensor_batches = rearrange_micro_data_proto(max_token_len, mini_batch)
            else:
                micro_batches = mini_batch.chunk(math.ceil(mini_batch.batch.batch_size[0] / micro_batch_size))
                num_micro_batches = len(micro_batches)

            mini_batch_entropy = []
            mini_batch_log_prob = []
            with torch.no_grad():
                for i, micro_batch_proto in enumerate(micro_batches):
                    if use_dynamic_bsz:
                        non_tensor_batch = non_tensor_batches[i]
                        assert micro_batch_proto.device == torch.device('cpu')
                        micro_batch = micro_batch_proto.cuda()
                    else:
                        non_tensor_batch = micro_batch_proto.non_tensor_batch
                        assert micro_batch_proto.batch.device == torch.device('cpu')
                        micro_batch = micro_batch_proto.batch.cuda()
                    entropy, log_probs, _ = self._forward_micro_batch(micro_batch=micro_batch,
                                                                      temperature=temperature,
                                                                      compute_entropy=True,
                                                                      non_tensor_batch=non_tensor_batch)
                    mini_batch_log_prob.append(log_probs)
                    mini_batch_entropy.append(entropy)
            # release root module unshard memory
            if isinstance(self.actor_module, FSDP):
                self.actor_module._handle.reshard(True)

            mini_log_prob = torch.cat(mini_batch_log_prob, dim=0)
            mini_entropy = torch.cat(mini_batch_entropy, dim=0)
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == mini_entropy.size(
                    0), f"{len(indices)} vs. {mini_entropy.size()} vs. {mini_log_prob.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                mini_log_prob = mini_log_prob[revert_indices]
                mini_entropy = mini_entropy[revert_indices]

            log_prob_lst.append(mini_log_prob)
            entropy_lst.append(mini_entropy)

        log_probs = torch.concat(log_prob_lst, dim=0)
        entropy = torch.concat(entropy_lst, dim=0)
        return entropy, log_probs

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def train_one_step(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        global_step = data.meta_info.get('global_step')
        batch = data.batch

        # todo: put it into batch.meta_info
        entropy_coeff = self.config.entropy_coeff
        if entropy_coeff <= 0.:
            compute_entropy = False
        else:
            compute_entropy = True

        # only use nullcontext for non-reentrant gradient checkpointing
        metrics_exec_context = nullcontext() if self.enable_non_reentrant_recompute else self.metrics_context
        metrics = {}

        with self.profiler_context as p, metrics_exec_context:
            if self.config.use_dynamic_bsz:
                micro_batches, _, _ = rearrange_micro_batches(batch=batch, max_token_len=self.config.ppo_max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = batch.split(self.config.ppo_micro_batch_size)
            self._optimizer_zero_grad()

            for i, micro_data in enumerate(micro_batches):
                assert micro_data.device == torch.device('cpu')
                micro_data = micro_data.cuda()  # actor device is cpu when using offload
                full_entropy, log_prob, seqlen = self._forward_micro_batch(micro_batch=micro_data,
                                                                           temperature=temperature,
                                                                           compute_entropy=compute_entropy)
                loss, micro_data_metric = self.loss_fn(self.config, micro_data, full_entropy, log_prob)

                if self.config.use_dynamic_bsz:
                    loss = loss * (len(micro_data) / self.config.ppo_mini_batch_size)
                else:
                    loss = loss / self.gradient_accumulation
                loss.backward()

                micro_data_metric['actor/seqlen'] = seqlen
                append_to_dict(metrics, micro_data_metric)

            grad_norm = self._optimizer_step()

            if self.config.get('update_gate_ema', False):
                # update gate_ema
                update_gate_ema(self.actor_module)

            data_metric = {
                'actor/grad_norm': grad_norm.detach().item(),
                'actor/#micro_batch_update': len(micro_batches),
            }
            append_to_dict(metrics, data_metric)

            profile_step(p, global_step)
            self.memory_profiler.step()
        return metrics

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        if not self.config.use_dynamic_bsz:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
            self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
            if self.gradient_accumulation > 2 and not isinstance(self.profiler_context, nullcontext):
                raise ValueError(
                    f'Number of {self.gradient_accumulation=} is too large when turn on profile. Try to turn off profile or reduce ppo_mini_batch_size.'
                )
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        global_step = data.meta_info.get('global_step')

        # compute batch full token count
        response_length = data.batch['responses'].size(1)
        batch_full_token_count_mask = data.batch['attention_mask'][:, -response_length:]
        if 'overlong_mask' in data.batch.keys():
            batch_full_token_count_mask *= data.batch['overlong_mask'].unsqueeze(-1)
        batch_full_token_count = max(1, batch_full_token_count_mask.sum().item())

        dataloader = make_mini_step_dataloader(data, self.config.ppo_mini_batch_size, return_dataproto=True)
        metrics = {}

        first_mini_ppo_kl_sum = 0
        # only use nullcontext for non-reentrant gradient checkpointing
        metrics_exec_context = nullcontext() if self.enable_non_reentrant_recompute else self.metrics_context

        for batch_idx, mini_batch in enumerate(dataloader):
            with self.profiler_context as p, metrics_exec_context:
                if self.config.use_dynamic_bsz:
                    indices, micro_batches, non_tensor_batches = rearrange_micro_data_proto(
                        self.config.ppo_max_token_len, mini_batch)
                else:
                    # split batch into micro_batches
                    micro_chunk_size = math.ceil(mini_batch.batch.batch_size[0] / self.config.ppo_micro_batch_size)
                    micro_batches = mini_batch.chunk(micro_chunk_size)
                self._optimizer_zero_grad()

                minibatch_early_stop = False

                # compute minibatch full token count
                mini_batch_full_token_count_mask = mini_batch.batch['attention_mask'][:, -response_length:]
                if 'overlong_mask' in mini_batch.batch.keys():
                    mini_batch_full_token_count_mask *= mini_batch.batch['overlong_mask'].unsqueeze(-1)
                mini_batch_full_token_count = max(1, mini_batch_full_token_count_mask.sum().item())

                for i, micro_batch in enumerate(micro_batches):
                    if self.config.use_dynamic_bsz:
                        micro_data = micro_batch
                        non_tensor_batch = non_tensor_batches[i]
                    else:
                        micro_data = micro_batch.batch
                        non_tensor_batch = micro_batch.non_tensor_batch
                    assert micro_data.device == torch.device('cpu')
                    micro_data = micro_data.cuda()  # actor device is cpu when using offload
                    entropy_coeff = self.config.entropy_coeff
                    if entropy_coeff <= 0.:
                        compute_entropy = False
                    else:
                        compute_entropy = True

                    full_entropy, log_prob, seqlen = self._forward_micro_batch(micro_batch=micro_data,
                                                                               temperature=temperature,
                                                                               compute_entropy=compute_entropy,
                                                                               non_tensor_batch=non_tensor_batch)
                    policy_loss, micro_data_metric = self.loss_fn(self.config, micro_data, full_entropy, log_prob)

                    if self.config.early_stop_by_kl != 0 and micro_data_metric[
                            'actor/ppo_kl'] > self.config.early_stop_by_kl and batch_idx > 0:
                        minibatch_early_stop = True
                        break

                    if self.config.loss_average_method in ['token', 'sample', 'constant']:
                        if self.config.use_dynamic_bsz:
                            loss = policy_loss * (len(micro_data) / self.config.ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                    elif self.config.loss_average_method == 'minibatch':
                        loss = policy_loss / mini_batch_full_token_count
                    elif self.config.loss_average_method == 'batch':
                        loss = policy_loss / batch_full_token_count * len(dataloader)
                    else:
                        raise NotImplementedError(f'loss_average_method {loss_average_method} not implemented.')
                    loss.backward()

                    micro_data_metric['actor/seqlen'] = seqlen

                    if batch_idx == 0:
                        first_mini_ppo_kl_sum += micro_data_metric['actor/ppo_kl_sum']

                    append_to_dict(metrics, micro_data_metric)
                    if self.config.gc_freq == "micro":
                        gc.collect()

                if minibatch_early_stop:
                    print(f'early stop at {batch_idx}!!!')
                    break

                if not isinstance(self.metrics_context, nullcontext):
                    training_stats = sync_training_stats(self.metrics_context, self.actor_model_config,
                                                         self.actor_module, self.config.fsdp_size)
                    append_to_dict(metrics, training_stats)

                grad_norm = self._optimizer_step()

                if self.config.gc_freq == "mini":
                    gc.collect()

                if self.config.get('update_gate_ema', False):
                    # update gate_ema
                    update_gate_ema(self.actor_module)

                data_metric = {
                    'actor/grad_norm': grad_norm.detach().item(),
                    'actor/#micro_batch_update': len(micro_batches),
                }
                append_to_dict(metrics, data_metric)

                profile_step(p, global_step)
                self.memory_profiler.step()

        append_to_dict(metrics, {'first_mini_ppo_kl_sum': first_mini_ppo_kl_sum})

        self._optimizer_zero_grad()
        return metrics


def default_loss_fn(config, micro_data, full_entropy, log_prob):
    responses = micro_data['responses']
    response_length = responses.size(1)
    attention_mask = micro_data['attention_mask']
    if config.use_model_output_mask:
        loss_mask = micro_data['model_output_mask']
        response_mask = loss_mask[:, -response_length:]
    else:
        response_mask = attention_mask[:, -response_length:]

    use_rollout_log_probs = config.get("use_rollout_log_probs", False)
    if use_rollout_log_probs:
        # use ewma if use_rollout_log_probs: importance sampling by rollout_logprob, clip by old_log_prob
        use_ewma_loss = True
        old_log_prob = micro_data['rollout_log_probs']
        ref_log_prob = micro_data['old_log_probs']
    else:
        use_ewma_loss = config.use_ewma_loss
        old_log_prob = micro_data['old_log_probs']
        ref_log_prob = micro_data.get('ref_log_prob', None)

    advantages = micro_data['advantages']
    upgo_advantages = micro_data['upgo_advantages']
    overlong_mask = micro_data.get('overlong_mask', None)

    clip_ratio = config.clip_ratio
    clip_ratio_low = clip_ratio
    clip_ratio_high = clip_ratio
    if config.clip_ratio_low:
        clip_ratio_low = config.clip_ratio_low
    if config.clip_ratio_high:
        clip_ratio_high = config.clip_ratio_high
    clip_ratio2 = config.clip_ratio2
    scale_pg_by_kl = config.scale_pg_by_kl
    scale_pg_by_local_kl = config.scale_pg_by_local_kl
    entropy_coeff = config.entropy_coeff
    upgo_loss_weight = config.upgo_loss_weight
    kl_loss_weight = config.kl_loss_weight
    offpolicy_kl_loss_weight = config.offpolicy_kl_loss_weight
    lm_loss_weight = config.lm_loss_weight
    kl_penalty_type = config.kl_penalty
    loss_average_method = config.loss_average_method
    loss_average_constant = config.loss_average_constant

    total_loss, pg_loss, upgo_loss, pg_clipfrac, pg_clipfrac_hi, pg_clipfrac_lo, pg_clipfrac2, ppo_kl, ppo_kl_sum = core_algos.compute_policy_loss(
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
        loss_average_constant=loss_average_constant)

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
        eos_ids = micro_data['eos_ids']
        raw_scores = micro_data['token_level_scores']
        lm_loss = core_algos.compute_lm_loss(log_prob, raw_scores, eos_ids, loss_average_method=loss_average_method)
    else:
        lm_loss = torch.zeros((), device=pg_loss.device)

    if entropy_coeff <= 0.:
        compute_entropy = False
    else:
        compute_entropy = True

    if compute_entropy:
        entropy_loss = verl_F.masked_mean(full_entropy, response_mask)
    else:
        entropy_loss = torch.zeros((), device=pg_loss.device)

    policy_loss = total_loss - entropy_loss * entropy_coeff + kl_loss_weight * kl_loss + lm_loss_weight * lm_loss + offpolicy_kl_loss_weight * offpolicy_kl_loss

    metrics = {
        # 'actor/entropy': entropy_loss.detach().item(),
        'actor/pg_loss': pg_loss.detach().item(),
        'actor/upgo_loss': upgo_loss.detach().item(),
        'actor/kl_loss': kl_loss.detach().item(),
        'actor/offpolicy_kl_loss': offpolicy_kl_loss.detach().item(),
        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
        'actor/pg_clipfrac_hi': pg_clipfrac_hi.detach().item(),
        'actor/pg_clipfrac_lo': pg_clipfrac_lo.detach().item(),
        'actor/pg_clipfrac2': pg_clipfrac2.detach().item(),
        'actor/ppo_kl': ppo_kl.detach().item(),
        'actor/ppo_kl_sum': ppo_kl_sum.detach().item(),
        'actor/tokens_per_micro_batch_update': attention_mask.sum().detach().item(),
        'actor/lm_loss': lm_loss.detach().item(),
    }
    return policy_loss, metrics


def make_mini_step_dataloader(data, ppo_mini_batch_size, return_dataproto=False):
    # make minibatch iterator
    # dataloader = self._make_minibatch_iterator(data=data)
    select_keys = [
        'responses', 'input_ids', 'attention_mask', 'old_log_probs', 'advantages', 'upgo_advantages', 'off_policy_steps'
    ]
    for opt_key in [
            'ref_log_prob', 'rollout_log_probs', 'overlong_mask', 'eos_ids', 'token_level_scores', 'model_output_mask'
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
