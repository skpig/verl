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
from typing import Iterable

import torch
from tensordict import TensorDict
from torch import nn
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
from alpha_seed.workers.hybrid_engine.fsdp_ulysses import ulysses_pad_and_slice_inputs

from alpha_seed import core_algos

# Note that this one doesn't change the original order
from .utils import rearrange_micro_batches
# Note that this one should only used for training because it changes the original order
from alpha_seed.utils.seqlen_balance import rearrange_micro_batches as rearrange_micro_batches_train

from contextlib import nullcontext

__all__ = ['DataParallelPPOActor']

try:
    from verl.utils.debug import get_profiler_context, MemoryProfiler
except:
    print('Cannot find profile utilities. Please use latest verl master')
    raise


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_rmpad = self.config.get('use_rmpad', False)
        if torch.distributed.get_rank() == 0:
            print(f'Actor use_rmpad={self.use_rmpad}')

        if hasattr(self.config, 'profile'):
            # refernce doesn't need debug
            self.profiler_context = get_profiler_context(filename=self.config.profile.filename,
                                                         profile_on_ranks=self.config.profile.profile_on_ranks,
                                                         default_hdfs_dir=self.config.profile.default_hdfs_dir,
                                                         upload_to_mlx=self.config.profile.upload_to_mlx,
                                                         enable=self.config.profile.enable,
                                                         wait=10)
            self.memory_profiler = MemoryProfiler(filename=self.config.profile.filename + 'memory',
                                                  enable=torch.distributed.get_rank() == 0 and
                                                  self.config.profile.enable,
                                                  upload_to_mlx=self.config.profile.upload_to_mlx,
                                                  wait=10)

        self.compute_entropy_loss = torch.compile(core_algos.compute_entropy_loss, dynamic=True)
        self.entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature):
        from flash_attn.bert_padding import index_first_axis, rearrange

        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.use_rmpad:
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
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, sp_size)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
                # forward
                output = self.actor_module(input_ids=input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False)

                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)

                batch_size, seqlen = input_ids.shape

                # TODO: we should carefully determine whether to turn on inplace_backward
                full_log_probs_rmpad = -cross_entropy_loss(logits_rmpad, input_ids_rmpad_rolled,
                                                           inplace_backward=False)[0]  # (total_nnz,)
                if sp_size > 1:
                    full_log_probs_rmpad = gather_outputs(full_log_probs_rmpad,
                                                          gather_dim=0,
                                                          padding_dim=0,
                                                          unpad_dim_size=total_nnz)
                full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1),
                                        indices=indices,
                                        batch=batch_size,
                                        seqlen=seqlen)
                log_probs = full_output.squeeze(-1)[:, -response_length - 1:-1]  # [batch_size, response_length]

                # compute entropy loss
                full_response_mask = attention_mask.clone()
                full_response_mask[:, :-response_length] = 0  # set the prompt part to zero
                full_response_mask_rmpad = index_first_axis(
                    rearrange(full_response_mask.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices).squeeze(-1)  # (total_nnz,)

                if sp_size > 1:
                    entropy = self.entropy_from_logits(logits_rmpad)
                    entropy = gather_outputs(entropy, gather_dim=0, padding_dim=0, unpad_dim_size=total_nnz)
                    entropy_loss = verl_F.masked_mean(entropy, mask=full_response_mask_rmpad)
                else:
                    entropy_loss = self.compute_entropy_loss(logits_rmpad, full_response_mask_rmpad)  # (total_nnz,)

            else:
                assert NotImplementedError('Only support rmpad mode')

                sp_size = get_ulysses_sequence_parallel_world_size()
                if sp_size > 1:
                    raise NotImplementedError("ulysses sequence parallelism w/o use_rmpad is not supported yet")
                output = self.actor_module(input_ids=micro_batch['input_ids'],
                                           attention_mask=micro_batch['attention_mask'],
                                           position_ids=micro_batch['position_ids'],
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy_loss = self.compute_entropy_loss(logits, attention_mask[:, -response_length:])

            return entropy_loss, log_probs

    def _make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        select_keys = [
            'responses', 'input_ids', 'attention_mask', 'old_log_probs', 'ref_log_prob', 'advantages', 'upgo_advantages'
        ]
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs,
                                  dataloader_kwargs={'shuffle': self.config.shuffle})

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> DataProto:
        # set to eval
        self.actor_module.eval()

        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len']
        else:
            micro_batch_size = data.meta_info['micro_batch_size']
            micro_batch_size = 1
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask']
        batch = data.select(batch_keys=select_keys).batch
        if use_dynamic_bsz:
            (micro_batches, num_micro_batches) = rearrange_micro_batches(batch=data.batch, max_token_len=max_token_len)
        else:
            # split batch into micro_batches
            micro_batches = batch.split(micro_batch_size)
            num_micro_batches = len(micro_batches)
        log_probs_lst = []
        for i, micro_batch in enumerate(micro_batches):
            with torch.inference_mode():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            if i < num_micro_batches:
                log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

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
        dataloader = self._make_minibatch_iterator(data=data)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            with self.profiler_context as p:
                if self.config.use_dynamic_bsz:
                    (micro_batches,
                     num_micro_batches) = rearrange_micro_batches_train(batch=data.batch,
                                                                        max_token_len=self.config.ppo_max_token_len)
                else:
                    # split batch into micro_batches
                    micro_batches = data.batch.split(self.config.ppo_micro_batch_size)
                self.actor_optimizer.zero_grad()

                for i, micro_data in enumerate(micro_batches):
                    micro_data = micro_data.cuda()  # actor device is cpu when using offload
                    responses = micro_data['responses']
                    response_length = responses.size(1)
                    attention_mask = micro_data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = micro_data['old_log_probs']
                    ref_log_prob = micro_data['ref_log_prob']
                    advantages = micro_data['advantages']
                    upgo_advantages = micro_data['upgo_advantages']

                    clip_ratio = self.config.clip_ratio
                    clip_ratio2 = self.config.clip_ratio2
                    scale_pg_by_kl = self.config.scale_pg_by_kl
                    entropy_coeff = self.config.entropy_coeff
                    upgo_loss_weight = self.config.upgo_loss_weight
                    kl_loss_weight = self.config.kl_loss_weight
                    kl_penalty = self.config.kl_penalty

                    entropy_loss, log_prob = self._forward_micro_batch(micro_batch=micro_data, temperature=temperature)

                    total_loss, pg_loss, upgo_loss, pg_clipfrac, pg_clipfrac2, ppo_kl, ppo_kl_sum = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob,
                        ref_log_prob=ref_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        upgo_advantages=upgo_advantages,
                        eos_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange2=clip_ratio2,
                        scale_pg_by_kl=scale_pg_by_kl,
                        upgo_loss_weight=upgo_loss_weight)

                    if kl_loss_weight > 0.0:
                        kl_loss = core_algos.compute_kl_loss(log_prob, ref_log_prob, response_mask, kl_penalty)
                    else:
                        kl_loss = torch.zeros(()).to(pg_loss.device)
                    policy_loss = total_loss - entropy_loss * entropy_coeff + kl_loss_weight * kl_loss

                    if self.config.use_dynamic_bsz:
                        if i >= num_micro_batches:
                            # fake data
                            loss = policy_loss * 0.0
                        else:
                            loss = policy_loss * (len(micro_data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_data_metric = {
                        'actor/entropy': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/upgo_loss': upgo_loss.detach().item(),
                        'actor/kl_loss': kl_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/pg_clipfrac2': pg_clipfrac2.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                        'actor/ppo_kl_sum': ppo_kl_sum.detach().item(),
                        'actor/tokens_per_micro_batch_update': attention_mask.sum().detach().item(),
                    }
                    append_to_dict(metrics, micro_data_metric)

                grad_norm = self._optimizer_step()
                data_metric = {
                    'actor/grad_norm': grad_norm.detach().item(),
                    'actor/#micro_batch_update': len(micro_batches)
                }
                append_to_dict(metrics, data_metric)

                p.step()
                self.memory_profiler.step()

        self.actor_optimizer.zero_grad()
        return metrics
