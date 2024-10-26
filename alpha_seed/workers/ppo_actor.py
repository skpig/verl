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

from verl import DataProto
from verl.trainer.ppo.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, log_probs_from_logits_response_rmpad, get_unpad_data
from flash_attn.bert_padding import pad_input, unpad_input
from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import slice_input_tensor, gather_outputs

from alpha_seed import core_algos

__all__ = ['DataParallelPPOActor']


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

    def _forward_micro_batch(self, micro_batch, temperature):
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.use_rmpad:
                input_ids = micro_batch['input_ids']
                attention_mask = micro_batch['attention_mask']
                position_ids = micro_batch['position_ids']
                input_ids_rmpad = unpad_input(input_ids.unsqueeze(-1),
                                              attention_mask=attention_mask)[0]  # (totol_nnz, 1)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # handle ulysses sequence parallelism
                if (sp_size := get_ulysses_sequence_parallel_world_size()) > 1:
                    _, total_s = input_ids_rmpad.shape
                    pad_size = (sp_size - total_s % sp_size) % sp_size
                    if pad_size > 0:
                        # append a placeholder sequence
                        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=0)
                        attention_mask = torch.nn.functional.pad(attention_mask, (0, 0, 0, 1), value=0)
                        attention_mask[-1, :pad_size] = 1
                    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)

                # Note that in rmpad implementation, we don't need position_ids.
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=attention_mask,
                                           position_ids=None,
                                           use_cache=False)

                # handle ulysses sequence parallelism
                if get_ulysses_sequence_parallel_world_size() > 1:
                    if pad_size > 0:
                        # remove the trailing placeholder sequence
                        attention_mask = attention_mask[:-1]
                    output.logits = gather_outputs(output.logits, gather_dim=1, padding_dim=1, unpad_dim_size=total_s)

                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad = logits_rmpad / temperature
                log_probs = log_probs_from_logits_response_rmpad(input_ids=input_ids,
                                                                 attention_mask=attention_mask,
                                                                 logits_rmpad=logits_rmpad,
                                                                 response_length=response_length)
                logits = logits_rmpad
            else:
                output = self.actor_module(input_ids=micro_batch['input_ids'],
                                           attention_mask=micro_batch['attention_mask'],
                                           position_ids=micro_batch['position_ids'],
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits / temperature
                logits = logits[:, -response_length - 1:-1]
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
            return logits, log_probs

    def _make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
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

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        micro_batches = batch.split(micro_batch_size)
        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        dataloader = self._make_minibatch_iterator(data=data)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            micro_batches = data.batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                clip_ratio2 = self.config.clip_ratio2
                entropy_coeff = self.config.entropy_coeff

                logits, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, pg_clipfrac2, ppo_kl, ppo_kl_sum = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    eos_mask=response_mask,
                    cliprange=clip_ratio,
                    cliprange2=clip_ratio2)

                if self.use_rmpad:
                    full_response_mask = attention_mask.clone()
                    full_response_mask[:, :-response_length] = 0  # set the prompt part to zero
                    full_response_mask_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                        full_response_mask.unsqueeze(-1), attention_mask=attention_mask)
                    full_response_mask_rmpad = full_response_mask_rmpad.squeeze(-1)  # (total_nnz)
                    entropy_loss = core_algos.compute_entropy_loss(logits, full_response_mask_rmpad)  # (total_nnz,)
                else:
                    entropy_loss = core_algos.compute_entropy_loss(logits, response_mask)
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/pg_clipfrac2': pg_clipfrac2.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                    'actor/ppo_kl_sum': ppo_kl_sum.detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
