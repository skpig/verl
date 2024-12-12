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
Implement a multiprocess PPOCritic
"""

from typing import Iterable

import torch
import torch.distributed
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo.critic import BasePPOCritic
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.model import compute_position_id_with_mask

from tensordict import TensorDict

from alpha_seed.workers.hybrid_engine.fsdp_ulysses import ulysses_pad_and_slice_inputs
from alpha_seed import core_algos

from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import gather_outputs
# Note that this one doesn't change the original order
from .utils import rearrange_micro_batches
# Note that this one should only used for training because it changes the original order
from alpha_seed.utils.seqlen_balance import rearrange_micro_batches as rearrange_micro_batches_train
from contextlib import nullcontext

__all__ = ['DataParallelPPOCritic']

try:
    from verl.utils.debug import get_profiler_context, MemoryProfiler
except:
    print('Cannot find profile utilities. Please use latest verl master')
    raise


class DataParallelPPOCritic(BasePPOCritic):

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer

        self.use_rmpad = self.config.get('use_rmpad', False)
        if torch.distributed.get_rank() == 0:
            print(f'Critic use_rmpad={self.use_rmpad}')

        if not self.config.use_dynamic_bsz:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0, f'{self.config.ppo_mini_batch_size=}, {self.config.ppo_micro_batch_size=}'
            self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size

        self.profiler_context = get_profiler_context(filename=self.config.profile.filename,
                                                     profile_on_ranks=self.config.profile.profile_on_ranks,
                                                     default_hdfs_dir=self.config.profile.default_hdfs_dir,
                                                     upload_to_mlx=self.config.profile.upload_to_mlx,
                                                     enable=self.config.profile.enable,
                                                     wait=10)
        self.memory_profiler = MemoryProfiler(filename=self.config.profile.filename + 'memory',
                                              enable=torch.distributed.get_rank() == 0 and self.config.profile.enable,
                                              upload_to_mlx=self.config.profile.upload_to_mlx,
                                              wait=10)

        self.value_loss = torch.compile(core_algos.compute_value_loss, disable=True)

    def _forward_micro_batch(self, micro_batch: TensorDict, response_length):
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.use_rmpad:
                input_ids = micro_batch['input_ids'].to(torch.int64)
                batch, seqlen = input_ids.shape
                attention_mask = micro_batch['attention_mask'].to(torch.int64)
                position_ids = compute_position_id_with_mask(attention_mask)
                input_ids_rmpad, indices, _, _ = unpad_input(input_ids.unsqueeze(-1),
                                                             attention_mask=attention_mask)  # (totol_nnz, 1)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # handle ulysses sequence parallelism
                sp_size = get_ulysses_sequence_parallel_world_size()
                total_s = input_ids_rmpad.size(1)
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size)
                # forward
                values_rmpad = self.critic_module(input_ids=input_ids_rmpad,
                                                  position_ids=position_ids_rmpad,
                                                  use_cache=False).logits  # (1, total_nnz / sp_size, 1)
                values_rmpad = values_rmpad.squeeze(0).squeeze(-1)  # (total_nnz / sp_size)
                # handle ulysses sequence parallelism
                if sp_size > 1:
                    values_rmpad = gather_outputs(values_rmpad, gather_dim=0, padding_dim=0, unpad_dim_size=total_s)
                # pad it back
                values = pad_input(values_rmpad.unsqueeze(-1), indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
            else:
                assert NotImplementedError('Only support rmpad mode')

                sp_size = get_ulysses_sequence_parallel_world_size()
                if sp_size > 1:
                    raise NotImplementedError("ulysses sequence parallelism w/o use_rmpad is not supported yet")
                output = self.critic_module(input_ids=micro_batch['input_ids'],
                                            attention_mask=micro_batch['attention_mask'],
                                            position_ids=micro_batch['position_ids'],
                                            use_cache=False)  # prevent model thinks we are generating
                values = output.logits
            values = values[:, -response_length - 1:-1]
            values = values.float()
            return values

    def _make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        select_keys = ['input_ids', 'attention_mask', 'values', 'returns']
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs if not data.meta_info.get('phasic_update', False) else
                                  self.config.phasic_critic_epochs,
                                  dataloader_kwargs={'shuffle': self.config.shuffle})

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        self.critic_optimizer.step()
        return grad_norm

    def compute_values(self, data: DataProto) -> torch.Tensor:
        response_length = data.meta_info['response_length']

        self.critic_module.eval()

        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len']
        else:
            micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['input_ids', 'attention_mask']
        batch = data.select(batch_keys=select_keys).batch
        if use_dynamic_bsz:
            (micro_batches, num_micro_batches) = rearrange_micro_batches(batch=data.batch, max_token_len=max_token_len)
        else:
            # split batch into micro_batches
            micro_batches = batch.split(micro_batch_size)
            num_micro_batches = len(micro_batches)
        values_lst = []
        for i, micro_batch in enumerate(micro_batches):
            assert micro_batch.device == torch.device('cpu')
            micro_batch = micro_batch.cuda()  # actor device is cpu when using offload
            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch, response_length=response_length)
            if i < num_micro_batches:
                values_lst.append(values)
        values = torch.concat(values_lst, dim=0)
        return values

    def update_critic(self, data: DataProto):
        response_length = data.meta_info['response_length']

        self.critic_module.train()

        metrics = {}

        dataloader = self._make_minibatch_iterator(data)

        if not self.config.use_dynamic_bsz:
            if self.gradient_accumulation > 2 and not isinstance(self.profiler_context, nullcontext):
                raise ValueError(
                    f'Number of {self.gradient_accumulation=} is too large when turn on profile. Try to turn off profile or reduce ppo_mini_batch_size.'
                )

        for batch_idx, data in enumerate(dataloader):
            with self.profiler_context as p:
                if self.config.use_dynamic_bsz:
                    (micro_batches,
                     num_micro_batches) = rearrange_micro_batches_train(batch=data.batch,
                                                                        max_token_len=self.config.ppo_max_token_len)
                else:
                    # split batch into micro_batches
                    micro_batches = data.batch.split(self.config.ppo_micro_batch_size)
                self.critic_optimizer.zero_grad()

                for i, micro_data in enumerate(micro_batches):
                    assert micro_data.device == torch.device('cpu')
                    micro_data = micro_data.cuda()  # critic device is cpu when using offload
                    attention_mask = micro_data['attention_mask']
                    values = micro_data['values']
                    returns = micro_data['returns']

                    eos_mask = attention_mask[:, -response_length - 1:-1]

                    vpreds = self._forward_micro_batch(micro_data, response_length=response_length)

                    # assert not torch.any(torch.isnan(vpreds)).item()

                    vf_loss, vf_clipfrac = self.value_loss(vpreds=vpreds,
                                                           values=values,
                                                           returns=returns,
                                                           eos_mask=eos_mask,
                                                           cliprange_value=self.config.cliprange_value)
                    if self.config.use_dynamic_bsz:
                        if i >= num_micro_batches:
                            # fake data
                            loss = vf_loss * 0.0
                        else:
                            loss = vf_loss * (len(micro_data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = vf_loss / self.gradient_accumulation
                    loss.backward()

                    micro_data_metric = {
                        'critic/vf_loss': vf_loss.detach().item(),
                        'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                        'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
                        'critic/tokens_per_micro_batch_update': attention_mask.sum().detach().item(),
                    }

                    append_to_dict(metrics, micro_data_metric)

                grad_norm = self._optimizer_step()
                data_metric = {
                    'critic/grad_norm': grad_norm.detach().item(),
                    'critic/#micro_batch_update': len(micro_batches)
                }
                append_to_dict(metrics, data_metric)

                p.step()
                self.memory_profiler.step()

        self.critic_optimizer.zero_grad()
        return metrics
