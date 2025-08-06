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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import math

from mono_rl import DataProto
from verl.workers.critic import BasePPOCritic
from mono_rl.models.seed_models.modeling_vlm import get_image_keys
import verl.utils.torch_functional as verl_F
from alpha_seed.core_algos import compute_value_loss
from omegaconf import OmegaConf, DictConfig
from mono_rl.worker.engine.fsdp.models.model import FSDPModel
from alpha_seed.utils.functional import append_dict_items_to_dict

__all__ = ['DataParallelPPOCritic']


class DataParallelPPOCritic(BasePPOCritic):

    def __init__(self, as_config: DictConfig, model_engine: FSDPModel):
        super().__init__(as_config)
        self.engine = model_engine

    def _make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        select_keys = ['input_ids', 'responses', 'attention_mask', 'values', 'returns']
        for opt_key in ['overlong_mask', 'model_output_mask']:
            if opt_key in data.batch:
                select_keys.append(opt_key)
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs if not data.meta_info.get('phasic_update', False) else
                                  self.config.phasic_critic_epochs,
                                  dataloader_kwargs={'shuffle': self.config.shuffle})

    def compute_values(self, data: DataProto) -> torch.Tensor:
        select_keys = ['responses', 'input_ids', 'attention_mask']
        image_keys = get_image_keys(data.non_tensor_batch)
        selected_data = data.select(batch_keys=select_keys, non_tensor_batch_keys=image_keys)
        values_lst = []
        # Note: mismatched data order (here vs. upldate critic) can lead to
        # mismatched values. In order to match them, we need to split
        # batch into mini batches (same with training).
        chunk_size = math.ceil(selected_data.batch.batch_size[0] / self.config.ppo_mini_batch_size)
        for _, mini_batch in enumerate(selected_data.chunk(chunk_size)):
            output_proto = self.engine.forward_backward_step(data=mini_batch, forward_only=True)
            if isinstance(self.engine.model_module, FSDP):
                self.engine.model_module._handle.reshard(True)  # release memory
            values_lst.append(output_proto.batch['values'])
        values = torch.concat(values_lst, dim=0)
        return values

    def update_critic(self, data: DataProto):

        self.engine.optimizer_zero_grad()
        config_dict = OmegaConf.to_container(self.config, resolve=True)

        response_length = data.batch['responses'].size(1)
        config_dict["response_length"] = response_length

        if self.config.shuffle:
            dataloader = self._make_minibatch_iterator(data)
            chunk_size = math.ceil(data.batch.batch_size[0] / self.config.ppo_mini_batch_size)
        else:
            dataloader = make_mini_step_dataloader(data, self.config.ppo_mini_batch_size, return_dataproto=True)
            chunk_size = len(dataloader)

        metrics = {}
        seq_level_vf_lst = []
        self.engine.set_loss(vf_loss_fn, config_dict)

        for batch_idx, mini_batch in enumerate(dataloader):
            # Set the loss function of update actor function for every mini batch
            output_proto = self.engine.forward_backward_step(data=mini_batch, forward_only=False)
            if batch_idx < chunk_size:
                seq_level_vf_lst.append(output_proto.batch['seq_vf'])

            metrics_opt = self.engine.optimizer_step()
            self.engine.optimizer_zero_grad()

            print(f"[debug][critic] {metrics_opt['grad_norm']=}")

            data_metric = {
                'critic/grad_norm': metrics_opt['grad_norm'],  # NOTE: grad_norm is a float from monorl
                'critic/#micro_batch_update': output_proto.meta_info["metrics"].pop('#micro_batch_update'),
                **output_proto.meta_info["metrics"],
            }
            append_dict_items_to_dict(metrics, data_metric)

        seq_vf = torch.cat(seq_level_vf_lst)
        self.engine.optimizer_zero_grad()

        return seq_vf, metrics


def vf_loss_fn(config, output, micro_data):
    """ 
    The loss function used for the critic model
    """
    vpreds = output["values"]
    values = micro_data["values"]
    returns = micro_data["returns"]
    response_length = config.get("response_length", 512)
    attention_mask = micro_data["attention_mask"]
    eos_mask = attention_mask[:, -response_length - 1:-1]
    overlong_mask = micro_data.get("overlong_mask", None)
    loss_average_method = config.get("critic_loss_average_method", "sample")

    cliprange_value_low = config.get("cliprange_value")
    cliprange_value_high = config.get("cliprange_value")
    if config.get("cliprange_value_low", None) is not None:
        cliprange_value_low = config.get("cliprange_value_low")
    if config.get("cliprange_value_high", None) is not None:
        cliprange_value_high = config.get("cliprange_value_high")

    vf_loss, vf_clipfrac, seq_vf = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        eos_mask=eos_mask,
        cliprange_value_low=cliprange_value_low,
        cliprange_value_high=cliprange_value_high,
        overlong_mask=overlong_mask,
        loss_average_method=loss_average_method,
    )

    if config.get("use_dynamic_bsz", False):
        vf_loss = vf_loss * (len(micro_data) / config["ppo_mini_batch_size"])
    else:
        gradient_accumulation = config["ppo_mini_batch_size"] // config["ppo_micro_batch_size"]
        vf_loss = vf_loss / gradient_accumulation

    # NOTE(hpguo) `seq_vf` tensor will be moved to DataProto.batch in monorl `forward_backward_step`
    metrics = {
        "critic/vf_loss": vf_loss.detach().item(),
        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
        "critic/vpred_mean": verl_F.masked_mean(vpreds, eos_mask).detach().item(),
        "critic/tokens_per_micro_batch_update": attention_mask.sum().detach().item(),
        "seq_vf": seq_vf,
    }
    return vf_loss, metrics


def make_mini_step_dataloader(data, ppo_mini_batch_size, return_dataproto=False):
    select_keys = ['input_ids', 'responses', 'attention_mask', 'values', 'returns']
    for opt_key in ['overlong_mask', 'model_output_mask']:
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
