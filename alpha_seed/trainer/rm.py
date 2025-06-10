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

import os

import logging
import warnings
import functools
import re
from typing import Optional, Tuple
from alpha_seed.trainer.utils.lineage import report_checkpoint_saved, report_job_config, report_data_loaded, report_trial_started, safely_do
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from torch.utils.data import DataLoader, DistributedSampler
from codetiming import Timer
from omegaconf import OmegaConf
from torch.nn import functional as F
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

import hdfs_io
import seed_models
from seed_models.utils.count_flops import FlopsCounter
from pprint import pprint

from tensordict import TensorDict
import verl.utils.torch_functional as verl_F
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tracking import Tracking
from mono_rl import DataProto
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.utils.debug import log_gpu_memory_usage

from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch, get_parallel_plan
from alpha_seed.workers.fsdp.initialize import create_mesh, parallel_load_safetensors, parallel_init_fsdp_fn, meta_device_init
from alpha_seed.workers.fsdp.extensions import register_dtensor_save_hook, parallelize_module
from alpha_seed.workers.fsdp.clip_grad_norm import clip_grad_norm_
from alpha_seed.utils.observility.training_stats import all_reduce

from alpha_seed.utils.dataset.rm_dataset import RMDataset
from alpha_seed.utils.dataset.rl_dataset import collate_fn
from alpha_seed.workers.hybrid_engine.fsdp_gather import DataGatherManager, ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

from mono_rl.single_controller import Worker
from mono_rl.single_controller import register, Dispatch

from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.bert_padding import index_first_axis, rearrange
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

from dist_attn.ulysses.parallel_states import set_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_world_size

from omnistore import FSDPCheckpointer


class ReduceLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx: torch.autograd.Function, loss: torch.Tensor, num_valid_tokens: torch.Tensor) -> torch.Tensor:
        if num_valid_tokens == 0:
            loss = torch.nan_to_num(loss)

        group = get_ulysses_sequence_parallel_group()
        dist.all_reduce(loss, group=group)
        dist.all_reduce(num_valid_tokens, group=group)
        ctx.save_for_backward(num_valid_tokens)
        return loss / num_valid_tokens

    @staticmethod
    def backward(ctx: torch.autograd.Function,
                 grad_output: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        global_num_tokens, = ctx.saved_tensors
        grad_output = get_ulysses_sequence_parallel_world_size() * grad_output / global_num_tokens
        return grad_output, None


def reduce_sequence_parallel_loss(loss: torch.Tensor, num_valid_tokens: torch.Tensor) -> torch.Tensor:
    return ReduceLoss.apply(loss, num_valid_tokens)


class RMTrainer(object):

    def __init__(self, config):
        self.config = config
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.sp_size = config.model.sp_size
        self.tp_size = config.model.tp_size

        safely_do(lambda: report_job_config(config), rank=self.rank)()

        assert self.world_size % (
            config.model.sp_size * config.model.tp_size
        ) == 0, f"world_size {self.world_size} % (sp_size {config.model.sp_size} * tp_size {config.model.tp_size}) != 0"
        self.dp_size = self.world_size // config.model.sp_size // config.model.tp_size

        self.fsdp_mesh, self.tp_mesh, self.sp_mesh, self.gather_mesh = create_mesh(-1, self.tp_size, self.sp_size)
        self.gather_manager = DataGatherManager(self.gather_mesh, self.sp_mesh)

        if self.sp_size > 1:
            set_ulysses_sequence_parallel_group(self.sp_mesh.get_group())

        local_model_path = copy_local_path_from_hdfs(src=self.config.model.path, verbose=True)
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path,
                                                       trust_remote_code=self.config.model.trust_remote_code)

        if self.config.data.chat_template is None:
            from mono_rl.utils.seed import CHAT_TEMPLATE
            self.tokenizer.chat_template = CHAT_TEMPLATE

        # normalize dp size
        self._normalize_config_bsz()

        safely_do(
            lambda: report_data_loaded(train_files=self.config.data.train_files, val_files=self.config.data.val_files),
            rank=self.rank)()

        self._build_dataloader()
        # build model
        override_model_config = OmegaConf.to_container(config.model.override_config)
        safely_do(lambda: report_trial_started(checkpoint_paths=[self.config.model.path]), rank=self.rank)()
        self._build_model_optimizer(override_model_config)

    def _normalize_config_bsz(self):
        assert self.config.data.train_batch_size % self.dp_size == 0, f"train_batch_size {self.config.data.train_batch_size} % dp_size {self.dp_size} != 0"
        assert self.config.data.micro_batch_size % self.dp_size == 0, f"micro_batch_size {self.config.data.micro_batch_size} % dp_size {self.dp_size} != 0"

        self.train_batch_size = self.config.data.train_batch_size // self.dp_size
        self.micro_batch_size = self.config.data.micro_batch_size // self.dp_size

    def _build_dataloader(self):
        config = self.config
        # build dataset
        self.train_dataset = RMDataset(parquet_files=config.data.train_files,
                                       tokenizer=self.tokenizer,
                                       key=config.data.key,
                                       max_length=config.data.max_seq_length,
                                       max_response_num=config.data.max_response_num,
                                       truncation=config.data.truncation)

        self.val_dataset = RMDataset(parquet_files=config.data.val_files,
                                     tokenizer=self.tokenizer,
                                     key=config.data.key,
                                     max_length=config.data.max_seq_length,
                                     max_response_num=config.data.max_response_num,
                                     truncation=config.data.truncation)

        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=self.dp_size,
                                                rank=self.rank // self.sp_size // self.tp_size,
                                                drop_last=True)

        self.val_sampler = DistributedSampler(self.val_dataset,
                                              shuffle=True,
                                              num_replicas=self.dp_size,
                                              rank=self.rank // self.sp_size // self.tp_size,
                                              drop_last=True)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.train_batch_size,
                                           sampler=self.train_sampler,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.micro_batch_size,
                                         sampler=self.val_sampler,
                                         drop_last=True,
                                         collate_fn=collate_fn)

    def _build_model_optimizer(self, override_model_config):
        local_model_path = copy_local_path_from_hdfs(src=self.config.model.path, verbose=True)

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)

        with meta_device_init(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = AutoConfig.from_pretrained(local_model_path,
                                                num_labels=1,
                                                classifier_dropout=0.0,
                                                trust_remote_code=self.config.model.trust_remote_code)
            for k, v in override_model_config.items():
                setattr(config, k, v)
            setattr(config, "id2label", {0: "LABEL_0"})
            setattr(config, "label2id", {"LABEL_0": 0})
            architecture = config.architectures[0].replace('ForCausalLM', 'ForTokenClassification')
            setattr(config, "architectures", [architecture])

            if self.rank == 0:
                pprint(config)

            # monkey patch
            apply_monkey_patch(config, verbose=self.rank == 0)
            model = AutoModelForTokenClassification.from_config(config=config,
                                                                torch_dtype=torch.float32,
                                                                attn_implementation="flash_attention_2")

            # enable recompute
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

            nparams = sum(p.numel() for p in model.parameters())
            print(f"number of parameters before parallelization: {nparams / (1e9):.2f}B")

            shard_plan = get_parallel_plan(config, self.tp_mesh)
            shard_plan = parallelize_module(model, shard_plan, self.tp_mesh)

            nparams = sum(p.numel() for p in model.parameters())
            print(f"number of parameters after parallelization: {nparams / (1e9):.2f}B")
            self.flops_counter = FlopsCounter(config)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)
        auto_wrap_policy = get_fsdp_wrap_policy(module=model)

        shards = parallel_load_safetensors(local_model_path)
        init_fn = parallel_init_fsdp_fn(model, shards)

        if (not self.config.model.offload_params) or (self.train_batch_size != self.micro_batch_size) \
            or self.config.model.use_dynamic_bsz:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.offload_params)

        self.fsdp_model = FSDP(model,
                               use_orig_params=True,
                               param_init_fn=init_fn,
                               auto_wrap_policy=auto_wrap_policy,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               cpu_offload=cpu_offload,
                               forward_prefetch=True,
                               sync_module_states=False,
                               device_id=torch.cuda.current_device(),
                               device_mesh=self.fsdp_mesh)
        if len(shards) > 0:
            warnings.warn(
                "detected some parameter is not loaded in the model. Ignore this warning if you shrink the model layers."
            )
            shards.clear()

        register_dtensor_save_hook(self.fsdp_model, shard_plan)

        self.optimizer = optim.AdamW([param for param in self.fsdp_model.parameters() if param.requires_grad],
                                     lr=self.config.optim.lr,
                                     betas=self.config.optim.betas,
                                     weight_decay=self.config.optim.weight_decay,
                                     fused=True)

        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * self.config.trainer.total_epochs

        if self.rank == 0:
            print(
                f"Number of steps/epoch {steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}",
            )

        num_warmup_steps = int(total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = verl_F.get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                                   num_warmup_steps=num_warmup_steps,
                                                                   num_training_steps=total_steps,
                                                                   min_lr_ratio=self.config.optim.min_lr_ratio)

    def _compute_loss(self, micro_batch: TensorDict):
        response_num = micro_batch['response_num']
        input_ids = micro_batch['input_ids'].to(torch.int64)  # (bsz, n_resp, max_len)
        attention_mask = micro_batch['attention_mask'].to(torch.int64)  # (bsz, n_resp, max_len)
        gt_scores = micro_batch['scores'].to(torch.int64)  # (bsz, n_resp)
        batch_size, num_responses, seqlen = input_ids.shape
        input_ids_lst, attention_mask_lst, gt_score_lst = [], [], []
        for n, i, a, g in zip(response_num, input_ids, attention_mask, gt_scores):
            input_ids_lst.append(i[:n])
            attention_mask_lst.append(a[:n])
            gt_score_lst.append(g[:n])
        input_ids = torch.cat(input_ids_lst, dim=0)  # (totol_bsz, max_len)
        attention_mask = torch.cat(attention_mask_lst, dim=0)  # (totol_bsz, max_len)
        flat_gt_scores = torch.cat(gt_score_lst, dim=0)  # (totol_bsz)
        total_response_num = input_ids.size(0)

        if self.config.model.use_rmpad:
            position_ids = compute_position_id_with_mask(attention_mask)
            input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                input_ids.unsqueeze(-1), attention_mask=attention_mask)  # (totol_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                  indices).transpose(0, 1)

            # handle ulysses sequence parallelism
            total_nnz = input_ids_rmpad.size(1)
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad, position_ids_rmpad, self.sp_size)
            kwargs = {
                'input_ids': input_ids_rmpad,
                'position_ids': position_ids_rmpad,
                'output_hidden_states': False,
            }
            logits_rmpad = self.fsdp_model(**kwargs, use_cache=False).logits.squeeze(0)  # (total_nnz)
            # gather output from sp
            if self.sp_size > 1:
                logits_rmpad = gather_outpus_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
            # pad it back
            logits = pad_input(logits_rmpad.unsqueeze(-1), indices=indices, batch=total_response_num,
                               seqlen=seqlen).squeeze(-1)
        else:
            logits = self.fsdp_model(input_ids=input_ids, attention_mask=None,
                                     use_cache=False).logits  # (total_nnz, max_len)
        last_non_pad_token_idxs = (attention_mask.cumsum(dim=-1)).argmax(dim=-1)
        flat_rewards = logits[torch.arange(total_response_num), last_non_pad_token_idxs]  # (total_response_num)
        # chosen_reward_mean = torch.mean(flat_rewards[flat_gt_scores > 2])
        # rejected_reward_mean = torch.mean(flat_rewards[flat_gt_scores <= 2])
        reward_mean = torch.mean(flat_rewards)
        l2_loss = torch.mean(flat_rewards**2)
        rewards = []
        scores_diff = gt_scores.view(batch_size, num_responses, 1) - gt_scores.view(
            batch_size, 1, num_responses)  # (bsz, n_resp, n_resp)
        pairwise_mask = (scores_diff > 0).to(torch.float32)  # (bsz, n_resp, n_resp)
        for i, n in enumerate(response_num):
            rewards.append(
                torch.cat([flat_rewards[:n]] + [flat_rewards[n - 1:n]] * (self.config.data.max_response_num - n)))
            flat_rewards = flat_rewards[n:]
            pairwise_mask[i, n:, :] = 0
            pairwise_mask[i, :, n:] = 0
        pairwise_mask = pairwise_mask.view(batch_size, -1)  # (bsz, n_resp * n_resp)
        rewards = torch.cat(rewards, dim=0)  # (bsz, n_resp)
        pos_rewards = rewards.view(batch_size, num_responses, 1)
        neg_rewards = rewards.view(batch_size, 1, num_responses)
        rewards_diff = (pos_rewards - neg_rewards).view(batch_size,
                                                        -1)  # (bsz, n_resp, n_resp) -> (bsz, n_resp * n_resp)
        if self.config.trainer.margin:
            logsig_loss = -F.logsigmoid(rewards_diff - self.config.trainer.margin)
        else:
            logsig_loss = -F.logsigmoid(rewards_diff)
        logsig_loss = torch.sum(logsig_loss * pairwise_mask, dim=-1) / (torch.sum(pairwise_mask, dim=-1) + 1e-8
                                                                       )  # (bsz,)
        loss = torch.mean(logsig_loss) * batch_size / self.train_batch_size
        if self.config.trainer.center_rewards_coeff > 0:
            loss += self.config.trainer.center_rewards_coeff * l2_loss * batch_size / self.train_batch_size
        pairwise_acc = torch.sum((rewards_diff > 0) * pairwise_mask) / (torch.sum(pairwise_mask) + 1e-8)

        return loss, pairwise_acc, reward_mean

    def training_step(self, batch_data: DataProto):
        self.fsdp_model.train()

        self.optimizer.zero_grad()

        batch_data.to(torch.cuda.current_device())

        with Timer(name='train_step', logger=None) as timer:
            # split batch into micro_batches
            micro_batches = batch_data.batch.split(self.micro_batch_size)

            log_gpu_memory_usage('Before train')
            total_loss, pairwise_acc_lst, reward_mean_lst = 0.0, [], []
            for micro_batch in micro_batches:
                loss, pairwise_acc, reward_mean = self._compute_loss(micro_batch)
                loss.backward()
                total_loss += loss.item()
                pairwise_acc_lst.append(pairwise_acc)
                reward_mean_lst.append(reward_mean)

            log_gpu_memory_usage('Before optimizer step')
            grad_norm = clip_grad_norm_(self.fsdp_model, max_norm=self.config.optim.max_grad_norm).item()
            pairwise_acc = torch.mean(torch.stack(pairwise_acc_lst)).item()
            reward_mean = torch.mean(torch.stack(reward_mean_lst)).item()
            self.optimizer.step()
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            log_gpu_memory_usage('After optimizer step')

        delta_time = timer.last
        global_num_tokens = torch.sum(batch_data.batch['attention_mask'], dim=-1).view(-1).tolist()
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        mfu = estimated_flops / promised_flops / self.sp_size

        total_loss, grad_norm, pairwise_acc, reward_mean, mfu = \
            all_reduce([total_loss, grad_norm, pairwise_acc, reward_mean, mfu], op="mean")
        return {
            'train/loss': total_loss,
            'train/pairwise_acc': pairwise_acc,
            'train/lr(1e-4)': lr * 1e4,
            'train/grad_norm': grad_norm,
            'train/num_micro_batches': len(micro_batches),
            'train/reward_mean': reward_mean,
            'train/mfu': mfu,
        }

    def validation_step(self, batch_data: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss, pairwise_acc, reward_mean = self._compute_loss(batch_data)
            loss = loss * self.train_batch_size / self.micro_batch_size
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(pairwise_acc, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(reward_mean, op=torch.distributed.ReduceOp.AVG)
        return loss, pairwise_acc, reward_mean

    def save_checkpoint(self, step: int, epoch: int = -1):
        base_dir = os.path.join(self.config.trainer.default_hdfs_dir, "checkpoints")

        FSDPCheckpointer.save(
            base_dir,
            {"model": self.fsdp_model},
            global_steps=step,
        )

        path = os.path.join(base_dir, f'global_step_{step}')

        safely_do(lambda: report_checkpoint_saved(
            default_hdfs_path=self.config.trainer.default_hdfs_dir, path=path, step=step, epoch=epoch, omnistore={}),
                  rank=self.rank)()

    def fit(self):
        if self.rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger,
                                config=OmegaConf.to_container(self.config, resolve=True))

        global_step = 0
        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in self.train_dataloader:
                batch = DataProto.from_single_dict(data)
                with Timer(name="train/elapsed_time_per_step", logger=None) as timer:
                    metric = self.training_step(batch)

                metric.update({"train/elapsed_time_per_step": timer.last})

                # validation
                if global_step % self.config.trainer.eval_interval == 0:
                    val_losses, pairwise_accs, reward_means = [], [], []
                    for data in self.val_dataloader:
                        data = TensorDict(data, batch_size=self.micro_batch_size).cuda()
                        val_loss, pairwise_acc, reward_mean = self.validation_step(data)
                        val_losses.append(val_loss)
                        pairwise_accs.append(pairwise_acc)
                        reward_means.append(reward_mean)
                    if self.rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        pairwise_acc = torch.mean(torch.stack(pairwise_accs))
                        reward_mean = torch.mean(torch.stack(reward_means))
                        metric.update({
                            'val/loss': val_loss.detach().item(),
                            'val/pairwise_acc': pairwise_acc.detach().item(),
                            'val/reward_mean': reward_mean.detach().item()
                        })
                    torch.distributed.barrier()

                if self.rank == 0:
                    tracking.log(data=metric, step=global_step)

                global_step += 1

            # save checkpoint
            self.save_checkpoint(step=global_step, epoch=epoch)

            if self.rank == 0:
                local_path = os.path.join(self.config.trainer.default_local_dir, "huggingface")
                os.makedirs(local_path, exist_ok=True)
                self.fsdp_model.module.config.save_pretrained(local_path)
                self.tokenizer.save_pretrained(local_path)
                hdfs_io.copy(src=local_path,
                             dst=os.path.join(self.config.trainer.default_hdfs_dir, "checkpoints",
                                              f"global_step_{global_step}"))

        dist.barrier()


class RayRMTrainer(Worker):

    def __init__(self, config):
        super().__init__()
        self.config = config
        dist.init_process_group(backend="nccl")
        self.trainer = RMTrainer(self.config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def fit(self):
        self.trainer.fit()
