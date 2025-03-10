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
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.utils.data import DataLoader, DistributedSampler
from codetiming import Timer
from omegaconf import OmegaConf

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import hdfs_io
import seed_models
from seed_models.utils.count_flops import FlopsCounter
from pprint import pprint

from tensordict import TensorDict
import verl.utils.torch_functional as verl_F
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tracking import Tracking
from verl import DataProto
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch
from alpha_seed.workers.actors.initialize import create_mesh, parallel_load_safetensors, parallel_init_fsdp_fn, meta_device_init
from alpha_seed.workers.actors.checkpoint.extensions import register_dtensor_save_hook
from alpha_seed.workers.actors import activation_offload
from alpha_seed.workers.actors.offload import offload_fsdp_optimizer, load_fsdp_optimizer
from alpha_seed.models.transformers.parallel import apply_parallel_plan
from alpha_seed.models.transformers.ops import clip_grad_norm_
from alpha_seed.utils.observility.training_stats import all_reduce

from alpha_seed.utils.dataset.sft_dataset import SFTDataset
from alpha_seed.utils.dataset.rl_dataset import collate_fn
from alpha_seed.workers.hybrid_engine.fsdp_gather import DataGatherManager, ulysses_pad_and_slice_inputs

from single_controller.base.worker import Worker
from single_controller.base.decorator import register, Dispatch

from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.bert_padding import index_first_axis, rearrange
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

from dist_attn.ulysses.parallel_states import set_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_world_size

import omnistore


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


class SFTTrainer(object):

    def __init__(self, config):
        self.config = config
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.sp_size = config.model.sp_size
        self.tp_size = config.model.tp_size

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
            from verl.utils.seed import CHAT_TEMPLATE
            self.tokenizer.chat_template = CHAT_TEMPLATE

        # normalize dp size
        self._normalize_config_bsz()

        self._build_dataloader()
        # build model
        override_model_config = OmegaConf.to_container(config.model.override_config)
        self._build_model_optimizer(override_model_config)

    def _normalize_config_bsz(self):
        assert self.config.data.train_batch_size % self.dp_size == 0, f"train_batch_size {self.config.data.train_batch_size} % dp_size {self.dp_size} != 0"
        assert self.config.data.micro_batch_size % self.dp_size == 0, f"micro_batch_size {self.config.data.micro_batch_size} % dp_size {self.dp_size} != 0"

        self.train_batch_size = self.config.data.train_batch_size // self.dp_size
        self.micro_batch_size = self.config.data.micro_batch_size // self.dp_size

    def _build_dataloader(self):
        config = self.config
        # build dataset
        self.train_dataset = SFTDataset(parquet_files=config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        key=config.data.key,
                                        max_length=config.data.max_seq_length,
                                        truncation=config.data.truncation)

        self.val_dataset = SFTDataset(parquet_files=config.data.val_files,
                                      tokenizer=self.tokenizer,
                                      key=config.data.key,
                                      max_length=config.data.max_seq_length,
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
            config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
            for k, v in override_model_config.items():
                setattr(config, k, v)

            if self.rank == 0:
                pprint(config)

            # monkey patch
            apply_monkey_patch(config, verbose=self.rank == 0)
            model = AutoModelForCausalLM.from_config(config=config,
                                                     torch_dtype=torch.float32,
                                                     attn_implementation="flash_attention_2")
            # enable recompute
            if self.config.model.enable_gradient_checkpointing:
                if self.config.model.act_offload:
                    torch.utils.checkpoint.CheckpointFunction = activation_offload.CheckpointFunction

                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': self.config.model.act_offload})

            nparams = sum(p.numel() for p in model.parameters())
            print(f"number of parameters before parallelization: {nparams / (1e9):.2f}B")

            shard_plan = apply_parallel_plan(model, config, self.tp_mesh)

            nparams = sum(p.numel() for p in model.parameters())
            print(f"number of parameters after parallelization: {nparams / (1e9):.2f}B")
            self.flops_counter = FlopsCounter(config)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)
        auto_wrap_policy = get_fsdp_wrap_policy(module=model)

        shards = parallel_load_safetensors(local_model_path) if self.config.model.omnistore_path is None else {}
        init_fn = parallel_init_fsdp_fn(model, shards)

        self.fsdp_model = FSDP(model,
                               use_orig_params=True,
                               param_init_fn=init_fn,
                               auto_wrap_policy=auto_wrap_policy,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               cpu_offload=None,
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

        if self.config.model.omnistore_path is not None:
            _lazy_init(self.fsdp_model, self.fsdp_model)
            omnistore.FSDPCheckpointer.load(self.config.model.omnistore_path, {
                "model": self.fsdp_model,
            })

        from alpha_seed.trainer.optim import get_optimizer_from_config
        self.optimizer = get_optimizer_from_config(self.fsdp_model.parameters(), self.config.optim)

        self.act_offload_ctx = activation_offload.get_offload_context(self.config.model.act_offload, self.fsdp_model)

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
        input_ids = micro_batch['input_ids'].to(torch.int64)
        attention_mask = micro_batch['attention_mask'].to(torch.int64)
        loss_mask = micro_batch['loss_mask'].to(torch.int64)
        micro_batch_size = len(micro_batch)
        if self.config.model.use_rmpad:
            position_ids = compute_position_id_with_mask(attention_mask)
            input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                input_ids.unsqueeze(-1), attention_mask=attention_mask)  # (totol_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
            loss_mask, _, _, _ = unpad_input(loss_mask.unsqueeze(-1), attention_mask=attention_mask)
            loss_mask = loss_mask.transpose(0, 1)  # (1, total_nnz)
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                  indices).transpose(0, 1)

            # handle ulysses sequence parallelism
            total_nnz = input_ids_rmpad.size(1)
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad, position_ids_rmpad, self.sp_size)
            input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.sp_size)
            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            loss_mask, _, _ = ulysses_pad_and_slice_inputs(loss_mask, None, self.sp_size)
            loss_mask = loss_mask.squeeze(0)
            # batch_size, seqlen = input_ids.shape
            kwargs = {
                'input_ids': input_ids_rmpad,
                'position_ids': position_ids_rmpad,
                'output_hidden_states': False,
            }
            if self.config.model.fuse_lm_head_ce_loss:
                kwargs.update({'fuse_lm_head_ce_loss': True, 'labels': input_ids_rmpad_rolled})
                with self.act_offload_ctx:
                    loss = self.fsdp_model(**kwargs, use_cache=False).loss
            else:
                vocab_size = self.fsdp_model.module.config.vocab_size
                with self.act_offload_ctx:
                    logits = self.fsdp_model(**kwargs, use_cache=False).logits.reshape(-1, vocab_size)

                loss = cross_entropy_loss(logits, input_ids_rmpad_rolled, inplace_backward=True)[0]

            # since gather_manager gathers data from all sp/tp ranks
            loss = torch.sum(loss * loss_mask) * micro_batch_size / self.train_batch_size
            if self.sp_size > 1:
                num_valid_tokens = loss_mask.sum()
                if num_valid_tokens == 0:
                    print(f"local num_valid_tokens is zero on rank {self.rank}")
                loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
            else:
                num_valid_tokens = loss_mask.sum()
                loss /= num_valid_tokens
        else:
            labels = torch.where(loss_mask == 1, input_ids, -100)  # ignored_label_index
            with self.act_offload_ctx:
                loss = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    use_cache=False).loss * micro_batch_size / self.train_batch_size

        return loss

    def training_step(self, batch_data: DataProto):
        self.fsdp_model.train()

        self.optimizer.zero_grad()

        batch_data.to(torch.cuda.current_device())

        with Timer(name='train_step', logger=None) as timer:
            if self.config.model.use_dynamic_bsz:
                micro_batches, _, _ = rearrange_micro_batches(batch=batch_data.batch,
                                                              max_token_len=self.config.data.max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = batch_data.batch.split(self.micro_batch_size)

            total_loss = 0.0
            for micro_batch in micro_batches:
                loss = self._compute_loss(micro_batch)
                loss.backward()
                total_loss += loss.item()

            if self.config.optim.state_offload:
                load_fsdp_optimizer(self.optimizer, torch.cuda.current_device())

            grad_norm = clip_grad_norm_(self.fsdp_model, max_norm=self.config.optim.max_grad_norm).item()
            self.optimizer.step()
            self.lr_scheduler.step()

            if self.config.optim.state_offload:
                offload_fsdp_optimizer(self.optimizer)

        delta_time = timer.last
        global_num_tokens = torch.sum(batch_data.batch['attention_mask'], dim=-1).view(-1).tolist()
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        mfu = estimated_flops / promised_flops / (self.sp_size * self.tp_size)
        seqlen = sum(global_num_tokens) / len(global_num_tokens)

        lr = self.lr_scheduler.get_last_lr()[0]
        total_loss, grad_norm, mfu, seqlen = all_reduce([total_loss, grad_norm, mfu, seqlen], op="mean")
        return {
            'train/loss': total_loss,
            'train/lr(1e-4)': lr * 1e4,
            'train/grad_norm': grad_norm,
            'train/num_micro_batches': len(micro_batches),
            'train/mfu': mfu,
            'train/seqlen_avg': seqlen,
        }

    def validation_step(self, batch_data: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss(batch_data) * self.train_batch_size / self.micro_batch_size
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def save_checkpoint(self, step):
        omnistore.FSDPCheckpointer.save(
            os.path.join(self.config.trainer.default_hdfs_dir, "checkpoints"),
            {"model": self.fsdp_model},
            global_steps=step,
        )

    def fit(self):
        if self.rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger)

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
                    val_losses = []
                    for data in self.val_dataloader:
                        data = TensorDict(data, batch_size=self.micro_batch_size).cuda()
                        val_loss = self.validation_step(data)
                        val_losses.append(val_loss)
                    if self.rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric.update({'val/loss': val_loss.detach().item()})
                    torch.distributed.barrier()

                if self.rank == 0:
                    tracking.log(data=metric, step=global_step)

                global_step += 1

            # save checkpoint
            self.save_checkpoint(step=global_step)

        if self.rank == 0:
            local_path = os.path.join(self.config.trainer.default_local_dir, "huggingface")
            os.makedirs(local_path, exist_ok=True)
            self.fsdp_model.module.config.save_pretrained(local_path)
            self.tokenizer.save_pretrained(local_path)
            hdfs_io.copy(src=local_path, dst=self.config.trainer.default_hdfs_dir)

        dist.barrier()


class RaySFTTrainer(Worker):

    def __init__(self, config):
        super().__init__()
        self.config = config
        dist.init_process_group(backend="nccl")
        self.trainer = SFTTrainer(self.config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def fit(self):
        self.trainer.fit()
