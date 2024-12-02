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
The main entry point to run the PPO algorithm
"""

from filelock import FileLock
import shutil
import warnings
import os
import logging
import hdfs_io
import ray
import torch
import torch.distributed
from omegaconf import DictConfig, open_dict, OmegaConf
from typing import List
from typing import Union

import verl.utils.torch_functional as verl_F
from single_controller.base import Worker
from single_controller.base.decorator import register, Dispatch
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, load_fsdp_grad, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, load_fsdp_param_and_grad
from verl.utils.import_utils import import_external_libs
from verl.utils.debug import log_gpu_memory_usage
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from verl.utils.torch_functional import broadcast_dict_tensor, allgather_dict_tensors
from verl.utils.model import compute_position_id_with_mask
import numpy as np

from alpha_seed.workers.hybrid_engine.hsdp import create_device_mesh
from alpha_seed.workers.hybrid_engine.fsdp_ulysses import FSDPUlyssesShardingManager
from .initialize import get_device_init_context, create_init_fn
from alpha_seed.workers.utils import rearrange_micro_batches
from dist_attn.ulysses.parallel_states import set_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_world_size
from dist_attn.ulysses.ops import slice_input_tensor, gather_outputs
from alpha_seed.workers.ppo_actor import DataParallelPPOActor
from alpha_seed.workers.ppo_critic import DataParallelPPOCritic

from seed_models.utils.count_flops import FlopsCounter

from codetiming import Timer

from datetime import timedelta

from .checkpoint import CheckpointManager

logger = logging.getLogger(__file__)


@ray.remote
class AsyncActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()

        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            timeout = timedelta(minutes=int(os.getenv('NCCL_TIMEOUT', 60)))
            torch.distributed.init_process_group(backend="nccl", timeout=timeout)

        # build device mesh
        self.master_address = os.getenv('MASTER_ADDR', 'localhost')
        self.master_port = os.getenv('MASTER_PORT', '12345')

        print(f'Master address: {self.master_address}, Master port: {self.master_port}')
        world_size = torch.distributed.get_world_size()

        # Note that here we assume actor and refernce policy have the same device mesh
        self.device_mesh = create_device_mesh(config.actor.fsdp_size, role)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref', 'standalone_rollout']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_standalone_rollout = self.role in ['standalone_rollout']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        # build device mesh for ulysses parallel. Note that we need to split the naming for actor and ref
        # to handle the case that actor and ref can colocate or not colocate
        # for actor
        if self._is_actor:
            sp_size = config.actor.ulysses_sequence_parallel_size
            if sp_size > 1:
                self.actor_ulysses_sp_device_mesh = init_device_mesh('cuda',
                                                                     mesh_shape=(world_size // sp_size, sp_size),
                                                                     mesh_dim_names=['dp', 'sp'])
            else:
                self.actor_ulysses_sp_device_mesh = None

            self.actor_ulysses_sharding_manager = FSDPUlyssesShardingManager(self.actor_ulysses_sp_device_mesh)

        if self._is_ref:
            sp_size = config.ref.ulysses_sequence_parallel_size
            if sp_size > 1:
                self.ref_ulysses_sp_device_mesh = init_device_mesh('cuda',
                                                                   mesh_shape=(world_size // sp_size, sp_size),
                                                                   mesh_dim_names=['dp', 'sp'])
            else:
                self.ref_ulysses_sp_device_mesh = None

            self.ref_ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ref_ulysses_sp_device_mesh)

        # normalize config
        if self._is_actor:
            sp_size = config.actor.ulysses_sequence_parallel_size
            self.config.actor.ppo_mini_batch_size //= world_size // sp_size
            self.config.actor.ppo_micro_batch_size //= world_size // sp_size
        if self._is_rollout or self._is_standalone_rollout:
            sp_size = config.actor.ulysses_sequence_parallel_size
            self.config.rollout.micro_batch_size //= world_size  # for xperf-gpt
            self.config.rollout.log_prob_micro_batch_size //= world_size // sp_size
        if self._is_ref:
            sp_size = config.ref.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size //= world_size // sp_size
        self.save_sequences = self.config.rollout.get('save_sequences', None)
        self.load_sequences = self.config.rollout.get('load_sequences', None)

    def _sequence_uuid(self):
        """Encode model ckpt, seqlen info for sequence generation, used for performance profiling

        TODO(haibin.lin): encode dataset info into uuid"""
        model_path = self.config.model.path.split('/')[-1]
        num_bon = self.config.rollout.num_bon
        max_token_len = self.config.rollout.max_token_len
        response_length = self.config.rollout.response_length
        prompt_length = self.config.rollout.prompt_length
        my_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        fields = [
            model_path, 'num_bon', num_bon, 'max_token_len', max_token_len, 'response_length', response_length,
            'prompt_length', prompt_length, 'rank', my_rank, world_size
        ]
        uuid = '_'.join([str(x) for x in fields])
        return uuid

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_master_addr(self):
        key = "standalone_master_addr" if self._is_standalone_rollout else "hybrid_master_addr"
        out = DataProto.from_dict(tensors={'mock': torch.tensor([[0]])}, meta_info={key: self.master_address})
        return out

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_rmpad=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False,
                               role='actor'):
        if self.rank == 0:
            print(f'Build model and optimizer for {role}')

        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.float32
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        setattr(actor_model_config, '_moe_implementation', 'fused')
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        if use_rmpad:
            # optimize the model via rmpad
            from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch
            assert apply_monkey_patch(
                config=actor_model_config,
                verbose=self.rank == 0), f'Cannot find rmpad version of {actor_model_config.model_type}'

        init_context = get_device_init_context(use_meta_tensor=True)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                actor_module.train()
                if self.rank == 0:
                    print(actor_module)
                    print('Enable actor gradient checkpointing')
                    if hasattr(actor_module, 'transformer'):
                        model = actor_module.transformer
                    elif hasattr(actor_module, 'model'):
                        model = actor_module.model
                    else:
                        model = None
                    if model is not None:
                        print(
                            f'{model.gradient_checkpointing=}, {model.training=}, {model._gradient_checkpointing_func=}'
                        )
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self.rank == 0:
            print(f'wrap_policy: {auto_wrap_policy}')

        cpu_offload = None

        if role == 'actor':
            if self.config.actor.fsdp_config.param_offload:
                cpu_offload = CPUOffload(offload_params=True)
        elif role == 'ref':
            if self.config.ref.fsdp_config.param_offload:
                cpu_offload = CPUOffload(offload_params=True)

        # we only support ZeRO3 of hybrid DP+FSDP or full FSDP
        if self.device_mesh.ndim == 1:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif self.device_mesh.ndim == 2:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise NotImplementedError(f"get device mesh ndim={self.device_mesh.ndim}, but only support 1 or 2")

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(actor_module,
                                 param_init_fn=create_init_fn(actor_module),
                                 use_orig_params=False,
                                 auto_wrap_policy=auto_wrap_policy,
                                 device_id=torch.cuda.current_device(),
                                 sharding_strategy=sharding_strategy,
                                 mixed_precision=mixed_precision,
                                 sync_module_states=True,
                                 forward_prefetch=True,
                                 device_mesh=self.device_mesh,
                                 cpu_offload=cpu_offload)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if role == 'actor':
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.95)),
                                          eps=optim_config.get('eps', 1e-08),
                                          weight_decay=optim_config.get('weight_decay', 0.1))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps = int(optim_config.get('lr_warmup_steps', -1))
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, hybrid_master_address=None, standalone_master_address=None):
        assert self.config.rollout.name == 'xperf_gpt'

        import xperf_gpt
        xperf_gpt.load_xperf_gpt()

        from alpha_seed.workers.streaming_service.streaming_rollout import AsyncXPerfGPTRollout
        from alpha_seed.workers.hybrid_engine import FSDPXPerfGPTShardingManager

        log_gpu_memory_usage('Before AsyncXPerfGPTRollout init', logger=logger)
        rollout = AsyncXPerfGPTRollout(config=self.config.rollout,
                                       tokenizer=self.tokenizer,
                                       model_hf_config=self.actor_model_config,
                                       is_standalone=self._is_standalone_rollout)
        log_gpu_memory_usage('After AsyncXPerfGPTRollout init', logger=logger)
        sharding_manager = FSDPXPerfGPTShardingManager(module=self.actor_module_fsdp,
                                                       model_config=self.actor_model_config,
                                                       inference_engine=rollout.inference_engine,
                                                       device_mesh=rollout.device_mesh,
                                                       standalone=self._is_standalone_rollout)
        if hybrid_master_address is not None and standalone_master_address is not None:
            sharding_manager.setup_standalone_rollout_comm(hybrid_master_address, standalone_master_address)
        else:
            # not support for the case that contains standalone rollout
            sharding_manager.release_param_and_cache()
            log_gpu_memory_usage('After AsyncXPerfGPTRollout release parameter and kv cache', logger=logger)
        log_gpu_memory_usage('After FSDPXPerfGPTShardingManager init', logger=logger)
        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device: str, model: bool = True, optimizer: bool = True):
        if self._is_actor and self.config.actor.fsdp_config.param_offload:
            return
        if self._is_ref and self.config.ref.fsdp_config.param_offload:
            return
        assert device in ("cuda", "cpu")
        if device == "cuda":
            device = torch.cuda.current_device()
            if self._is_actor:
                if model:
                    load_fsdp_param_and_grad(self.actor_module_fsdp, device)
                if optimizer:
                    load_fsdp_optimizer(self.actor_optimizer, device)
            if self._is_ref:
                if model:
                    load_fsdp_param_and_grad(self.ref_module_fsdp, device)
        elif device == "cpu":
            if self._is_actor:
                if model:
                    offload_fsdp_param_and_grad(self.actor_module_fsdp)
                if optimizer:
                    offload_fsdp_optimizer(self.actor_optimizer)
            if self._is_ref:
                if model:
                    offload_fsdp_param_and_grad(self.ref_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self, hybrid_master_address=None, standalone_master_address=None):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_rmpad = self.config.model.get('use_rmpad', False)
        use_ce_loss_fusion = self.config.model.get('use_ce_loss_fusion', False)

        if self._is_actor or self._is_rollout or self._is_standalone_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                use_rmpad=use_rmpad,
                trust_remote_code=self.config.model.get('trust_remote_code', False),
                role='actor' if self._is_actor else 'rollout')

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module
            assert self.actor_module.config.num_attention_heads % self.config.actor.ulysses_sequence_parallel_size == 0, \
                f'invalid ulysses sequence parallel size: {self.actor_module.config.num_attention_heads=} % {self.config.actor.ulysses_sequence_parallel_size=} != 0'

        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_rmpad = use_rmpad
                self.config.actor.use_ce_loss_fusion = use_ce_loss_fusion
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               use_rmpad=use_rmpad,
                                                               override_model_config=override_model_config,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False),
                                                               role='ref')[0]
            self.ref_module_fsdp.eval()

            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_rmpad = use_rmpad
                self.config.ref.use_ce_loss_fusion = use_ce_loss_fusion
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_rollout or self._is_standalone_rollout:
            self.rollout, self.sharding_manager = self._build_rollout(hybrid_master_address, standalone_master_address)
            self.rollout_async = None

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = CheckpointManager(model=self.actor.actor_module,
                                                        optimizer=self.actor.actor_optimizer,
                                                        lr_scheduler=self.actor_lr_scheduler,
                                                        tokenizer=self.tokenizer)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def update_standalone_rollout(self):
        assert self._is_rollout or self._is_standalone_rollout
        self.sharding_manager.update_standalone_rollout()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_actor
        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.actor_ulysses_sharding_manager:
            data = self.actor_ulysses_sharding_manager.preprocess_data(data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            data = self.actor_ulysses_sharding_manager.postprocess_data(data)

        self.actor_lr_scheduler.step()
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics['actor/lr(1e-4)'] = lr * 1e4

        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def old_log_probs(self, prompts: DataProto):
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_actor

        output = prompts
        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['temperature'] = prompts.meta_info['generation_kwargs']['temperature']
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.use_dynamic_bsz
            if self.config.rollout.use_dynamic_bsz:
                output.meta_info['max_token_len'] = self.config.rollout.max_token_len
            else:
                output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            with self.actor_ulysses_sharding_manager:
                output = self.actor_ulysses_sharding_manager.preprocess_data(output)
                old_entropy, old_log_probs = self.actor.compute_log_prob(data=output)
                output.batch['old_log_probs'] = old_log_probs
                output.batch['old_entropy'] = old_entropy
                output = self.actor_ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    def _load_sequences_offline(self):
        """load pre-generated sequences from hdfs"""
        uuid = self._sequence_uuid()
        fname = f'{uuid}.pt'
        from hdfs_io.hdfs_io import hcopy
        if not os.path.exists(fname):
            hcopy(f'{self.load_sequences}/{fname}', fname)
        data = torch.load(fname, map_location='cpu')
        output = DataProto(**data)
        if self.rank == 0:
            print("loaded pre-generated sequences", flush=True)
        return output

    def _save_sequences_offline(self, output):
        # TODO(haibin.lin): save to hdfs with hdfs_io
        uuid = self._sequence_uuid()
        output_to_save = {
            'batch': output.batch,
            'non_tensor_batch': output.non_tensor_batch,
            'meta_info': output.meta_info
        }
        fname = f'{uuid}.pt'
        torch.save(output_to_save, fname)
        from hdfs_io.hdfs_io import hcopy, hmkdir
        hmkdir(self.save_sequences)
        hcopy(fname, f'{self.save_sequences}/{fname}')
        print('Saved sequences and shutting down... Summary:', summerize_data(output.batch), uuid, flush=True)
        torch.distributed.barrier()
        # TODO(haibin.lin): typically we should throw an exception instead,
        # for perf tuning we directly quit from here
        exit()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')

        assert self._is_rollout

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        with self.sharding_manager:
            log_gpu_memory_usage('After entering sharding manager', logger=logger)
            prompts = self.sharding_manager.preprocess_data(prompts)

            if self.load_sequences:
                output = self._load_sequences_offline()
            else:
                generator = self.rollout.generate_sequences(prompts=prompts)
                output = next(generator)

                if self.save_sequences:
                    self._save_sequences_offline(output)

            output = self.sharding_manager.postprocess_data(output)

        output = output.to('cpu')
        torch.distributed.barrier()
        torch.cuda.empty_cache()

        log_gpu_memory_usage('After rollout generation', logger=logger)
        # clear kv cache
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences_put(self, prompts: DataProto):
        prompts = prompts.to('cuda')
        assert self._is_standalone_rollout

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        prompts = self.sharding_manager.preprocess_data(prompts)
        prompts.meta_info["complete_ratio"] = 1
        self.rollout_async = self.rollout.generate_sequences(prompts=prompts, is_async=True)
        next(self.rollout_async)
        return prompts

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences_get(self, prompts: DataProto):
        assert self._is_standalone_rollout
        output = next(self.rollout_async)
        log_gpu_memory_usage('After rollout generation standalone get', logger=logger)
        output = self.sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # clear kv cache
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['use_dynamic_bsz'] = self.config.ref.use_dynamic_bsz
        if self.config.ref.use_dynamic_bsz:
            data.meta_info['max_token_len'] = self.config.ref.max_token_len
        else:
            data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.train_generate_kwargs.temperature

        log_gpu_memory_usage('Before reference recompute log prob', logger=logger)

        with self.ref_ulysses_sharding_manager:
            data = self.ref_ulysses_sharding_manager.preprocess_data(data)
            _, output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ref_ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # reset FSDP buffer after forward
        self.ref_policy.actor_module._handle.reshard(True)
        log_gpu_memory_usage('After reference recompute log prob', logger=logger)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, hdfs_path=None, version='v1'):
        assert self._is_actor
        # TODO: support omnistore
        self.checkpoint_manager.load_checkpoint(version, hdfs_path, device_mesh=self.device_mesh)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def save_checkpoint(self, local_path, hdfs_path=None, version='v1'):
        # TODO: support omnistore
        assert self._is_actor
        self.checkpoint_manager.save_checkpoint(version, local_path, hdfs_path, self.device_mesh)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_param_and_cache(self):
        self.sharding_manager.release_param_and_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_ref_ema(self):
        """
        Update the reference policy via ema
        """
        assert self._is_actor and self._is_ref

        beta = self.config.ref.ema
        assert beta >= 0 and beta <= 1

        if beta == 1:
            # this is a small optimization, we can skip the copy
            return

        for (name, param), (name_ema, param_ema) in zip(self.actor_module_fsdp.named_parameters(),
                                                        self.ref_module_fsdp.named_parameters()):
            assert name == name_ema
            with torch.no_grad():
                # Note that the param here is sharded
                # Note (zhangchi.usc1992) this may be running on CPU and potentially slow
                param_ema.copy_(param.to(param_ema.device) * (1 - beta) + beta * param_ema)


def summerize_data(data: Union[dict, tuple, list], name: str = 'summary', level: int = 0, show_value=False) -> str:
    """Return the summary of a Tensor dict/tuple.

    Example::

      >>> data = (torch.ones(32,3,224,224), torch.zeros(32, 768))
      >>> label = torch.ones(32, 1)
      >>> data_batch = {'data': data, 'label': label}
      >>> summerize_data(data_batch)
      summary: dict, len: 2
        data: <class 'tuple'>, len: 2
          0: Tensor, len: torch.Size([32, 3, 224, 224]), dtype: torch.float32, val: 1.0
          1: Tensor, len: torch.Size([32, 768]), dtype: torch.float32, val: 0.0
        label: Tensor, len: torch.Size([32, 1]), dtype: torch.float32, val: 1.0

    """
    import torch
    indentation = '  ' * level
    summary = ''
    if isinstance(data, dict) or hasattr(data, 'items'):
        summary += indentation + f'{name}: dict, len: {len(data)}\n'
        for k, v in data.items():
            summary += summerize_data(v, k, level + 1)
    elif isinstance(data, (tuple, list)):
        summary += indentation + f'{name}: {type(data)}, len: {len(data)}\n'
        for idx, v in enumerate(data):
            summary += summerize_data(v, idx, level + 1)
    elif isinstance(data, torch.Tensor):
        summary += indentation + f'{name}: Tensor, len: {data.size()}, dtype: {data.dtype}'
        if show_value:
            summary += f', val: {data.detach().cpu().numpy().reshape(-1)[0]}\n'
        else:
            summary += '\n'
    else:
        summary += indentation + f'{name}: {type(data)}'
        if show_value:
            summary += f', val: {data}\n'
        else:
            summary += '\n'
    return summary
