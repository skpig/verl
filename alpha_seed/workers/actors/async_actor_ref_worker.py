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

from contextlib import nullcontext
from filelock import FileLock
import shutil
import warnings
import os
import logging
import hdfs_io
from functools import partial
import ray
import torch
import torch.distributed
from torch.utils.checkpoint import noop_context_fn
from omegaconf import DictConfig, open_dict, OmegaConf
from typing import List
from typing import Union

import verl.utils.torch_functional as verl_F
from single_controller.base import Worker
from single_controller.base.decorator import register, Dispatch
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from .offload import offload_fsdp_model_to_cpu, load_fsdp_model_to_gpu
from alpha_seed.workers.actors.offload import offload_fsdp_optimizer, load_fsdp_optimizer
from verl.utils.import_utils import import_external_libs
from verl.utils.debug import log_gpu_memory_usage
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from verl.utils.torch_functional import broadcast_dict_tensor, allgather_dict_tensors
import numpy as np

from alpha_seed.utils import ndtimeline
from alpha_seed.workers.hybrid_engine.fsdp_gather import DataGatherManager
from alpha_seed.models.transformers.parallel import apply_parallel_plan
from .initialize import (create_mesh, parallel_init_fsdp_fn, parallel_load_safetensors, meta_device_init,
                         cleanup_local_tmp_folder_safetensors_files)
from .checkpoint.extensions import register_dtensor_save_hook
from alpha_seed.workers.ppo_actor import DataParallelPPOActor
from alpha_seed.utils.kernels.persist_gemm import deploy_persist_gemm, undelopy_persist_gemm
from alpha_seed.models.transformers.parallel.collectives import get_memory
from alpha_seed.utils.observility.training_stats import MetricsTorchDispatchMode, metrics_context_fn
from alpha_seed.utils.observility import get_profiler_context_wrapped

from seed_models.utils.count_flops import FlopsCounter

from codetiming import Timer

from datetime import timedelta

from .checkpoint import CheckpointManagerWrapper

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

        self.role = role
        assert self.role in [
            'actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref', 'standalone_rollout',
            'standalone_validator'
        ]

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_standalone_rollout = self.role in ['standalone_rollout']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']
        self._is_standalone_validator = self.role in ['standalone_validator']

        self.actor_strategy = config.actor.strategy
        self.ref_strategy = config.ref.strategy

        # actor model
        if self.actor_strategy == 'fsdp':
            actor_fsdp_size = config.actor.fsdp_size
            actor_sp_size = config.actor.ulysses_sequence_parallel_size
            actor_tp_size = config.actor.tp_size
            actor_meshes = create_mesh(fsdp_size=actor_fsdp_size, tp_size=actor_tp_size, sp_size=actor_sp_size)
            self.actor_fsdp_mesh = actor_meshes[0]
            self.actor_tp_mesh = actor_meshes[1]  # shared for both train and inference
            self.actor_sp_mesh = actor_meshes[2]
            self.actor_gather_mesh = actor_meshes[3]
            self.actor_gather_manager = DataGatherManager(self.actor_gather_mesh, self.actor_sp_mesh)
            if torch.distributed.get_rank():
                print(
                    f"Created actor with fsdp_size={self.actor_fsdp_mesh.shape}, tp_size={self.actor_tp_mesh.size()}, "
                    f"actor sp_size={self.actor_sp_mesh.size()}")
            if actor_tp_size > 1:
                if not config.actor.fsdp_config.use_orig_params:
                    raise ValueError(
                        "enable tensor / expert parallelism must set actor.fsdp_config.use_orig_params=True")
        elif self.actor_strategyy == 'megatron':
            # implement 3D parallel self.actor_gather_manager. We still assume that data is chunked in data parallel.
            # We first need to perform allgather in model parallel group so that data in each tp/pp/cp group is identical.
            # Then, we chunk data according to context parallel rank
            # In this way, the API of FSDP and Megatron can be identical
            raise NotImplementedError

        # reference model
        if self._is_ref:
            if self.ref_strategy == 'fsdp':
                ref_fsdp_size = config.ref.fsdp_size
                ref_sp_size = config.ref.ulysses_sequence_parallel_size
                ref_tp_size = config.ref.tp_size
                ref_meshes = create_mesh(fsdp_size=ref_fsdp_size, tp_size=ref_tp_size, sp_size=ref_sp_size)
                self.ref_fsdp_mesh = ref_meshes[0]
                self.ref_tp_mesh = ref_meshes[1]
                self.ref_sp_mesh = ref_meshes[2]
                self.ref_gather_mesh = ref_meshes[3]
                self.ref_gather_manager = DataGatherManager(self.ref_gather_mesh, self.ref_sp_mesh)
                if torch.distributed.get_rank():
                    print(
                        f"Created reference with fsdp_size={self.ref_fsdp_mesh.shape}, tp_size={self.ref_tp_mesh.size()}, "
                        f"infer sp_size={self.ref_sp_mesh.size()}")
                if ref_tp_size > 1:
                    if not config.ref.fsdp_config.use_orig_params:
                        raise ValueError(
                            "enable tensor / expert parallelism must set ref.fsdp_config.use_orig_params=True")
            elif self.ref_strategy == 'megatron':
                raise NotImplementedError

        profile_fname = f"trace_{self.role}_rank{self.rank}.json"

        self.profiler_context = get_profiler_context_wrapped(filename=profile_fname,
                                                             profile_on_ranks=[0],
                                                             upload_to_mlx=True,
                                                             enable=False,
                                                             wait=0,
                                                             warmup=0,
                                                             active=1)
        if config.actor.get("sm_margin", 0) > 0:
            deploy_persist_gemm(int(config.actor.get("sm_margin", 0)))

        # normalize config
        if self._is_actor:
            if self.actor_strategy == 'fsdp':
                sp_size = config.actor.ulysses_sequence_parallel_size
                self.config.actor.ppo_mini_batch_size //= (world_size // sp_size // actor_tp_size)
                self.config.actor.ppo_micro_batch_size //= (world_size // sp_size // actor_tp_size)
            elif self.actor_strategy == 'megatron':
                raise NotImplementedError

        if self._is_rollout or self._is_standalone_rollout:
            if self.actor_strategy == 'fsdp':
                sp_size = config.actor.ulysses_sequence_parallel_size
                self.config.rollout.micro_batch_size //= world_size  # for xperf-gpt
                self.config.rollout.log_prob_micro_batch_size //= (world_size // sp_size // actor_tp_size)
            elif self.actor_strategy == 'megatron':
                raise NotImplementedError

        if self._is_ref:
            if self.ref_strategy == 'fsdp':
                sp_size = config.ref.ulysses_sequence_parallel_size
                self.config.ref.log_prob_micro_batch_size //= (world_size // sp_size // ref_tp_size)
            elif self.ref_strategy == 'megatron':
                raise NotImplementedError

        self._model_initialized = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_rollout_callback_function(self, eos_callback_fn):
        self.rollout.set_rollout_callback_function(eos_callback_fn=eos_callback_fn)

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
        # TODO: ignore pulling model file if resuming ckpt
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
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

        with meta_device_init(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_config(actor_model_config,
                                                            torch_dtype=torch_dtype,
                                                            attn_implementation='flash_attention_2',
                                                            trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if self.config.remove_o_bias:
                from seed_models import P6DenseForCausalLM
                if isinstance(actor_module, P6DenseForCausalLM):
                    for layer in actor_module.model.layers:
                        if layer.self_attn.o_proj.bias is not None:
                            layer.self_attn.o_proj.bias.requires_grad = False

            enable_training_stats = self.config.actor.enable_training_stats
            metrics_context = MetricsTorchDispatchMode() if enable_training_stats else nullcontext()

            if enable_gradient_checkpointing:
                use_reentrant = self.config.actor.act_offload
                if self.config.actor.act_offload:
                    # doc link: https://bytedance.us.larkoffice.com/docx/NiWVd0QgoopepBxBXmDuHJKwsNe
                    from alpha_seed.workers.actors import activation_offload
                    torch.utils.checkpoint.CheckpointFunction = activation_offload.CheckpointFunction

                # this is a specialization for seed m8 to get avoid of
                # non-deterministic recompute of gate
                if actor_module.config.model_type == "seed_m8":
                    from seed_models.models.m8.modeling_m8 import M8DecoderLayer
                    from torch.utils.checkpoint import checkpoint
                    gradient_checkpointing_kwargs = {
                        'use_reentrant':
                            use_reentrant,
                        "context_fn":
                            partial(metrics_context_fn, metrics_context) if
                            (enable_training_stats and not use_reentrant) else noop_context_fn,
                    }
                    recompute_fn = partial(checkpoint, **gradient_checkpointing_kwargs)
                    for layer in actor_module.transformer.h:
                        assert isinstance(layer, M8DecoderLayer)
                        layer._gradient_checkpointing_func = recompute_fn
                else:
                    actor_module.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={
                            'use_reentrant':
                                use_reentrant,
                            "context_fn":
                                partial(metrics_context_fn, metrics_context) if (
                                    enable_training_stats and not use_reentrant) else noop_context_fn,
                        })
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
                        print(f'{model.gradient_checkpointing=}, {model.training=}')
        # use shard plan
        tp_mesh = self.ref_tp_mesh if role == 'ref' else self.actor_tp_mesh
        shard_plan = apply_parallel_plan(actor_module, actor_module.config, tp_mesh)

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

        from alpha_seed.models.transformers.monkey_patch import get_ignore_modules_in_mixed_precision

        mp_config = dict(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )
        if self.config.update_gate_ema:
            mp_config['_module_classes_to_ignore'] = get_ignore_modules_in_mixed_precision(
                actor_model_config.model_type)
        mixed_precision = MixedPrecision(**mp_config)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self.rank == 0:
            print(f'wrap_policy: {auto_wrap_policy}')

        cpu_offload = None

        if role == 'actor':
            if self.config.actor.fsdp_config.param_offload:
                # NOTE: CPUOffload needs to cooperate with FSDP.no_sync() in gradient accumulation,
                # which will lead to more memory consumption as gradients keep unshard in between micro-batches.
                # temporarily disbale this for more investigation
                cpu_offload = CPUOffload(offload_params=False)
        elif role == 'ref':
            if self.config.ref.fsdp_config.param_offload:
                cpu_offload = CPUOffload(offload_params=True)
        elif role == 'rollout':
            # rollout only, requires cpu_offload
            cpu_offload = CPUOffload(offload_params=True)

        # we only support ZeRO3 of hybrid DP+FSDP or full FSDP
        fsdp_mesh = self.ref_fsdp_mesh if role == 'ref' else self.actor_fsdp_mesh
        if fsdp_mesh.ndim == 1:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif fsdp_mesh.ndim == 2:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise NotImplementedError(f"role: {role}: get device mesh ndim={fsdp_mesh.ndim}, but only support 1 or 2")

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(actor_module,
                                 param_init_fn=parallel_init_fsdp_fn(actor_module,
                                                                     parallel_load_safetensors(local_path)),
                                 use_orig_params=self.config.actor.fsdp_config.use_orig_params,
                                 auto_wrap_policy=auto_wrap_policy,
                                 device_id=torch.cuda.current_device(),
                                 sharding_strategy=sharding_strategy,
                                 mixed_precision=mixed_precision,
                                 sync_module_states=False,
                                 forward_prefetch=True,
                                 device_mesh=fsdp_mesh,
                                 cpu_offload=cpu_offload)

        register_dtensor_save_hook(actor_module_fsdp, shard_plan)

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

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config, metrics_context

    def _build_rollout(self):
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

        if self.actor_strategy == 'fsdp':
            sharding_manager = FSDPXPerfGPTShardingManager(module=self.actor_module_fsdp,
                                                           model_config=self.actor_model_config,
                                                           inference_engine=rollout.inference_engine,
                                                           device_mesh=rollout.device_mesh,
                                                           standalone=self._is_standalone_rollout or
                                                           self._is_standalone_validator,
                                                           only_bind_once=self.role == "rollout")
        elif self.actor_strategy == 'megatron':
            raise NotImplementedError

        sharding_manager.release_param_and_cache()
        log_gpu_memory_usage('After AsyncXPerfGPTRollout release parameter and kv cache', logger=logger)
        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device: str, model: bool = True, optimizer: bool = True):
        assert device in ("cuda", "cpu")
        if device == "cuda":
            device = torch.cuda.current_device()
            if self._is_actor or self._is_standalone_rollout or self._is_standalone_validator:
                if self.actor_strategy == 'fsdp':
                    if not self.config.actor.fsdp_config.param_offload:
                        if model:
                            load_fsdp_model_to_gpu(self.actor_module_fsdp)
                        if optimizer and self.actor_optimizer is not None:
                            load_fsdp_optimizer(self.actor_optimizer, device)
                elif self.actor_strategy == 'megatron':
                    raise NotImplementedError

            if self._is_ref:
                if self.ref_strategy == 'fsdp':
                    if model and not self.config.ref.fsdp_config.param_offload:
                        load_fsdp_model_to_gpu(self.ref_module_fsdp)
                elif self.ref_strategy == 'megatron':
                    raise NotImplementedError

        elif device == "cpu":
            if self._is_actor or self._is_standalone_rollout or self._is_standalone_validator:
                if self.actor_strategy == 'fsdp':
                    if not self.config.actor.fsdp_config.param_offload:
                        if model:
                            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                        if optimizer and self.actor_optimizer is not None:
                            offload_fsdp_optimizer(self.actor_optimizer)
                elif self.actor_strategy == 'megatron':
                    raise NotImplementedError
            if self._is_ref:
                if self.ref_strategy == 'fsdp':
                    if model and not self.config.ref.fsdp_config.param_offload:
                        offload_fsdp_model_to_cpu(self.ref_module_fsdp)
                elif self.ref_strategy == 'megatron':
                    raise NotImplementedError

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self, remove_safetensors_after_init=False):
        if self._model_initialized:
            return
        with self.profiler_context:
            self._init_model()
        self._model_initialized = True
        if remove_safetensors_after_init:
            cleanup_local_tmp_folder_safetensors_files(self.actor_model_config._name_or_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def setup_standalone_worker_comm(self, hybrid_master_address, standalone_master_address, port, role):
        self.sharding_manager.setup_standalone_worker_comm(hybrid_master_address, standalone_master_address, port, role)

    def _init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_rmpad = self.config.model.get('use_rmpad', False)
        use_ce_loss_fusion = self.config.model.get('use_ce_loss_fusion', False)

        if self._is_actor or self._is_rollout or self._is_standalone_rollout or self._is_standalone_validator:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            if self.actor_strategy == 'fsdp':
                self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config, self.metrics_context = self._build_model_optimizer(
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

            elif self.actor_strategy == 'megatron':
                # TODO: build megatron model
                raise NotImplementedError

        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            if self.actor_strategy == 'fsdp':
                with open_dict(self.config.actor):
                    self.config.actor.use_rmpad = use_rmpad
                    self.config.actor.use_ce_loss_fusion = use_ce_loss_fusion
                enable_non_reentrant_recompute = self.config.model.get('enable_gradient_checkpointing',
                                                                       False) and not self.config.actor.act_offload
                self.actor = DataParallelPPOActor(config=self.config.actor,
                                                  actor_module=self.actor_module_fsdp,
                                                  actor_optimizer=self.actor_optimizer,
                                                  actor_model_config=self.actor_model_config,
                                                  enable_non_reentrant_recompute=enable_non_reentrant_recompute,
                                                  metrics_context=self.metrics_context)
            elif self.actor_strategy == 'megatron':
                # TODO: build megatron actor
                raise NotImplementedError

        if self._is_ref:
            if self.ref_strategy == 'fsdp':
                self.ref_module_fsdp = self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=self.config.ref.fsdp_config,
                    optim_config=None,
                    use_rmpad=use_rmpad,
                    override_model_config=override_model_config,
                    enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                    trust_remote_code=self.config.model.get('trust_remote_code', False),
                    role='ref')[0]
                self.ref_module_fsdp.eval()

                OmegaConf.set_struct(self.config.ref, True)
                with open_dict(self.config.ref):
                    self.config.ref.use_rmpad = use_rmpad
                    self.config.ref.use_ce_loss_fusion = use_ce_loss_fusion
                self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
            elif self.ref_strategy == 'megatron':
                # TODO: build megatron actor
                raise NotImplementedError

        if self._is_rollout or self._is_standalone_rollout or self._is_standalone_validator:
            self.rollout, self.sharding_manager = self._build_rollout()
            self.rollout_async = None

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            if self.actor_strategy == 'fsdp':
                self.checkpoint_manager = CheckpointManagerWrapper(model=self.actor.actor_module,
                                                                   optimizer=self.actor.actor_optimizer,
                                                                   lr_scheduler=self.actor_lr_scheduler,
                                                                   tokenizer=self.tokenizer)
            elif self.actor_strategy == 'megatron':
                # TODO: build megatron checkpoint manager
                raise NotImplementedError

        if self._is_ref:
            if self.ref_strategy == 'fsdp':
                self.checkpoint_manager_ref = CheckpointManagerWrapper(model=self.ref_policy.actor_module,
                                                                       optimizer=None,
                                                                       lr_scheduler=None,
                                                                       tokenizer=self.tokenizer)
            elif self.ref_strategy == 'megatron':
                # TODO: build megatron checkpoint manager
                raise NotImplementedError

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_eos_callback_fn(self, eos_callback_fn):
        self.eos_callback_fn = eos_callback_fn

        def make_eos_call_back_fn(device_mesh):
            from xperf_gpt.inference.session import Query

            def eos_callback_fn(query: Query):
                if device_mesh is None:
                    tp_rank = 0
                else:
                    tp_rank = device_mesh['tp'].get_local_rank()

                if tp_rank == 0:
                    # only happens on tp rank zero
                    self.eos_callback_fn(query)

            return eos_callback_fn

        self.rollout.set_rollout_callback_function(eos_callback_fn=make_eos_call_back_fn(self.rollout.device_mesh))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def update_standalone_worker(self, role):
        assert self._is_rollout or self._is_standalone_rollout or self._is_standalone_validator
        self.sharding_manager.update_standalone_worker(role)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        torch.cuda.reset_peak_memory_stats()
        # data = data.to('cuda')

        assert self._is_actor
        # data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        # note optimizer offload will be managed inside `update_policy`
        if self.config.actor.train_memory_offload:
            self.to("cuda", model=True, optimizer=False)

        with self.actor_gather_manager:
            data = self.actor_gather_manager.preprocess_data(data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            data = self.actor_gather_manager.postprocess_data(data)

        self.actor_lr_scheduler.step()
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics['actor/lr(1e-4)'] = lr * 1e4

        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        max_memory_allocated, max_memory_reserved = get_memory()
        output = DataProto(
            meta_info={
                'metrics': metrics,
                'memory/actor_max_allocated': max_memory_allocated,
                'memory/actor_max_reserved': max_memory_reserved
            })
        output = output.to('cpu')
        if self.config.actor.train_memory_offload:
            self.to("cpu", model=True, optimizer=True)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def old_log_probs(self, prompts: DataProto):
        prompts = prompts.to('cpu')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_actor

        output = prompts
        if self._is_actor and recompute_log_prob:

            if self.config.actor.train_memory_offload:
                self.to("cuda", model=True, optimizer=False)

            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['temperature'] = prompts.meta_info['generation_kwargs']['temperature']
            # align with the training config
            output.meta_info['use_dynamic_bsz'] = self.config.actor.use_dynamic_bsz
            if self.config.actor.use_dynamic_bsz:
                output.meta_info['max_token_len'] = self.config.actor.ppo_max_token_len
            else:
                output.meta_info['micro_batch_size'] = self.config.actor.ppo_micro_batch_size
            with self.actor_gather_manager:
                output = self.actor_gather_manager.preprocess_data(output)
                old_entropy, old_log_probs = self.actor.compute_log_prob(data=output)
                output.batch['old_log_probs'] = old_log_probs
                output.batch['old_entropy'] = old_entropy
                output = self.actor_gather_manager.postprocess_data(output)

            if self.config.actor.train_memory_offload:
                self.to("cpu", model=True, optimizer=False)

        output = output.to('cpu')

        # clear kv cache
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        torch.cuda.reset_peak_memory_stats()
        prompts = prompts.to('cuda')

        assert self._is_rollout or self._is_standalone_validator

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)

        # xperf needs parameters from actor
        if self.config.actor.train_memory_offload:
            self.to("cuda", model=True, optimizer=False)

        with self.sharding_manager:

            # after parameters go to xperf, offload actor model to CPU
            if self.config.actor.train_memory_offload:
                self.to("cpu", model=True, optimizer=False)

            log_gpu_memory_usage('After entering sharding manager', logger=logger)
            prompts = self.sharding_manager.preprocess_data(prompts)

            generator = self.rollout.generate_sequences(prompts=prompts)
            output = next(generator)

            output = self.sharding_manager.postprocess_data(output)

        max_memory_allocated, max_memory_reserved = get_memory()
        output.meta_info.update({
            'memory/gen_max_allocated': max_memory_allocated,
            'memory/gen_max_reserved': max_memory_reserved
        })
        output = output.to('cpu')
        # torch.distributed.barrier()

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
        prompts.meta_info["complete_ratio"] = 0
        self.rollout_async = self.rollout.generate_sequences(prompts=prompts, is_async=True)
        next(self.rollout_async)
        return prompts

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences_get(self):
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
        torch.cuda.reset_peak_memory_stats()
        assert self._is_ref

        # data = data.to('cuda')

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['use_dynamic_bsz'] = self.config.ref.use_dynamic_bsz
        if self.config.ref.use_dynamic_bsz:
            data.meta_info['max_token_len'] = self.config.ref.max_token_len
        else:
            data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.train_generate_kwargs.temperature

        log_gpu_memory_usage('Before reference recompute log prob', logger=logger)

        with self.ref_gather_manager:
            data = self.ref_gather_manager.preprocess_data(data)
            _, output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ref_gather_manager.postprocess_data(output)

        # reset FSDP buffer after forward
        self.ref_policy.actor_module._handle.reshard(True)
        log_gpu_memory_usage('After reference recompute log prob', logger=logger)

        max_memory_allocated, max_memory_reserved = get_memory()
        output.meta_info.update({
            'memory/ref_max_allocated': max_memory_allocated,
            'memory/ref_max_reserved': max_memory_reserved
        })
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, hdfs_path=None, version='v1', enable_shm=False, model='actor'):
        # TODO: remove the following line once megatron ckpt manager is implemented
        if self.actor_strategy in ['megatron']:
            # TODO(fix me)
            return

        if model == 'actor':
            assert self._is_actor
            ckpt_manager = self.checkpoint_manager
        elif model == 'ref':
            assert self._is_ref
            ckpt_manager = self.checkpoint_manager_ref
        else:
            raise ValueError(f'Unknown {model=}')

        if self.config.actor.train_memory_offload:
            self.to("cuda")
        ckpt_manager.load_checkpoint(version=version,
                                     hdfs_path=hdfs_path,
                                     device_mesh=self.actor_fsdp_mesh,
                                     role='actor',
                                     enable_shm=enable_shm)
        if self.config.actor.train_memory_offload:
            self.to("cpu")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def save_checkpoint(self,
                        local_path,
                        hdfs_path=None,
                        version='v1',
                        global_step=0,
                        ckpt_global_uploader_ref=None,
                        enable_shm=False,
                        model='actor'):
        # TODO: remove the following line once megatron ckpt manager is implemented
        if self.actor_strategy in ['megatron']:
            # TODO(fix me)
            return

        # TODO: support omnistore
        if model == 'actor':
            assert self._is_actor
            ckpt_manager = self.checkpoint_manager
        elif model == 'ref':
            assert self._is_ref
            ckpt_manager = self.checkpoint_manager_ref
        else:
            raise ValueError(f'Unknown {model=}')

        if self.config.actor.train_memory_offload:
            self.to("cuda")

        ckpt_manager.save_checkpoint(version=version,
                                     local_path=local_path,
                                     hdfs_path=hdfs_path,
                                     device_mesh=self.actor_fsdp_mesh,
                                     role=model,
                                     global_step=global_step,
                                     ckpt_global_uploader_ref=ckpt_global_uploader_ref,
                                     enable_shm=enable_shm)
        if self.config.actor.train_memory_offload:
            self.to("cpu")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_param_and_cache(self):
        self.sharding_manager.release_param_and_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_ref_ema(self):
        """
        Update the reference policy via ema
        """
        if self.actor_strategy in ['megatron']:
            # TODO(fix me)
            return

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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def upload_process_group(self, trigger_timestamp):
        if self.actor_strategy in ['megatron']:
            # TODO(fix me)
            return
        ndtimeline.upload_process_group(trigger_timestamp, ndtimeline.DumpType.initial.value)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def do_ndtimeline_action(self, action, *args, **kwargs):
        if self.actor_strategy in ['megatron']:
            # TODO(fix me)
            return
        ndtimeline.do_ndtimeline_action(action, *args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_ndtimeline(self):
        if self.actor_strategy in ['megatron']:
            # TODO(fix me)
            return

        if self._is_actor or self._is_rollout:
            mocked_fsdp_shape = list(self.actor_fsdp_mesh.shape)
            mocked_fsdp_shape[-1] *= self.actor_tp_mesh.size()
            mocked_fsdp_shape = tuple(mocked_fsdp_shape)
        elif self._is_ref:
            mocked_fsdp_shape = list(self.ref_fsdp_mesh.shape)
            mocked_fsdp_shape[-1] *= self.ref_tp_mesh.size()
            mocked_fsdp_shape = tuple(mocked_fsdp_shape)
        elif self._is_standalone_rollout:
            mocked_fsdp_shape = (self.config.streaming_rollout_args.n_gpus_per_node *
                                 self.config.streaming_rollout_args.nnodes,)
        elif self._is_standalone_validator:
            mocked_fsdp_shape = (self.config.streaming_validator_args.n_gpus_per_node *
                                 self.config.streaming_validator_args.nnodes,)
        else:
            mocked_fsdp_shape = (-1,)
        ndtimeline.init_with_ray(self.config.get("use_cuda_timer", False), mocked_fsdp_shape, self)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reinit(self, config: DictConfig, role: str):
        import gc
        if self._model_initialized:
            if self._is_actor or self._is_standalone_rollout or self._is_standalone_validator:
                del self.actor_module_fsdp
                del self.actor_optimizer
            if self._is_rollout or self._is_standalone_rollout or self._is_standalone_validator:
                del self.rollout
                del self.sharding_manager
            if self._is_ref:
                del self.ref_module_fsdp
            self._model_initialized = False
        gc.collect()
        torch.cuda.empty_cache()
        self.__init__(config, role)


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
