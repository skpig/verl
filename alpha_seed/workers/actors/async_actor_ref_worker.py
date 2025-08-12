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

import time
from typing import Union, List
import warnings
import os
import logging
import ray
import torch
import torch.distributed
from omegaconf import DictConfig
import gc

from mono_rl.single_controller import Worker
from mono_rl.single_controller import register, Dispatch, Execute
from mono_rl import DataProto
from alpha_seed.workers.fsdp.offload import (offload_fsdp_optimizer, load_fsdp_optimizer, offload_fsdp_model_to_cpu,
                                             load_fsdp_model_to_gpu)

from alpha_seed.workers.megatron.offload import (offload_megatron_model_to_cpu, load_megatron_model_to_gpu)
from verl.utils.debug import log_gpu_memory_usage
from alpha_seed.workers.xperf_rollout.component.query import Query
from alpha_seed.workers.xperf_rollout.utils.layout_convert_helper import offload_to_device

from alpha_seed.utils import ndtimeline
from alpha_seed.workers.hybrid_engine.fsdp_gather import DataGatherManager
from alpha_seed.workers.fsdp.initialize import cleanup_local_tmp_folder_safetensors_files
from alpha_seed.workers.ppo_actor import DataParallelPPOActor
from alpha_seed.workers.xperf_rollout.profiler.visualizer import visualize_metrics
from alpha_seed.utils.kernels.persist_gemm import deploy_persist_gemm
from mono_rl.models.seed_models.parallel.collectives import get_memory
from mono_rl.models.seed_models.modeling_vlm import add_pixel_values_to_inflight_query
from alpha_seed.utils.dataset.vlm_rl_dataset import load_and_transform_save_image
from alpha_seed.utils.observility import get_profiler_context_wrapped
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoProcessor

from seed_models.utils.count_flops import FlopsCounter

from codetiming import Timer

from datetime import timedelta

from .checkpoint import CheckpointManagerWrapper
from ..streaming_service.streaming_utils import get_free_port_for_nccl_primitive
from ..xperf_rollout.session import LoadMetric

# mariana dependency
try:
    from megatron.core import parallel_state as mpu
    from alpha_seed.workers.ppo_actor_megatron import MegatronPPOActor
except:
    pass

from alpha_seed.utils.mono_rl.config import actor_config_to_mono_config, ref_config_to_mono_config
from mono_rl.worker import Role
from mono_rl.worker.engine.fsdp.models.model import FSDPModel

logger = logging.getLogger(__file__)


@ray.remote
class AsyncActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, enable_actor_critic_spatial_mux: bool = False):
        super().__init__()

        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.config = config
        if not torch.distributed.is_initialized():
            timeout = timedelta(seconds=int(os.getenv('NCCL_TIMEOUT', 3600)))
            torch.distributed.init_process_group(backend="nccl", timeout=timeout)
        # build device mesh
        self.master_address = os.getenv('MASTER_ADDR', 'localhost')
        self.master_port = os.getenv('MASTER_PORT', '12345')

        print(f'Master address: {self.master_address}, Master port: {self.master_port}')

        self.role = role
        assert self.role in [
            'actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref', 'standalone_rollout',
            'standalone_validator', 'rollout_server'
        ], f'Invalid role: {self.role}'

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_standalone_rollout = self.role in ['standalone_rollout']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']
        self._is_standalone_validator = self.role in ['standalone_validator']

        self.actor_strategy = config.actor.strategy
        self.ref_strategy = config.ref.strategy
        self.local_path = None
        self.hybrid_rollout_addresses = None
        self.enable_actor_critic_spatial_mux = enable_actor_critic_spatial_mux

        self._is_valid_actor = self._is_actor
        if self.enable_actor_critic_spatial_mux:
            actor_world_size = torch.distributed.get_world_size() // 2
            if self.rank >= actor_world_size:
                self._is_valid_actor = False

        if self.actor_strategy in ('fsdp', 'vescale-fsdp2'):
            actor_mono_config = self._get_actor_mono_config()
            self.actor_engine = FSDPModel(actor_mono_config.engine)
            self.actor_fsdp_mesh = self.actor_engine.fsdp_mesh
            self.actor_tp_mesh = self.actor_engine.tp_mesh
            self.actor_oe_mesh = self.actor_engine.oe_mesh
            self.actor_sp_mesh = self.actor_engine.sp_mesh
            self.actor_gather_mesh = self.actor_engine.gather_mesh
            self.actor_train_mesh = self.actor_engine.train_mesh
            if self._is_valid_actor:
                self.actor_gather_manager = DataGatherManager(self.actor_gather_mesh, self.actor_sp_mesh)
        elif self.actor_strategy == 'megatron':
            # implement 3D parallel self.actor_gather_manager. We still assume that data is chunked in data parallel.
            # We first need to perform allgather in model parallel group so that data in each tp/pp/cp group is identical.
            # Then, we chunk data according to context parallel rank
            # In this way, the API of FSDP and Megatron can be identical
            # FIXME: the MegatronPPoActor API is not aligned yet, need fix in both alpha_seed and mono_rl
            from alpha_seed.workers.hybrid_engine.megatron_gather import MegatronDataGatherManager
            from mono_rl.worker.engine.megatron.model import MegatronModel
            actor_mono_config = actor_config_to_mono_config(self.config.actor, self.config.model)
            self.actor_engine = MegatronModel(actor_mono_config.engine)
            self.actor_gather_manager = MegatronDataGatherManager()

        if self._is_ref:
            if self.ref_strategy in ('fsdp', 'vescale-fsdp2'):
                ref_mono_config = self._get_ref_mono_config()
                self.ref_engine = FSDPModel(ref_mono_config.engine)
                self.ref_fsdp_mesh = self.ref_engine.fsdp_mesh
                self.ref_tp_mesh = self.ref_engine.tp_mesh
                self.ref_oe_mesh = self.ref_engine.oe_mesh
                self.ref_sp_mesh = self.ref_engine.sp_mesh
                self.ref_gather_mesh = self.ref_engine.gather_mesh
                self.ref_train_mesh = self.ref_engine.train_mesh
                self.ref_gather_manager = DataGatherManager(self.ref_gather_mesh, self.ref_sp_mesh)
            elif self.ref_strategy == 'megatron':
                from mono_rl.worker.engine.megatron.model import MegatronModel
                ref_mono_config = ref_config_to_mono_config(self.config.ref, self.config.model)
                self.ref_engine = MegatronModel(ref_mono_config.engine)
                # we assume that ref shares the same device mesh
                self.ref_gather_manager = self.actor_gather_manager

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

        self._model_initialized = False

        self.binding_timer = Timer(logger=None)

    def _get_actor_mono_config(self):
        actor_mono_config = actor_config_to_mono_config(self.config.actor, self.config.model)
        if not self._is_actor:
            actor_mono_config.engine.fsdp.param_offload = True  # Set param offload to True for rollout in mono config
            actor_mono_config.engine.fsdp.model_type = "bf16"
        else:
            actor_mono_config.engine.fsdp.optim_offload = True
        if self.enable_actor_critic_spatial_mux:
            actor_mono_config.engine.fsdp.spatial_mux_type = "first_half"
        # set use_rmpad and use_ce_loss_fusion
        actor_mono_config.engine.model.use_rmpad = self.config.model.get('use_rmpad', True)
        actor_mono_config.engine.model.use_ce_loss_fusion = self.config.model.get('use_ce_loss_fusion', False)
        return actor_mono_config

    def _get_ref_mono_config(self):
        ref_mono_config = actor_config_to_mono_config(self.config.ref, self.config.model)
        ref_mono_config.engine.fsdp.model_type = "bf16"
        ref_mono_config.engine.model.use_rmpad = self.config.model.get('use_rmpad', True)
        ref_mono_config.engine.model.use_ce_loss_fusion = self.config.model.get('use_ce_loss_fusion', False)
        return ref_mono_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_rollout_callback_function(self, eos_callback_fn):
        self.rollout.set_rollout_callback_function(eos_callback_fn=eos_callback_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_master_addr(self):
        key = "standalone_master_addr" if self._is_standalone_rollout else "hybrid_master_addr"
        out = DataProto.from_dict(tensors={'mock': torch.tensor([[0]])}, meta_info={key: self.master_address})
        return out

    def _get_actor_model_config_mariana(self, model_path):
        from verl.utils.fs import copy_local_path_from_hdfs
        # TODO: add local_path, tokenizer, processer into monorl mariana
        local_path = copy_local_path_from_hdfs(model_path)
        actor_model_config = AutoConfig.from_pretrained(local_path)
        return actor_model_config, local_path

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def delete_local_tmp_folder_safetensors_files(self):
        if int(os.getenv("RAY_LOCAL_RANK", "0")) != 0:
            return

        # get local tmp folder from actor model config
        folder_path = self.actor_model_config._name_or_path
        if not os.path.isdir(folder_path):
            return

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.safetensors'):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f'failed to remove safetensors file {file_path}, exception {e} will be ignored')

    def _build_rollout(self):
        assert self.config.rollout.name == 'xperf_gpt'

        import xperf_gpt
        xperf_gpt.load_xperf_gpt()

        from alpha_seed.workers.streaming_service.streaming_rollout import AsyncXPerfGPTRollout
        from alpha_seed.workers.hybrid_engine import FSDPXPerfGPTShardingManager, MegatronXPerfGPTShardingManager

        log_gpu_memory_usage('Before AsyncXPerfGPTRollout init')

        # actually, we just need hf_config in order to build rollout
        rollout = AsyncXPerfGPTRollout(config=self.config.rollout, role=self.role)
        rollout.initialize(local_path=self.local_path,
                           is_standalone=self._is_standalone_rollout or self._is_standalone_validator)
        rollout.setup_rollout()
        log_gpu_memory_usage('After AsyncXPerfGPTRollout init')

        # Note that in standalone case, model is None.
        weights_communicator = self.config.rollout.weights_communicator
        if self.actor_strategy in ('fsdp', 'vescale-fsdp2'):
            sharding_manager = FSDPXPerfGPTShardingManager(
                module=self.actor_module_fsdp,
                model_config=self.actor_model_config,
                inference_engine=rollout.inference_engine,
                device_mesh=rollout.device_mesh,
                standalone=self._is_standalone_rollout or self._is_standalone_validator,
                only_bind_once=self.role == "rollout",
                backend='fsdp',
                weights_communicator=weights_communicator,
                enable_actor_critic_spatial_mux=self.enable_actor_critic_spatial_mux)
        elif self.actor_strategy == 'megatron':
            sharding_manager = MegatronXPerfGPTShardingManager(module=self.actor_module_mariana,
                                                               model_config=self.actor_model_config,
                                                               inference_engine=rollout.inference_engine,
                                                               device_mesh=rollout.device_mesh,
                                                               standalone=self._is_standalone_rollout or
                                                               self._is_standalone_validator,
                                                               only_bind_once=self.role == "rollout",
                                                               backend='megatron')
        else:
            raise NotImplementedError

        sharding_manager.release_param_and_cache()
        log_gpu_memory_usage('After AsyncXPerfGPTRollout release parameter and kv cache')
        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device: str, model: bool = True, optimizer: bool = True):
        assert device in ("cuda", "cpu")
        if device == "cuda":
            device = torch.cuda.current_device()
            if self._is_actor:
                if self.actor_strategy == 'fsdp':
                    if not self.config.actor.fsdp_config.param_offload:
                        if model and self.actor_module_fsdp:
                            load_fsdp_model_to_gpu(self.actor_module_fsdp)
                        if optimizer and self.actor_optimizer is not None:
                            load_fsdp_optimizer(self.actor_optimizer, device)
                elif self.actor_strategy == 'vescale-fsdp2':
                    if not self.config.actor.fsdp_config.param_offload:
                        if model and self.actor_module_fsdp:
                            self.actor_module_fsdp.to('cuda')
                        if optimizer and self.actor_optimizer is not None:
                            load_fsdp_optimizer(self.actor_optimizer, device)
                elif self.actor_strategy == 'megatron':
                    assert model
                    # we only load grad when we want to load optimizer for training
                    load_megatron_model_to_gpu(models=self.actor_module_mariana, load_grad=optimizer)

            if self._is_ref:
                if self.ref_strategy == 'fsdp':
                    if model and not self.config.ref.fsdp_config.param_offload:
                        load_fsdp_model_to_gpu(self.ref_module_fsdp)
                elif self.ref_strategy == 'vescale-fsdp2':
                    if model and not self.config.ref.fsdp_config.param_offload:
                        self.ref_module_fsdp.to('cuda')
                elif self.ref_strategy == 'megatron':
                    if model:
                        # we never load grad for ref model
                        load_megatron_model_to_gpu(self.ref_module_mariana, load_grad=False)

        elif device == "cpu":
            if self._is_actor:
                if self.actor_strategy == 'fsdp':
                    if not self.config.actor.fsdp_config.param_offload:
                        if model and self.actor_module_fsdp:
                            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                        if optimizer and self.actor_optimizer is not None:
                            offload_fsdp_optimizer(self.actor_optimizer)
                elif self.actor_strategy == 'vescale-fsdp2':
                    if not self.config.actor.fsdp_config.param_offload:
                        if model and self.actor_module_fsdp:
                            self.actor_module_fsdp.to('cpu')
                        if optimizer and self.actor_optimizer is not None:
                            offload_fsdp_optimizer(self.actor_optimizer)
                elif self.actor_strategy == 'megatron':
                    if model:
                        offload_megatron_model_to_cpu(models=self.actor_module_mariana)

            if self._is_ref:
                if self.ref_strategy == 'fsdp':
                    if model and not self.config.ref.fsdp_config.param_offload:
                        offload_fsdp_model_to_cpu(self.ref_module_fsdp)
                elif self.ref_strategy == 'vescale-fsdp2':
                    if model and not self.config.ref.fsdp_config.param_offload:
                        self.ref_module_fsdp.to('cpu')
                elif self.ref_strategy == 'megatron':
                    if model:
                        offload_megatron_model_to_cpu(models=self.ref_module_mariana)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self, remove_safetensors_after_init=False, from_scratch=True):
        if self._model_initialized:
            return
        with self.profiler_context:
            self._init_model(from_scratch)
        self._normalize_config()
        self._model_initialized = True
        if remove_safetensors_after_init:
            cleanup_local_tmp_folder_safetensors_files(self.actor_model_config._name_or_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def setup_as_server(self, ifname=None):
        return self.sharding_manager.weights_communicator.setup_as_server(ifname)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def setup_as_relay(self, ifname=None):
        return self.sharding_manager.weights_communicator.setup_as_server(ifname)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def setup_as_client(self, role, source_addresses, all_rollout_addresses):
        # source_addresses: length等于自己的world size，根据自己的rank一一对应一个address即可
        # all_rollout_addresses: length等于hybrid rollout的world size
        self.hybrid_rollout_addresses = all_rollout_addresses
        source_address = source_addresses[self.rank]
        self.sharding_manager.weights_communicator.setup_as_client(role, source_address)

    def _normalize_config(self):
        config = self.config
        world_size = torch.distributed.get_world_size()
        actor_tp_size = config.actor.tp_size
        ref_tp_size = config.ref.tp_size
        # normalize config
        if self._is_actor:
            if self.enable_actor_critic_spatial_mux:
                world_size = world_size // 2
            if self.actor_strategy in ('fsdp', 'vescale-fsdp2'):
                sp_size = config.actor.ulysses_sequence_parallel_size
                tp_size = 1 if self.actor_strategy == 'vescale-fsdp2' else actor_tp_size
                self.config.actor.ppo_mini_batch_size //= (world_size // sp_size // tp_size)
                self.config.actor.ppo_micro_batch_size //= (world_size // sp_size // tp_size)
            elif self.actor_strategy == 'megatron':
                # we import here to remove the mariana as necessary dependency
                dp_size = mpu.get_data_parallel_world_size()
                self.config.actor.ppo_mini_batch_size //= dp_size
                self.config.actor.ppo_micro_batch_size //= dp_size

        if self._is_ref:
            if self.ref_strategy in ('fsdp', 'vescale-fsdp2'):
                sp_size = config.ref.ulysses_sequence_parallel_size
                tp_size = 1 if self.actor_strategy == 'vescale-fsdp2' else ref_tp_size
                self.config.ref.log_prob_micro_batch_size //= (world_size // sp_size // tp_size)
                self.config.ref.ppo_mini_batch_size //= (world_size // sp_size // tp_size)
            elif self.ref_strategy == 'megatron':
                dp_size = mpu.get_data_parallel_world_size()
                self.config.ref.log_prob_micro_batch_size //= dp_size

    def _init_model(self, from_scratch):
        # This is used to import external_lib into the huggingface systems
        log_gpu_memory_usage("Before actor initialized")

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        self.actor_module_fsdp, self.actor_module_mariana, self.actor_optimizer, self.actor_lr_scheduler = None, None, None, None
        self.ref_module_fsdp, self.ref_module_mariana = None, None

        # Setup attrs that every role needs, including actor_model_config, local_path, tokenizer, processor
        # For actor, rolllout, and ref_policy, some of the four attributes might be overridden later in this function
        if self.actor_strategy in ('fsdp', 'vescale-fsdp2'):
            from mono_rl.worker.engine.fsdp.utils import get_hf_model_config
            self.actor_model_config, self.local_path = get_hf_model_config(self.config.model.path,
                                                                           override_model_config,
                                                                           from_scratch=from_scratch)
        elif self.actor_strategy == 'megatron':
            self.actor_model_config, self.local_path = self._get_actor_model_config_mariana(self.config.model.path)

        trust_remote_code = self.config.model.get('trust_remote_code', False)

        tokenizer_path = os.path.join(self.local_path, 'tokenizer')
        if not os.path.exists(tokenizer_path):
            tokenizer_path = self.local_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # NOTE: the following for role specific attributes over-writing
        if self._is_actor and not self._is_valid_actor or self._is_standalone_rollout or self._is_standalone_validator:
            pass  # NOTE: no need to set extra fields for standalone rollout and validators.
        elif self._is_valid_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self.actor_strategy in ('fsdp', 'vescale-fsdp2'):
                self.actor_engine.init_model(from_scratch=from_scratch, build_optimizer=self._is_actor)
                self.actor = DataParallelPPOActor(as_config=self.config.actor, model_engine=self.actor_engine)
                self.actor_module_fsdp, self.actor_model_config = self.actor_engine.model_module, self.actor_engine.model_config
            elif self.actor_strategy == 'megatron':
                self.actor_engine.init_model(from_scratch=from_scratch)
                self.actor = MegatronPPOActor(as_config=self.config.actor, model_engine=self.actor_engine)
                self.actor_module_mariana, self.actor_model_config = self.actor_engine.model_module, self.actor_engine.actor_model_config
            if self._is_actor:
                self.actor_optimizer, self.actor_lr_scheduler = self.actor_engine.optimizer, self.actor_engine.lr_scheduler
            self.image_manager = self.actor_engine.image_manager

        if self._is_ref:
            from_scratch_ref = True if self.config.ref.ema == 1 else from_scratch
            if self.ref_strategy in ('fsdp', 'vescale-fsdp2'):
                self.ref_engine.init_model(from_scratch=from_scratch_ref, build_optimizer=False)
                self.ref_policy = DataParallelPPOActor(as_config=self.config.ref, model_engine=self.ref_engine)
                self.ref_module_fsdp = self.ref_engine.model_module
            elif self.ref_strategy == 'megatron':
                self.ref_engine.init_model(from_scratch=from_scratch_ref)
                self.ref_policy = MegatronPPOActor(as_config=self.config.ref, model_engine=self.ref_engine)
                self.ref_module_mariana = self.ref_engine.model_module
                self.ref_gather_manager = self.actor_gather_manager

        if self.config.actor.train_memory_offload:
            self.to("cpu")
        log_gpu_memory_usage("After actor initialized")

        if self._is_rollout or self._is_standalone_rollout or self._is_standalone_validator:
            self.rollout, self.sharding_manager = self._build_rollout()
            self.rollout_async = None

        if self._is_valid_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            actor_model_to_save = self.actor_module_fsdp if self.actor_strategy != 'megatron' else self.actor_module_mariana
            self.checkpoint_manager = CheckpointManagerWrapper(strategy=self.actor_strategy,
                                                               model=actor_model_to_save,
                                                               optimizer=self.actor_optimizer,
                                                               lr_scheduler=self.actor_lr_scheduler,
                                                               hf_config=self.actor_model_config,
                                                               tokenizer=self.tokenizer,
                                                               device_mesh=self.actor_train_mesh,
                                                               processor=self.processor)

        if self._is_ref:
            ref_model_to_save = self.ref_module_fsdp if self.ref_strategy != 'megatron' else self.ref_module_mariana
            self.checkpoint_manager_ref = CheckpointManagerWrapper(
                strategy=self.ref_strategy,
                model=ref_model_to_save,
                optimizer=None,
                lr_scheduler=None,
                hf_config=self.actor_model_config,  # same for actor and ref
                tokenizer=self.tokenizer,
                device_mesh=self.ref_train_mesh,
                processor=self.processor)

        ndtimeline.init_with_ray(self)
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_eos_callback_fn(self, eos_callback_fn):
        self.rollout.set_rollout_callback_function(eos_callback_fn=eos_callback_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def update_standalone_worker(self, role):
        # Note(zhangchi.usc1992)
        # self.sharding_manager.standalone indicates that it is a standalone_rollout or standalone_validator
        # if it is streaming rollout, the weight is latest, so no need to bind weight again.

        log_gpu_memory_usage(f'Before update_standalone_worker {role=}')
        assert self._is_rollout or self._is_standalone_rollout or self._is_standalone_validator
        if not self.sharding_manager.standalone and self.config.actor.train_memory_offload:
            # 把hybrid rollout的参数从cpu->cuda
            self.to("cuda", model=True, optimizer=False)
        # sharding_manager.__enter__ 会把 FSDP 的weights格式转换到megatron的格式
        # 然后才做下面的收发，发送之后就不用在standalone rollout里转

        # TODO(zhangchi.usc1992): we have a redundant weight binding here for standalone validator
        # Try to remove it by introduing an argument
        with self.sharding_manager:
            # hybrid rollout send
            # standalone rollout recv
            # 总共收发 standalone world_size 次
            self.sharding_manager.weights_communicator.update_standalone_worker(role)
        if not self.sharding_manager.standalone and self.config.actor.train_memory_offload:
            # 再把hybrid rollout的参数卸载回cpu
            self.to("cpu", model=True, optimizer=False)

        log_gpu_memory_usage(f'After update_standalone_worker {role=}')

    @register(execute_mode=Execute.RANK_ZERO, blocking=True)
    def update_standalone_worker_end(self):
        # this function should only be invoked by standalone worker
        assert self._is_rollout or self._is_standalone_rollout or self._is_standalone_validator
        # 通知所有actor server退出weights transfer
        self.sharding_manager.weights_communicator.update_standalone_worker_end(self.hybrid_rollout_addresses)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=True)
    def load_and_transform_save_image(self, prompts: DataProto):
        prompts = load_and_transform_save_image(prompts,
                                                self.tokenizer,
                                                self.processor,
                                                self.image_manager,
                                                max_prompt_length=self.config.rollout.prompt_length,
                                                truncation=self.config.rollout.vlm.truncation)
        return prompts

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    def update_actor(self, data: DataProto):
        torch.cuda.reset_peak_memory_stats()
        data = data.to('cpu')

        if not self._is_valid_actor:
            return DataProto()

        assert self._is_actor
        # data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy')

        # note optimizer offload will be managed inside `update_policy`
        if self.config.actor.train_memory_offload:
            self.to("cuda", model=True, optimizer=False if self.actor_strategy == 'fsdp' else True)

        # add necessary meta_info keys to the data proto
        data.meta_info['role'] = Role.Actor
        data.meta_info["response_length"] = data.batch["responses"].shape[1]
        data.meta_info['compute_entropy'] = (self.config.actor.entropy_coeff > 0)

        with self.actor_gather_manager:
            data = self.actor_gather_manager.preprocess_data(data)

            with Timer(name='update_policy', logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            kwargs = {}
            if 'global_img_token_num' in data.meta_info:
                kwargs['images_seqlens'] = data.meta_info['global_img_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time, **kwargs)
            world_size = self.world_size // 2 if self.enable_actor_critic_spatial_mux else self.world_size
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / world_size

            data = self.actor_gather_manager.postprocess_data(data)

        if self.actor_strategy in ('fsdp', 'vescale-fsdp2'):
            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
        elif self.actor_strategy == 'megatron':
            self.actor_lr_scheduler[0].step(1)
            lr = self.actor_lr_scheduler[0].get_lr()
        metrics['actor/lr(1e-4)'] = lr * 1e4

        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        max_memory_allocated, max_memory_reserved = get_memory(group=self.actor_train_mesh.get_group())
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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_loss_fn(self, loss_fn):
        self.acotr_loss_fn = loss_fn
        self.actor.set_loss_fn(loss_fn)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def train_actor(self, data: DataProto):
        torch.cuda.reset_peak_memory_stats()
        assert self._is_actor
        log_gpu_memory_usage('Before update policy', logger=logger)

        # note optimizer offload will be managed inside `update_policy`
        if self.config.actor.train_memory_offload:
            self.to("cuda", model=True, optimizer=False)

        # add necessary meta_info keys to the data proto
        data.meta_info['role'] = Role.Actor
        data.meta_info["response_length"] = data.batch["responses"].shape[1]
        data.meta_info['compute_entropy'] = (self.config.actor.entropy_coeff > 0)

        with self.actor_gather_manager:
            data = self.actor_gather_manager.preprocess_data(data)

            with Timer(name='train_actor', logger=None) as timer:
                metrics = self.actor.train_one_step(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            kwargs = {}
            if 'global_img_token_num' in data.meta_info:
                kwargs['images_seqlens'] = data.meta_info['global_img_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time, **kwargs)
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            data = self.actor_gather_manager.postprocess_data(data)

        # lr schuelder step only once every global step
        if data.meta_info.get('lr_scheduler_step', False):
            if self.actor_strategy == 'fsdp':
                self.actor_lr_scheduler.step()
                lr = self.actor_lr_scheduler.get_last_lr()[0]
            elif self.actor_strategy == 'megatron':
                self.actor_lr_scheduler[0].step(1)
                lr = self.actor_lr_scheduler[0].get_lr()
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

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    def old_log_probs(self, prompts: DataProto):
        log_gpu_memory_usage('Before old_log_probs')
        prompts = prompts.to('cpu')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        if not self._is_valid_actor:
            output = prompts
            if recompute_log_prob:
                output.batch['old_log_probs'] = torch.empty_like(output.batch["responses"],
                                                                 device="cpu",
                                                                 dtype=torch.bfloat16)
                output.batch['old_entropy'] = torch.empty_like(output.batch["responses"],
                                                               device="cpu",
                                                               dtype=torch.bfloat16)
            return output

        output = prompts
        if self._is_actor and recompute_log_prob:

            if self.config.actor.train_memory_offload:
                self.to("cuda", model=True, optimizer=False)

            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['temperature'] = prompts.meta_info['generation_kwargs']['temperature']
            # align with the training config
            output.meta_info['use_dynamic_bsz'] = self.config.actor.use_dynamic_bsz
            if self.config.actor.use_dynamic_bsz:
                output.meta_info['micro_batch_tokens'] = self.config.actor.ppo_max_token_len
            else:
                output.meta_info['micro_batch_size'] = self.config.actor.ppo_micro_batch_size
            output.meta_info['response_length'] = output.batch["responses"].shape[1]
            output.meta_info['compute_entropy'] = True
            output.meta_info['role'] = Role.Actor

            with self.actor_gather_manager:
                reuse_old_experts = self.config.actor.reuse_old_experts
                output = self.actor_gather_manager.preprocess_data(output)
                old_entropy, old_log_probs, acceptance_matrix, old_experts = self.actor.compute_log_prob(
                    data=output, reuse_old_experts=reuse_old_experts)
                output.batch['old_log_probs'] = old_log_probs
                output.batch['old_entropy'] = old_entropy
                for j in range(len(acceptance_matrix)):
                    output.batch[f'acceptance_matrix_{j}'] = acceptance_matrix[j]
                if reuse_old_experts:
                    output.batch['old_experts'] = old_experts
                output = self.actor_gather_manager.postprocess_data(output)

            if self.config.actor.train_memory_offload:
                self.to("cpu", model=True, optimizer=False)

        output = output.to('cpu')
        # clear kv cache
        log_gpu_memory_usage('After recompute log prob')
        return output

    def _run_sequence(self, prompts: DataProto, mode: str = "rollout"):
        torch.cuda.reset_peak_memory_stats()
        prompts = prompts.to('cuda')
        assert self._is_rollout or self._is_standalone_validator
        prompts.batch = prompts.batch.cuda()
        log_gpu_memory_usage(f'Rollout[{mode}]: Before load training memory')
        # xperf needs parameters from actor
        if self.config.actor.train_memory_offload:
            self.to("cuda", model=True, optimizer=False)
        log_gpu_memory_usage(f'Rollout[{mode}]: Before entering sharding manager')
        self.binding_timer.start()
        with self.sharding_manager:
            binding_time = self.binding_timer.stop()
            log_gpu_memory_usage(f'Rollout[{mode}]: After entering sharding manager')
            # after parameters go to xperf, offload actor model to CPU
            if self.config.actor.train_memory_offload:
                self.to("cpu", model=True, optimizer=False)
            log_gpu_memory_usage(f'Rollout[{mode}]: After offload train parameters')
            prompts = self.sharding_manager.preprocess_data(prompts)
            generator = self.rollout.generate_sequences(prompts=prompts, mode=mode)
            output = next(generator)
            output = self.sharding_manager.postprocess_data(output)
            log_gpu_memory_usage(f'Rollout[{mode}]: After sequence computation')

        log_gpu_memory_usage(f'Rollout[{mode}]: After release kv cache')

        max_memory_allocated, max_memory_reserved = get_memory()
        output.meta_info.update({
            'memory/gen_max_allocated': max_memory_allocated,
            'memory/gen_max_reserved': max_memory_reserved,
            'timing/weight_binding': binding_time
        })
        output = output.to('cpu')
        # clear kv cache
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rollout_log_probs(self, prompts: DataProto):
        return self._run_sequence(prompts, mode="log_probs")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        return self._run_sequence(prompts, mode="rollout")

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
        prompts = prompts.to('cpu')
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
        log_gpu_memory_usage('Before compute_ref_log_prob')
        data = data.to('cpu')

        torch.cuda.reset_peak_memory_stats()
        assert self._is_ref

        data = data.to('cuda')

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['use_dynamic_bsz'] = self.config.ref.use_dynamic_bsz
        if self.config.ref.use_dynamic_bsz:
            data.meta_info['micro_batch_tokens'] = self.config.ref.max_token_len
        else:
            data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.train_generate_kwargs.temperature
        data.meta_info['response_length'] = data.batch["responses"].shape[1]
        data.meta_info['compute_entropy'] = True
        data.meta_info['role'] = Role.Ref

        log_gpu_memory_usage('Before reference recompute log prob', logger=logger)

        with self.ref_gather_manager:
            data = self.ref_gather_manager.preprocess_data(data)
            _, output, _, _ = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ref_gather_manager.postprocess_data(output)

        # reset FSDP buffer after forward
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(self.ref_policy.engine.model_module, FSDP):
            self.ref_policy.engine.model_module._handle.reshard(True)
        log_gpu_memory_usage('After reference recompute log prob')

        max_memory_allocated, max_memory_reserved = get_memory()
        output.meta_info.update({
            'memory/ref_max_allocated': max_memory_allocated,
            'memory/ref_max_reserved': max_memory_reserved
        })
        output = output.to('cpu')
        return output

    def _save_load_checkpoint_helper(self, version, model):
        """verify settings and prepare input params for saving and loading ckpt
        """
        device_mesh = None
        if model == 'actor':
            assert self._is_actor or self._is_rollout
            ckpt_manager = self.checkpoint_manager
            parallel_strategy = self.actor_strategy
            if hasattr(self, 'actor_fsdp_mesh'):
                device_mesh = self.actor_fsdp_mesh
        elif model == 'ref':
            assert self._is_ref
            ckpt_manager = self.checkpoint_manager_ref
            parallel_strategy = self.ref_strategy
            if hasattr(self, 'ref_fsdp_mesh'):
                device_mesh = self.ref_fsdp_mesh
        else:
            raise ValueError(f'Unknown {model=}')

        if parallel_strategy == 'megatron':
            if version != 'omnistore':
                raise NotImplementedError('Only the OmniStore ckpt manager supports megatron strategy currently')
        elif parallel_strategy not in ('fsdp', 'vescale-fsdp2'):
            raise NotImplementedError(f'Saving / loading ckpt is not supported for strategy {parallel_strategy}')
        return ckpt_manager, parallel_strategy, device_mesh

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, hdfs_path=None, version='v1', enable_shm=False, model='actor'):
        if model == "actor" and not self._is_valid_actor:
            return

        ckpt_manager, parallel_strategy, device_mesh = self._save_load_checkpoint_helper(version, model)

        if self.config.actor.train_memory_offload:
            self.to("cuda")
        ckpt_manager.load_checkpoint(version=version,
                                     hdfs_path=hdfs_path,
                                     device_mesh=device_mesh,
                                     role=model,
                                     strategy=parallel_strategy,
                                     enable_shm=enable_shm)
        if self.config.actor.train_memory_offload:
            self.to("cpu")
        torch.cuda.empty_cache()
        log_gpu_memory_usage(f'After loading checkpoint of {model}')

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def save_checkpoint(self,
                        local_path,
                        hdfs_path=None,
                        version='v1',
                        global_step=0,
                        ckpt_global_uploader_ref=None,
                        enable_shm=False,
                        model='actor'):
        if model == "actor" and not self._is_valid_actor:
            return

        ckpt_manager, parallel_strategy, device_mesh = self._save_load_checkpoint_helper(version, model)

        if self.config.actor.train_memory_offload:
            self.to("cuda")
        ckpt_manager.save_checkpoint(version=version,
                                     local_path=local_path,
                                     hdfs_path=hdfs_path,
                                     device_mesh=device_mesh,
                                     role=model,
                                     strategy=parallel_strategy,
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
    def do_ndtimeline_action(self, action, *args, **kwargs):
        if self.actor_strategy in ['megatron']:
            # TODO(fix me)
            return
        ndtimeline.do_ndtimeline_action(action, *args, **kwargs)

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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def add_inflight_queries(self, queries: List[Query]):
        ret = []
        queries = add_pixel_values_to_inflight_query(queries, self.image_manager)
        for q in queries:
            qid = self.rollout.add_inflight_query(q)
            ret.append(qid)
        return ret

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def abort_queries(self, query_ids: List[str], not_after: float):
        self.rollout.abort_queries(query_ids, not_after)

    # 只在dp_size=1的情况下调用，所以这里rank0执行即可
    @register(execute_mode=Execute.RANK_ZERO, blocking=True)
    def get_history_ids(self):
        return self.rollout.get_valid_history_ids()

    # 只在dp_size=1的情况下调用，所以这里rank0执行即可，DP_COMPUTE与此参数暂不兼容
    @register(execute_mode=Execute.RANK_ZERO, blocking=True)
    def get_all_queries(self, query_type: str):
        return self.rollout.get_all_queries(query_type)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def release_running_queries_and_return_metrics(self):
        metrics = self.rollout.inference_engine.infer_scheduler.metrics
        visualize_metrics(metrics)
        self.rollout.inference_engine.empty_cache()
        return metrics

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=True)
    def get_load_metrics(self) -> LoadMetric:
        return self.rollout.get_load_metrics()

    @register(execute_mode=Execute.RANK_ZERO)
    def get_master_free_port(self) -> int:
        return get_free_port_for_nccl_primitive()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def setup_standalone_worker_comm(self, hybrid_master_address, standalone_master_address, port, role):
        self.sharding_manager.weights_communicator.setup_standalone_worker_comm(hybrid_master_address,
                                                                                standalone_master_address, port, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def toggle_inference_server_state(self, sleep):
        """
            Toggle the inference server state between running and sleeping modes.

            This method **must be used in a paired fashion**:
                - Call with `sleep=False` to enable the inference engine.
                - Later, call with `sleep=True` to properly disable and offload resources.

            Unpaired usage will lead to inconsistent engine state, potential deadlocks, or GPU memory inconsistency.
        """
        if self.config.rollout.mode == "batch":
            # If running in batch mode, toggling is unnecessary.
            return
        if sleep:
            # 让engine停下来
            while (self.rollout.gen_loop_exited.is_set()):
                time.sleep(1)
            with self.rollout.inference_engine.update_weights_lock:
                self.rollout.stop_event.set()
            # Clear the "weights loaded" flag, so engine won't proceed until reloaded.
            self.rollout.weights_loaded.clear()
            # Make sure the engine is fully stopped before offloading the
            self.rollout.gen_loop_exited.wait()
            # offload weights
            self.sharding_manager.__exit__(None, None, None)
            return
        assert (self.rollout.inference_engine.status == "idle")
        if self.config.actor.train_memory_offload:
            self.to("cuda", model=True, optimizer=False)
        self.sharding_manager.__enter__()
        with self.rollout.inference_engine.update_weights_lock:
            self.rollout.stop_event.clear()
            # 通知 engine weights loaded
            self.rollout.weights_loaded.set()


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
