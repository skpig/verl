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

import gc
import warnings
import os
import logging
import ray
import torch
import torch.distributed

from mono_rl.single_controller import Worker
from mono_rl.single_controller import register, Dispatch
from mono_rl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.torch_functional import get_constant_schedule_with_warmup
from mono_rl.models.seed_models.parallel.collectives import get_memory
from alpha_seed.workers.fsdp.initialize import cleanup_local_tmp_folder_safetensors_files
from alpha_seed.workers.fsdp.offload import offload_fsdp_optimizer, load_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_model_to_gpu
from alpha_seed.workers.megatron.offload import offload_megatron_model_to_cpu, load_megatron_model_to_gpu
from alpha_seed.workers.hybrid_engine.fsdp_gather import DataGatherManager
from alpha_seed.workers.ppo_critic import DataParallelPPOCritic
from alpha_seed.utils import ndtimeline

from seed_models.utils.count_flops import FlopsCounter
from transformers import AutoTokenizer, AutoProcessor

from codetiming import Timer

from datetime import timedelta

from .checkpoint import CheckpointManagerWrapper
from tensordict import TensorDict
from alpha_seed.utils.mono_rl.config import critic_config_to_mono_config
from mono_rl.worker.engine.fsdp.models.model import FSDPModel
from mono_rl.worker import Role

# mariana dependency
try:
    from megatron.core import parallel_state as mpu
    from alpha_seed.workers.ppo_critic_megatron import MegatronPPOCritic
except:
    pass

logger = logging.getLogger(__file__)


@ray.remote
class CriticWorker(Worker):

    def __init__(self, config, enable_actor_critic_spatial_mux=False):
        super().__init__()

        warnings.simplefilter(action='ignore', category=FutureWarning)

        import torch.distributed
        if not torch.distributed.is_initialized():
            timeout = timedelta(seconds=int(os.getenv('NCCL_TIMEOUT', 3600)))
            torch.distributed.init_process_group(backend="nccl", timeout=timeout)
        self.config = config
        self.role = "critic"

        self.critic_strategy = config.strategy
        self.enable_actor_critic_spatial_mux = enable_actor_critic_spatial_mux

        self._is_valid_critic = True
        if self.enable_actor_critic_spatial_mux:
            critic_world_size = torch.distributed.get_world_size() // 2
            if self.rank < critic_world_size:
                self._is_valid_critic = False

        assert self.critic_strategy in ['fsdp', 'megatron', 'vescale-fsdp2']

        world_size = torch.distributed.get_world_size()
        if self.enable_actor_critic_spatial_mux:
            world_size = world_size // 2

        # normalize config
        if self.critic_strategy in ('fsdp', 'vescale-fsdp2'):
            mono_config = critic_config_to_mono_config(self.config)
            mono_config.engine.fsdp.optim_offload = True
            if self.enable_actor_critic_spatial_mux:
                mono_config.engine.fsdp.spatial_mux_type = "last_half"
            self.critic_engine = FSDPModel(mono_config.engine)
            self.fsdp_mesh = self.critic_engine.fsdp_mesh
            self.tp_mesh = self.critic_engine.tp_mesh
            self.oe_mesh = self.critic_engine.oe_mesh
            self.sp_mesh = self.critic_engine.sp_mesh
            self.gather_mesh = self.critic_engine.gather_mesh
            self.train_mesh = self.critic_engine.train_mesh
            if self._is_valid_critic:
                self.gather_manager = DataGatherManager(self.gather_mesh, self.sp_mesh)
            # normalize config here
            sp_size = config.ulysses_sequence_parallel_size
            tp_size = config.tp_size
            tp_size = 1 if self.critic_strategy == 'vescale-fsdp2' else tp_size
            self.config.ppo_mini_batch_size //= (world_size // sp_size // tp_size)
            self.config.ppo_micro_batch_size //= (world_size // sp_size // tp_size)
        elif self.critic_strategy == 'megatron':
            from mono_rl.worker.engine.megatron.model import MegatronModel
            mono_config = critic_config_to_mono_config(self.config)
            # create and init monorl fsdp model engine
            self.critic_engine = MegatronModel(mono_config.engine)
            # get the device meshes from monorl megatron model engine and construct the data gather manager
            from alpha_seed.workers.hybrid_engine.megatron_gather import MegatronDataGatherManager
            self.gather_manager = MegatronDataGatherManager()

        self._model_initialized = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device: str, model=True, optimizer=True, model_empty_cache=True):
        assert device in ("cuda", "cpu")
        if self.critic_strategy == 'fsdp':
            if self.config.model.fsdp_config.param_offload:
                return
            if device == "cuda":
                if model and self.critic_module:
                    load_fsdp_model_to_gpu(self.critic_module)
                if optimizer and self.critic_optimizer:
                    load_fsdp_optimizer(self.critic_optimizer, torch.cuda.current_device())
                gc.collect()
            elif device == "cpu":
                if model and self.critic_module:
                    offload_fsdp_model_to_cpu(self.critic_module, model_empty_cache)
                if optimizer and self.critic_optimizer:
                    offload_fsdp_optimizer(self.critic_optimizer)
        elif self.critic_strategy == 'vescale-fsdp2':
            if self.config.model.fsdp_config.param_offload:
                return
            if device == 'cuda':
                if model and self.critic_module:
                    self.critic_module.to(torch.cuda.current_device(), non_blocking=True)
                if optimizer and self.critic_optimizer:
                    load_fsdp_optimizer(self.critic_optimizer)
            elif device == "cpu":
                if model and self.critic_module:
                    self.critic_module.to('cpu', non_blocking=True)
                if optimizer and self.critic_optimizer:
                    offload_fsdp_optimizer(self.critic_optimizer)
        elif self.critic_strategy == 'megatron':
            if device == 'cuda':
                load_megatron_model_to_gpu(models=self.critic_module, load_grad=optimizer)
                gc.collect()
            elif device == 'cpu':
                offload_megatron_model_to_cpu(models=self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self, remove_safetensors_after_init=False, from_scratch=True):
        if self._model_initialized:
            return

        self.critic_module, self.critic_optimizer, self.critic_model_config = None, None, None

        if self.critic_strategy in ('fsdp', 'vescale-fsdp2') and self._is_valid_critic:
            self.critic_engine.init_model(from_scratch=from_scratch, build_optimizer=True)
            # TODO: update lr_scheduler
            self.critic_module, self.critic_optimizer, self.critic_model_config = self.critic_engine.model_module, self.critic_engine.optimizer, self.critic_engine.model_config
            self.critic = DataParallelPPOCritic(as_config=self.config, model_engine=self.critic_engine)
        elif self.critic_strategy == 'megatron':
            # FIXME: the MegatronPPoCritic API is not aligned yet, need fix in both alpha_seed and mono_rl
            self.critic_engine.init_model(from_scratch=from_scratch)
            self.critic = MegatronPPOCritic(as_config=self.config, model_engine=self.critic_engine)
            # TODO: Fix the assignment below for megatron
            self.critic_module, self.critic_model_config = self.critic_engine.model_module, self.critic_engine.model_config

        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.
        # TODO: move this part to FSDP engine
        trust_remote_code = self.config.model.get('trust_remote_code', False)
        if self._is_valid_critic:
            tokenizer_path = os.path.join(self.critic_engine.local_path, 'tokenizer')
            if not os.path.exists(tokenizer_path):
                tokenizer_path = self.critic_engine.local_path
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
            self.processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)

        # update critic lr scheduler as the customized scheduler with warmup steps
        total_steps = self.config.optim.get('total_training_steps', 0)
        num_warmup_steps = int(self.config.optim.get('lr_warmup_steps', -1))
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = self.config.optim.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
        if self.rank == 0:
            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        if self._is_valid_critic:
            self.critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.critic_optimizer,
                                                                         num_warmup_steps=num_warmup_steps)
            self.flops_counter = FlopsCounter(self.critic_model_config)
            if self.rank == 0:
                print(self.critic_model_config)
            self.checkpoint_manager = CheckpointManagerWrapper(strategy=self.critic_strategy,
                                                               model=self.critic_module,
                                                               optimizer=self.critic_optimizer,
                                                               lr_scheduler=self.critic_lr_scheduler,
                                                               hf_config=self.critic_model_config,
                                                               tokenizer=self.tokenizer,
                                                               device_mesh=self.train_mesh,
                                                               processor=self.processor)

        if self.config.train_memory_offload:
            self.to("cpu")
        torch.cuda.empty_cache()
        ndtimeline.init_with_ray(self)
        self._model_initialized = True
        if remove_safetensors_after_init and self._is_valid_critic:
            cleanup_local_tmp_folder_safetensors_files(self.critic_model_config._name_or_path)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    def compute_values(self, data: DataProto):
        data = data.to('cpu')

        if not self._is_valid_critic:
            output = DataProto.from_dict(
                tensors={'values': torch.empty_like(data.batch["responses"], device="cpu", dtype=torch.bfloat16)})
            return output

        # Note we don't offload to cpu after compute_values
        # as it next will update critic
        if self.config.train_memory_offload:
            self.to("cuda", model=True, optimizer=False)

        data.meta_info["role"] = Role.Critic
        data.meta_info['response_length'] = data.batch["responses"].shape[1]
        micro_batch_size = self.config.infer_micro_batch_size
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        data.meta_info['compute_entropy'] = False

        if self.config.use_dynamic_bsz:
            data.meta_info['micro_batch_tokens'] = self.config.get('infer_ppo_max_token_len',
                                                                   self.config.ppo_max_token_len)
        else:
            data.meta_info['micro_batch_size'] = micro_batch_size
        with self.gather_manager:
            data = self.gather_manager.preprocess_data(data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.gather_manager.postprocess_data(output)
        output = output.to('cpu')

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    def update_critic(self, data: DataProto):
        torch.cuda.reset_peak_memory_stats()
        log_gpu_memory_usage('Before Critic update')
        data = data.to('cpu')

        if self._is_valid_critic:
            # optimizer will be loaded just before the step to save
            # forward & backward memory
            if self.config.train_memory_offload:
                self.to("cuda",
                        model=True,
                        optimizer=False if self.critic_strategy in ('fsdp', 'vescale-fsdp2') else True)

            data.meta_info["role"] = Role.Critic
            data.meta_info['response_length'] = data.batch["responses"].shape[1]
            data.meta_info['compute_entropy'] = False

            with self.gather_manager:
                data = self.gather_manager.preprocess_data(data)

                with Timer(name='update_critic', logger=None) as timer:
                    seq_vf, metrics = self.critic.update_critic(data=data)
                delta_time = timer.last

                global_num_tokens = data.meta_info['global_token_num']
                kwargs = {}
                if 'global_img_token_num' in data.meta_info:
                    kwargs['images_seqlens'] = data.meta_info['global_img_token_num']
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time,
                                                                                    **kwargs)
                world_size = self.world_size // 2 if self.enable_actor_critic_spatial_mux else self.world_size
                metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / world_size

                if self.critic_strategy in ('fsdp', 'vescale-fsdp2'):
                    self.critic_lr_scheduler.step()
                    lr = self.critic_lr_scheduler.get_last_lr()[0]
                elif self.critic_strategy == 'megatron':
                    self.critic_lr_scheduler[0].step(1)
                    lr = self.critic_lr_scheduler[0].get_lr()

                metrics['critic/lr(1e-4)'] = lr * 1e4

                max_memory_allocated, max_memory_reserved = get_memory(group=self.train_mesh.get_group())
                output = DataProto(batch=TensorDict(source={'seq_vf': seq_vf}, batch_size=(seq_vf.shape[0],)),
                                   meta_info={
                                       'metrics': metrics,
                                       'memory/critic_max_allocated': max_memory_allocated,
                                       'memory/critic_max_reserved': max_memory_reserved
                                   })
                output = self.gather_manager.postprocess_data(output)

            if self.config.train_memory_offload:
                self.to("cpu", model_empty_cache=False)
            output = output.to('cpu')

        if self.enable_actor_critic_spatial_mux:
            gather_obj = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gather_obj, output.meta_info if self._is_valid_critic else {})

            if not self._is_valid_critic:
                batch_size = data.batch["responses"].shape[0]
                output = DataProto(batch=TensorDict(
                    source={'seq_vf': torch.empty((batch_size), device="cpu", dtype=torch.float)},
                    batch_size=batch_size),
                                   meta_info=gather_obj[self.train_mesh.size()])
                return output

        log_gpu_memory_usage('After Critic update')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, hdfs_path=None, version='v1', enable_shm=False):
        if not self._is_valid_critic:
            return

        if self.config.train_memory_offload:
            self.to("cuda")
        fsdp_mesh = self.fsdp_mesh if self.critic_strategy in ('fsdp', 'vescale-fsdp2') else None
        self.checkpoint_manager.load_checkpoint(version=version,
                                                hdfs_path=hdfs_path,
                                                device_mesh=fsdp_mesh,
                                                role='critic',
                                                strategy=self.critic_strategy,
                                                enable_shm=enable_shm)
        if self.config.train_memory_offload:
            self.to("cpu")
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After loading checkpoint of critic')

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def save_checkpoint(self,
                        local_path,
                        hdfs_path=None,
                        version='v1',
                        global_step=0,
                        ckpt_global_uploader_ref=None,
                        enable_shm=False):
        if not self._is_valid_critic:
            return

        if self.config.train_memory_offload:
            self.to("cuda")
        self.checkpoint_manager.save_checkpoint(
            version=version,
            local_path=local_path,
            hdfs_path=hdfs_path,
            device_mesh=self.fsdp_mesh if self.critic_strategy in ('fsdp', 'vescale-fsdp2') else None,
            role='critic',
            strategy=self.critic_strategy,
            global_step=global_step,
            ckpt_global_uploader_ref=ckpt_global_uploader_ref,
            enable_shm=enable_shm)
        if self.config.train_memory_offload:
            self.to("cpu")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def do_ndtimeline_action(self, action, *args, **kwargs):
        ndtimeline.do_ndtimeline_action(action, *args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reinit(self, config):
        import gc
        if self._model_initialized:
            del self.critic_module
            del self.critic_optimizer
            self._model_initialized = False
        gc.collect()
        torch.cuda.empty_cache()
        self.__init__(config)
