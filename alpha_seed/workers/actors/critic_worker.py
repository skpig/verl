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

import json
import gc
from filelock import FileLock
import shutil
import warnings
import os
import logging
import hdfs_io
import ray
import torch
import torch.distributed

import verl.utils.torch_functional as verl_F
from single_controller.base import Worker
from single_controller.base.decorator import register, Dispatch
from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from alpha_seed.models.transformers.parallel import apply_parallel_plan
from alpha_seed.models.transformers.parallel.collectives import get_memory
from .initialize import create_mesh
from .initialize import parallel_init_fsdp_fn, parallel_load_safetensors, meta_device_init, cleanup_local_tmp_folder_safetensors_files
from .checkpoint.extensions import register_dtensor_save_hook
from .offload import (offload_fsdp_model_to_cpu, load_fsdp_model_to_gpu, offload_megatron_model_to_cpu,
                      load_megatron_model_to_gpu)
from alpha_seed.workers.actors.offload import offload_fsdp_optimizer, load_fsdp_optimizer
from verl.utils.import_utils import import_external_libs
from verl.utils.debug import log_gpu_memory_usage

from alpha_seed.workers.hybrid_engine.fsdp_gather import DataGatherManager
from alpha_seed.workers.ppo_critic import DataParallelPPOCritic
from alpha_seed.utils import ndtimeline
from alpha_seed.utils.ckpt import download_minimal_required_files

from seed_models.utils.count_flops import FlopsCounter

from codetiming import Timer

from datetime import timedelta

from .checkpoint import CheckpointManagerWrapper

logger = logging.getLogger(__file__)


@ray.remote
class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()

        warnings.simplefilter(action='ignore', category=FutureWarning)

        import torch.distributed
        if not torch.distributed.is_initialized():
            timeout = timedelta(minutes=int(os.getenv('NCCL_TIMEOUT', 60)))
            torch.distributed.init_process_group(backend="nccl", timeout=timeout)

        self.config = config
        self.role = "critic"

        self.critic_strategy = config.strategy

        assert self.critic_strategy in ['fsdp', 'megatron']

        world_size = torch.distributed.get_world_size()

        if self.critic_strategy == 'fsdp':
            fsdp_size = config.fsdp_size
            sp_size = config.ulysses_sequence_parallel_size
            tp_size = config.tp_size
            meshes = create_mesh(fsdp_size=fsdp_size, tp_size=tp_size, sp_size=sp_size, tp_outside=config.tp_outside)
            # Deprecated case: critic model is saved as ShardedTensor
            # we will always use full FSDP
            self.fsdp_mesh = None
            if not config.NO_DEVICE_MESH:
                self.fsdp_mesh = meshes[0]
            self.tp_mesh = meshes[1]
            self.sp_mesh = meshes[2]
            self.gather_mesh = meshes[3]
            self.gather_manager = DataGatherManager(self.gather_mesh, self.sp_mesh)

            # normalize config
            self.config.ppo_mini_batch_size //= (world_size // sp_size // tp_size)
            self.config.ppo_micro_batch_size //= (world_size // sp_size // tp_size)
        elif self.critic_strategy == 'megatron':
            # implement 3D parallel self.actor_gather_manager. We still assume that data is chunked in data parallel.
            # We first need to perform allgather in model parallel group so that data in each tp/pp/cp group is identical.
            # Then, we chunk data according to context parallel rank
            # In this way, the API of FSDP and Megatron can be identical
            from alpha_seed.workers.hybrid_engine.megatron_gather import MegatronDataGatherManager
            self.gather_manager = MegatronDataGatherManager()

        self._model_initialized = False

    def _build_critic_model_optimizer(self, config, from_scratch=True):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim

        local_path = download_minimal_required_files(config.model.path, from_scratch, torch.distributed.get_rank(),
                                                     torch.distributed.get_world_size())
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.
        # TODO: support loading critic weights from RM. Support using AutoModelForTokenClassification
        from transformers import AutoTokenizer

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        use_rmpad = self.config.get('use_rmpad', False)
        if use_rmpad:
            # optimize the model via rmpad
            from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch
            assert apply_monkey_patch(
                config=critic_model_config,
                verbose=self.rank == 0), f'Cannot find rmpad version of {critic_model_config.model_type}'

        with meta_device_init(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, '_moe_implementation', 'fused')
            # NOTE: this is to support loading sft models directly
            setattr(critic_model_config, "id2label", {0: "LABEL_0"})
            setattr(critic_model_config, "label2id", {"LABEL_0": 0})
            critic_module = AutoModelForTokenClassification.from_config(critic_model_config,
                                                                        torch_dtype=torch_dtype,
                                                                        attn_implementation='flash_attention_2',
                                                                        trust_remote_code=trust_remote_code)
            # reset score head parameter
            # critic_module.score.reset_parameters()
            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.enable_gradient_checkpointing:
                # doc link: https://bytedance.us.larkoffice.com/docx/NiWVd0QgoopepBxBXmDuHJKwsNe
                use_reentrant = self.config.act_offload
                critic_module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': use_reentrant})
                critic_module.train()
                if self.rank == 0:
                    print(critic_module)
                    if hasattr(critic_module, 'transformer'):
                        model = critic_module.transformer
                    elif hasattr(critic_module, 'model'):
                        model = critic_module.model
                    else:
                        model = None
                    if model is not None:
                        print(
                            f'{model.gradient_checkpointing=}, {model.training=}, {model._gradient_checkpointing_func=}'
                        )
        shard_plan = apply_parallel_plan(critic_module, critic_module.config, self.tp_mesh)

        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        if self.rank == 0:
            print_model_size(critic_module)

        fsdp_config = self.config.model.fsdp_config

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

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP')

        cpu_offload = None
        if self.config.model.fsdp_config.param_offload:
            # NOTE: CPUOffload needs to cooperate with FSDP.no_sync() in gradient accumulation,
            # which will lead to more memory consumption as gradients keep unshard in between micro-batches.
            # temporarily disbale this for more investigation
            cpu_offload = CPUOffload(offload_params=False)

        # we only support ZeRO3 of hybrid DP+FSDP or full FSDP
        if self.fsdp_mesh is None or self.fsdp_mesh.ndim == 1:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif self.fsdp_mesh.ndim == 2:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise NotImplementedError(f"get device mesh ndim={self.fsdp_mesh.ndim}, but only support 1 or 2")

        shard_states = parallel_load_safetensors(local_path) if from_scratch else {}
        if torch.distributed.get_rank() == 0:
            print(f"init fsdp from_scratch={from_scratch}")
        critic_module = FSDP(critic_module,
                             param_init_fn=parallel_init_fsdp_fn(critic_module, shard_states),
                             use_orig_params=True,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=sharding_strategy,
                             device_mesh=self.fsdp_mesh,
                             mixed_precision=mixed_precision,
                             forward_prefetch=True,
                             sync_module_states=False,
                             cpu_offload=cpu_offload)

        register_dtensor_save_hook(critic_module, shard_plan, self.config.tp_outside)

        log_gpu_memory_usage('After critic FSDP', logger=logger)

        from alpha_seed.trainer.optim import get_optimizer_from_config
        critic_optimizer = get_optimizer_from_config(critic_module.parameters(), config.optim)

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps = int(config.optim.get('lr_warmup_steps', -1))
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler, critic_model_config

    def _build_critic_model_optimizer_mariana(self, config):
        from alpha_seed.models.mariana.checkpoint_utils import load_partial_pretrain
        from alpha_seed.models.mariana.config_utils import convert_hf_config_to_mariana, update_megatron_config
        from alpha_seed.models.mariana.modeling_mariana import convert_gate_to_fp32
        from alpha_seed.models.mariana.optimizer_utils import configure_optimizers
        from transformers import AutoTokenizer, AutoConfig

        from mariana.utils.megatron import initialize_megatron_args
        from verl.utils.fs import copy_local_path_from_hdfs

        from mariana.models.text.config import TrainConfig, MegatronConfig

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        # TODO: ignore pulling model file if resuming ckpt
        local_path = copy_local_path_from_hdfs(config.model.path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        critic_model_config = AutoConfig.from_pretrained(local_path)

        megatron_config = MegatronConfig(**self.config.mariana.megatron)
        local_path = copy_local_path_from_hdfs(config.model.path)

        model_config = convert_hf_config_to_mariana(hf_config=critic_model_config,
                                                    model_implementation=self.config.mariana.model_implementation)

        # vpp size
        update_megatron_config(model_config,
                               megatron_config,
                               vpp_size=self.config.mariana.megatron.virtual_pipeline_parallel_size)

        if not torch.distributed.is_initialized():
            # Note(zhangchi.usc1992): very important! We only build megatron world once
            initialize_megatron_args(model_config, megatron_config)

        # step 3: build model and optimizer
        def megatron_model_provider(pre_process=True, post_process=True):
            """Build the policy model."""
            from alpha_seed.models.mariana.modeling_mariana import MarianaForTokenClassification
            model = MarianaForTokenClassification(model_config,
                                                  megatron_config,
                                                  pre_process=pre_process,
                                                  post_process=post_process)
            return model

        from megatron.training import get_model
        from megatron.model import ModelType

        # model_kwargs
        model_kwargs = {}
        # this returns model chunk for each pp stage
        # note that for reference policy, we actually don't need wrap_with_ddp. We do so that offload API can be unified.
        models = get_model(megatron_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True, **model_kwargs)
        convert_gate_to_fp32(models)

        # load checkpoint. Note that we should load ckpt before optimizer. Otherwise, the fp32 params will be wrong.
        # we assume the megatron_merge_state.pt in the same folder as hf
        # ckpt_path = 'hdfs://haruna/home/byte_data_seed/ssd_hldy/user/tiantianfan1/sft/M8_680m_SFT/checkpoints/global_epoch_2/megatron_merge_states.pt'
        # ckpt_local_path = copy_local_path_from_hdfs(ckpt_path)
        # load_partial_pretrain(models,
        #                       partial_pretrain=ckpt_local_path,
        #                       model_config=model_config,
        #                       download_in_shards=True)

        # switch to use omnistore
        # the original ckpt is under local_path/meta_info.json

        ckpt_meta_info_json_path = os.path.join(local_path, 'meta_info.json')
        assert os.path.exists(
            ckpt_meta_info_json_path
        ), 'Please make sure the huggingface checkpoint stores the upstream path. If not, please re-convert it using latest seed-models'

        with open(ckpt_meta_info_json_path, 'r') as f:
            ckpt_meta_info = json.load(f)

        assert 'omnistore_ckpt_path' in ckpt_meta_info
        ckpt_path = ckpt_meta_info['omnistore_ckpt_path']
        import omnistore
        ckpt_state = {"model": models}
        # load model and optimizer
        omnistore.MegatronCheckpointer.load(
            path=ckpt_path,
            enable_shm_download_ckpt_tmp=False,
            checkpoint_state=ckpt_state,
            loader_in_split_mode=False,
        )

        # build optimizer
        optim_config = self.config.optim

        total_steps = optim_config.get('total_training_steps', 0)
        total_steps = 100000

        assert total_steps > 0

        num_warmup_steps = int(optim_config.get('lr_warmup_steps', -1))
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        optimizers, lr_schedulers = configure_optimizers(
            models=models,
            train_iters=total_steps,
            lr_warmup_iters=num_warmup_steps,
            lr=optim_config.lr,
            adam_betas=optim_config.betas,
            adam_eps=optim_config.eps,
            weight_decay=optim_config.weight_decay,
        )

        # If resume_optimizer is false, copy bf16 weights in model to optimizer
        # to avoid loss error issues.
        optimizers[0].reload_model_params()

        from alpha_seed.workers.actors.offload import offload_megatron_model_to_cpu

        offload_megatron_model_to_cpu(models=models)  # everything is on CPU

        return models, optimizers, lr_schedulers, critic_model_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device: str, model=True, optimizer=True, model_empty_cache=True):
        assert device in ("cuda", "cpu")
        if self.critic_strategy == 'fsdp':
            if self.config.model.fsdp_config.param_offload:
                return
            if device == "cuda":
                if model:
                    load_fsdp_model_to_gpu(self.critic_module)
                if optimizer:
                    load_fsdp_optimizer(self.critic_optimizer, torch.cuda.current_device())
                gc.collect()
            elif device == "cpu":
                if model:
                    offload_fsdp_model_to_cpu(self.critic_module, model_empty_cache)
                if optimizer:
                    offload_fsdp_optimizer(self.critic_optimizer)
        elif self.critic_strategy == 'megatron':
            if device == 'cuda':
                load_megatron_model_to_gpu(models=self.critic_module, load_grad=optimizer)
                gc.collect()
            elif device == 'cpu':
                offload_megatron_model_to_cpu(models=self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, remove_safetensors_after_init=False, from_scratch=True):
        if self._model_initialized:
            return
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        if self.critic_strategy == 'fsdp':
            self.critic_module, self.critic_optimizer, self.critic_lr_scheduler, self.critic_model_config = self._build_critic_model_optimizer(
                self.config, from_scratch=from_scratch)

            self.critic = DataParallelPPOCritic(config=self.config,
                                                critic_module=self.critic_module,
                                                critic_optimizer=self.critic_optimizer)
        elif self.critic_strategy == 'megatron':
            from alpha_seed.workers.ppo_critic_megatron import MegatronPPOCritic
            self.critic_module, self.critic_optimizer, self.critic_lr_scheduler, self.critic_model_config = self._build_critic_model_optimizer_mariana(
                self.config)
            self.critic = MegatronPPOCritic(config=self.config,
                                            module=self.critic_module,
                                            optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        if self.rank == 0:
            print(self.critic_model_config)

        self.checkpoint_manager = CheckpointManagerWrapper(strategy=self.critic_strategy,
                                                           model=self.critic_module,
                                                           optimizer=self.critic_optimizer,
                                                           lr_scheduler=self.critic_lr_scheduler,
                                                           hf_config=self.critic_model_config,
                                                           tokenizer=self.tokenizer)

        if self.config.train_memory_offload:
            self.to("cpu")
        torch.cuda.empty_cache()
        ndtimeline.init_with_ray(self)
        self._model_initialized = True
        if remove_safetensors_after_init:
            cleanup_local_tmp_folder_safetensors_files(self.critic_model_config._name_or_path)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        # data = data.to('cuda')
        # Note we don't offload to cpu after compute_values
        # as it next will update critic
        if self.config.train_memory_offload:
            self.to("cuda", model=True, optimizer=False)

        micro_batch_size = self.config.infer_micro_batch_size
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        if self.config.use_dynamic_bsz:
            data.meta_info['max_token_len'] = self.config.get('infer_ppo_max_token_len', self.config.ppo_max_token_len)
        else:
            data.meta_info['micro_batch_size'] = micro_batch_size
        with self.gather_manager:
            data = self.gather_manager.preprocess_data(data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.gather_manager.postprocess_data(output)
        output = output.to('cpu')

        # torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        torch.cuda.reset_peak_memory_stats()
        # data = data.to('cuda')

        log_gpu_memory_usage('Before Critic update')

        # optimizer will be loaded just before the step to save
        # forward & backward memory
        if self.config.train_memory_offload:
            self.to("cuda", model=True, optimizer=False if self.critic_strategy == 'fsdp' else True)

        with self.gather_manager:
            data = self.gather_manager.preprocess_data(data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            if self.critic_strategy == 'fsdp':
                self.critic_lr_scheduler.step()
                lr = self.critic_lr_scheduler.get_last_lr()[0]
            elif self.critic_strategy == 'megatron':
                self.critic_lr_scheduler[0].step(1)
                lr = self.critic_lr_scheduler[0].get_lr()

            metrics['critic/lr(1e-4)'] = lr * 1e4

            max_memory_allocated, max_memory_reserved = get_memory()
            output = DataProto(batch=None,
                               meta_info={
                                   'metrics': metrics,
                                   'memory/critic_max_allocated': max_memory_allocated,
                                   'memory/critic_max_reserved': max_memory_reserved
                               })
            output = self.gather_manager.postprocess_data(output)

        if self.config.train_memory_offload:
            self.to("cpu", model_empty_cache=False)
        output = output.to('cpu')

        log_gpu_memory_usage('After Critic update')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, hdfs_path=None, version='v1', enable_shm=False):
        if self.config.train_memory_offload:
            self.to("cuda")
        self.checkpoint_manager.load_checkpoint(version=version,
                                                hdfs_path=hdfs_path,
                                                device_mesh=self.fsdp_mesh if self.critic_strategy == 'fsdp' else None,
                                                role='critic',
                                                strategy=self.critic_strategy,
                                                enable_shm=enable_shm)
        if self.config.train_memory_offload:
            self.to("cpu")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def save_checkpoint(self,
                        local_path,
                        hdfs_path=None,
                        version='v1',
                        global_step=0,
                        ckpt_global_uploader_ref=None,
                        enable_shm=False):
        if self.config.train_memory_offload:
            self.to("cuda")
        self.checkpoint_manager.save_checkpoint(version=version,
                                                local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                device_mesh=self.fsdp_mesh if self.critic_strategy == 'fsdp' else None,
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
