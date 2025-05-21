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
import warnings
import os
import logging
import ray
import torch
import torch.distributed

from single_controller.base import Worker
from single_controller.base.decorator import register, Dispatch
from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.import_utils import import_external_libs
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_constant_schedule_with_warmup
from verl.utils.model import print_model_size
from alpha_seed.models.transformers.parallel.collectives import get_memory
from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch, get_parallel_plan
from alpha_seed.workers.fsdp.initialize import (create_mesh, meta_device_init,
                                                cleanup_local_tmp_folder_safetensors_files)
from alpha_seed.trainer.optim import get_optimizer_from_config
from alpha_seed.workers.fsdp.offload import offload_fsdp_optimizer, load_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_model_to_gpu
from alpha_seed.workers.megatron.offload import offload_megatron_model_to_cpu, load_megatron_model_to_gpu
from alpha_seed.workers.hybrid_engine.fsdp_gather import DataGatherManager
from alpha_seed.workers.ppo_critic import DataParallelPPOCritic
from alpha_seed.utils import ndtimeline
from alpha_seed.utils.ckpt import download_minimal_required_files

from seed_models.utils.count_flops import FlopsCounter
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from codetiming import Timer

from datetime import timedelta

from .checkpoint import CheckpointManagerWrapper
from tensordict import TensorDict

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

        assert self.critic_strategy in ['fsdp', 'megatron', 'vescale-fsdp2']

        world_size = torch.distributed.get_world_size()

        if self.critic_strategy in ('fsdp', 'vescale-fsdp2'):
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
        local_path = download_minimal_required_files(config.model.path, from_scratch, torch.distributed.get_rank(),
                                                     torch.distributed.get_world_size())
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.
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
        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        architectures = [
            arch.replace('ForCausalLM', 'ForTokenClassification') for arch in critic_model_config.architectures
        ]
        setattr(critic_model_config, 'architectures', architectures)

        use_rmpad = self.config.get('use_rmpad', False)
        if use_rmpad:
            # optimize the model via rmpad
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

            if self.config.freeze_gate:
                from seed_models import M8ForTokenClassification, M8ForSequenceClassification
                if isinstance(critic_module, M8ForTokenClassification) or isinstance(
                        critic_module, M8ForSequenceClassification):
                    for layer in critic_module.transformer.h:
                        if layer.mlp.moe.gate is not None:
                            layer.mlp.moe.gate.requires_grad = False

        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')
            print_model_size(critic_module)

        strategy = self.critic_strategy
        if strategy == "fsdp":
            from alpha_seed.workers.fsdp import fully_shard
        elif strategy == "vescale-fsdp2":
            from alpha_seed.workers.vescale.fully_shard import fully_shard
        else:
            raise RuntimeError(f"[critic]: Unknown strategy for fsdp: {strategy}")

        if strategy == 'fsdp' and config.model.fsdp_config.param_offload:
            # NOTE: CPUOffload needs to cooperate with FSDP.no_sync() in gradient accumulation,
            # which will lead to more memory consumption as gradients keep unshard in between micro-batches.
            # temporarily disbale this for more investigation
            raise NotImplementedError("CPUOffload is not supported for trainable model")

        act_offload_kwargs = dict(
            offload_threshold=config.get('act_offload_threshold', 1024 * 1024),
            offload_upbound=config.act_offload_upbound,
            buffer_size=config.act_offload_buff_size,
        )

        if hasattr(critic_module, "vision_encoder"):
            block_cls = critic_module.language_model._no_split_modules + critic_module.vision_encoder._no_split_modules
        else:
            block_cls = critic_module._no_split_modules[0]

            # also wrap MLP for M10
            if critic_model_config.model_type == 'seed_m10':
                block_cls = block_cls + ['M10MLP']

        if self.rank == 0:
            print(f'FSDP wrap module cls: {block_cls}')

        critic_module, _ = fully_shard(
            model=critic_module,
            block_cls=block_cls,
            fsdp_mesh=self.fsdp_mesh,
            tp_plan=get_parallel_plan(critic_model_config, self.tp_mesh),
            tp_mesh=self.tp_mesh,
            tp_outside=config.tp_outside,
            recompute=config.model.enable_gradient_checkpointing,
            act_offload=config.act_offload,
            param_offload=config.model.fsdp_config.param_offload,
            weights=local_path if from_scratch else None,
            act_offload_kwargs=act_offload_kwargs,
        )
        log_gpu_memory_usage('After critic FSDP')

        # create critic optimizer
        critic_optimizer = get_optimizer_from_config(
            [param for param in critic_module.parameters() if param.requires_grad], config.optim)

        # enbale optimizer offload
        if strategy == 'fsdp' and not config.model.fsdp_config.param_offload:
            critic_optimizer.register_step_pre_hook(
                lambda optim, args, kwargs: load_fsdp_optimizer(optim, torch.cuda.current_device()))
            critic_optimizer.register_step_post_hook(lambda optim, args, kwargs: offload_fsdp_optimizer(optim))
        elif strategy == 'vescale-fsdp2':
            from alpha_seed.workers.vescale.fully_shard import register_dtensor_hook
            from vescale.parallel.fsdp2.extension.optimizer_offload import apply_optimizer_offload
            register_dtensor_hook(critic_module, critic_optimizer)
            if not config.model.fsdp_config.param_offload:
                apply_optimizer_offload(critic_module,
                                        critic_optimizer,
                                        get_seqlen_fn=lambda args, kwargs: kwargs["input_ids"].numel())

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps = int(config.optim.get('lr_warmup_steps', -1))
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

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

        # TODO(zhangchi.usc1992): this logic is VERY VERY hacky as the upstream mariana
        # lacks huggingface folder checkpoint format
        ckpt_meta_info_json_path = os.path.join(local_path, 'meta_info.json')

        if os.path.exists(ckpt_meta_info_json_path):
            # we read from huggingface
            with open(ckpt_meta_info_json_path, 'r') as f:
                ckpt_meta_info = json.load(f)
            assert 'omnistore_ckpt_path' in ckpt_meta_info
            ckpt_path = ckpt_meta_info['omnistore_ckpt_path']
            config_path = local_path
        else:
            config_path = local_path
            # Note(zhangchi.usc1992) make sure the config_path does not end with '/', which is guaranteed by copy_local_path_from_hdfs
            ckpt_path = os.path.dirname(config.model.path)
            # config_path = os.path.join(local_path, 'huggingface')
            assert os.path.exists(config_path), \
                'Please make sure the huggingface checkpoint stores the upstream path. If not, please re-convert it using 0306 seed-models'

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = AutoTokenizer.from_pretrained(config_path)
        critic_model_config = AutoConfig.from_pretrained(config_path)
        architectures = [
            arch.replace('ForCausalLM', 'ForTokenClassification') for arch in critic_model_config.architectures
        ]
        setattr(critic_model_config, 'architectures', architectures)

        megatron_config = MegatronConfig(**self.config.mariana.megatron)

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

        import omnistore
        ckpt_state = {"model": models}
        # load model and optimizer
        omnistore.MegatronCheckpointer.load(
            path=ckpt_path,
            enable_shm_download_ckpt_tmp=False,
            checkpoint_state=ckpt_state,
            loader_in_split_mode=False,
            ignore_model_keys={r'.*\.score_head\.weight$', r'.*\.score_head\.bias$'}
            if self.config.load_score_head is False else None,
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

        from alpha_seed.workers.megatron.offload import offload_megatron_model_to_cpu

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
        elif self.critic_strategy == 'vescale-fsdp2':
            if self.config.model.fsdp_config.param_offload:
                return
            if device == 'cuda':
                if model:
                    self.critic_module.to(torch.cuda.current_device(), non_blocking=True)
                if optimizer:
                    load_fsdp_optimizer(self.critic_optimizer)
            elif device == "cpu":
                if model:
                    self.critic_module.to('cpu', non_blocking=True)
                if optimizer:
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
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        if self.critic_strategy in ('fsdp', 'vescale-fsdp2'):
            self.critic_module, self.critic_optimizer, self.critic_lr_scheduler, self.critic_model_config = self._build_critic_model_optimizer(
                self.config, from_scratch=from_scratch)
            self.critic = DataParallelPPOCritic(config=self.config,
                                                critic_module=self.critic_module,
                                                critic_optimizer=self.critic_optimizer,
                                                critic_model_config=self.critic_model_config)
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
            self.to("cuda", model=True, optimizer=False if self.critic_strategy in ('fsdp', 'vescale-fsdp2') else True)

        with self.gather_manager:
            data = self.gather_manager.preprocess_data(data)

            with Timer(name='update_critic', logger=None) as timer:
                seq_vf, metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            kwargs = {}
            if 'global_img_token_num' in data.meta_info:
                kwargs['images_seqlens'] = data.meta_info['global_img_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time, **kwargs)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            if self.critic_strategy in ('fsdp', 'vescale-fsdp2'):
                self.critic_lr_scheduler.step()
                lr = self.critic_lr_scheduler.get_last_lr()[0]
            elif self.critic_strategy == 'megatron':
                self.critic_lr_scheduler[0].step(1)
                lr = self.critic_lr_scheduler[0].get_lr()

            metrics['critic/lr(1e-4)'] = lr * 1e4

            max_memory_allocated, max_memory_reserved = get_memory()
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

        log_gpu_memory_usage('After Critic update')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, hdfs_path=None, version='v1', enable_shm=False):
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
