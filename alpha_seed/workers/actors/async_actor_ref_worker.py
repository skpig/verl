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

import copy
from typing import Union, List
import json
from contextlib import nullcontext
import warnings
import os
import logging
import ray
import torch
import torch.distributed
from omegaconf import DictConfig, open_dict
import gc

from single_controller.base import Worker
from single_controller.base.decorator import register, Dispatch
from verl import DataProto
from alpha_seed.utils.functional import update_model_config, get_text_config
from verl.utils.model import print_model_size
from verl.single_controller.base.decorator import Execute
from alpha_seed.workers.fsdp.offload import (offload_fsdp_optimizer, load_fsdp_optimizer, offload_fsdp_model_to_cpu,
                                             load_fsdp_model_to_gpu)
from alpha_seed.workers.megatron.offload import (offload_megatron_model_to_cpu, load_megatron_model_to_gpu,
                                                 offload_megatron_optimizer, load_megatron_optimizer)
from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch, get_parallel_plan, get_ignore_modules_in_mixed_precision
from verl.utils.import_utils import import_external_libs
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.torch_functional import get_constant_schedule_with_warmup
from alpha_seed.trainer.optim import get_optimizer_from_config
from alpha_seed.workers.xperf_rollout.component.query import Query
from alpha_seed.workers.xperf_rollout.utils.layout_convert_helper import offload_to_device

from alpha_seed.utils import ndtimeline
from alpha_seed.workers.hybrid_engine.fsdp_gather import DataGatherManager
from alpha_seed.workers.fsdp.initialize import (create_mesh, meta_device_init,
                                                cleanup_local_tmp_folder_safetensors_files)
from alpha_seed.workers.ppo_actor import DataParallelPPOActor
from alpha_seed.utils.kernels.persist_gemm import deploy_persist_gemm
from alpha_seed.models.transformers.parallel.collectives import get_memory
from alpha_seed.utils.observility.training_stats import MetricsTorchDispatchMode
from alpha_seed.utils.observility import get_profiler_context_wrapped
from alpha_seed.utils.ckpt import download_minimal_required_files
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForVision2Seq

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
        if not torch.distributed.is_initialized():
            timeout = timedelta(minutes=int(os.getenv('NCCL_TIMEOUT', 60)))
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
        # actor model
        if self.actor_strategy in ('fsdp', 'vescale-fsdp2'):
            actor_fsdp_size = config.actor.fsdp_size
            actor_sp_size = config.actor.ulysses_sequence_parallel_size

            actor_tp_size = config.actor.tp_size
            actor_meshes = create_mesh(fsdp_size=actor_fsdp_size,
                                       tp_size=actor_tp_size,
                                       sp_size=actor_sp_size,
                                       tp_outside=config.actor.tp_outside)
            self.actor_fsdp_mesh = actor_meshes[0]
            self.actor_tp_mesh = actor_meshes[1]  # shared for both train and inference
            self.actor_sp_mesh = actor_meshes[2]
            self.actor_gather_mesh = actor_meshes[3]
            self.actor_gather_manager = DataGatherManager(self.actor_gather_mesh, self.actor_sp_mesh)
            if torch.distributed.get_rank() == 0:
                print(
                    f"Created actor with fsdp_size={self.actor_fsdp_mesh.shape}, tp_size={self.actor_tp_mesh.size()}, "
                    f"actor sp_size={self.actor_sp_mesh.size()}")
        elif self.actor_strategy == 'megatron':
            # implement 3D parallel self.actor_gather_manager. We still assume that data is chunked in data parallel.
            # We first need to perform allgather in model parallel group so that data in each tp/pp/cp group is identical.
            # Then, we chunk data according to context parallel rank
            # In this way, the API of FSDP and Megatron can be identical
            from alpha_seed.workers.hybrid_engine.megatron_gather import MegatronDataGatherManager
            self.actor_gather_manager = MegatronDataGatherManager()

        # reference model
        if self._is_ref:
            if self.ref_strategy in ('fsdp', 'vescale-fsdp2'):
                ref_fsdp_size = config.ref.fsdp_size
                ref_sp_size = config.ref.ulysses_sequence_parallel_size
                ref_tp_size = config.ref.tp_size
                ref_meshes = create_mesh(fsdp_size=ref_fsdp_size,
                                         tp_size=ref_tp_size,
                                         sp_size=ref_sp_size,
                                         tp_outside=config.ref.tp_outside)
                self.ref_fsdp_mesh = ref_meshes[0]
                self.ref_tp_mesh = ref_meshes[1]
                self.ref_sp_mesh = ref_meshes[2]
                self.ref_gather_mesh = ref_meshes[3]
                self.ref_gather_manager = DataGatherManager(self.ref_gather_mesh, self.ref_sp_mesh)
                if torch.distributed.get_rank():
                    print(
                        f"Created reference with fsdp_size={self.ref_fsdp_mesh.shape}, tp_size={self.ref_tp_mesh.size()}, "
                        f"infer sp_size={self.ref_sp_mesh.size()}")
            elif self.ref_strategy == 'megatron':
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
                               role='actor',
                               from_scratch=True):
        if self.rank == 0:
            print(f'Build model and optimizer for {role}')

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        # TODO: ignore pulling model file if resuming ckpt
        self.local_path = download_minimal_required_files(model_path, from_scratch, torch.distributed.get_rank(),
                                                          torch.distributed.get_world_size())

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)
        torch_dtype = torch.float32 if self._is_actor else torch.bfloat16

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

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

        actor_module_fsdp = None
        actor_optimizer = None
        actor_lr_scheduler = None
        metrics_context = None

        # we only need actor_model_config in rollout
        if self._is_standalone_rollout or self._is_standalone_validator:
            return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config, metrics_context

        if use_rmpad:
            # optimize the model via rmpad
            assert apply_monkey_patch(
                config=actor_model_config,
                verbose=self.rank == 0), f'Cannot find rmpad version of {actor_model_config.model_type}'

        with meta_device_init(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            AutoModel = AutoModelForVision2Seq if actor_model_config.model_type == 'seed_vl' else AutoModelForCausalLM
            actor_module = AutoModel.from_config(actor_model_config,
                                                 torch_dtype=torch_dtype,
                                                 attn_implementation='flash_attention_2',
                                                 trust_remote_code=trust_remote_code)
            if hasattr(actor_model_config, "vision_config") and actor_model_config.vision_config.freeze_vit:
                actor_module.vision_encoder.requires_grad_(False)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if self.config.remove_o_bias:
                from seed_models import P6DenseForCausalLM
                if isinstance(actor_module, P6DenseForCausalLM):
                    for layer in actor_module.model.layers:
                        if layer.self_attn.o_proj.bias is not None:
                            layer.self_attn.o_proj.bias.requires_grad = False

            if self.config.freeze_gate:
                from seed_models import M8ForCausalLM
                if isinstance(actor_module, M8ForCausalLM):
                    for layer in actor_module.transformer.h:
                        if layer.mlp.moe.gate is not None:
                            layer.mlp.moe.gate.requires_grad = False

            enable_training_stats = self.config.actor.enable_training_stats
            metrics_context = MetricsTorchDispatchMode() if enable_training_stats else nullcontext()

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)
        if self.rank == 0:
            print(actor_module)
            print_model_size(actor_module)

        fsdp_mesh = self.ref_fsdp_mesh if role == 'ref' else self.actor_fsdp_mesh
        tp_mesh = self.ref_tp_mesh if role == 'ref' else self.actor_tp_mesh
        tp_outside = self.config.ref.tp_outside if role == "ref" else self.config.actor.tp_outside

        strategy = self.ref_strategy if role == "ref" else self.actor_strategy
        if strategy == 'fsdp':
            from alpha_seed.workers.fsdp.fully_shard import fully_shard
        elif strategy == 'vescale-fsdp2':
            from alpha_seed.workers.vescale.fully_shard import fully_shard
        else:
            raise RuntimeError(f"[{role}]: Unknown strategy for fsdp: {strategy}")

        # set up parameter cpu offload
        if role == 'actor':
            if strategy == 'fsdp' and self.config.actor.fsdp_config.param_offload:
                # NOTE: CPUOffload needs to cooperate with FSDP.no_sync() in gradient accumulation,
                # which will lead to more memory consumption as gradients keep unshard in between micro-batches.
                # temporarily disbale this for more investigation
                raise NotImplementedError("CPUOffload for trainable model is not supported in torch FSDP, "
                                          "please use strategy=vescale-fsdp2 instead.")
            param_offload = self.config.actor.fsdp_config.param_offload
        elif role == 'ref':
            param_offload = self.config.ref.fsdp_config.param_offload
        elif role == 'rollout':
            param_offload = True

        # get ignored modules
        ignored_modules = None
        if self.config.update_gate_ema:
            ignored_modules = get_ignore_modules_in_mixed_precision(actor_model_config.model_type)

        if not from_scratch:
            warnings.filterwarnings("ignore", "state not found in", category=UserWarning)

        act_offload_kwargs = dict(
            offload_threshold=self.config.get('act_offload_threshold', 1024 * 1024),
            offload_upbound=self.config.get('act_offload_upbound', None),
            buffer_size=self.config.get('act_offload_buff_size', 40),
        )
        if hasattr(actor_module, "vision_encoder"):
            block_cls = actor_module.language_model._no_split_modules + actor_module.vision_encoder._no_split_modules
        else:
            block_cls = actor_module._no_split_modules

            # also wrap MLP for M10
            if actor_model_config.model_type == 'seed_m10':
                block_cls = block_cls + ['M10MLP']

        if self.rank == 0:
            print(f'FSDP wrap module cls: {block_cls}')

        actor_module_fsdp, metrics_context = fully_shard(
            model=actor_module,
            block_cls=block_cls,
            fsdp_mesh=fsdp_mesh,
            tp_plan=get_parallel_plan(actor_model_config, tp_mesh),
            tp_mesh=tp_mesh,
            tp_outside=tp_outside,
            recompute=enable_gradient_checkpointing,
            act_offload=self.config.actor.act_offload if role == 'actor' else False,
            param_offload=param_offload,
            weights=self.local_path if from_scratch else None,
            ignored_modules=ignored_modules,
            enable_training_stats=enable_training_stats,
            act_offload_kwargs=act_offload_kwargs)
        log_gpu_memory_usage(f'After {role} FSDP init')

        # create optimizer for actor
        actor_optimizer = None
        actor_lr_scheduler = None
        if role == 'actor':
            actor_optimizer = get_optimizer_from_config(
                [param for param in actor_module_fsdp.parameters() if param.requires_grad], optim_config)

            # enable optimizer offload
            if strategy == 'fsdp' and not param_offload:
                actor_optimizer.register_step_pre_hook(
                    lambda optim, args, kwargs: load_fsdp_optimizer(optim, torch.cuda.current_device()))
                actor_optimizer.register_step_post_hook(lambda optim, args, kwargs: offload_fsdp_optimizer(optim))
            elif strategy == 'vescale-fsdp2':
                from alpha_seed.workers.vescale.fully_shard import register_dtensor_hook
                from vescale.parallel.fsdp2.extension.optimizer_offload import apply_optimizer_offload
                register_dtensor_hook(actor_module_fsdp, actor_optimizer)
                if not param_offload:
                    apply_optimizer_offload(actor_module_fsdp,
                                            actor_optimizer,
                                            get_seqlen_fn=lambda args, kwargs: kwargs["input_ids"].numel())

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps = int(optim_config.get('lr_warmup_steps', -1))
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)

        assert get_text_config(actor_model_config).num_attention_heads % self.config.actor.ulysses_sequence_parallel_size == 0, \
            f'invalid ulysses sequence parallel size: {get_text_config(actor_model_config).num_attention_heads=} % {self.config.actor.ulysses_sequence_parallel_size=} != 0'

        log_gpu_memory_usage('After actor optimizer init')
        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config, metrics_context

    def _build_model_optimizer_mariana(self, model_path, role='actor'):
        assert role in ['actor', 'ref']

        from alpha_seed.models.mariana.checkpoint_utils import load_partial_pretrain
        from alpha_seed.models.mariana.config_utils import convert_hf_config_to_mariana, update_megatron_config
        from alpha_seed.models.mariana.modeling_mariana import convert_gate_to_fp32
        from alpha_seed.models.mariana.optimizer_utils import configure_optimizers

        from mariana.utils.megatron import initialize_megatron_args
        from verl.utils.fs import copy_local_path_from_hdfs

        from mariana.models.text.config import TrainConfig, MegatronConfig

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        # TODO: ignore pulling model file if resuming ckpt
        local_path = copy_local_path_from_hdfs(model_path)
        self.local_path = local_path

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
            ckpt_path = os.path.dirname(model_path)
            # config_path = os.path.join(local_path, 'huggingface')
            assert os.path.exists(config_path), \
                'Please make sure the huggingface checkpoint stores the upstream path. If not, please re-convert it using 0306 seed-models'

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = AutoTokenizer.from_pretrained(config_path)
        actor_model_config = AutoConfig.from_pretrained(config_path)

        if self._is_standalone_rollout or self._is_standalone_validator:
            return None, None, None, actor_model_config

        megatron_config = MegatronConfig(**self.config.mariana.megatron)

        model_config = convert_hf_config_to_mariana(hf_config=actor_model_config,
                                                    model_implementation=self.config.mariana.model_implementation)

        # vpp size
        update_megatron_config(model_config,
                               megatron_config,
                               vpp_size=self.config.mariana.megatron.virtual_pipeline_parallel_size)

        if role == 'actor':
            #Note(zhangchi.usc1992): very important! We only build megatron world once
            initialize_megatron_args(model_config, megatron_config)

        # step 3: build model and optimizer
        def megatron_model_provider(pre_process=True, post_process=True):
            """Build the policy model."""
            from alpha_seed.models.mariana.modeling_mariana import MarianaForCausalLM
            model = MarianaForCausalLM(model_config,
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

        import omnistore
        ckpt_state = {"model": models}
        # load model and optimizer
        omnistore.MegatronCheckpointer.load(
            path=ckpt_path,
            enable_shm_download_ckpt_tmp=False,
            checkpoint_state=ckpt_state,
            loader_in_split_mode=False,
        )

        if role == 'actor':
            # build optimizer
            optim_config = self.config.actor.optim

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

        else:
            optimizers = None
            lr_schedulers = None

        offload_megatron_model_to_cpu(models=models)  # everything is on CPU

        log_gpu_memory_usage(head='After offload_megatron_model_to_cpu in init')

        return models, optimizers, lr_schedulers, actor_model_config

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
            sharding_manager = FSDPXPerfGPTShardingManager(module=self.actor_module_fsdp,
                                                           model_config=self.actor_model_config,
                                                           inference_engine=rollout.inference_engine,
                                                           device_mesh=rollout.device_mesh,
                                                           standalone=self._is_standalone_rollout or
                                                           self._is_standalone_validator,
                                                           only_bind_once=self.role == "rollout",
                                                           backend='fsdp',
                                                           weights_communicator=weights_communicator)
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
                        if model:
                            load_fsdp_model_to_gpu(self.actor_module_fsdp)
                        if optimizer and self.actor_optimizer is not None:
                            load_fsdp_optimizer(self.actor_optimizer, device)
                elif self.actor_strategy == 'vescale-fsdp2':
                    if not self.config.actor.fsdp_config.param_offload:
                        if model:
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
            # clean cpu memory
            gc.collect()

        elif device == "cpu":
            if self._is_actor:
                if self.actor_strategy == 'fsdp':
                    if not self.config.actor.fsdp_config.param_offload:
                        if model:
                            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                        if optimizer and self.actor_optimizer is not None:
                            offload_fsdp_optimizer(self.actor_optimizer)
                elif self.actor_strategy == 'vescale-fsdp2':
                    if not self.config.actor.fsdp_config.param_offload:
                        if model:
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
            if self.actor_strategy in ('fsdp', 'vescale-fsdp2'):
                sp_size = config.actor.ulysses_sequence_parallel_size
                self.config.actor.ppo_mini_batch_size //= (world_size // sp_size // actor_tp_size)
                self.config.actor.ppo_micro_batch_size //= (world_size // sp_size // actor_tp_size)
            elif self.actor_strategy == 'megatron':
                # we import here to remove the mariana as necessary dependency
                dp_size = mpu.get_data_parallel_world_size()
                self.config.actor.ppo_mini_batch_size //= dp_size
                self.config.actor.ppo_micro_batch_size //= dp_size

        # TODO(zhangchi.usc1992): this is useless. correct me if this is wrong
        # if self._is_rollout or self._is_standalone_rollout:
        #     if self.actor_strategy == 'fsdp':
        #         sp_size = config.actor.ulysses_sequence_parallel_size
        #         self.config.rollout.micro_batch_size //= world_size  # for xperf-gpt
        #         self.config.rollout.log_prob_micro_batch_size //= (world_size // sp_size // actor_tp_size)
        #     elif self.actor_strategy == 'megatron':
        #         dp_size = mpu.get_data_parallel_world_size()
        #         self.config.rollout.micro_batch_size //= world_size  # for xperf-gpt
        #         self.config.rollout.log_prob_micro_batch_size //= dp_size

        if self._is_ref:
            if self.ref_strategy in ('fsdp', 'vescale-fsdp2'):
                sp_size = config.ref.ulysses_sequence_parallel_size
                self.config.ref.log_prob_micro_batch_size //= (world_size // sp_size // ref_tp_size)
                self.config.ref.ppo_mini_batch_size //= (world_size // sp_size // ref_tp_size)
            elif self.ref_strategy == 'megatron':
                dp_size = mpu.get_data_parallel_world_size()
                self.config.ref.log_prob_micro_batch_size //= dp_size

    def _init_model(self, from_scratch):
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

            if self.actor_strategy in ('fsdp', 'vescale-fsdp2'):
                self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config, self.metrics_context = self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                    use_rmpad=use_rmpad,
                    trust_remote_code=self.config.model.get('trust_remote_code', False),
                    role='actor' if self._is_actor else 'rollout',
                    from_scratch=from_scratch)

                assert get_text_config(self.actor_model_config).num_attention_heads % self.config.actor.ulysses_sequence_parallel_size == 0, \
                    f'invalid ulysses sequence parallel size: {get_text_config(self.actor_model_config).num_attention_heads=} % {self.config.actor.ulysses_sequence_parallel_size=} != 0'

            elif self.actor_strategy == 'megatron':
                # TODO: build megatron model
                self.actor_module_mariana, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer_mariana(
                    model_path=self.config.model.path, role='actor')

        # load from checkpoint
        if self._is_actor or self._is_rollout:
            OmegaConf.set_struct(self.config.actor, True)
            if self.actor_strategy in ('fsdp', 'vescale-fsdp2'):
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
                self.actor = MegatronPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_mariana,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_ref:
            from_scratch_ref = True if self.config.ref.ema == 1 else from_scratch
            if self.ref_strategy in ('fsdp', 'vescale-fsdp2'):
                self.ref_module_fsdp = self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=self.config.ref.fsdp_config,
                    optim_config=None,
                    use_rmpad=use_rmpad,
                    override_model_config=override_model_config,
                    enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                    trust_remote_code=self.config.model.get('trust_remote_code', False),
                    role='ref',
                    from_scratch=from_scratch_ref)[0]
                self.ref_module_fsdp.eval()

                OmegaConf.set_struct(self.config.ref, True)
                with open_dict(self.config.ref):
                    self.config.ref.use_rmpad = use_rmpad
                    self.config.ref.use_ce_loss_fusion = use_ce_loss_fusion
                self.ref_policy = DataParallelPPOActor(config=self.config.ref,
                                                       actor_module=self.ref_module_fsdp,
                                                       actor_model_config=self.actor_model_config)
            elif self.ref_strategy == 'megatron':
                # TODO: build megatron actor
                self.ref_module_mariana = self._build_model_optimizer_mariana(model_path=self.config.model.path,
                                                                              role='ref')[0]
                # TODO: make it eval
                # for each model chunk
                self.ref_policy = MegatronPPOActor(config=self.config.ref, actor_module=self.ref_module_mariana)

        if self.config.actor.train_memory_offload:
            self.to("cpu")
        log_gpu_memory_usage("After actor initialized")

        if self._is_rollout or self._is_standalone_rollout or self._is_standalone_validator:
            self.rollout, self.sharding_manager = self._build_rollout()
            self.rollout_async = None

        if self._is_actor or self._is_rollout:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = CheckpointManagerWrapper(strategy=self.actor_strategy,
                                                               model=self.actor.actor_module,
                                                               optimizer=self.actor.actor_optimizer,
                                                               lr_scheduler=self.actor_lr_scheduler,
                                                               hf_config=self.actor_model_config,
                                                               tokenizer=self.tokenizer)

        if self._is_ref:
            self.checkpoint_manager_ref = CheckpointManagerWrapper(
                strategy=self.ref_strategy,
                model=self.ref_policy.actor_module,
                optimizer=None,
                lr_scheduler=None,
                hf_config=self.actor_model_config,  # same for actor and ref
                tokenizer=self.tokenizer)

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

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        torch.cuda.reset_peak_memory_stats()
        # data = data.to('cuda')

        assert self._is_actor
        # data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy')

        # note optimizer offload will be managed inside `update_policy`
        if self.config.actor.train_memory_offload:
            self.to("cuda", model=True, optimizer=False if self.actor_strategy == 'fsdp' else True)

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
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

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

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def old_log_probs(self, prompts: DataProto):
        log_gpu_memory_usage('Before old_log_probs')

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
        log_gpu_memory_usage('After recompute log prob')
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        torch.cuda.reset_peak_memory_stats()
        prompts = prompts.to('cuda')

        assert self._is_rollout or self._is_standalone_validator

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)

        log_gpu_memory_usage('Before load training memory')

        # xperf needs parameters from actor
        if self.config.actor.train_memory_offload:
            self.to("cuda", model=True, optimizer=False)

        log_gpu_memory_usage('Before entering sharding manager')

        self.binding_timer.start()

        with self.sharding_manager:

            binding_time = self.binding_timer.stop()

            log_gpu_memory_usage('After entering sharding manager')
            # after parameters go to xperf, offload actor model to CPU
            if self.config.actor.train_memory_offload:
                self.to("cpu", model=True, optimizer=False)

            log_gpu_memory_usage('After offload train parameters')
            prompts = self.sharding_manager.preprocess_data(prompts)

            generator = self.rollout.generate_sequences(prompts=prompts)
            output = next(generator)

            output = self.sharding_manager.postprocess_data(output)

            log_gpu_memory_usage('After generate sequences')

        log_gpu_memory_usage('After release kv cache')

        max_memory_allocated, max_memory_reserved = get_memory()
        output.meta_info.update({
            'memory/gen_max_allocated': max_memory_allocated,
            'memory/gen_max_reserved': max_memory_reserved,
            'timing/weight_binding': binding_time
        })
        output = output.to('cpu')
        # torch.distributed.barrier()

        log_gpu_memory_usage('After rollout generation')
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
        log_gpu_memory_usage('Before compute_ref_log_prob')

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
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(self.ref_policy.actor_module, FSDP):
            self.ref_policy.actor_module._handle.reshard(True)
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
        for q in queries:
            qid = self.rollout.add_inflight_query(q)
            ret.append(qid)
        return ret

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def get_inflight_query(self, query_id):
        return await self.rollout.get_inflight_query(query_id)

    # 只在dp_size=1的情况下调用，所以这里rank0执行即可，DP_COMPUTE与此参数暂不兼容
    @register(execute_mode=Execute.RANK_ZERO, blocking=True)
    def get_all_queries(self, query_type: str):
        return self.rollout.get_all_queries(query_type)

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
        if self.config.rollout.mode == "batch":
            return
        if sleep:
            # 让engine停下来
            with self.rollout.inference_engine.update_weights_lock:
                self.rollout.stop_event.set()
            # 让engine等待下一次weights loaded
            self.rollout.weights_loaded.clear()
            # 等待engine完全退出gen loop
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
            # 通知engine weights loaded
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
