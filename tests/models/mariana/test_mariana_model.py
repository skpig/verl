"""
This script runs a mariana model forward and forward + backward + optimizer step

torchrun --nproc_per_node 4 --standalone tests/models/mariana/test_mariana_model.py
"""

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

os.environ['MEGATRON_NCCL_TIMEOUT_SECOND'] = '18000'
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
os.environ['MARIANA_DISABLE_ROPE_REGISTER_INV_FREQ'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'

import torch
import torch.distributed as dist
import hydra
import logging

from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import AutoTokenizer, AutoConfig
from megatron.core import parallel_state as mpu

from mono_rl import DataProto

from alpha_seed.models.mariana.checkpoint_utils import load_partial_pretrain
from alpha_seed.models.mariana.config_utils import convert_hf_config_to_mariana, update_megatron_config
from alpha_seed.models.mariana.modeling_mariana import convert_gate_to_fp32
from alpha_seed.models.mariana.optimizer_utils import configure_optimizers
from alpha_seed.workers.ppo_actor_megatron import MegatronPPOActor

from mariana.utils.megatron import initialize_megatron_args
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.debug import log_gpu_memory_usage

from mariana.models.text.config import TrainConfig, MegatronConfig

from alpha_seed.workers.megatron.offload import (
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
)


@hydra.main(config_path='.', config_name='config', version_base=None)
def main(config: DictConfig):
    # step 1: construct a model_config given hf_config
    megatron_config = MegatronConfig(**config.megatron)
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    hf_config = AutoConfig.from_pretrained(local_path)

    model_config = convert_hf_config_to_mariana(hf_config=hf_config, model_implementation=config.model_implementation)

    # vpp size
    update_megatron_config(model_config, megatron_config, vpp_size=config.megatron.virtual_pipeline_parallel_size)

    # step 2: build megatron world
    initialize_megatron_args(model_config, megatron_config)

    # step 3: build model and optimizer
    def megatron_model_provider(pre_process=True, post_process=True):
        """Build the policy model."""
        from alpha_seed.models.mariana.modeling_mariana import MarianaForCausalLM
        model = MarianaForCausalLM(model_config, megatron_config, pre_process=pre_process, post_process=post_process)
        return model

    from megatron.training import get_model, get_raw_model, wrap_model
    from megatron.model import ModelType

    log_gpu_memory_usage(head='Before model init')

    # model_kwargs
    model_kwargs = {}
    # this returns model chunk for each pp stage
    models = get_model(megatron_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True, **model_kwargs)

    log_gpu_memory_usage(head='After model init')

    convert_gate_to_fp32(models)

    log_gpu_memory_usage(head='After convert_gate_to_fp32')

    # load checkpoint. Note that we should load ckpt before optimizer. Otherwise, the fp32 params will be wrong.
    # we assume the megatron_merge_state.pt in the same folder as hf
    # ckpt_path = 'hdfs://haruna/home/byte_data_seed/hdd_hldy/user/huakai.dev/ckpt/moe/680M_MOE_M8_D7/680M_M8_D7_2.25T_mixCT.32K/megatron_merge_states.pt'
    # ckpt_local_path = copy_local_path_from_hdfs(ckpt_path)
    # load_partial_pretrain(models, partial_pretrain=ckpt_local_path, model_config=model_config, download_in_shards=True)

    # build optimizer
    optim_config = config.actor_rollout_ref.actor.optim

    # hardcode for now
    optim_config.total_training_steps = 1000
    optim_config.lr_warmup_steps = 10

    optimizers, lr_schedulers = configure_optimizers(
        models=models,
        train_iters=optim_config.total_training_steps,
        lr_warmup_iters=optim_config.lr_warmup_steps,
        lr=optim_config.lr,
        adam_betas=optim_config.betas,
        adam_eps=optim_config.eps,
        weight_decay=optim_config.weight_decay,
    )

    log_gpu_memory_usage(head='After optimizer init')

    ckpt_path = 'hdfs://haruna/home/byte_data_seed/ssd_hldy/user/tiantianfan1/sft/M8_680m_SFT/checkpoints/global_step_2198'
    import omnistore

    ckpt_state = {"model": models}
    # load model and optimizer
    omnistore.MegatronCheckpointer.load(
        path=ckpt_path,
        enable_shm_download_ckpt_tmp=False,
        checkpoint_state=ckpt_state,
        loader_in_split_mode=False,
    )

    log_gpu_memory_usage(head='After omnistore load')

    optimizers[0].reload_model_params()

    log_gpu_memory_usage(head='After optimizer reload')

    actor = MegatronPPOActor(config=config.actor_rollout_ref.actor, actor_module=models, actor_optimizer=optimizers)

    # step 4: generate random data. Currently we only support TP. So we have to make sure the data is identical on every TP
    from verl.utils.model import compute_position_id_with_mask, create_random_mask
    from verl.utils.torch_functional import masked_mean

    batch_size = 2
    max_prompt_length = 4
    max_response_length = 4

    log_gpu_memory_usage(head='After constructing MegatronPPOActor')

    input_ids = torch.randint(low=0,
                              high=hf_config.vocab_size,
                              size=(batch_size, max_prompt_length + max_response_length),
                              dtype=torch.int64,
                              device='cuda')
    # broadcast to tp region to make sure that each tp contains the same data
    attention_mask = create_random_mask(input_ids=input_ids,
                                        max_ratio_of_valid_token=0.8,
                                        max_ratio_of_left_padding=0.2,
                                        min_ratio_of_valid_token=0.6)

    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)

    # dist.broadcast(input_ids, src=mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
    # dist.broadcast(attention_mask,
    #                src=mpu.get_tensor_model_parallel_src_rank(),
    #                group=mpu.get_tensor_model_parallel_group())

    # dist.broadcast(input_ids,
    #                src=mpu.get_pipeline_model_parallel_first_rank(),
    #                group=mpu.get_pipeline_model_parallel_group())
    # dist.broadcast(attention_mask,
    #                src=mpu.get_pipeline_model_parallel_first_rank(),
    #                group=mpu.get_pipeline_model_parallel_group())

    # dist.broadcast(input_ids, src=mpu.get_context_parallel_global_ranks())

    response_mask = attention_mask[:, -max_response_length:]

    data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'responses': input_ids[:, -max_response_length:],
        'old_log_probs': torch.randn(batch_size, max_response_length, dtype=torch.float32, device='cuda'),
        'advantages': torch.randn(batch_size, max_response_length, dtype=torch.float32, device='cuda'),
        'upgo_advantages': torch.randn(batch_size, max_response_length, dtype=torch.float32, device='cuda'),
        'off_policy_steps': torch.randn(batch_size, max_response_length, dtype=torch.float32, device='cuda'),
    }
    data = DataProto.from_single_dict(data=data,
                                      meta_info={
                                          'response_length': max_response_length,
                                          'use_dynamic_bsz': True,
                                          'max_token_len': 8
                                      })
    config.actor_rollout_ref.actor.ppo_max_token_len = 8

    # step 5: perform forward
    entropy, logprobs = actor.compute_log_prob(data=data)

    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        print(masked_mean(entropy, response_mask))
        print(masked_mean(logprobs, response_mask))

    metrics = actor.update_policy(data=data)

    log_gpu_memory_usage(head='Before offload_megatron_model_to_cpu')

    offload_megatron_model_to_cpu(models=models)  # offload model and grad
    # no need to offload optimizer as they are always on CPU

    log_gpu_memory_usage(head='After offload_megatron_model_to_cpu')

    load_megatron_model_to_gpu(models=models, load_grad=False)  # for inference

    entropy, logprobs = actor.compute_log_prob(data=data)

    load_megatron_model_to_gpu(models=models, load_grad=True)  # for training

    metrics = actor.update_policy(data=data)

    log_gpu_memory_usage(head='After second training')

    # if dist.get_rank() == 0:
    #     from IPython import embed
    #     embed()
    # dist.barrier()


if __name__ == "__main__":
    main()
