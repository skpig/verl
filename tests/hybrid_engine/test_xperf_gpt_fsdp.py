"""
We instantiate a mariana models. Wrap it using FSDP with FULL_SHARD. Then, feed the weights from FSDP model to XPerfGPT and perform generation
using TP

torchrun --nproc-per-node=8 --standalone tests/hybrid_engine/test_xperf_gpt_fsdp.py \
    actor_rollout_ref.actor.strategy=megatron \
    mariana.megatron.tensor_parallel_size=2 \
    mariana.megatron.pipeline_parallel_size=4 \
    mariana.megatron.virtual_pipeline_parallel_size=7

"""

import os
import numpy as np

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['USE_SESSION_CACHE'] = '0'
os.environ['MARIANA_DISABLE_ROPE_REGISTER_INV_FREQ'] = '1'

import seed_models  # noqa

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

import torch
import torch.distributed
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision

from torch.distributed.device_mesh import init_device_mesh

import hydra

from alpha_seed.workers.megatron.offload import offload_megatron_model_to_cpu, load_megatron_model_to_gpu


@hydra.main(config_path='../../tasks/config', config_name='ppo_trainer')
def main(global_config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

    p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
    p6dense_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf'
    # p6dense_path1 = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/models/p6dense-0.5B-Instruct'
    p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'
    m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf'
    p6_path_qwen = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/qwen2.5_32b_v3.1.2_o1-mini-monologue_241201_hf'
    m10_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/m10_680m_new'
    from mono_rl.utils.seed import CHAT_TEMPLATE
    from omegaconf import OmegaConf

    model_path = copy_local_path_from_hdfs(m10_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    raw_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
    tokenizer.chat_template = raw_template

    backend = global_config.actor_rollout_ref.actor.strategy

    config = AutoConfig.from_pretrained(model_path)

    if backend == 'fsdp':
        with torch.device('cpu'):
            setattr(config, '_moe_implementation', 'fused')

            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                         torch_dtype=torch.float32,
                                                         attn_implementation="flash_attention_2",
                                                         config=config)

            config = model.config
            print(config)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(module=model)
        print(auto_wrap_policy)

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            model,
            use_orig_params=True,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=False,
            device_id=torch.cuda.current_device(),
            device_mesh=device_mesh)
        from torch.distributed.fsdp import StateDictType
        FSDP.set_state_dict_type(actor_module_fsdp, StateDictType.SHARDED_STATE_DICT)
    elif backend == 'megatron':
        from alpha_seed.models.mariana.checkpoint_utils import load_partial_pretrain
        from alpha_seed.models.mariana.config_utils import convert_hf_config_to_mariana, update_megatron_config
        from alpha_seed.models.mariana.modeling_mariana import convert_gate_to_fp32
        from alpha_seed.models.mariana.optimizer_utils import configure_optimizers

        from mariana.utils.megatron import initialize_megatron_args

        from mariana.models.text.config import MegatronConfig

        # TODO: ignore pulling model file if resuming ckpt

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        megatron_config = MegatronConfig(**global_config.mariana.megatron)

        model_config = convert_hf_config_to_mariana(hf_config=config,
                                                    model_implementation=global_config.mariana.model_implementation)

        # vpp size
        update_megatron_config(model_config,
                               megatron_config,
                               vpp_size=global_config.mariana.megatron.virtual_pipeline_parallel_size)

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
        models = get_model(megatron_model_provider, ModelType.encoder_or_decoder, True, **model_kwargs)

        convert_gate_to_fp32(models)

        # load checkpoint. Note that we should load ckpt before optimizer. Otherwise, the fp32 params will be wrong.
        # we assume the megatron_merge_state.pt in the same folder as hf
        # ckpt_path = 'hdfs://haruna/home/byte_data_seed/ssd_hldy/user/tiantianfan1/sft/M8_680m_SFT/checkpoints/global_epoch_2/megatron_merge_states.pt'
        # ckpt_local_path = copy_local_path_from_hdfs(ckpt_path)
        # load_partial_pretrain(models,
        #                       partial_pretrain=ckpt_local_path,
        #                       model_config=model_config,
        #                       download_in_shards=True)

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

        # offload
        offload_megatron_model_to_cpu(models=models)

    from alpha_seed.workers.streaming_service.streaming_rollout import AsyncXPerfGPTRollout

    import xperf_gpt

    xperf_gpt.load_xperf_gpt()

    from xperf_gpt.inference.session import Query

    def make_eos_call_back_fn(device_mesh):

        def eos_callback_fn(query: Query):
            if device_mesh is None:
                tp_rank = 0
            else:
                tp_rank = device_mesh['tp'].get_local_rank()

            if tp_rank == 0:
                print(f'Rank: {torch.distributed.get_rank()}, {query.meta_info}')

        return eos_callback_fn

    rollout = AsyncXPerfGPTRollout(config=global_config.actor_rollout_ref.rollout)
    rollout.initialize(local_path=model_path)
    rollout.setup_rollout()

    if backend == 'fsdp':
        from alpha_seed.workers.hybrid_engine.fsdp_xperfgpt import FSDPXPerfGPTShardingManager
        sharding_manager = FSDPXPerfGPTShardingManager(module=actor_module_fsdp,
                                                       model_config=config,
                                                       inference_engine=rollout.inference_engine,
                                                       device_mesh=rollout.device_mesh,
                                                       backend=backend)
    elif backend == 'megatron':
        from alpha_seed.workers.hybrid_engine.fsdp_xperfgpt import MegatronXPerfGPTShardingManager
        sharding_manager = MegatronXPerfGPTShardingManager(module=models,
                                                           model_config=config,
                                                           inference_engine=rollout.inference_engine,
                                                           device_mesh=rollout.device_mesh,
                                                           backend=backend)

    eos_callback_fn = make_eos_call_back_fn(rollout.device_mesh)
    rollout.set_rollout_callback_function(eos_callback_fn=eos_callback_fn)

    from mono_rl import DataProto

    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    chat = [{'role': 'user', 'content': prompt}]

    sentences = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    # sentences = "小炒肉怎么做"
    input_data = tokenizer(sentences, return_tensors='pt').to('cuda')

    input_ids = input_data['input_ids']
    attention_mask = input_data['attention_mask']

    data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'off_policy_steps': torch.zeros(input_ids.shape[0], global_config.data.max_response_length),
        'rollout_log_probs': torch.randn(input_ids.shape[0], global_config.data.max_response_length),
        'probs_gt_threshold_num': torch.zeros(input_ids.shape[0], global_config.data.max_response_length),
        'probs_lt_threshold_sum': torch.zeros(input_ids.shape[0], global_config.data.max_response_length),
    }

    non_tensors = {
        'oj_feature':
            np.array([f'rank_{torch.distributed.get_rank()}' for i in range(input_ids.shape[0])], dtype=object)
    }

    data = DataProto.from_dict(
        data,
        non_tensors=non_tensors,
        meta_info={'generation_kwargs': global_config.actor_rollout_ref.rollout.train_generate_kwargs})

    if backend == 'megatron':
        load_megatron_model_to_gpu(models=models, load_grad=False)
    with sharding_manager:
        if backend == 'megatron':
            offload_megatron_model_to_cpu(models=models)
        data = sharding_manager.preprocess_data(data)
        output = next(rollout.generate_sequences(data))
        output = sharding_manager.postprocess_data(output)

    output_ids = output.batch['input_ids']

    if torch.distributed.get_rank() == 0:
        text_out = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        print(text_out[0].replace(tokenizer.pad_token, ''))

        # from IPython import embed
        # embed()

    torch.distributed.barrier()


if __name__ == '__main__':
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    main()
