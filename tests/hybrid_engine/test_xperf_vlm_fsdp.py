"""
We instantiate a VLM seed_models models. Wrap it using FSDP with FULL_SHARD. Then, feed the weights from FSDP model to XPerfGPT and perform generation
using TP
"""

import os
import numpy as np

from alpha_seed.workers.streaming_service.streaming_rollout import AsyncXPerfGPTRollout
from alpha_seed.workers.hybrid_engine.fsdp_xperfgpt import FSDPXPerfGPTShardingManager

import warnings
import seed_models  # noqa

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from alpha_seed.workers.fsdp.initialize import parallel_load_safetensors, parallel_init_fsdp_fn, meta_device_init, create_mesh
import torch
import torch.distributed
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForVision2Seq
from torch.distributed.device_mesh import init_device_mesh
from verl import DataProto
from omegaconf import OmegaConf
import xperf_gpt

from alpha_seed.utils.dataset.vlm_rl_dataset import RLHFDatasetVL
from transformers import AutoProcessor
from alpha_seed.utils.dataset.vlm_rl_dataset import collate_fn
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

local_rank, rank, world_size = initialize_global_process_group()

device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

vlm_path = "hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/lingyue/checkpoints/xperf/m8_vlm_15b_msv571_longct_cotv5_2_hf"
vlm_path = "hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/lingyue/checkpoints/xperf/m8_2b5_32k_seedvit_400m_baseline_openthought_8k_simplified_sys_fix_dropout_rope50"
model_path = copy_local_path_from_hdfs(vlm_path)
print(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"

max_response_length = 8192
rollout_config = OmegaConf.create({
    "prompt_length": 8192,
    "response_length": max_response_length,
    "micro_batch_size": 128,
    "tensor_model_parallel_size": 8,
    "train_generate_kwargs": {
        "do_sample": False,
        "top_k": -1,
        "top_p": 0.7,
        "temperature": 1.0,
        "min_p": -1,
        "eta_epsilon": 0,
    },
    "enable_paged_attention": True,
    "max_ctx_batch_size": 1,
    "rollout_pool": {},
    "schedule_strategy": "default",
    "gpu_memory_utilization": 0.9,
    "profile": {
        "enable": False,
        "filename": "test",
        "profile_first_n_execs": 3,
        "profile_on_ranks": [0],
        "profile_at_steps": [0],
        "default_hdfs_dir": None,
        "upload_to_mlx": False,
    },
    "xperf_custom": {
        "enable": False,
        "backbone": "m8",
    },
    "plugin": {
        "enable": False
    }
})

print(
    f"rank {torch.distributed.get_rank()} before load_xperf_gpt {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB memory",
    flush=True,
)
xperf_gpt.load_xperf_gpt()

config = AutoConfig.from_pretrained(model_path)
# set __init_sub_process so that we can quit the process after testing
# AsyncXPerfGPTRollout.__init_sub_process = lambda: None
AsyncXPerfGPTRollout._AsyncXPerfGPTRollout__init_sub_process = lambda x: None
rollout = AsyncXPerfGPTRollout(config=rollout_config)
rollout.initialize(local_path=model_path, is_standalone=False)
rollout.setup_rollout()

with meta_device_init():
    config = AutoConfig.from_pretrained(model_path)
    setattr(config, "_moe_implementation", "fused")

    model = AutoModelForVision2Seq.from_config(config,
                                               torch_dtype=torch.bfloat16,
                                               attn_implementation="flash_attention_2")

from alpha_seed.models.transformers.monkey_patch import get_parallel_plan
from alpha_seed.workers.fsdp import fully_shard

meshes = create_mesh(8, 1, 1)
fsdp_mesh, tp_mesh = meshes[:2]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    actor_module_fsdp, _ = fully_shard(
        model=model,
        block_cls=model.language_model._no_split_modules + model.vision_encoder._no_split_modules,
        fsdp_mesh=fsdp_mesh,
        tp_plan=get_parallel_plan(config, tp_mesh),
        tp_mesh=tp_mesh,
        recompute=True,
        act_offload=True,
        param_offload=True,
        weights=model_path,
    )

processor = AutoProcessor.from_pretrained(model_path)
dataset = RLHFDatasetVL(
    parquet_files=
    "hdfs://haruna/home/byte_data_seed/hl_lq/iccv/user/lingyue/data/rl/vlm_rl_alphaseed_math_stem_5.0.parquet",
    tokenizer=tokenizer,
    prompt_key="prompt",
    answer_key="answer",
    image_key="img",
    use_ref_answer=False,
    max_prompt_length=8192,
    multi_prompts="none",
    num_prompts_per_data=1,
    processor=processor,
)

sampler = SequentialSampler(data_source=dataset)
train_batch_size = 1
train_dataloader = DataLoader(
    dataset=dataset,
    batch_size=train_batch_size,
    shuffle=None,
    drop_last=True,
    collate_fn=collate_fn,
    sampler=sampler,
)
train_iter = iter(train_dataloader)
batch = next(train_iter)

sharding_manager = FSDPXPerfGPTShardingManager(
    module=actor_module_fsdp,
    model_config=config,
    inference_engine=rollout.inference_engine,
    device_mesh=rollout.device_mesh,
)

input_ids = batch["input_ids"]
attention_mask = batch["attention_mask"]

data = {"input_ids": input_ids, "attention_mask": attention_mask}
data["rollout_log_probs"] = torch.zeros(
    input_ids.shape[0],
    max_response_length,
    dtype=torch.bfloat16,
    device=input_ids.device,
).fill_(-1)
data["off_policy_steps"] = torch.zeros(
    input_ids.shape[0],
    max_response_length,
    dtype=torch.bfloat16,
    device=input_ids.device,
).fill_(-1)

data["probs_lt_threshold_sum"] = torch.zeros(
    input_ids.shape[0],
    max_response_length,
    dtype=torch.bfloat16,
    device=input_ids.device,
).fill_(-1)
data["probs_gt_threshold_num"] = torch.zeros(
    input_ids.shape[0],
    max_response_length,
    dtype=torch.bfloat16,
    device=input_ids.device,
).fill_(-1)

non_tensors = {
    "oj_feature": np.array(
        [f"rank_{torch.distributed.get_rank()}" for i in range(input_ids.shape[0])],
        dtype=object,
    )
}
non_tensor_keys = ["pixel_values", "image_grid_hw"]
for key in non_tensor_keys:
    non_tensors[key] = batch[key]

data = DataProto.from_dict(
    data,
    non_tensors=non_tensors,
    meta_info={"generation_kwargs": rollout_config.train_generate_kwargs},
)


def generate_sequences(prompts):
    complete_ratio = prompts.meta_info.get('complete_ratio', 1)
    max_new_tokens = prompts.meta_info.get('generation_kwargs').get('max_new_tokens', rollout_config.response_length)

    prompt_ids = prompts.batch['input_ids']  # (bs, prompt_length)
    batch_size = prompt_ids.shape[0]
    # left-padded attention_mask
    off_turn_off_policy_steps = prompts.batch["off_policy_steps"]
    first_non_one_indices = (prompt_ids != tokenizer.pad_token_id).int().argmax(dim=1)
    rmv_padding_prompt_ids = [row[index:].tolist() for row, index in zip(prompt_ids, first_non_one_indices)]

    # (zhangchi.usc1992) note, here we pass all the non_tensor_batch and meta_info to the inference engine as prompt_meta_info.
    prompt_meta_info = [{
        "off_policy_steps": max(off_policy_step)
    } for off_policy_step in off_turn_off_policy_steps.tolist()]
    for key, value in prompts.non_tensor_batch.items():
        for i in range(batch_size):
            prompt_meta_info[i][key] = value[i]
        # self.input_queue.put(
        # (rmv_padding_prompt_ids, complete_ratio, prompts.meta_info['generation_kwargs'], prompt_meta_info))
    query_pool = rmv_padding_prompt_ids
    import copy
    original_query_pool = copy.deepcopy(query_pool)
    generation_kwargs = prompts.meta_info['generation_kwargs']
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
    from alpha_seed.workers.streaming_service.streaming_rollout import omegaconf_config_to_py_obj
    generation_kwargs = omegaconf_config_to_py_obj(generation_kwargs)
    rollout.inference_engine.set_generator_strategy(**generation_kwargs)
    rollout.reset_status()
    rollout.inference_engine.execute(query_pool,
                                     complete_ratio=complete_ratio,
                                     stop_event=rollout.stop_event if rollout.is_standalone else None,
                                     prompt_meta_info=prompt_meta_info)

    response_outputs = []
    response_log_probs = []
    response_probs_gt_threshold_num = []
    response_probs_lt_threshold_sum = []
    is_finished = []
    off_policy_steps = []
    for prompt, v in zip(original_query_pool, rollout.inference_engine.get_inorder_responses()):
        if torch.distributed.get_rank() == 0:
            print(f"=============== idx: {v.idx} input: {v.input_prompt}\noutput: {v.output_prompt}", flush=True)
        response_outputs.append((v.input_ids + v.new_token_ids)[len(prompt):])
        response_log_probs.append(v.new_token_log_probs)
        response_probs_gt_threshold_num.append(v.probs_gt_threshold_num)
        response_probs_lt_threshold_sum.append(v.probs_lt_threshold_sum)
        is_finished.append(v.is_finished)
        off_policy_steps.append([-1] * len(v.new_token_log_probs))
    return response_outputs


with sharding_manager:
    data = sharding_manager.preprocess_data(data)
    output = generate_sequences(data)
    response0 = tokenizer.decode(output[0], skip_special_tokens=True)
    assert 'boxed' in response0
