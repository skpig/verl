"""
Adapted from tests/hybrid_engine/test_vlm.py for pure text M8 680m model
"""

import pytest
import ray
from omegaconf import OmegaConf
from tests.test_utils import ray_fixture, gpu_allocator, get_config, get_tokenizer, create_rollout_manager
from verl.utils.fs import copy_local_path_from_hdfs
from mono_rl import DataProto
import numpy as np
import uuid


def get_batch(config, tokenizer):
    from alpha_seed.utils.dataset.rl_dataset import RLHFDataset
    from alpha_seed.utils.dataset.rl_dataset import collate_fn
    from torch.utils.data import DataLoader
    from torch.utils.data import SequentialSampler

    dataset = RLHFDataset(
        parquet_files=
        "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet",
        tokenizer=tokenizer,
        prompt_key="prompt",
        answer_key="answer",
        use_ref_answer=False,
        max_prompt_length=8192,
        multi_prompts="none",
        num_prompts_per_data=1,
    )

    sampler = SequentialSampler(data_source=dataset)
    train_batch_size = 8
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
    batch = DataProto.from_single_dict(batch)
    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    batch.non_tensor_batch['rollout_id'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    return batch


@pytest.mark.parametrize("gpu_allocator", [8], indirect=True)
def test_text_gen(monkeypatch, gpu_allocator, ray_fixture):
    monkeypatch.setenv('ARNOLD_HDFS_NATIVE', 'true')
    override_config = OmegaConf.create({
        'data': {
            'max_prompt_length': 8192,
            'max_response_length': 8192,
        },
        'actor_rollout_ref': {
            "model": {
                "path": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf"
            },
            "rollout": {
                "tensor_model_parallel_size": 8,
                "enable_paged_attention": True,
                "max_ctx_batch_size": 1,
                "num_slots": 256,
                "gpu_memory_utilization": 0.5,
                "xperf_triton": {
                    "model_name": "M8",
                    "enable": True,
                    "max_batch_size": 16,
                },
            }
        },
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 8,
            "project_name": "alpha_seed_test",
            "experiment_name": "hybrid_engine_text",
            "logger": ['console'],
            "default_hdfs_dir": "./"
        }
    })
    config = get_config(override_config)
    tokenizer = get_tokenizer(config)
    tokenizer.padding_side = 'left'
    batch = get_batch(config, tokenizer)

    rollout_manager = create_rollout_manager(config)
    batch, _ = ray.get(rollout_manager.val_generate_async.remote(batch))
    prompt0_len = batch.batch['attention_mask'][0].sum()
    input_ids = batch.batch['input_ids'][batch.batch['input_ids'] != tokenizer.pad_token_id]
    input_ids = input_ids[input_ids > 0]
    response0 = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(response0)
