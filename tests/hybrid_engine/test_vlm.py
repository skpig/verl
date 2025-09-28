# pytest -vvv -s tests/hybrid_engine/test_vlm.py
import pytest
import ray
from omegaconf import OmegaConf
from tests.test_utils import ray_fixture, gpu_allocator, get_config, get_tokenizer, create_rollout_manager
from alpha_seed.utils.dataset.vlm_rl_dataset import load_and_transform_save_image
from mono_rl.utils.dataset.dist_data_util import get_dist_data_manager, init_or_get_dist_data_manager
from transformers import AutoProcessor
from verl.utils.fs import copy_local_path_from_hdfs
from mono_rl import DataProto
import numpy as np
import uuid


def get_batch(config, tokenizer, processor, model_path):
    from alpha_seed.utils.dataset.vlm_rl_dataset import RLHFDatasetVL
    from alpha_seed.utils.dataset.vlm_rl_dataset import collate_fn
    from torch.utils.data import DataLoader
    from torch.utils.data import SequentialSampler

    dataset = RLHFDatasetVL(
        parquet_files="hdfs://harunawl/home/byte_data_seed_wl/user/caisonghua/mmathcot_v4_hard_w_sys_for_rl_10.parquet",
        tokenizer=tokenizer,
        prompt_key="prompt",
        answer_key="answer",
        use_ref_answer=False,
        max_prompt_length=8192,
        multi_prompts="none",
        num_prompts_per_data=1,
        processor=processor,
        config=config,
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
    batch = DataProto.from_single_dict(batch)
    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    batch.non_tensor_batch['rollout_id'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    return batch


@pytest.mark.parametrize("gpu_allocator", [8], indirect=True)
def test_vlm_gen(monkeypatch, gpu_allocator, ray_fixture):
    override_config = OmegaConf.create({
        'data': {
            'max_prompt_length': 8192,
            'max_response_length': 8192,
            'image_key': "img",
            'think_template': 'v2',
        },
        'actor_rollout_ref': {
            "model": {
                "path": "hdfs://haruna/home/byte_data_seed/lf_lq/user/caisonghua/m8_vlm_680m_seedvit"
            },
            "rollout": {
                "tensor_model_parallel_size": 8,
                "enable_paged_attention": True,
                "max_ctx_batch_size": 1,
                "gpu_memory_utilization": 0.5,
            }
        },
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 8,
            "project_name": "alpha_seed_test",
            "experiment_name": "hybrid_engine_vlm",
            "logger": ['console'],
            "default_hdfs_dir": "./"
        },
        "elastic": {
            "resource_pools": {
                "stable_pool_names": []
            }
        }
    })
    config = get_config(override_config)
    tokenizer = get_tokenizer(config)
    tokenizer.padding_side = 'left'
    dist_data_manager = init_or_get_dist_data_manager('')
    model_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    processor = AutoProcessor.from_pretrained(model_path)
    batch = get_batch(config, tokenizer, processor, model_path)

    rollout_manager = create_rollout_manager(config)
    batch = load_and_transform_save_image(batch, tokenizer, processor, dist_data_manager, max_prompt_length=8192)
    batch, _ = ray.get(rollout_manager.val_generate_async.remote(batch))
    prompt0_len = batch.batch['attention_mask'][0].sum()
    input_ids = batch.batch['input_ids'][batch.batch['input_ids'] != tokenizer.pad_token_id]
    input_ids = input_ids[input_ids > 0]
    response0 = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(response0)
