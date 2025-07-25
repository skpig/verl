import copy
import time
import uuid
import ray

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from verl import DataProto

from alpha_seed.workers.streaming_service.rollout_proxy import RolloutWorkerGroupProxy
from tests.rollout.test_rollout_manager import decode_output
from tests.test_utils import gpu_allocator, ray_fixture, set_common_envs, get_config, get_tokenizer, create_rollout_manager


def mock_save_dataproto(data: DataProto, prefix: str = ''):
    print(f"mock savedataproto prefix={prefix}")


def get_dataproto(config, tokenizer):
    sys_prompt = r"""Solve the following math problem."""
    qa_list = [
        (f"{sys_prompt}\nCalculate 1 + 2.", "3"),
        (f"{sys_prompt}\nCalculate 3 + 5.", "8"),
        (f"{sys_prompt}\nCalculate 5.3 + 2.4.", "7.7"),
        (f"{sys_prompt}\nWhat's the sum of 100 and 201.", "301"),
    ]
    qa_list = qa_list * 16

    data = []
    for question, answer in qa_list:
        prompt = [{"role": "user", "content": question}]
        reward_model = {'style': 'rule-lighteval/MATH', 'ground_truth': str(answer)}
        data.append({"prompt": prompt, "reward_model": reward_model})
    df = pd.DataFrame(data)

    sentences = tokenizer.apply_chat_template(
        [prompt for prompt in df.prompt.tolist()],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_data = tokenizer(
        sentences,
        padding="max_length",
        return_tensors="pt",
        padding_side='left',
        max_length=config.data.max_prompt_length,
    )

    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    batch = DataProto.from_dict(data)
    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    batch.non_tensor_batch['rollout_id'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
    batch.non_tensor_batch['reward_model'] = np.array(df['reward_model'].tolist(), dtype=object)
    batch.check_consistency()
    return batch


def get_common_config():
    override_config = OmegaConf.create({
        "data": {
            "train_batch_size": 4,
        },
        "actor_rollout_ref": {
            "model": {
                "path": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf"
            },
            "rollout": {
                "tensor_model_parallel_size": 2,
                "mode": "batch",
                "weights_communicator": "nccl",
                "complete_ratio": 1.0,
                "rollout_pool": {
                    "warmup_step": 0,
                },
                "gpu_memory_utilization": 0.5
            },
        },
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 2,
            "project_name": "alpha_seed_test",
            "experiment_name": "rollout_manager",
            "logger": ['console'],
        },
        "streaming_rollout": {
            "nnodes": 0,
            "n_gpus_per_node": 2,
        },
        "streaming_validator": {
            "nnodes": 0,
            "n_gpus_per_node": 2,
        },
    })
    return get_config(override_config)


def _check_score(out_text, batch):

    def reward_fn(solution_str: str, ground_truth: str):
        return 1.0 if ground_truth in solution_str else -1.0

    total_score = 0
    for text, item in zip(out_text, batch.chunk(len(batch))):
        reward_model = item.non_tensor_batch['reward_model'][0]
        score = reward_fn(solution_str=text, ground_truth=reward_model['ground_truth'])
        total_score += score

    # 对的超过一半即可
    assert total_score > 0


@pytest.mark.parametrize("complete_ratio", [1.0, 0.0, 0.5])
@pytest.mark.parametrize("weights_communicator", ["ucx"])
@pytest.mark.parametrize("gpu_allocator", [8], indirect=True)
def test_train_elastic_generate(set_common_envs, gpu_allocator, ray_fixture, complete_ratio, weights_communicator):
    if complete_ratio == 1.0:
        pytest.skip("complete_ratio == 1.0 does not support atm")

    config = get_common_config()
    config.actor_rollout_ref.rollout.mode = 'server'
    config.actor_rollout_ref.rollout.weights_communicator = weights_communicator
    config.actor_rollout_ref.rollout.complete_ratio = complete_ratio
    config.streaming_rollout.elastic.enable = True
    config.streaming_rollout.elastic.min_replicas = 1
    config.streaming_rollout.elastic.max_replicas = 3
    config.streaming_rollout.nnodes = 1
    config.streaming_validator.nnodes = 0

    tokenizer = get_tokenizer(config)
    batch = get_dataproto(config, tokenizer)
    rollout_manager = create_rollout_manager(config)

    input_batch = copy.deepcopy(batch)
    try:
        # step 1
        batch = copy.deepcopy(input_batch)
        batch = rollout_manager.train_generate(batch,
                                               step=0,
                                               save_dataproto_fn=mock_save_dataproto,
                                               is_warmup_step=complete_ratio == 0.)
        # scale up
        elastic_replica = rollout_manager.train_standalone_wg.replicas

        # step 2
        batch = copy.deepcopy(input_batch)
        batch = rollout_manager.train_generate(batch,
                                               step=1,
                                               save_dataproto_fn=mock_save_dataproto,
                                               is_warmup_step=False)

        # wait for gen a bit
        time.sleep(4)

        # scale down
        ray.get(elastic_replica.scale_down(1))
        time.sleep(4)

        # step 3
        batch = copy.deepcopy(input_batch)
        batch = rollout_manager.train_generate(batch,
                                               step=2,
                                               save_dataproto_fn=mock_save_dataproto,
                                               is_warmup_step=False)
        out_text = decode_output(batch, tokenizer)
        print(out_text)
        _check_score(out_text, batch)
    except:
        raise
    finally:
        rollout_manager.stop_servers()
