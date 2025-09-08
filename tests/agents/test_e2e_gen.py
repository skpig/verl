import json
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from mono_rl import DataProto
import copy
import pandas as pd
import numpy as np
import uuid
import base64
from tests.test_utils import ray_fixture, gpu_allocator, set_common_envs, get_config, get_tokenizer, create_rollout_manager


def get_dataproto(config, tokenizer):
    prompt_tmpl = (Path(__file__).parent / "plugin_system_prompt_v1.md").read_text()
    import datetime
    now = datetime.datetime.now()
    qa_list = [
        ("Calculate 1 + 2.", "3"),
        ("Which year is today?", f"{now.year}"),
        ("Calculate 3.4 + 5.2.", "8.6"),
        ("Which month is today?", f"{now.month}"),
        ("Which day in a month is today?", f"{now.day}"),
        ("Calculate 5.3 + 4.2.", "9.5"),
        ("What's the sum of 100 and 201", "301"),
        ("What's the sum of 100 and current year?", f"{now.year + 100}"),
    ]

    data = []
    for question, answer in qa_list:
        prompt = [{"role": "user", "content": prompt_tmpl.replace("__QUESTION__", question)}]
        reward_model = {'style': 'custom', 'ground_truth': str(answer)}
        env_kwargs = {"env_type": "basic", "env_args": {"round_ndigits": 3}}
        env_str = f"example_env@{json.dumps(env_kwargs)}"
        data.append({"prompt": prompt, "reward_model": reward_model, "agent_env": env_str})
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
    batch.non_tensor_batch['agent_env'] = np.array(df['agent_env'].tolist(), dtype=object)
    batch.check_consistency()
    return batch


def get_plugin_enabled_common_config():
    override_config = OmegaConf.create({
        "data": {
            "train_batch_size": 8,
            "max_prompt_length": 512,
            "max_response_length": 2048,
        },
        "actor_rollout_ref": {
            "model": {
                "path": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf"
            },
            "rollout": {
                "name": "xperf_gpt",
                "tensor_model_parallel_size": 2,
                "mode": "batch",
                "complete_ratio": 1.0,
                "enable_paged_attention": False,
                "plugin": {
                    "enable": True,
                    "names": ["example_plugin"],
                },
                "gpu_memory_utilization": 0.2,
            },
        },
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 2,
            "project_name": "alpha_seed_test",
            "experiment_name": "e2e_plugin_batch",
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
        "misc": {
            "aiomonitor": {
                "enable": True,
            },
        },
    })
    return get_config(override_config=override_config)


@pytest.mark.parametrize("mode", ["server", "batch"])
@pytest.mark.parametrize("is_standalone", [True, False])
@pytest.mark.parametrize("gpu_allocator", [4], indirect=True)
def test_plugin_val_gen(set_common_envs, gpu_allocator, ray_fixture, mode, is_standalone):
    override_config = get_plugin_enabled_common_config()
    override_config.actor_rollout_ref.rollout.mode = mode
    if is_standalone:
        override_config.streaming_validator.nnodes = 1
    config = get_config(override_config)
    tokenizer = get_tokenizer(config)
    batch = get_dataproto(config, tokenizer)
    rollout_manager = create_rollout_manager(config)
    metrics = {}
    try:
        batch = rollout_manager.val_generate(batch, metrics=metrics, is_async=is_standalone)
    finally:
        rollout_manager.stop_servers()
    assert metrics[f'rollout/{"standalone" if is_standalone else "hybrid"}/plugin/example_plugin_success'] > 0
    responses = batch.batch['input_ids'][:, config.data.max_prompt_length:]
    text_out = rollout_manager.tokenizer.batch_decode(responses, skip_special_tokens=True)
    print(text_out)


def mock_save_dataproto(data: DataProto, prefix: str = ''):
    print(f"mock savedataproto prefix={prefix}")


@pytest.mark.parametrize("mode", ["server", "batch"])
@pytest.mark.parametrize("gpu_allocator", [4], indirect=True)
def test_plugin_train_gen(set_common_envs, gpu_allocator, ray_fixture, mode):
    override_config = get_plugin_enabled_common_config()
    override_config.actor_rollout_ref.rollout.mode = mode
    override_config.actor_rollout_ref.rollout.complete_ratio = 0.5 if mode == "batch" else 1.0
    override_config.streaming_rollout.nnodes = 1

    config = get_config(override_config)
    tokenizer = get_tokenizer(config)
    input_batch = get_dataproto(config, tokenizer)
    rollout_manager = create_rollout_manager(config)
    metrics = {}

    last_extra_data = None
    try:
        for i in range(2):
            batch = copy.deepcopy(input_batch)
            if last_extra_data is not None:
                batch.non_tensor_batch['extra_data'] = last_extra_data
            batch = rollout_manager.train_generate(batch,
                                                   step=i,
                                                   save_dataproto_fn=mock_save_dataproto,
                                                   is_warmup_step=False,
                                                   metrics=metrics)
            assert batch.batch['model_output_mask'].shape == batch.batch['responses'].shape
            import dill
            from alpha_seed.workers.xperf_rollout.component.query_plugin import EnvStates
            extra_data = batch.non_tensor_batch['extra_data']
            last_extra_data = extra_data
            for extra_data_item in extra_data:
                env_states_b64 = extra_data_item['env_states']
                env_state_bytes = base64.b64decode(env_states_b64)
                env_state = dill.loads(env_state_bytes)
                assert isinstance(env_state, list)
                assert isinstance(env_state[0], EnvStates)
                assert 'resume_state' not in extra_data_item, f"finished items shouldn't contain resume_state"
    finally:
        rollout_manager.stop_servers()


if __name__ == "__main__":
    import ray
    ray.init()
    test_plugin_val_gen(set_common_envs, [4], ray_fixture, "batch", True)
