import json
from omegaconf import OmegaConf
from pathlib import Path
from alpha_seed.workers.agents.envs import create_agent_envs_from_str
import torch
import torch.distributed as dist
from functools import partial
from tests.test_utils import get_config, PytestXdistEnv


def get_plugin_config(override_config=None):
    default_conf_path = (Path(__file__).parent.parent.parent / "tasks/config/ppo_trainer.yaml")
    default_conf = OmegaConf.load(default_conf_path)
    default_plugin_config = default_conf.actor_rollout_ref.rollout.plugin
    if override_config is None:
        return default_plugin_config
    config = OmegaConf.merge(default_plugin_config, override_config)
    return config


def get_basic_example_env():
    kwargs = {'env_type': 'basic', 'env_args': {'round_ndigits': 2}}
    env = create_agent_envs_from_str(f'example_env@{json.dumps(kwargs)}')[0]
    return env


def setup_dist(rank, world_size, backend='nccl'):
    xdist_env = PytestXdistEnv()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://localhost:{12135 + xdist_env.worker_id}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )


def teardown_dist():
    dist.barrier()
    dist.destroy_process_group()


def dist_worker(rank, world_size, target):
    setup_dist(rank, world_size)
    try:
        target()
    finally:
        teardown_dist()


def get_bbpe_tokenizer():
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    local_path = copy_local_path_from_hdfs(
        "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/binxingyan/bbpe155k-v6.4.3-ml.pret")
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    return tokenizer


def get_math_test_dataproto(config, tokenizer):
    import pandas as pd
    import numpy as np
    import uuid
    from mono_rl import DataProto

    sys_prompt = r"""Solve the following math problem."""
    qa_list = [
        (f"{sys_prompt}\nCalculate 1 + 2.", "3"),
        (f"{sys_prompt}\nCalculate 3 + 5.", "8"),
        (f"{sys_prompt}\nCalculate 5.3 + 2.4.", "7.7"),
        (f"{sys_prompt}\nWhat's the sum of 100 and 201.", "301"),
    ]

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
