import json
import os
from pathlib import Path
import pytest
import copy
from functools import partial
import time

from single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup

from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tracking import Tracking
from verl.single_controller.ray.base import create_colocated_worker_cls

from transformers import AutoTokenizer

import torch

import ray
import pandas as pd
import numpy as np
import uuid

from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from alpha_seed.workers.actors.rollout_pool import RolloutPool
from alpha_seed.workers.streaming_service.streaming_rollout import RemoteAsyncXPerfGPTRollout
from alpha_seed.workers.streaming_service.rollout_manager import RolloutManager


@pytest.fixture(scope='function')
def ray_fixture():
    ray.init()
    yield
    ray.shutdown()


def set_common_envs(monkeypatch):
    monkeypatch.setenv('TOKENIZERS_PARALLELISM', "false")
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    monkeypatch.setenv("XPERF_DUMP_NAN", "0")


def get_dataproto(config, tokenizer):
    sys_prompt = r"""Solve the following math problem."""
    qa_list = [
        (f"{sys_prompt}\nCalculate 1 + 2.", "3"),
        (f"{sys_prompt}\nCalculate 3 + 5.", "8"),
        (f"{sys_prompt}\nCalculate 5.3 + 2.4.", "7.7"),
        (f"{sys_prompt}\nWhat's the sum of 100 and 201.", "301"),
    ]

    data = []
    reward_model = []
    for question, answer in qa_list:
        prompt = [{"role": "user", "content": question}]
        reward_model.append({'style': 'rule-lighteval/MATH', 'ground_truth': str(answer)})
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
    batch.non_tensor_batch['reward_model'] = np.array(reward_model, dtype=object)
    batch.check_consistency()
    return batch


def get_common_config():
    default_conf_path = (Path(__file__).parent.parent.parent / "tasks/config/ppo_trainer.yaml")
    default_conf = OmegaConf.load(default_conf_path)
    override_config = OmegaConf.create({
        "data": {
            "train_batch_size": 4,
        },
        "actor_rollout_ref": {
            "model": {
                # "path": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf"
                "path":
                    "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf"
            },
            "rollout": {
                "tensor_model_parallel_size": 2,
                "mode": "batch",
                "complete_ratio": 1.0,
                "rollout_pool": {
                    "warmup_step": 0,
                },
                "gpu_memory_utilization": 0.2
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

    config = OmegaConf.merge(default_conf, override_config)
    return config


def get_logger(config):
    logger = Tracking(project_name=config.trainer.project_name,
                      experiment_name=config.trainer.experiment_name,
                      default_backend=config.trainer.logger,
                      config=OmegaConf.to_container(config, resolve=True))
    return logger


def get_tokenizer(config):
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    return tokenizer


def decode_output(output, tokenizer):
    output_ids = output.batch['input_ids']
    text_out = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return text_out


def _check_score(out_text, batch):

    def reward_fn(solution_str: str, ground_truth: str):
        return 1.0 if ground_truth in solution_str else -1.0

    for text, item in zip(out_text, batch.chunk(len(batch))):
        reward_model = item.non_tensor_batch['reward_model'][0]
        score = reward_fn(solution_str=text, ground_truth=reward_model['ground_truth'])
        assert score == 1.0


def _create_rollout_wg_common(actor_rollout_ref_config, ngpus: int, role: str, name: str, is_server: bool):
    resource_pool = RayResourcePool(process_on_nodes=[ngpus], use_gpu=True, name_prefix=name)
    rollout_cls_with_init = RayClassWithInitArgs(
        cls=RemoteAsyncXPerfGPTRollout if is_server else AsyncActorRolloutRefWorker,
        config=actor_rollout_ref_config,
        role=role)
    class_dict = {name: rollout_cls_with_init}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
    wg_dict = wg.spawn(prefix_set=class_dict.keys())
    wg = wg_dict[name]

    ray.get(wg.init_model())
    return wg


def create_hybrid_wg(config):
    assert config.trainer.nnodes == 1
    return _create_rollout_wg_common(config.actor_rollout_ref,
                                     ngpus=config.trainer.n_gpus_per_node,
                                     role='rollout',
                                     name='hybrid_rollout',
                                     is_server=False)


def create_streaming_rollout_wg(config):
    if config.streaming_rollout.nnodes == 0:
        return None
    is_server = config.actor_rollout_ref.rollout.mode == 'server'
    return _create_rollout_wg_common(config.actor_rollout_ref,
                                     ngpus=config.streaming_rollout.n_gpus_per_node,
                                     role='standalone_rollout',
                                     name='streaming_rollout',
                                     is_server=is_server)


def create_streaming_validator_wg(config):
    if config.streaming_validator.nnodes == 0:
        return None
    is_server = config.actor_rollout_ref.rollout.mode == 'server'
    return _create_rollout_wg_common(config.actor_rollout_ref,
                                     ngpus=config.streaming_validator.n_gpus_per_node,
                                     role='standalone_validator',
                                     name='streaming_validator',
                                     is_server=is_server)


def create_rollout_pool(config):
    rollout_pool = RolloutPool.get_or_create_actor(config)
    return rollout_pool


class TestContext:

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(config)
        self.tokenizer = get_tokenizer(config)

        hybrid_wg = create_hybrid_wg(config)
        streaming_rollout_wg = create_streaming_rollout_wg(config)
        streaming_validator_wg = create_streaming_validator_wg(config)
        rollout_pool = create_rollout_pool(config)
        self.rollout_manager = RolloutManager(config, logger=self.logger, tokenizer=self.tokenizer)
        self.rollout_manager.initialize(hybrid_wg,
                                        rollout_pool=rollout_pool,
                                        train_standalone_wg=streaming_rollout_wg,
                                        val_standalone_wg=streaming_validator_wg)


@pytest.mark.parametrize("complete_ratio", [1.0, 0.0, 0.5])
@pytest.mark.parametrize("is_server", [True])
def test_train_generate(monkeypatch, ray_fixture, complete_ratio, is_server):
    if is_server and (0.0 < complete_ratio < 1.0):
        pytest.skip("skip is_server and 0<complete_ratio<1")

    set_common_envs(monkeypatch)
    config = get_common_config()
    has_standalone = complete_ratio < 1.0
    config.actor_rollout_ref.rollout.mode = 'server' if is_server else 'batch'
    config.actor_rollout_ref.rollout.complete_ratio = complete_ratio
    config.streaming_rollout.nnodes = 1 if has_standalone else 0
    config.streaming_validator.nnodes = 0

    ctx = TestContext(config)
    batch = get_dataproto(config, ctx.tokenizer)

    def mock_save_dataproto(data: DataProto, prefix: str = ''):
        print(f"mock savedataproto prefix={prefix}")

    input_batch = copy.deepcopy(batch)

    try:
        for i in range(3):
            start = time.time()
            batch = copy.deepcopy(input_batch)
            is_warmup_step = i == 0
            batch = ctx.rollout_manager.train_generate(batch,
                                                       step=i,
                                                       save_dataproto_fn=mock_save_dataproto,
                                                       is_warmup_step=is_warmup_step)
            if is_warmup_step:
                warmup_elapsed = time.time() - start
                assert batch is None
            else:
                out_text = decode_output(batch, ctx.tokenizer)
                print(out_text)
                _check_score(out_text, batch)
                if complete_ratio < 1.0:
                    # mock training to overlap
                    time.sleep(warmup_elapsed)
    except:
        raise
    finally:
        ctx.rollout_manager.stop_servers()


@pytest.mark.parametrize("is_standalone", [True, False])
@pytest.mark.parametrize("is_server", [True, False])
def test_val_generate(monkeypatch, ray_fixture, is_standalone, is_server):
    set_common_envs(monkeypatch)
    config = get_common_config()
    config.actor_rollout_ref.rollout.mode = 'server' if is_server else 'batch'
    config.streaming_rollout.nnodes = 0
    config.streaming_validator.nnodes = 1 if is_standalone else 0

    ctx = TestContext(config)
    batch = get_dataproto(config, ctx.tokenizer)
    batch = ctx.rollout_manager.val_generate(batch, is_async=is_standalone)
    out_text = decode_output(batch, ctx.tokenizer)
    print(out_text)
    ctx.rollout_manager.stop_servers()
    _check_score(out_text, batch)
