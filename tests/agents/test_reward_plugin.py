from typing import *
import pytest
import ray

from omegaconf import OmegaConf

import asyncio
from mono_rl import DataProto
import pandas as pd
import numpy as np
import dill
import uuid

import inspect
from transformers import AutoTokenizer
from alpha_seed.workers.agents.plugins import BasePlugin, PluginResponse, PluginRequireMetaInfo
from alpha_seed.workers.agents.envs import BaseEnv

from tests.test_utils import gpu_allocator, ray_fixture, set_common_envs, get_config, get_tokenizer, \
    create_rollout_manager, PytestXdistEnv, create_rollout_manager_with_wgs
from .utils import get_math_test_dataproto
from alpha_seed.utils.reward_score.utils import Verifier


async def reward_fn(response: str, reward_model: Dict, config: Dict, tokenizer: AutoTokenizer):
    from tasks.main_ppo import post_process_solution_str
    ground_truth = reward_model['ground_truth']
    reward_style = reward_model['style']
    config = OmegaConf.create(config)
    solution_str_post_proc = post_process_solution_str(config=config,
                                                       solution_str=response,
                                                       eos_token=tokenizer.eos_token)
    data_uid = str(uuid.uuid4())
    compute_score_fn = Verifier.get_verifier(reward_style, config=config, with_external=False).compute_score_client

    score_fn_inputs = {
        "solution_str": solution_str_post_proc,
        "ground_truth": ground_truth,
        "config": config,
        'data_uid': data_uid,
        "solution_len": None,
        "solution_ids": None,
        'rm_name': None,
        'pause_tokens_index': None
    }
    if inspect.iscoroutinefunction(compute_score_fn):
        score = await compute_score_fn(**score_fn_inputs)
    else:
        score = compute_score_fn(**score_fn_inputs)
    return score


class RewardEnv(BaseEnv):

    def __init__(self):
        self._reward = -1.0

    def action_supported(self, action):
        return True

    async def step(self, action: str, reward_model: Dict, config: Dict, tokenizer: AutoTokenizer) -> str:
        self._reward = await reward_fn(response=action, reward_model=reward_model, config=config, tokenizer=tokenizer)
        return ''

    @property
    def reward(self) -> float:
        return self._reward


class RewardPlugin(BasePlugin, PluginRequireMetaInfo):

    def __init__(self, tokenizer, **kwargs):
        super().__init__(call_begin_tag='',
                         call_end_tag=tokenizer.eos_token,
                         result_begin_tag='',
                         result_end_tag='',
                         tokenizer=tokenizer,
                         **kwargs)

    async def __call__(self, call_str: str, envs: List[BaseEnv], meta_info: Dict) -> PluginResponse:
        if len(call_str) == 0:
            # return empty to end sequence
            return PluginResponse.success(output='')
        assert len(envs) == 1 and isinstance(envs[0], RewardEnv)
        reward_model = meta_info['reward_model']
        config = meta_info['extra_data']['config']

        try:
            await envs[0].step(call_str, reward_model=reward_model, config=config, tokenizer=self.tokenizer)
        except:
            pass
        return PluginResponse.success(output='')


def get_test_config():
    override_config = OmegaConf.create({
        "actor_rollout_ref": {
            "model": {
                "path": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf"
                # "path": "hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/jiangchengquan/models/qwen3_0.6b_p6d"
            },
            "rollout": {
                "tensor_model_parallel_size": 2,
                "mode": "server",
                "weights_communicator": "nccl",
                "complete_ratio": 1.0,
                "rollout_pool": {
                    "warmup_step": 0,
                },
                "gpu_memory_utilization": 0.2,
                "plugin": {
                    'enable': True,
                    'names': ['reward'],
                    'pause_condition': 'on_trigger',
                },
            },
        },
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 4,
            "project_name": "alpha_seed_test",
            "experiment_name": "reward_with_plugin",
            "logger": ['console'],
        },
        "rollout_server": {
            "handler": "general/single_turn"
        }
    })
    return get_config(override_config)


def custom_create_plugin_from_name(name: str, **kwargs):
    assert name == 'reward', f"{name}"
    return RewardPlugin(**kwargs)


def custom_create_agent_envs_from_str(env_strs: List[str], **kwargs) -> List[BaseEnv]:
    assert env_strs == ['reward_env@{}'], f"{env_strs}"
    return [RewardEnv()]


def compute_score_with_env(env_state_bytes, *args, **kwargs):
    env_states = dill.loads(env_state_bytes)
    return env_states[0].reward


def custom_select_rm_score_fn(reward_style, with_external=True):
    assert reward_style == 'rule-lighteval/MATH'
    if with_external:
        return compute_score_with_env
    else:
        from alpha_seed.utils.reward_score import math_v1
        return math_v1.compute_score


class CustomVerifier(Verifier, reward_style="custom"):

    def __init__(self, config=None, tokenizer=None, reward_style="rule-lighteval/MATH", with_external=True) -> None:
        super().__init__(config=config, tokenizer=tokenizer)
        assert reward_style == 'rule-lighteval/MATH', f"{reward_style}"
        self.reward_style = reward_style
        self.with_external = with_external

    def compute_score(self, *args, **kwargs) -> float:
        if self.with_external:
            return compute_score_with_env(*args, **kwargs)
        else:
            from alpha_seed.utils.reward_score import math_v1
            return math_v1.compute_score(*args, **kwargs)


def apply_patch(actor):
    import torch
    import alpha_seed.workers.agents.plugins.plugin_manager
    import alpha_seed.workers.xperf_rollout.component.query_plugin
    import tasks.main_ppo
    alpha_seed.workers.agents.plugins.plugin_manager.create_plugin_from_name = custom_create_plugin_from_name
    alpha_seed.workers.xperf_rollout.component.query_plugin.create_agent_envs_from_str = custom_create_agent_envs_from_str

    return DataProto.from_dict({'dummy': torch.tensor([1])})


@pytest.mark.parametrize("gpu_allocator", [4], indirect=True)
def test_reward_with_plugin(monkeypatch, set_common_envs, gpu_allocator, ray_fixture):
    config = get_test_config()
    tokenizer = get_tokenizer(config)
    batch = get_math_test_dataproto(config, tokenizer)
    batch.non_tensor_batch['agent_env'] = np.array([['reward_env@{}'] for _ in range(len(batch))], dtype=object)

    rollout_manager, (hybrid_wg, _, _) = create_rollout_manager_with_wgs(config)
    apply_patch(None)
    hybrid_wg.execute_with_func_generator(apply_patch)

    try:
        batch, _ = ray.get(rollout_manager.val_generate_async.remote(batch, is_async=False))
        for item in batch.chunk(len(batch)):
            reward_model = item.non_tensor_batch['reward_model'][0]
            extra_data = item.non_tensor_batch['extra_data'][0]
            import base64
            env_state_bytes = base64.b64decode(extra_data['env_states'])
            rm_fn = CustomVerifier(config=config, tokenizer=tokenizer).compute_score_client
            score = rm_fn(ground_truth=reward_model['ground_truth'], env_state_bytes=env_state_bytes)
            response = tokenizer.decode(batch.batch['input_ids'][0], skip_special_tokens=True)
            print("response:", response)
            print("score", score)
            # assert score == 1.0
    except:
        raise
    finally:
        ray.get(rollout_manager.stop_servers.remote())
