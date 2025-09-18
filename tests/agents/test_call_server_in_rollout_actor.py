from typing import *
from pathlib import Path
import pytest
import copy
import time

import ray
from omegaconf import OmegaConf

from mono_rl import DataProto
import pandas as pd
import numpy as np
import uuid

from alpha_seed.workers.agents.plugins import BasePlugin, PluginResponse, PluginRequireMetaInfo
from alpha_seed.workers.agents.envs import BaseEnv
from alpha_seed.workers.streaming_service.streaming_utils import chat_completions

from tests.test_utils import gpu_allocator, ray_fixture, set_common_envs, get_config, get_tokenizer, \
    create_rollout_manager, PytestXdistEnv, create_rollout_manager_with_wgs
from .utils import get_math_test_dataproto


class SummaryPlugin(BasePlugin, PluginRequireMetaInfo):

    def __init__(self, tokenizer, **kwargs):
        super().__init__(call_begin_tag='',
                         call_end_tag=tokenizer.eos_token,
                         result_begin_tag='<summary>',
                         result_end_tag='</summary>',
                         tokenizer=tokenizer,
                         **kwargs)

    async def __call__(self, call_str: str, envs: List[BaseEnv], meta_info: Dict) -> PluginResponse:
        if len(call_str) == 0:
            # return empty to end sequence
            return PluginResponse.success(output='')
        summary_prompt = f'''Summarize given message. DO NOT think, direct output the summarized result. The message is:\n{call_str}'''
        chat = [{'role': 'user', 'content': summary_prompt}]
        token_ids = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        data = {"prompt": token_ids}
        new_meta_info = {}
        new_meta_info['uid'] = str(uuid.uuid4())
        new_meta_info['reward_model'] = None
        new_meta_info['generation_kwargs'] = copy.copy(meta_info['generation_kwargs'])
        new_meta_info['generation_kwargs']['plugin_config'] = None  # disable plugin in summary query
        server_meta = meta_info['server_meta']
        config = OmegaConf.create({
            'prompt_length': 512,
            'response_length': 512,
        })

        completion = await chat_completions(data, new_meta_info, config, server_meta['host'], server_meta['port'])

        from alpha_seed.workers.agents.handlers import DataPack
        data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
        text = self.tokenizer.decode(data_pack.response_outputs[0], skip_special_tokens=False)

        return PluginResponse.success(output=self.format_ret(text))


def custom_create_plugin_from_name(name: str, **kwargs):
    assert name == 'summary'
    return SummaryPlugin(**kwargs)


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
                    'names': ['summary'],
                    'pause_condition': 'on_eos',
                },
            },
        },
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 4,
            "project_name": "alpha_seed_test",
            "experiment_name": "call_server_in_rollout_actor",
            "logger": ['console'],
        },
        "rollout_server": {
            "handler": "general/single_turn"
        }
    })
    return get_config(override_config)


def apply_patch(actor):
    import torch
    import alpha_seed.workers.agents.plugins.plugin_manager
    alpha_seed.workers.agents.plugins.plugin_manager.create_plugin_from_name = custom_create_plugin_from_name
    return DataProto.from_dict({'dummy': torch.tensor([1])})


@pytest.mark.parametrize("gpu_allocator", [4], indirect=True)
def test_summarize(monkeypatch, set_common_envs, gpu_allocator, ray_fixture):
    config = get_test_config()
    tokenizer = get_tokenizer(config)
    batch = get_math_test_dataproto(config, tokenizer)

    rollout_manager, (hybrid_wg, _, _) = create_rollout_manager_with_wgs(config)
    hybrid_wg.execute_with_func_generator(apply_patch)

    try:
        batch, _ = ray.get(rollout_manager.val_generate_async.remote(batch, is_async=False))
        out_text = tokenizer.batch_decode(batch.batch['input_ids'], skip_special_tokens=True)
        print(out_text)
    except:
        raise
    finally:
        ray.get(rollout_manager.stop_servers.remote())
