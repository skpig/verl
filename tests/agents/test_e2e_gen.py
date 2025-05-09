import json
import os
from pathlib import Path
import pytest

from single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup

from omegaconf import OmegaConf

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs

from transformers import AutoTokenizer

import torch

import ray
import pandas as pd
import numpy as np

from utils import get_config


@pytest.fixture(scope='function')
def ray_fixture():
    ray.init()
    yield
    ray.shutdown()


def get_data(enable_plugin):
    questions = [
        "Calculate 1 + 2",
        "What time is it?",
        "Calculate 3.14 + 5.29",
        "Which year is today?",
        "Which month is today?",
        "Which date is today?",
        "Calculate 5.32 + 4.23",
        "What's the sum of 100 and 201",
    ]

    prompt_tmpl = (Path(__file__).parent / "plugin_system_prompt_v1.md").read_text()
    data = []
    for question in questions:
        content = prompt_tmpl.replace("__QUESTION__", question) if enable_plugin else question
        prompt = [{"role": "user", "content": content}]
        env_kwargs = {"env_type": "basic", "env_args": {"round_ndigits": 3}}
        env_str = f"example_env@{json.dumps(env_kwargs)}"
        data.append({"prompt": prompt, "agent_env": env_str})
    df = pd.DataFrame(data)
    return df


def to_dataproto(df, tokenizer, config):
    sentences = tokenizer.apply_chat_template(
        [prompt for prompt in df.prompt.tolist()],
        add_generation_prompt=True,
        tokenize=False,
    )

    input_data = tokenizer(sentences, padding=True, return_tensors="pt")

    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    model_output_mask = torch.zeros_like(attention_mask)

    def get_dummy_tensor(dtype=torch.bfloat16):
        response_length = config.actor_rollout_ref.rollout.response_length
        return torch.zeros(input_ids.shape[0], response_length, dtype=dtype).fill_(-1)

    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "model_output_mask": model_output_mask,
        "off_policy_steps": get_dummy_tensor(),
        "rollout_log_probs": get_dummy_tensor(),
        "probs_gt_threshold_num": get_dummy_tensor(),
        "probs_lt_threshold_sum": get_dummy_tensor(),
    }
    non_tensors = {
        "agent_env": np.array(df.agent_env.tolist(), dtype=object),
        "resume_state": np.array([None] * input_ids.shape[0], dtype=object),
    }

    gen_kwargs = config.actor_rollout_ref.rollout.train_generate_kwargs
    return DataProto.from_dict(
        data,
        non_tensors=non_tensors,
        meta_info={"generation_kwargs": gen_kwargs},
    )


def decode_output(output, tokenizer):
    output_ids = output.batch['input_ids']
    text_out = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return text_out


@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("enable_plugin", [True, False])
def test_hybrid_engine(world_size, enable_plugin, monkeypatch, ray_fixture):
    monkeypatch.setenv('TOKENIZERS_PARALLELISM', "false")
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    monkeypatch.setenv("XPERF_DUMP_NAN", "0")

    df = get_data(enable_plugin)
    assert len(df) % world_size == 0

    override_config = OmegaConf.create({
        "actor_rollout_ref": {
            "model": {
                "path": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf"
            },
            "rollout": {
                "name": "xperf_gpt",
                "enable_paged_attention": True,
                "plugin": {
                    "enable": enable_plugin,
                    "names": ["example_plugin"],
                },
            },
        }
    })
    config = get_config(override_config)
    model_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    batch = to_dataproto(df=df, tokenizer=tokenizer, config=config)
    resource_pool = RayResourcePool(process_on_nodes=[world_size], use_gpu=True)
    from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
    rollout_cls_with_init = RayClassWithInitArgs(cls=AsyncActorRolloutRefWorker,
                                                 config=config.actor_rollout_ref,
                                                 role="rollout")
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=rollout_cls_with_init)
    ray.get(wg.init_model())

    batch.meta_info.update({
        'complete_ratio': 1.0,
    })

    # mock hybrid gen
    out_batch = wg.generate_sequences(batch)
    out_text = decode_output(out_batch, tokenizer)


def _setup_rollout_server(config, tokenizer):
    import asyncio
    import threading
    from contextlib import suppress

    assert config.rollout_server.nnodes == 1

    resource_pool = RayResourcePool(process_on_nodes=[config.rollout_server.n_gpus_per_node], use_gpu=True)
    from alpha_seed.workers.streaming_service.streaming_rollout import RemoteAsyncXPerfGPTRollout
    from single_controller.ray.base import create_colocated_worker_cls
    rollout_cls_with_init = RayClassWithInitArgs(cls=RemoteAsyncXPerfGPTRollout,
                                                 config=config.actor_rollout_ref,
                                                 role="rollout_server")
    class_dict = {'rollout_server': rollout_cls_with_init}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
    wg_dict = wg.spawn(prefix_set=class_dict.keys())
    wg = wg_dict['rollout_server']
    wg.setup_rollout()

    server = None

    def start_background_loop(loop):
        asyncio.set_event_loop(loop)
        with suppress(asyncio.CancelledError):
            loop.run_forever()

    background_loop = asyncio.new_event_loop()
    background_thread = threading.Thread(target=start_background_loop, args=(background_loop,), daemon=True)
    background_thread.start()

    async def listen():
        from alpha_seed.workers.streaming_service.streaming_rollout_server import AsyncXPerfGPTRolloutServer
        server = AsyncXPerfGPTRolloutServer(config, tokenizer)
        async with server as rollout:
            rollout.attach_actors(wg)
            await asyncio.Future()

    asyncio.run_coroutine_threadsafe(listen(), background_loop)
    return server


# @pytest.mark.parametrize("world_size", [2])
# @pytest.mark.parametrize("enable_plugin", [False])
# def test_rollout_server(world_size, enable_plugin, monkeypatch, ray_fixture):
#     monkeypatch.setenv('TOKENIZERS_PARALLELISM', "false")
#     monkeypatch.setenv("NCCL_DEBUG", "WARN")
#     monkeypatch.setenv("XPERF_DUMP_NAN", "0")

#     df = get_data(enable_plugin)
#     assert len(df) % world_size == 0

#     override_config = OmegaConf.create(
#         {
#             "actor_rollout_ref": {
#                 "model": {
#                     "path": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf"
#                 },
#                 "rollout": {
#                     "name": "xperf_gpt",
#                     "enable_paged_attention": True,
#                     "plugin": {
#                         "enable": enable_plugin,
#                         "names": ["example_plugin"],
#                     },
#                 },
#             },
#             "rollout_server": {
#                 "nnodes": 1,
#                 "n_gpus_per_node": world_size,
#             }
#         }
#     )

#     config = get_config(override_config)
#     model_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     tokenizer.padding_side = "left"

#     batch = to_dataproto(df=df, tokenizer=tokenizer, config=config)
#     batch.meta_info.update( { 'complete_ratio': 1.0 })

#     server = _setup_rollout_server(config, tokenizer)
#     from alpha_seed.workers.agents.math.handler import process_single_batch
#     from alpha_seed.workers.agents import TaskContext

#     context = TaskContext(config, tokenizer, None, 0)

#     import asyncio
#     async def gen():
#         tasks = []
#         for _, item in enumerate(batch.chunk(len(batch))):
#             tasks.append(asyncio.create_task(process_single_batch(item, context)))

#         out_items = await asyncio.gather(*tasks)
#         return DataProto.concat(out_items)
#     out_batch = asyncio.run(gen())
#     import ipdb
#     ipdb.set_trace()

if __name__ == "__main__":
    pytest.main()
