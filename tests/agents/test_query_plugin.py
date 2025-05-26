from typing import *
import copy
import json
from transformers import AutoTokenizer
from pathlib import Path
from omegaconf import OmegaConf
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from alpha_seed.workers.xperf_rollout.component.query import Query, batch_sync_tp_queries
from alpha_seed.workers.xperf_rollout.component.query_plugin import TokenRole
from utils import get_plugin_config, dist_worker
import pytest


def get_tokenizer() -> AutoTokenizer:
    from verl.utils.fs import copy_local_path_from_hdfs
    path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/tokenizer/Qwen2.5-72B-tokenizer_fix_bos'
    local_path = copy_local_path_from_hdfs(path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    return tokenizer


def text_to_ids(text: str, tokenizer):
    return tokenizer(text, padding=False, return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()


def get_tp_group():
    tp_group = None if not dist.is_initialized() else dist.group.WORLD
    return tp_group


def extract_plugin_outputs(query: Query):
    output_tokens = query.output_tokens
    ret = []
    for o_range in query.plugin_query.output_ranges:
        if o_range.role == TokenRole.Tool:
            ret.append(output_tokens[o_range.start:o_range.end + 1])
    return ret


def mock_model_response(query: Query, resp_tokens: List[int]):
    tp_group = get_tp_group()
    for token in resp_tokens:
        query.add_token(token)
        if query.meet_pause_condition():
            query.reset_compute()
            while query.meet_pause_condition():
                batch_sync_tp_queries([query], tp_group=tp_group)
                query.try_resume_from_paused()
                # print('waiting...')
                time.sleep(0.1)


def get_query(config, prompt) -> Query:
    chat = [{'role': 'user', 'content': prompt}]
    tokenizer = get_tokenizer()
    prompt_str = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    input_ids = text_to_ids(prompt_str, tokenizer)
    query = Query(input_ids=copy.deepcopy(input_ids), input_prompt=prompt_str, idx=0, prefix_already_computed_len=0)

    env_kwargs = {"env_type": "basic", "env_args": {"round_ndigits": 3}}
    env_str = f"example_env@{json.dumps(env_kwargs)}"
    config = OmegaConf.to_container(config, resolve=True)
    query.meta_info = {'generation_kwargs': {'plugin_config': config}, 'extra_data': {'agent_env': [env_str],}}

    from dataclasses import dataclass

    @dataclass
    class MockInferenceSession:
        tokenizer: None
        tp_group: None

    sess = MockInferenceSession(tokenizer=tokenizer, tp_group=get_tp_group())
    query.attach_session(sess)
    return query


def _test_pause_on_eos_worker():
    override_config = OmegaConf.create({
        'enable': True,
        'names': ['example_plugin'],
        'pause_condition': "on_eos",
    })
    plugin_config = get_plugin_config(override_config)
    query = get_query(plugin_config, prompt="Calculate 1+2")
    tokenizer = query.plugin_query.tokenizer

    model_resp0 = "<think>Okay, I need to compute 1+2. I can call function Add to compute.</think>" + \
        """<plugin>Add(x=1, y=2)</plugin>""" + tokenizer.eos_token
    model_resp0_ids = text_to_ids(model_resp0, tokenizer)
    mock_model_response(query, model_resp0_ids)

    model_resp1 = "<answer>3</answer>" + tokenizer.eos_token
    model_resp1_ids = text_to_ids(model_resp1, tokenizer)
    mock_model_response(query, model_resp1_ids)

    full_response_tokens = (query.input_ids + query.new_token_ids)[query.original_input_len:]
    full_response = tokenizer.batch_decode([full_response_tokens])[0]
    expect_plugin_chat = [{'role': 'user', 'content': '<result>3</result>'}]
    expected_plugin_response = tokenizer.apply_chat_template(expect_plugin_chat,
                                                             tokenize=False,
                                                             add_generation_prompt=True)
    assert full_response == model_resp0 + expected_plugin_response + model_resp1

    plugin_tokens = extract_plugin_outputs(query)
    assert len(plugin_tokens) == 1
    assert tokenizer.batch_decode(plugin_tokens)[0] == expected_plugin_response


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_pause_on_eos(world_size):
    mp.spawn(dist_worker, args=(world_size, _test_pause_on_eos_worker), nprocs=world_size, join=True)


def _test_pause_on_trigger_worker():
    override_config = OmegaConf.create({
        'enable': True,
        'names': ['example_plugin'],
        'pause_condition': "on_trigger",
    })
    plugin_config = get_plugin_config(override_config)
    plugin_config = get_plugin_config(override_config)
    query = get_query(plugin_config, prompt="Calculate 1+2")
    tokenizer = query.plugin_query.tokenizer

    model_resp0 = "<think>Okay, I need to compute 1+2. I can call function Add to compute.</think>" + \
        """<plugin>Add(x=1, y=2)</plugin>"""
    model_resp0_ids = text_to_ids(model_resp0, tokenizer)
    mock_model_response(query, model_resp0_ids)
    model_resp1 = "<answer>3</answer>" + tokenizer.eos_token
    model_resp1_ids = text_to_ids(model_resp1, tokenizer)
    mock_model_response(query, model_resp1_ids)
    full_response_tokens = (query.input_ids + query.new_token_ids)[query.original_input_len:]
    full_response = tokenizer.batch_decode([full_response_tokens])[0]
    expected_plugin_response = "<result>3</result>"
    assert full_response == model_resp0 + expected_plugin_response + model_resp1
    plugin_tokens = extract_plugin_outputs(query)
    assert len(plugin_tokens) == 1
    assert tokenizer.batch_decode(plugin_tokens)[0] == expected_plugin_response


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_pause_on_trigger(world_size):
    mp.spawn(dist_worker, args=(world_size, _test_pause_on_trigger_worker), nprocs=world_size, join=True)


def test_resume_state():
    override_config = OmegaConf.create({
        'enable': True,
        'names': ['example_plugin'],
    })
    plugin_config = get_plugin_config(override_config)
    query = get_query(plugin_config, prompt="Calculate 1+2")

    import pickle
    state = query.get_resume_state()
    state_bytes = pickle.dumps(state)
    state = pickle.loads(state_bytes)
    query.set_resume_state(state)
