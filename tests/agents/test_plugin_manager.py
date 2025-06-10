import json
import omegaconf
import asyncio
from alpha_seed.workers.agents.plugins.plugin_manager import PluginManager, PluginResponse
from alpha_seed.workers.agents.envs import create_agent_envs_from_str
from utils import get_plugin_config, get_basic_example_env, get_bbpe_tokenizer


def test_call_from_sync():

    config = get_plugin_config(override_config=omegaconf.OmegaConf.create({
        "enable": True,
        "names": ['example_plugin']
    }))
    tokenizer = get_bbpe_tokenizer()
    mgr = PluginManager(config, tokenizer=tokenizer)

    env = get_basic_example_env()
    match_state = mgr.get_match_state()
    call_reqs = mgr.add_string_match(f"<plugin>Add(x=5.5, y=6.83)</plugin>", match_state)
    assert len(call_reqs) == 1
    fut0 = mgr.async_call(call_reqs[0], envs=[env], timeout=1.0)

    call_reqs = mgr.add_string_match(f"<plugin>Sleep(seconds=1.0)</plugin>", match_state)
    assert len(call_reqs) == 1
    fut1 = mgr.async_call(call_reqs[0], envs=[env], timeout=2.0)

    call_reqs = mgr.add_string_match(f"<plugin>Sleep(seconds=1.0)</plugin>", match_state)
    assert len(call_reqs) == 1
    fut2 = mgr.async_call(call_reqs[0], envs=[env], timeout=0.5)

    all_metrics = {}

    def update_metrics(new_metrics):
        for key, val in new_metrics.items():
            if key not in all_metrics:
                all_metrics[key] = val
            else:
                all_metrics[key] += val

    resp = fut0.result()
    update_metrics(resp.metrics)
    assert resp.plugin_resp.is_success() and resp.plugin_resp.output == "<result>12.33</result>"

    resp = fut1.result()
    update_metrics(resp.metrics)
    assert resp.plugin_resp.is_success() and resp.plugin_resp.output == '<result>Slept for 1.0 seconds.</result>'

    resp = fut2.result()
    update_metrics(resp.metrics)
    assert resp.plugin_resp.status == PluginResponse.Status.FAILED and resp.plugin_resp.output == 'Plugin example_plugin call failed: [TimeoutError()]'

    assert all_metrics['example_plugin_success'] == 2
    assert all_metrics['example_plugin_failed'] == 1
    assert len(all_metrics['example_plugin_elapsed']) == 2


def test_call_from_async():

    config = get_plugin_config(override_config=omegaconf.OmegaConf.create({
        "enable": True,
        "names": ['example_plugin']
    }))
    tokenizer = get_bbpe_tokenizer()
    mgr = PluginManager(config, tokenizer=tokenizer)

    env = get_basic_example_env()
    match_state = mgr.get_match_state()
    call_reqs = mgr.add_string_match(f"<plugin>Add(x=5.5, y=6.83)</plugin>", match_state)
    assert len(call_reqs) == 1

    async def func():
        return await mgr(call_reqs[0], envs=[env], timeout=1.0)

    resp = asyncio.run(func()).plugin_resp
    assert resp.is_success() and resp.output == "<result>12.33</result>"


def test_ordered_execution():
    """Test task can wait for another task to be done"""
    config = get_plugin_config(override_config=omegaconf.OmegaConf.create({
        "enable": True,
        "names": ['example_plugin']
    }))
    tokenizer = get_bbpe_tokenizer()
    mgr = PluginManager(config, tokenizer=tokenizer)

    kwargs = {'env_type': 'stateful', 'env_args': {}}
    env = create_agent_envs_from_str(f'example_env@{json.dumps(kwargs)}')[0]
    match_state = mgr.get_match_state()

    call_reqs = mgr.add_string_match(f"<plugin>Set_(val=1)</plugin>", match_state)
    assert len(call_reqs) == 1
    fut0 = mgr.async_call(call_reqs[0], envs=[env])

    call_reqs = mgr.add_string_match(f"<plugin>Get()</plugin>", match_state)
    assert len(call_reqs) == 1
    fut1 = mgr.async_call(call_reqs[0], envs=[env], deps=[fut0])
    res0 = fut0.result().plugin_resp
    res1 = fut1.result().plugin_resp
    assert res0.is_success() and res0.output == '<result>set value=1</result>'
    assert res1.is_success() and res1.output == '<result>get value=1</result>'
