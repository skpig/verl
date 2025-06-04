import json
import omegaconf
import asyncio
from alpha_seed.workers.agents.plugins.plugin_manager import PluginManager, PluginResponse
from alpha_seed.workers.agents.envs import create_agent_envs_from_str
from utils import get_plugin_config, get_basic_example_env


def test_call_from_sync():

    config = get_plugin_config(override_config=omegaconf.OmegaConf.create({
        "enable": True,
        "names": ['example_plugin']
    }))
    mgr = PluginManager(config)

    env = get_basic_example_env()
    match_state = mgr.get_match_state()
    call_str_dict, match_state = mgr.add_token_match(f"<plugin>Add(x=5.5, y=6.83)</plugin>", match_state)
    fut0 = mgr.async_call(call_str_dict, envs=[env], timeout=1.0)

    call_str_dict, match_state = mgr.add_token_match(f"<plugin>Sleep(seconds=1.0)</plugin>", match_state)
    fut1 = mgr.async_call(call_str_dict, envs=[env], timeout=2.0)

    call_str_dict, match_state = mgr.add_token_match(f"<plugin>Sleep(seconds=1.0)</plugin>", match_state)
    fut2 = mgr.async_call(call_str_dict, envs=[env], timeout=0.5)

    all_metrics = {}

    def update_metrics(new_metrics):
        for key, val in new_metrics.items():
            if key not in all_metrics:
                all_metrics[key] = val
            else:
                all_metrics[key] += val

    result, metrics = fut0.result()
    update_metrics(metrics)
    resp = result['example_plugin']
    assert resp.status == PluginResponse.Status.SUCCESS and resp.output == "<result>12.33</result>"

    result, metrics = fut1.result()
    update_metrics(metrics)
    resp = result['example_plugin']
    assert resp.status == PluginResponse.Status.SUCCESS and resp.output == '<result>Slept for 1.0 seconds.</result>'

    result, metrics = fut2.result()
    update_metrics(metrics)
    resp = result['example_plugin']
    assert resp.status == PluginResponse.Status.FAILED and resp.output == 'Plugin example_plugin call failed: [TimeoutError()]'

    assert all_metrics['example_plugin_success'] == 2
    assert all_metrics['example_plugin_failed'] == 1
    assert len(all_metrics['example_plugin_elapsed']) == 2


def test_call_from_async():

    config = get_plugin_config(override_config=omegaconf.OmegaConf.create({
        "enable": True,
        "names": ['example_plugin']
    }))
    mgr = PluginManager(config)

    env = get_basic_example_env()
    match_state = mgr.get_match_state()
    call_str_dict, match_state = mgr.add_token_match(f"<plugin>Add(x=5.5, y=6.83)</plugin>", match_state)

    async def func():
        return await mgr(call_str_dict, envs=[env], timeout=1.0)

    result, _ = asyncio.run(func())

    resp = result['example_plugin']
    assert resp.status == PluginResponse.Status.SUCCESS and resp.output == "<result>12.33</result>"


def test_ordered_execution():
    """Test task can wait for another task to be done"""
    config = get_plugin_config(override_config=omegaconf.OmegaConf.create({
        "enable": True,
        "names": ['example_plugin']
    }))
    mgr = PluginManager(config)

    kwargs = {'env_type': 'stateful', 'env_args': {}}
    env = create_agent_envs_from_str(f'example_env@{json.dumps(kwargs)}')[0]
    match_state = mgr.get_match_state()

    call_str_dict, match_state = mgr.add_token_match(f"<plugin>Set_(val=1)</plugin>", match_state)
    fut0 = mgr.async_call(call_str_dict, envs=[env])

    call_str_dict, match_state = mgr.add_token_match(f"<plugin>Get()</plugin>", match_state)
    fut1 = mgr.async_call(call_str_dict, envs=[env], deps=[fut0])
    res0, _ = fut0.result()
    res1, _ = fut1.result()
    assert res0['example_plugin'].output == '<result>set value=1</result>'
    assert res1['example_plugin'].output == '<result>get value=1</result>'
