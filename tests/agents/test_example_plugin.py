import asyncio
from alpha_seed.workers.agents.plugins import create_plugin_from_name
from alpha_seed.workers.agents.plugins.plugin_manager import PluginManager, PluginResponse
from .utils import get_basic_example_env, get_bbpe_tokenizer


def test_function_call(monkeypatch):
    tokenizer = get_bbpe_tokenizer()
    plugin = create_plugin_from_name('example_plugin', tokenizer=tokenizer)
    env = get_basic_example_env()

    async def main():
        match_state = plugin.get_match_state()
        call_str_list = plugin.add_string_match(f"<plugin>Add(x=5.5,y=6.83)</plugin>", match_state)
        fut0 = plugin(call_str_list[0], envs=[env])

        call_str_list = plugin.add_string_match(f"<plugin>Sleep(seconds=0.5)</plugin>", match_state)
        fut1 = plugin(call_str_list[0], envs=[env])

        resp0, resp1 = await asyncio.gather(fut0, fut1)
        assert resp0.status == PluginResponse.Status.SUCCESS and resp0.output == plugin.format_ret('12.33')
        assert resp1.status == PluginResponse.Status.SUCCESS and resp1.output == plugin.format_ret(
            'Slept for 0.5 seconds.')

    asyncio.run(main())


def test_exclude_think(monkeypatch):
    tokenizer = get_bbpe_tokenizer()
    plugin = create_plugin_from_name('example_plugin', exclude_think=True, tokenizer=tokenizer)
    env = get_basic_example_env()

    async def main():
        match_state = plugin.get_match_state()
        call_str_list = plugin.add_string_match(f"<plugin>Add(x=5.5,y=6.83)</plugin>", match_state)
        fut0 = plugin(call_str_list[0], envs=[env])
        resp0 = await fut0

        call_str_list = plugin.add_string_match(f"<think><plugin>Sleep(seconds=0.5)</plugin></think>", match_state)
        assert len(call_str_list) == 0

        assert resp0.status == PluginResponse.Status.SUCCESS and resp0.output == plugin.format_ret('12.33')

    asyncio.run(main())
