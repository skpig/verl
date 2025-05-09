import asyncio
from alpha_seed.workers.agents.plugins import create_plugin_from_name
from alpha_seed.workers.agents.plugins.plugin_manager import PluginManager, PluginResponse
from utils import get_basic_example_env


def test_function_call(monkeypatch):
    plugin = create_plugin_from_name('example_plugin')
    env = get_basic_example_env()

    async def main():
        match_state = plugin.get_match_state()
        call_str, match_state = plugin.add_token_match(f"<plugin>Add(x=5.5,y=6.83)</plugin>", match_state)
        fut0 = plugin(call_str, envs=[env])

        call_str, match_state = plugin.add_token_match(f"<plugin>Sleep(seconds=0.5)</plugin>", match_state)
        fut1 = plugin(call_str, envs=[env])

        resp0, resp1 = await asyncio.gather(fut0, fut1)
        assert resp0.status == PluginResponse.Status.SUCCESS and resp0.output == plugin.format_ret('12.33')
        assert resp1.status == PluginResponse.Status.SUCCESS and resp1.output == plugin.format_ret(
            'Slept for 0.5 seconds.')

    asyncio.run(main())
