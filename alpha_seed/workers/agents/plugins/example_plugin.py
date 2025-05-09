import asyncio
import json
from typing import *
from alpha_seed.workers.agents.envs import BaseEnv
from alpha_seed.workers.agents.plugins import BasePlugin, PluginResponse


class ExamplePlugin(BasePlugin):

    def __init__(self, **kwargs):
        super().__init__(call_begin_tag="<plugin>",
                         call_end_tag="</plugin>",
                         result_begin_tag="<result>",
                         result_end_tag="</result>")

    async def __call__(self, call_str: str, envs: List[BaseEnv]) -> PluginResponse:
        tasks = []
        for env in envs:
            if env.action_supported(action=call_str):
                tasks.append(asyncio.create_task(env.step(call_str)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        output = "\n".join([str(res) for res in results])
        return PluginResponse(
            status=PluginResponse.Status.SUCCESS,
            output=self.format_ret(output),
        )


def create_plugin(*args, **kwargs):
    return ExamplePlugin(*args, **kwargs)
