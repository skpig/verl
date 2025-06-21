from typing import *
import asyncio
import time
import contextlib
from alpha_seed.workers.agents.plugins import (
    BasePlugin,
    PluginRequireMetaInfo,
    PluginResponse,
    create_plugin_from_name,
)
import threading
from alpha_seed.workers.agents.envs import BaseEnv
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import json
from dataclasses import dataclass
import concurrent.futures


@dataclass
class PluginCallReq:
    name: str
    call_str: str


@dataclass
class PluginCallResp:
    name: str
    plugin_resp: PluginResponse
    metrics: Dict[str, List]


class AsyncTimer:

    def __init__(self):
        self.time = 0.0

    async def __call__(self, coro):
        start = time.perf_counter()
        result = await coro
        self.time = time.perf_counter() - start
        return result


class PluginManager:

    def __init__(self, config: Dict, tokenizer: AutoTokenizer):
        self._plugins: Dict[str, BasePlugin] = dict()

        def run_event_loop(loop):
            asyncio.set_event_loop(loop)
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_forever()

        self._loop = asyncio.new_event_loop()
        self._background_thread = threading.Thread(target=run_event_loop, args=(self._loop,), daemon=True)
        self._background_thread.start()

        args_dict = config['args']
        for name in config['names']:
            args = {} if args_dict is None else args_dict.get(name, {})
            self._plugins[name] = create_plugin_from_name(name,
                                                          tokenizer=tokenizer,
                                                          external_lib=config['plugin_external_lib'],
                                                          **args)

    def get_match_state(self) -> Dict:
        ret = dict()
        for name, plugin in self._plugins.items():
            ret[name] = plugin.get_match_state()
        return ret

    def add_token_match(self, token: str, state: Dict = None) -> List[PluginCallReq]:
        if state is None:
            state = self.get_match_state()
        call_reqs = []
        for name, plugin in self._plugins.items():
            matched = plugin.add_token_match(token, state=state[name])
            if matched is not None:
                call_reqs.append(PluginCallReq(name=name, call_str=matched))

        return call_reqs

    def add_string_match(self, text: str, state: Dict = None) -> List[PluginCallReq]:
        if state is None:
            state = self.get_match_state()
        call_reqs = []
        for name, plugin in self._plugins.items():
            all_matched = plugin.add_string_match(text, state=state[name])
            if len(all_matched) > 0:
                for matched in all_matched:
                    call_reqs.append(PluginCallReq(name=name, call_str=matched))
        return call_reqs

    @property
    def plugins(self) -> Dict[str, BasePlugin]:
        return self._plugins

    async def __call__(
        self,
        call_req: PluginCallReq,
        envs: List[BaseEnv],
        timeout: Union[float, None] = None,
        deps: List[asyncio.Future] = None,
        **kwargs,
    ) -> PluginCallResp:
        if deps is not None:
            await asyncio.gather(*deps, return_exceptions=True)

        name = call_req.name
        plugin = self._plugins[name]
        plugin_args = {
            'call_str': call_req.call_str,
            'envs': envs,
        }
        if isinstance(plugin, PluginRequireMetaInfo):
            plugin_args.update({
                'meta_info': kwargs['meta_info'],
            })
        task = asyncio.wait_for(asyncio.create_task(plugin(**plugin_args)), timeout=timeout)
        timer = AsyncTimer()
        metrics = dict()
        try:
            plugin_resp = await task
            metrics[f"{name}_elapsed"] = [timer.time]
            metrics[f"{name}_success"] = 1
        except Exception as e:
            metrics[f"{name}_failed"] = 1
            plugin_resp = PluginResponse.failed(output=plugin.format_ret(str(e)))

        return PluginCallResp(name, plugin_resp=plugin_resp, metrics=metrics)

    def async_call(
        self,
        call_req: PluginCallReq,
        envs: List[BaseEnv],
        timeout: Union[float, None] = None,
        deps: List[concurrent.futures.Future] = None,
        **kwargs,
    ) -> concurrent.futures.Future[List[PluginCallResp]]:

        aio_deps = None if deps is None else [asyncio.wrap_future(fut, loop=self._loop) for fut in deps]
        future = asyncio.run_coroutine_threadsafe(
            self.__call__(call_req, envs=envs, timeout=timeout, deps=aio_deps, **kwargs), self._loop)
        return future


PLUGIN_MANAGER_REGISTRY = None


def get_plugin_manager(config: OmegaConf, tokenizer: AutoTokenizer):
    assert isinstance(config, dict)
    global PLUGIN_MANAGER_REGISTRY
    if PLUGIN_MANAGER_REGISTRY is None:
        PLUGIN_MANAGER_REGISTRY = dict()
    config_str = json.dumps(config)
    if config_str not in PLUGIN_MANAGER_REGISTRY:
        PLUGIN_MANAGER_REGISTRY[config_str] = PluginManager(config=config, tokenizer=tokenizer)
    return PLUGIN_MANAGER_REGISTRY[config_str]
