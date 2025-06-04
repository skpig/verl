from typing import *
import asyncio
import time
import contextlib
from alpha_seed.workers.agents.plugins import (
    BasePlugin,
    PluginResponse,
    create_plugin_from_name,
)
import threading
from alpha_seed.workers.agents.envs import BaseEnv
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import json
import concurrent.futures


class AsyncTimer:

    def __init__(self):
        self.time = 0.0

    async def __call__(self, coro):
        start = time.perf_counter()
        result = await coro
        self.time = time.perf_counter() - start
        return result


class PluginManager:

    def __init__(self, config: Dict, tokenizer=None):
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
            self._plugins[name] = create_plugin_from_name(name, tokenizer=tokenizer, **args)

    def get_match_state(self) -> Dict:
        ret = dict()
        for name, plugin in self._plugins.items():
            ret[name] = plugin.get_match_state()
        return ret

    def add_token_match(self, token: str, state: Dict = None) -> Tuple[Dict[str, str], Dict]:
        """Extrace plugin call strings from text
        Returns a dict, key is the name of plugins (only with non-empty calls are included),
        value is matched call_str
        """
        if state is None:
            state = self.get_match_state()
        ret = dict()
        for name, plugin in self._plugins.items():
            matched, state[name] = plugin.add_token_match(token, state=state[name])
            if matched is not None:
                ret[name] = matched

        return ret, state

    @property
    def plugins(self) -> Dict[str, BasePlugin]:
        return self._plugins

    async def __call__(
        self,
        call_str_dict: Dict[str, str],
        envs: List[BaseEnv],
        timeout: Union[float, None] = None,
        deps: List[asyncio.Future] = None,
    ) -> Tuple[Dict[str, PluginResponse], Dict]:
        results_dict = dict()
        tasks = []
        timers = []
        names = []
        metrics = dict()

        if deps is not None:
            await asyncio.gather(*deps, return_exceptions=True)

        for name, call_str in call_str_dict.items():
            plugin = self._plugins[name]
            task = asyncio.wait_for(asyncio.create_task(plugin(call_str, envs=envs)), timeout=timeout)
            timer = AsyncTimer()
            tasks.append(timer(task))
            timers.append(timer)
            names.append(name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, timer, res in zip(names, timers, results):
            if (key := f"{name}_elapsed") not in metrics:
                metrics[key] = []

            if isinstance(res, Exception):
                if (key := f"{name}_failed") not in metrics:
                    metrics[key] = 0
                metrics[key] += 1
                msg = f"Plugin {name} call failed: [{repr(res)}]"
                response = PluginResponse(status=PluginResponse.Status.FAILED, output=msg)
            else:
                metrics[key].append(timer.time)
                if (key := f"{name}_success") not in metrics:
                    metrics[key] = 0
                metrics[key] += 1
                response = res
            results_dict[name] = response
        return results_dict, metrics

    def async_call(
        self,
        call_str_dict: Dict[str, str],
        envs: List[BaseEnv],
        timeout: Union[float, None] = None,
        deps: List[concurrent.futures.Future] = None,
    ) -> concurrent.futures.Future[Dict[str, PluginResponse], Dict]:

        aio_deps = None if deps is None else [asyncio.wrap_future(fut, loop=self._loop) for fut in deps]
        future = asyncio.run_coroutine_threadsafe(
            self.__call__(call_str_dict, envs=envs, timeout=timeout, deps=aio_deps), self._loop)
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
