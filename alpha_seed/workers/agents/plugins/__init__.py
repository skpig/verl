from typing import *
import os
from dataclasses import dataclass
from alpha_seed.workers.agents.envs import BaseEnv
from alpha_seed.workers.agents.plugins.tag_matcher import TagMatcher
from abc import ABC, abstractmethod


@dataclass
class PluginResponse:

    class Status:
        TIMEOUT: str = 'timeout'
        SUCCESS: str = 'success'
        FAILED: str = 'failed'

    status: Status = Status.SUCCESS
    output: str = ''

    def is_success(self) -> bool:
        return self.status == PluginResponse.Status.SUCCESS


class BasePlugin(ABC):

    def __init__(self, call_begin_tag, call_end_tag, result_begin_tag='', result_end_tag: str = ''):
        self.call_begin_tag = call_begin_tag
        self.call_end_tag = call_end_tag
        self.result_begin_tag = result_begin_tag
        self.result_end_tag = result_end_tag

    def get_match_state(self):
        return TagMatcher(start_tag=self.call_begin_tag, end_tag=self.call_end_tag)

    def add_token_match(self, token: str, state: TagMatcher = None) -> Tuple[Union[None, str], TagMatcher]:
        # NOTE: this function only return the last match
        if state is None:
            state = self.get_match_state()
        ret = None
        for c in list(token):
            matched = state.add_char_match(c)
            if matched is not None:
                ret = matched
        return ret, state

    def extract_all_call_str(self, text: str) -> List[str]:
        ret = []
        state = self.get_match_state()

        for c in list(text):
            matched = state.add_char_match(c)
            if matched is not None:
                ret.append(matched)
        return ret

    def format_ret(self, text: str) -> str:
        return self.result_begin_tag + text + self.result_end_tag

    @abstractmethod
    async def __call__(self, call_str: str, envs: List[BaseEnv]) -> PluginResponse:
        pass


def create_plugin_from_name(plugin_name, *args, **kwargs) -> BasePlugin:
    import importlib
    external_path = os.environ.get('EXTERNAL_PLUGIN_PATH', None)
    if (external_path is not None) and os.path.isfile((mod_path := os.path.join(external_path, f"{plugin_name}.py"))):
        spec = importlib.util.spec_from_file_location(plugin_name, mod_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(f".{plugin_name}", package="alpha_seed.workers.agents.plugins")
    return module.create_plugin(*args, **kwargs)
