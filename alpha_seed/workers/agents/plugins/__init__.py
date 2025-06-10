from typing import *
import os
from dataclasses import dataclass
from alpha_seed.workers.agents.envs import BaseEnv
from alpha_seed.workers.agents.plugins.tag_matcher import TagMatcher
from transformers import AutoTokenizer
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

    @staticmethod
    def success(output: str) -> 'PluginResponse':
        return PluginResponse(status=PluginResponse.Status.SUCCESS, output=output)

    @staticmethod
    def failed(output: str) -> 'PluginResponse':
        return PluginResponse(status=PluginResponse.Status.FAILED, output=output)

    @staticmethod
    def timeout(output: str) -> 'PluginResponse':
        return PluginResponse(status=PluginResponse.Status.TIMEOUT, output=output)


class BasePlugin(ABC):

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 call_begin_tag: str,
                 call_end_tag: str,
                 result_begin_tag: str = '',
                 result_end_tag: str = '',
                 **kwargs):
        self.tokenizer = tokenizer
        self.call_begin_tag = call_begin_tag
        self.call_end_tag = call_end_tag
        self.result_begin_tag = result_begin_tag
        self.result_end_tag = result_end_tag

    def get_match_state(self):
        return TagMatcher(start_tag=self.call_begin_tag, end_tag=self.call_end_tag)

    def add_token_match(self, token: str, state: TagMatcher = None) -> Union[None, str]:
        """Record a token
        Args:
            token: a token (typically created by tokenizer.tokenize or tokenizer.convert_ids_to_tokens, not tokenizer.decode)
            state: tag matcher, if None, a new tag matcher is created
        Returns:
            matched: if not None, the matched text between tags
        """
        if state is None:
            state = self.get_match_state()
        matched = state.add_token_match(token, tokenizer=self.tokenizer)
        return matched

    def add_string_match(self, text: str, state: TagMatcher = None) -> List[str]:
        """Record a string
        Args:
            text: string to record
            state: tag matcher, if None, a new tag matcher is created
        Returns:
            matched: a list of matched texts between tags
        """
        if state is None:
            state = self.get_match_state()
        all_matched = state.add_string_match(text, tokenizer=self.tokenizer)
        return all_matched

    def format_ret(self, text: str) -> str:
        return self.result_begin_tag + text + self.result_end_tag

    @abstractmethod
    async def __call__(self, call_str: str, envs: List[BaseEnv]) -> PluginResponse:
        pass


def create_plugin_from_name(plugin_name: str, tokenizer: AutoTokenizer, *args, **kwargs) -> BasePlugin:
    import importlib
    external_path = os.environ.get('EXTERNAL_PLUGIN_PATH', None)
    if (external_path is not None) and os.path.isfile((mod_path := os.path.join(external_path, f"{plugin_name}.py"))):
        spec = importlib.util.spec_from_file_location(plugin_name, mod_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(f".{plugin_name}", package="alpha_seed.workers.agents.plugins")
    return module.create_plugin(tokenizer=tokenizer, *args, **kwargs)
