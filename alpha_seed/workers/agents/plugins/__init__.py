from typing import *
import os
from dataclasses import dataclass
from alpha_seed.workers.agents import load_external_module
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


@dataclass
class PluginMatchState:
    main: TagMatcher
    exclude: Union[TagMatcher, None]


class BasePlugin(ABC):

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 call_begin_tag: str,
                 call_end_tag: str,
                 result_begin_tag: str = '',
                 result_end_tag: str = '',
                 exclude_begin_tag: str = '',
                 exclude_end_tag: str = '',
                 **kwargs):
        self.tokenizer = tokenizer
        self.call_begin_tag = call_begin_tag
        self.call_end_tag = call_end_tag
        self.result_begin_tag = result_begin_tag
        self.result_end_tag = result_end_tag
        # matches between <exclude_begin_tag, exclude_end_tag> are ignored
        self.exclude_begin_tag = exclude_begin_tag
        self.exclude_end_tag = exclude_end_tag

    def get_match_state(self) -> PluginMatchState:
        main_matcher = TagMatcher(start_tag=self.call_begin_tag, end_tag=self.call_end_tag)
        exclude_matcher = None if (len(self.exclude_begin_tag) == 0 or len(self.exclude_end_tag) == 0) else TagMatcher(
            start_tag=self.exclude_begin_tag, end_tag=self.exclude_end_tag)
        return PluginMatchState(main=main_matcher, exclude=exclude_matcher)

    def add_token_match(self, token: str, state: PluginMatchState = None) -> Union[None, str]:
        """Record a token
        Args:
            token: a token (typically created by tokenizer.tokenize or tokenizer.convert_ids_to_tokens, not tokenizer.decode)
            state: matchers, if None, a new state is created
        Returns:
            matched: if not None, the matched text between tags
        """
        if state is None:
            state = self.get_match_state()
        matched = state.main.add_token_match(token, tokenizer=self.tokenizer)
        if state.exclude is not None:
            _ = state.exclude.add_token_match(token, tokenizer=self.tokenizer)
            if state.exclude.is_matching_state:
                return None

        return matched

    def add_string_match(self, text: str, state: PluginMatchState = None) -> List[str]:
        """Record a string
        Args:
            text: string to record
            state: matchers, if None, a new state is created
        Returns:
            matched: a list of matched texts between tags
        """
        if state is None:
            state = self.get_match_state()
        tokens = self.tokenizer.tokenize(text)
        all_matched = []
        for token in tokens:
            matched = state.main.add_token_match(token, tokenizer=self.tokenizer)
            if state.exclude is not None:
                _ = state.exclude.add_token_match(token, tokenizer=self.tokenizer)
                if state.exclude.is_matching_state:
                    continue
            if matched is not None:
                all_matched.append(matched)
        return all_matched

    def format_ret(self, text: str) -> str:
        return self.result_begin_tag + text + self.result_end_tag

    @abstractmethod
    async def __call__(self, call_str: str, envs: List[BaseEnv]) -> PluginResponse:
        pass


class PluginRequireMetaInfo:
    pass


def create_plugin_from_name(plugin_name: str,
                            tokenizer: AutoTokenizer,
                            external_lib: str = None,
                            *args,
                            **kwargs) -> BasePlugin:
    module = load_external_module(package_name=plugin_name,
                                  external_lib=external_lib,
                                  external_path=os.environ.get('EXTERNAL_PLUGIN_PATH', None))
    if module is None:
        import importlib
        module = importlib.import_module(f".{plugin_name.replace('/', '.')}",
                                         package="alpha_seed.workers.agents.plugins")
    return module.create_plugin(tokenizer=tokenizer, *args, **kwargs)
