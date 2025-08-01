import json
from typing import List

import regex as re


class FunctionCall:

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class HermesToolParser:
    """Tool parser for Hermes format, adapted from verl"""

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.tool_call_start_token = config.rollout_server.tool_call_start_token
        self.tool_call_end_token = config.rollout_server.tool_call_end_token
        self.tool_call_regex = re.compile(
            re.escape(self.tool_call_start_token) + r"(.*?)" + re.escape(self.tool_call_end_token), re.DOTALL)

    async def extract_tool_calls(self, response_text: str) -> List[FunctionCall]:
        """Extract tool calls from response text"""
        if self.tool_call_start_token not in response_text or self.tool_call_end_token not in response_text:
            return []

        matches = self.tool_call_regex.findall(response_text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                if isinstance(function_call, list):
                    for f in function_call:
                        name, arguments = f["name"], f["arguments"] if "arguments" in f else f["parameters"]
                        function_calls.append(
                            FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
                else:
                    name, arguments = function_call["name"], function_call[
                        "arguments"] if "arguments" in function_call else function_call["parameters"]
                    function_calls.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
            except Exception as e:
                pass  # Skip invalid tool calls
        return function_calls
