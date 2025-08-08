import json
from typing import List

import regex as re

import ast


class FunctionCall:

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class ToolParser:
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
                tree = ast.parse(match)
                expr = tree.body[0].value
                assert isinstance(expr, ast.Call), "expr is not a ast.Call"
                func_name = expr.func.id
                kwargs_dict = {}
                for kw in expr.keywords:
                    key = kw.arg
                    if isinstance(kw.value, ast.Constant):
                        value = kw.value.value
                    elif isinstance(kw.value, ast.List):
                        value = [ast.literal_eval(item) for item in kw.value.elts]
                    elif isinstance(kw.value, ast.Dict):
                        keys = [ast.literal_eval(k) for k in kw.value.keys]
                        values = [ast.literal_eval(v) for v in kw.value.values]
                        value = dict(zip(keys, values))
                    else:
                        value = ast.literal_eval(ast.dump(kw.value))
                    kwargs_dict[key] = value
                function_calls.append(
                    FunctionCall(name=func_name, arguments=json.dumps(kwargs_dict, ensure_ascii=False)))
            except:
                try:
                    function_call = json.loads(match)
                    if isinstance(function_call, list):
                        function_call = function_call[0]
                    if 'arguments' in function_call:
                        name, arguments = function_call["name"], function_call["arguments"]
                    elif 'parameters' in function_call:
                        name, arguments = function_call["name"], function_call["parameters"]
                    else:
                        raise ValueError(f"Invalid function call format: {match}")
                    function_calls.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
                except Exception as e:
                    print(f"Error parsing function call: {match}. Error: {e}")
                    pass
        return function_calls
