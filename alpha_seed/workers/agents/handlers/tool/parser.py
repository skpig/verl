import json
from typing import List, Dict, Any, Optional

import regex as re
import json
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
        self.tool_call_use_xml = config.rollout_server.tool_call_use_xml
        self.tool_call_start_token_xml = config.rollout_server.tool_call_start_token_xml
        self.tool_call_end_token_xml = config.rollout_server.tool_call_end_token_xml
        self.tool_call_regex_xml = re.compile(
            re.escape(self.tool_call_start_token_xml) + r"(.*?)" + re.escape(self.tool_call_end_token_xml), re.DOTALL)

    def xlm_to_json(self, xlm_content):
        result = []
        FN_REGEX_PATTERN = r"<function=([^>]+)>(.*?)</function>"
        FN_PARAM_REGEX_PATTERN = r"<parameter=([^>]+)>(.*?)</parameter>"
        function_matches = re.finditer(FN_REGEX_PATTERN, xlm_content, re.DOTALL)
        for function_match in function_matches:
            fn_name = function_match.group(1)
            fn_body = function_match.group(2)
            arguments = {}

            for arg_match in re.finditer(FN_PARAM_REGEX_PATTERN, fn_body, re.DOTALL):
                arg_name = arg_match.group(1)
                arg_value = arg_match.group(2)
                arguments[arg_name] = arg_value
            result.append({'name': fn_name, 'parameters': arguments})
        return json.dumps(result, indent=4, ensure_ascii=False)

    async def extract_tool_calls(self, response_text: str) -> List[FunctionCall]:
        """Extract tool calls from response text"""
        if self.tool_call_use_xml:
            return await self.extract_tool_calls_xml(response_text)
        else:
            return await self.extract_tool_calls_json(response_text)

    async def extract_tool_calls_xml(self, response_text: str) -> List[FunctionCall]:
        if self.tool_call_start_token_xml not in response_text or self.tool_call_end_token_xml not in response_text:
            return []

        matches = self.tool_call_regex_xml.findall(response_text)
        function_calls = []
        for match in matches:
            try:
                match = self.xlm_to_json(match)
                try:
                    function_call = json.loads(match)
                except:
                    function_call = eval(match)
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
                print(f"Error parsing function call: {match}. Error: {e}")
        return function_calls

    async def extract_tool_calls_json(self, response_text: str) -> List[FunctionCall]:
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
                    try:
                        function_call = json.loads(match)
                    except:
                        function_call = eval(match)
                    if isinstance(function_call, list):
                        for f in function_call:
                            name, arguments = f["name"], f["arguments"] if "arguments" in f else f["parameters"]
                            function_calls.append(
                                FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
                    else:
                        name, arguments = function_call["name"], function_call[
                            "arguments"] if "arguments" in function_call else function_call["parameters"]
                        function_calls.append(
                            FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
                except Exception as e:
                    print(f"Error parsing function call: {match}. Error: {e}")
        return function_calls


async def _extract_messages_from_dataproto(item, max_prompt_length, tokenizer, tool_schemas) -> List[Dict]:
    """Extract messages from DataProto for chat template"""
    # For simplicity, assume it's a user message
    # In practice, you might need more sophisticated parsing
    empty_prompt = tokenizer.apply_chat_template([{
        "role": "user",
        "content": ""
    }],
                                                 tools=tool_schemas,
                                                 add_generation_prompt=True,
                                                 tokenize=False)
    empty_prompt_data = await tokenizer.batch_encode_plus_async([empty_prompt], add_special_tokens=False)
    remain_length = max(0, max_prompt_length - len(empty_prompt_data.input_ids[0]))
    if remain_length == 0:
        prompt = ""
    else:
        # 这里没有考虑有tool role的情况，一开始传进来的只有sp、user、assistant角色
        raw_prompt = list(filter(lambda x: x["role"] != "system", item.non_tensor_batch['raw_prompt'][0]))
        assert len(raw_prompt) % 2 == 1, "raw_prompt must be odd length"
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        initial_prompt = ""
        for i, p in enumerate(raw_prompt):
            p['content'] = p['content'].strip()
            if i != 0:
                initial_prompt += bos_token + p['role'] + "\n" + p['content'].strip()
            else:
                initial_prompt += p['content'].strip()
            if i != len(raw_prompt) - 1:
                initial_prompt += eos_token + "\n"

        prompt_data = await tokenizer.batch_encode_plus_async([initial_prompt], add_special_tokens=False)
        prompt_data = prompt_data.input_ids[0][-remain_length:]
        prompt = tokenizer.decode(prompt_data)
    messages = [{"role": "user", "content": prompt}]

    return messages


if __name__ == "__main__":
    from omegaconf import DictConfig
    import hdfs_io
    import asyncio
    from transformers import AutoTokenizer
    from alpha_seed.utils.chat_template import CHATML_TOOL_V5
    config = {
        "rollout_server": {
            "tool_call_start_token": "<tool_call>",
            "tool_call_end_token": "</tool_call>",
            "tool_call_start_token_xml": "<seed:tool_call>",
            "tool_call_end_token_xml": "</seed:tool_call>",
            "tool_call_use_xml": True
        }
    }
    config = DictConfig(config)
    hdfs_io.copy(
        src=
        "hdfs://haruna/home/byte_data_seed/ssd_hldy/user/songyuqing/cot_sft/bbpe155k-v6.4.3-ml.pret_add_code_cot_webgpt_fc_o1search_0220",
        dst="/opt/tiger")

    tokenizer = AutoTokenizer.from_pretrained("/opt/tiger/bbpe155k-v6.4.3-ml.pret_add_code_cot_webgpt_fc_o1search_0220")
    tokenizer.chat_template = CHATML_TOOL_V5
    tool_parser = ToolParser(tokenizer, config)

    tools = [{
        'name':
            'GlobalSearch',
        'description':
            '这是一个联网搜索工具，输入搜索问题，返回网页列表与对应的摘要信息。搜索问题应该简洁清晰，复杂问题应该拆解成多步并一步一步搜索。如果没有搜索到有用的页面，可以调整问题描述（如减少限定词、更换搜索思路）后再次搜索。搜索结果质量和语种有关，对于中文资源可以尝试输入中文问题，非中资源可以尝试使用英文或对应语种。',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': '搜索问题'
                }
            },
            'required': ['query']
        }
    }, {
        'name': 'complex_func',
        'description': 'xxx',
        'parameters': {
            'type': 'object',
            'properties': {
                'complex_para': {
                    'type': 'object',
                    'description': 'yyy'
                }
            },
            'required': ['complex_para']
        }
    }, {
        'name': 'Search_Plugin_new',
        'description': 'xxx',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'str',
                    'description': 'yyy'
                },
                'result_limits': {
                    'type': 'int',
                    'description': 'zzz'
                }
            },
            'required': ['query', 'result_limits']
        }
    }]
    s = """
<seed:tool_call>
<function=GlobalSearch>
<parameter=query>2025 Freight Focus article DAT Freight & Analytics</parameter>
</function>
</seed:tool_call>

调用单个工具：
<seed:tool_call>
<function=Search_Plugin_new>
<parameter=query>2025年全球人工智能市场规模预测</parameter>
<parameter=result_limits>15</parameter>
</function>
</seed:tool_call>

调用多个工具：
<seed:tool_call>
<function=LinkReader>
<parameter=description>总结这篇文章的核心观点</parameter>
<parameter=url>http://example.com/ai-report</parameter>
</function>
<function=Search_Plugin_new>
<parameter=query>文章作者的最新研究</parameter>
<parameter=result_limits>12</parameter>
</function>
</seed:tool_call>

调用复杂参数工具：
<seed:tool_call>
<function=complex_func>
<parameter=complex_para>
{"key1": "value1", "key2": "value2"}
</parameter>
</function>
</seed:tool_call>
    """
    for f in asyncio.run(tool_parser.extract_tool_calls(s)):
        print(f.name, f.arguments)
        print("=" * 20)
