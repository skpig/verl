import json
from typing import List, Dict

import regex as re
import xml.etree.ElementTree as ET
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
        # 预处理：为内容添加一个根标签，使XML格式完整
        xlm_content = f"<root>{xlm_content}</root>"

        # 预处理：将<function=xxx>格式转换为标准XML标签<function name="xxx">
        xlm_content = re.sub(r'<function=(.*?)>', r'<function name="\1">', xlm_content)
        xlm_content = re.sub(r'<parameter=(.*?)>', r'<parameter name="\1">', xlm_content)

        # 解析XML内容
        root = ET.fromstring(xlm_content)

        # 初始化结果列表
        result = []

        # 遍历所有function节点
        for function in root.findall('function'):
            func_info = {"name": function.attrib.get('name'), "parameters": {}}

            # 提取参数信息
            for param in function.findall('parameter'):
                param_name = param.attrib.get('name')
                try:
                    param_value = json.loads(param.text)
                except:
                    param_value = param.text
                func_info["parameters"][param_name] = param_value

            result.append(func_info)

        # 转换为JSON字符串并返回
        return json.dumps(result, ensure_ascii=False)

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
                    function_call = json.loads(match)
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
                    pass
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
