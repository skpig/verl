import json

import regex as re


class FunctionCall:

    def __init__(self, name: str, arguments: str, status: bool):
        self.name = name
        self.arguments = arguments
        self.status = status


def _prepare_code(call_str: str, images_bytes: list[bytes] | None = None) -> tuple[bool, str]:
    try:
        # print(f"call_str: {call_str}")
        signature_list = json.loads(call_str)
    except Exception as e:
        raise RuntimeError(f"parse json string {call_str} failed with error: {e}")

    assert isinstance(signature_list, list), "function call content must be a list"
    assert len(
        signature_list) == 1, "function call list must have length 1 (parallel function calls are not supported yet)"
    signature = signature_list[0]
    assert "name" in signature, '"name" not in function call.'
    assert "parameters" in signature, '"parameters" not in function call.'
    name = signature["name"]
    parameters = signature["parameters"]
    assert isinstance(name, str), '"name" must be a string'
    assert isinstance(parameters, dict), '"parameters" must be a dict'

    if "imgidx" in parameters:
        assert (images_bytes is not None) and (len(images_bytes) > 0), '"imgidx" is provided but no images are provided'
        parameters["image_bytes"] = images_bytes[parameters["imgidx"]]
        parameters.pop("imgidx")

    kwargs_parsed = []
    for k, v in parameters.items():
        kwargs_parsed.append(f"{k}={repr(v)}")
    kwargs_str = ", ".join(kwargs_parsed)
    code = f"{name}({kwargs_str})"
    # TODO: hardcode name here
    return FunctionCall(name='visual_cot', arguments=code, status=True)


class VisualCotParser:
    """Tool parser for Hermes format, adapted from verl"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token = "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"
        self.tool_call_start_token = "<|FunctionCallBegin|>"
        self.tool_call_end_token = "<|FunctionCallEnd|>" + self.eos_token
        self.tool_call_regex = re.compile(
            r"<\|FunctionCallBegin\|>(.*?)<\|FunctionCallEnd\|><\[EOS_never_used_51bce0c785ca2f68081bfa7d91973934\]>",
            re.DOTALL)
        self.result_begin_tag = "<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>tool name=plugin\n"
        self.result_end_tag = "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant\n"

    def _format_ret(self, text: str) -> str:
        return self.result_begin_tag + text + self.result_end_tag

    async def extract_tool_calls(self, response_text: str, images_bytes: list[bytes]):
        """Extract tool calls from response text"""
        if not response_text.endswith(self.eos_token):
            # session 里 decode 的时候丢弃了eos
            response_text += self.eos_token

        if self.tool_call_start_token not in response_text or self.tool_call_end_token not in response_text:
            return []

        matches = self.tool_call_regex.findall(response_text)
        function_calls = []
        for match in matches:
            try:
                function_call = _prepare_code(match, images_bytes)
                function_calls.append(function_call)
            except Exception as e:
                function_call = FunctionCall(name='visual_cot', arguments=str(e), status=False)
                function_calls.append(function_call)
        return function_calls
