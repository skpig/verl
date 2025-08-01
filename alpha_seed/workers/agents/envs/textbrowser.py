from typing import Dict, List

import os
import json
import time
import asyncio
import aiohttp
import numpy as np
from collections import defaultdict

from alpha_seed.workers.agents.envs import BaseEnv
from alpha_seed.workers.agents.envs.utils import parse_func_call_kwargs, truncate_str_by_tokens, is_url_blocked
from verl.tools.schemas import OpenAIFunctionToolSchema

PRINT_ERROR = os.getenv('AGENT_TEXTBROWSER_PRINT_ERROR', '0') == '1'
TRIAL_ID = os.getenv("MERLIN_JOB_ID", "0")
ARNOLD_TRIAL_OWNER = os.getenv("ARNOLD_TRIAL_OWNER", "zuoxiaochen.221")


def TextBrowserView(description: str, url: str) -> str:
    """
    这是一个链接浏览工具，可以打开链接（可以是网页、pdf等）并根据需求描述汇总页面上的所有相关信息。建议对所有有价值的链接都调用该工具来获取信息，有价值的链接包括但不限于如下几种：1.任务中明确提供的网址，2.搜索结果提供的带有相关摘要的网址，3. 之前调用TextBrowserView返回的内容中包含的且判断可能含有有用信息的网址。请尽量避免自己凭空构造链接。

    Args:
        description: 需求描述文本，详细描述在当前url内想要获取的内容
        url: 目标链接，应该是一个完整的url（以 http 开头）

    Returns:
        str: 页面上的相关信息汇总
    """
    pass


async def TextBrowserAPI(url: str, description: str, metrics: Dict[str, List], global_step: int) -> str:

    if is_url_blocked(url):
        return 'This url is blocked, please try another one.'

    _start_time = time.time()
    headers = {
        "api-key": "deadf37f-f228-45a3-8a8d-1c948415fd4a",
        "Content-Type": "application/json",
        "project_id": TRIAL_ID,
        "step": str(global_step),
        "user": f"{ARNOLD_TRIAL_OWNER}@bytedance.com"
    }
    input_params = {"url": url, "description": description, "is_offline": True}
    input_str = json.dumps(input_params, ensure_ascii=False)
    body = {"api_id": "6231", "name": "TextBrowserView", "input_params": input_str}

    content = ''
    for retry in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://gpt.bytedance.net/admin/prompt/apihub/fc_proxy",
                                        json=body,
                                        headers=headers,
                                        timeout=60) as resp:
                    resp = await resp.json()
                    content = resp.get('data', {}).get('model_final_text', '')
        except Exception as e:
            PRINT_ERROR and print(f'[call_textbrowser_apihub] Error: {e}')
        if content:
            break

    metrics['retry'].append(retry)
    metrics['time'].append(time.time() - _start_time)
    metrics['failure'].append(int(content == ''))

    return content


class TextBrowserEnv(BaseEnv):

    def __init__(self, tokenizer, **kwargs):
        self._call_count = 0
        self._call_history = []
        self._metrics = defaultdict(list)
        self.max_token_len = kwargs.get('max_token_len', int(os.getenv('AGENT_TEXTBROWSER_MAX_LEN', 8192)))
        self.tokenizer = tokenizer

    def action_supported(self, action: str) -> bool:
        func_name, _ = parse_func_call_kwargs(action)
        return func_name in ["TextBrowser", "TextBrowserView"]

    async def step(self, instance_id, tool_name, tool_args: dict, global_step: int) -> str:
        assert tool_name in ["TextBrowser", "TextBrowserView"]
        action = tool_args["url"] + "\n" + tool_args["description"]
        self._call_count += 1
        if action in self._call_history:
            return "This URL and description has been called before. Please try again with another URL or description."
        self._call_history.append(action)

        content = await TextBrowserAPI(**tool_args, metrics=self._metrics, global_step=global_step)
        content, content_token_len = truncate_str_by_tokens(text=content,
                                                            max_token_len=self.max_token_len,
                                                            tokenizer=self.tokenizer)
        self._metrics['len'].append(content_token_len)
        return content

    @property
    def metrics(self) -> dict:
        metrics = {"call_count": self._call_count}
        metrics.update({f'avg_{k}': np.mean(v) for k, v in self._metrics.items()})
        metrics.update({f'max_{k}': np.max(v) for k, v in self._metrics.items()})
        return {f"textbrowser_{k}": v for k, v in metrics.items()}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        from transformers.utils import get_json_schema
        schema = get_json_schema(TextBrowserView)
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
        return tool_schema


def create_from_env_str(env_str: str, **kwargs):
    prefix = "deep_research/textbrowser@"
    assert env_str.startswith(prefix)
    tokenizer = kwargs.get("tokenizer", None)
    assert tokenizer is not None, "Must provide a tokenizer for textbrowser env"
    env_args = json.loads(env_str[len(prefix):])
    return TextBrowserEnv(tokenizer=tokenizer, **env_args)


if __name__ == "__main__":
    import argparse
    import asyncio
    import time
    import json
    import hdfs_io
    from transformers import AutoTokenizer

    hdfs_io.copy(
        src=
        "hdfs://haruna/home/byte_data_seed/ssd_hldy/user/songyuqing/cot_sft/bbpe155k-v6.4.3-ml.pret_add_code_cot_webgpt_fc_o1search_0220",
        dst="/opt/tiger")

    tokenizer = AutoTokenizer.from_pretrained("/opt/tiger/bbpe155k-v6.4.3-ml.pret_add_code_cot_webgpt_fc_o1search_0220")

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str)
    parser.add_argument("--description", type=str, default="")
    args = parser.parse_args()
    print(args)
    env = create_from_env_str(f"deep_research/textbrowser@{json.dumps(vars(args))}", tokenizer=tokenizer)
    print(asyncio.run(env.step("", "TextBrowser", {"url": args.url, "description": args.description})))
    print(env.get_openai_tool_schema())
