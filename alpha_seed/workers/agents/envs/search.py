import os
import json
import time
import asyncio
from typing import Tuple

import aiohttp
import numpy as np
from collections import defaultdict

from alpha_seed.workers.agents.handlers.base_tool import BaseTool, ToolResult
from verl.tools.schemas import OpenAIFunctionToolSchema

from alpha_seed.workers.agents.envs import BaseEnv
from alpha_seed.workers.agents.envs.utils import truncate_str_by_tokens, parse_func_call_kwargs
from transformers import AutoTokenizer
import warnings
try:
    from seed.auth import apihub_auth_proxy
except ImportError:
    warnings.warn("apihub_auth_proxy is not available, must install byted-seed-sandbox latest version")
    apihub_auth_proxy = None
PRINT_ERROR = os.getenv("AGENT_SEARCH_PRINT_ERROR", "0") == "1"
SUBMITTER = os.getenv("ARNOLD_TRIAL_OWNER", "")
TRIAL_ID = os.getenv("MERLIN_JOB_ID", "0")
ARNOLD_TRIAL_OWNER = os.getenv("ARNOLD_TRIAL_OWNER", "zuoxiaochen.221")


async def apihub(query, search_engine, max_pages, global_step=0):
    if not query:
        return ''

    headers = {
        "api-key": "deadf37f-f228-45a3-8a8d-1c948415fd4a",
        "Content-Type": "application/json",
        "project-id": TRIAL_ID,
        "step": str(global_step),
        "user": f"{ARNOLD_TRIAL_OWNER}@bytedance.com"
    }
    input_params = {'search_engine': search_engine}

    if search_engine == "toutiao":
        input_params["query"] = query
        body = {"api_id": "6232", "name": "GlobalSearch", "input_params": json.dumps(input_params, ensure_ascii=False)}
    else:
        input_params["input_query"] = [query]
        input_params["search_engine"] = "usbing"
        body = {
            "api_id": "6228",
            "name": "SeedSearchTraining",
            "input_params": json.dumps(input_params, ensure_ascii=False)
        }

    pages = []

    retries = 0
    max_attempts = 3
    for i in range(max_attempts):
        retries = i
        try:
            async with aiohttp.ClientSession() as session:
                if apihub_auth_proxy:
                    async with apihub_auth_proxy.session_post(
                            session,
                            url="https://gpt.bytedance.net/admin/prompt/apihub/fc_proxy",
                            json=body,
                            headers=headers,
                            timeout=30) as resp:
                        resp = await resp.json()
                else:
                    async with session.post("https://gpt.bytedance.net/admin/prompt/apihub/fc_proxy",
                                            json=body,
                                            headers=headers,
                                            timeout=30) as resp:
                        resp = await resp.json()
        except Exception as e:
            PRINT_ERROR and print(f'[apihub] Error: {e}')
            continue

        resp_data = resp.get("data", {})
        if resp_data is None:
            PRINT_ERROR and print(f'[apihub] Error:', str(resp))
            continue

        pages = json.loads(resp['data']['result'])
        for page in pages:
            page['snippet'] = page['snippet'][:800]
            page["url"] = page["url"].replace("https://arxiv.org/abs", "https://arxiv.org/pdf"),

        if pages:
            break

    return pages[:max_pages], retries, max_attempts


def GlobalSearch(query: str) -> str:
    """
    这是一个联网搜索工具，输入搜索问题，返回网页列表与对应的摘要信息。搜索问题应该简洁清晰，复杂问题应该拆解成多步并一步一步搜索。如果没有搜索到有用的页面，可以调整问题描述（如减少限定词、更换搜索思路）后再次搜索。搜索结果质量和语种有关，对于中文资源可以尝试输入中文问题，非中资源可以尝试使用英文或对应语种。

    Args:
        query: 搜索问题

    Returns:
        str: 网页列表与对应的摘要信息
    """
    pass


async def SearchAPI(query: str, max_pages: int, search_engine: str, max_token_len: int, tokenizer: AutoTokenizer,
                    metrics: dict, global_step: int, **kwargs) -> Tuple[str, int, int]:
    """
    Access search engines to obtain information.

    Args:
        query: the search query
    """

    _start_time = time.time()
    snippets = f"Result from search query: {query}\nNo results found."

    if search_engine == "mix":
        usbing_resp, toutiao_resp = await asyncio.gather(
            apihub(query, search_engine="usbing", max_pages=max_pages, global_step=global_step),
            apihub(query, search_engine="toutiao", max_pages=max_pages, global_step=global_step))
        retries = usbing_resp[1] + toutiao_resp[1]
        max_attempts = usbing_resp[2] + toutiao_resp[2]
        pages = []
        url_set = set()
        for page in usbing_resp[0] + toutiao_resp[0]:
            if page["url"] not in url_set:
                pages.append(page)
                url_set.add(page["url"])
    else:
        pages, retries, max_attempts = await apihub(query, search_engine=search_engine, max_pages=max_pages)

    if pages:
        snippets = f"Result from search query: {query}\n"
        for page_idx, page in enumerate(pages):
            snippets += "<page{}>:\ntitle:{}\nsitename:{}\npublish_time:{}\nurl:{}\nsnippet:{}\n".format(
                page_idx, page["title"], page["sitename"], page["publish_time"], page["url"], page["snippet"])

    response, content_length = truncate_str_by_tokens(snippets, max_token_len, tokenizer)

    metrics['time'].append(time.time() - _start_time)
    metrics['len'].append(content_length)
    metrics['failure'].append(int(len(pages) == 0))

    return response, retries, max_attempts


class SearchEnv(BaseTool):

    def __init__(self, tokenizer, **kwargs):
        tool_schema = self.get_openai_tool_schema()
        super().__init__({}, tool_schema)
        self._call_count = 0
        self._call_history = []
        self._metrics = defaultdict(list)

        self.tokenizer = tokenizer

        self.max_pages = kwargs.get("max_pages", int(os.getenv("AGENT_SEARCH_MAX_PAGES", 10)))
        self.max_token_len = kwargs.get("max_token_len", int(os.getenv("AGENT_SEARCH_MAX_TOKEN_LEN", 4096)))
        self.search_engine = kwargs.get("search_engine", os.getenv("AGENT_SEARCH_ENGINE", "mix"))

        assert self.search_engine in ["toutiao", "bing", "usbing", "mix"], f"invalid search engine {self.search_engine}"

    def action_supported(self, action: str) -> bool:
        func_name, _ = parse_func_call_kwargs(action)
        return func_name in ["Search", "GlobalSearch"]

    async def execute(self, instance_id, tool_args: dict, **kwargs) -> ToolResult:
        tool_name: str = kwargs.get("tool_name")
        global_step: int = kwargs.get("global_step")
        assert tool_name in ["Search", "GlobalSearch"]
        action = tool_args["query"]
        self._call_count += 1
        if action in self._call_history:
            response = "This search query has been called before. Please try again with another query."
            retries, max_attempts = 0, 0
        else:
            self._call_history.append(action)

            tool_args.update({
                "max_pages": self.max_pages,
                "tokenizer": self.tokenizer,
                "max_token_len": self.max_token_len,
                "search_engine": self.search_engine,
                "metrics": self._metrics,
                "global_step": global_step,
            })

            response, retries, max_attempts = await SearchAPI(**tool_args)
        return ToolResult(response, retries, max_attempts)

    @property
    def metrics(self) -> dict:
        metrics = {"call_count": self._call_count}
        metrics.update({f'avg_{k}': np.mean(v) for k, v in self._metrics.items()})
        metrics.update({f'max_{k}': np.max(v) for k, v in self._metrics.items()})
        return {f"search_{k}": v for k, v in metrics.items()}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        from transformers.utils import get_json_schema
        schema = get_json_schema(GlobalSearch)
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
        return tool_schema


def create_from_env_str(env_str: str, **kwargs):
    prefix = "deep_research/search@"
    assert env_str.startswith(prefix)
    tokenizer = kwargs.get("tokenizer", None)
    assert tokenizer is not None, "Must provide a tokenizer for search env"
    env_args = json.loads(env_str[len(prefix):])
    return SearchEnv(tokenizer=tokenizer, **env_args)


if __name__ == "__main__":
    import argparse
    import asyncio
    import time
    import json
    import hdfs_io

    hdfs_io.copy(
        src=
        "hdfs://haruna/home/byte_data_seed/ssd_hldy/user/songyuqing/cot_sft/bbpe155k-v6.4.3-ml.pret_add_code_cot_webgpt_fc_o1search_0220",
        dst="/opt/tiger")

    tokenizer = AutoTokenizer.from_pretrained("/opt/tiger/bbpe155k-v6.4.3-ml.pret_add_code_cot_webgpt_fc_o1search_0220")

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str)
    args = parser.parse_args()
    print(args)
    env = create_from_env_str(f"deep_research/search@{json.dumps(vars(args))}", tokenizer=tokenizer)
    print(asyncio.run(env.execute("", {"query": f'Search(query="{args.query}")'}, tool_name="Search", global_step=0)))
    print(env.get_openai_tool_schema())
