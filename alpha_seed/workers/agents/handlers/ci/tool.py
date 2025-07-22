import functools
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
import re
import os
import time
import requests
from typing import Tuple, Any
import json
from bytedance import servicediscovery
import aiohttp
import asyncio

SANDBOX_PSM = 'data.aml.code_sandbox_arnold_celery.service.hl'


def jupyter_ci(code: str) -> str:
    """
    **代码解释器**：当你需要通过编写并运行代码实现目标时(例如通过代码进行数据分析、文本处理与分析、文件处理、绘制图表与图形等)，可以使用代码解释器JupyterCI_new。代码运行环境是一个支持对应program_language的沙盒环境(非联网环境，因此请勿进行网络请求或任何API的调用请求)。可以通过jupyter_mode选择是否通过jupyter模式运行代码。

    Args:
        code: python code to execute

    Returns:
        str: response from jupyter sandbox
    """
    pass


class JupyterCI(BaseTool):

    def __init__(self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None):
        if config is None:
            config = {}
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        self.jupyter_cell_timeout = os.getenv("jupyter_cell_timeout", config.get("jupyter_cell_timeout", 30))
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        from transformers.utils import get_json_schema
        schema = get_json_schema(jupyter_ci)
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)

        # tool_schema = {"type":"function","function":{"name":"JupyterCI_new","parameters":{"type":"object","properties":{"id":{"type":"string","description":"需要被运行的 <escapeShell> 代码块的ID。"},"program_language":{"type":"string","description":"代码的编程语言，默认为python。"},"jupyter_mode":{"type":"boolean","description":"是否使用Jupyter模式运行代码，默认为True。"}},"required":["id"]},"description":"**代码解释器**：当你需要通过编写并运行代码实现目标时(例如通过代码进行数据分析、文本处理与分析、文件处理、绘制图表与图形等)，可以使用代码解释器JupyterCI_new。在使用JupyterCI_new前, 你需要在**回复的文本中**以<escapeShell type=\"code\" id=[id]>[code_content]</escapeShell>的格式编写代码(其中[id]表示代码块对应id, [code_content]表示代码块内容), 并将代码块的id传入JupyterCI_new函数中。代码运行环境是一个支持对应program_language的沙盒环境(非联网环境，因此请勿进行网络请求或任何API的调用请求)。可以通过jupyter_mode选择是否通过jupyter模式运行代码。"}}
        return tool_schema

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        try:
            response = await self.submit_python_jupyter(parameters['code'], kwargs['ci_sandbox_psm'])
            if os.getenv('PRINT_JUPYTER_RESPONSE', '0') == '1':
                print('[doubao_code_interpreter] jupyter_response:', response)
            return response, 0, {}
        except Exception as e:
            # Must handle this
            print(f"[CI] Error: {e}")
            return f"CI Error: {e}", 0, {}

    async def submit_python_jupyter(self, code, psm):
        headers = {"Content-Type": "application/json"}
        #data = {"cells": action_dict['code_blocks']}
        data = {
            "cells": [code],
            'cell_timeout': self.jupyter_cell_timeout,
            'total_timeout': self.jupyter_cell_timeout * 3
        }

        # data = {"cells": action_dict['code_blocks'],\
        #         'cell_timeout': 1,
        #         'total_timeout': 2}

        # data["files"] = self.minimal_file_system
        data = json.dumps(data)
        eval_response = None
        rsp_str = ''
        plugin_failed_message = ''
        for _ in range(3):
            try:
                await self.get_endpoint(psm)
                print('current_end_point', self.sandbox_endpoint)
                url = f"{self.sandbox_endpoint}/run_jupyter"
                async with aiohttp.ClientSession() as session:
                    # request timeout should be smaller than sandbox timeout
                    async with session.post(url,
                                            headers=headers,
                                            data=data.encode("utf-8"),
                                            timeout=self.jupyter_cell_timeout * 4) as response:
                        eval_response = await response.json()
                # print('eval_response',eval_response)
                # import pdb
                # pdb.set_trace()
                #if eval_response["cells"] is not None and len(eval_response["cells"]) > 0:
                if eval_response["cells"][-1]["status"] == 'ok':
                    rsp_str = "\nSTDOUT:\n" + eval_response["cells"][-1]["stdout"] + "\n"
                    if len(eval_response["cells"][-1]["display"]) > 0:
                        rsp_str += eval_response["cells"][-1]["display"][-1]["text/plain"] + "\n"
                    break
                elif eval_response["cells"][-1]["status"] == 'TimeLimitExceeded':
                    rsp_str = 'TimeLimitExceeded'
                    break
                else:
                    # Combine the list into a single string
                    combined_traceback = "\n".join(eval_response["cells"][-1]["error"][0]["traceback"])

                    # Remove ANSI escape sequences
                    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
                    readable_traceback = ansi_escape.sub("", combined_traceback)
                    readable_traceback = await self.extract_user_error(readable_traceback)

                    rsp_str = "\nSTDERR:\n" + readable_traceback + "\n"
                    break
                # else:
                #     rsp_str = ""
                #     break

            except Exception as e:
                # if not ("Read timed out" in str(e)):
                #     if os.getenv('PRINT_JUPYTER_ERROR', '0') == '1':
                #         print("[doubao_code_interpreter] code response:", action_dict['code_blocks'])
                #         print("sandbox error: ", e)
                #         print("eval response:", eval_response)
                plugin_failed_message = str(e)
                # import pdb
                # pdb.set_trace()
                await asyncio.sleep(0.1)
                continue

        if rsp_str == "":
            rsp_str = "tool_call error:" + plugin_failed_message
        return rsp_str

    async def get_endpoint(self, psm):
        code_sandbox_psm = psm if psm else SANDBOX_PSM
        import servicediscovery.aio as servicediscovery
        for i in range(30):
            try:
                sd_result = await servicediscovery.get_one(code_sandbox_psm, address_family="v6")
                host = f"[{sd_result['Host']}]"
                port = sd_result["Port"]
                temp_endpoint = f"http://{host}:{port}"

                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{temp_endpoint}/v1/ping", timeout=5.0) as rsp:
                        assert rsp.status == 200
                        text = await rsp.text()
                        assert text == '"pong"'

                self.sandbox_endpoint = temp_endpoint
                break  # Success
            except Exception as e:
                # Optional: log the exception here
                print('Sandbox Get Endpoint Exception:', e)
                await asyncio.sleep(0.5)
                continue

    async def extract_user_error(self, traceback: str) -> str:
        traceback = traceback.strip()
        traceback = re.sub(r'\n{3,}', '\n\n', traceback)
        traceback_list = traceback.split("\n\n")

        final_error_info_list = []
        for tc_info in traceback_list:
            if "site_packages" in tc_info:
                continue
            tc_lines = tc_info.strip().split('\n')
            start_idx = -1
            for idx in range(len(tc_lines)):
                try:
                    assert tc_lines[idx].strip()
                except:
                    continue
                if (tc_lines[idx].startswith('Cell') or ('ipykernel' in tc_lines[idx]) or
                    (tc_lines[idx].startswith('Exception')) or ("Error:" in tc_lines[idx].split()[0]) or
                    (tc_lines[idx].startswith("OutOfBoundsDatetime"))):
                    start_idx = idx
                    break
            if start_idx != -1:
                final_error_info_list.append('\n'.join(tc_lines[start_idx:]))

        unique_error_info_list = []
        for ei in final_error_info_list[::-1]:
            if ei not in unique_error_info_list:
                unique_error_info_list.append(ei)
        unique_error_info_list = unique_error_info_list[::-1]

        if unique_error_info_list:
            return "\n\n".join(unique_error_info_list)
        return "No User Error Found."
