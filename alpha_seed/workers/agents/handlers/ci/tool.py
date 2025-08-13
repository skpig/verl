import functools

import async_timeout
from alpha_seed.workers.agents.handlers.base_tool import BaseTool
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


def JupyterCI_new(id: str, program_language: str = "python", jupyter_mode: bool = True) -> str:
    """
    **代码解释器**：当你需要通过编写并运行python代码实现目标时(例如通过代码进行数据分析、文本处理与分析、文件处理、绘制图表与图形等)，可以使用代码解释器JupyterCI_new。在使用JupyterCI_new前, 你需要在**回复的文本中**以<escapeShell type=\"code\" id=[id]>[code_content]</escapeShell>的格式编写代码(其中[id]表示代码块对应id, [code_content]表示代码块内容), 并将代码块的id传入JupyterCI_new函数中。代码运行环境是一个支持对应program_language的沙盒环境。目前只支持python代码，其他编程语言暂时不支持

    Args:
        id: 需要被运行的 <escapeShell> 代码块的ID。
        program_language: 代码的编程语言，默认为python。
        jupyter_mode: 是否使用Jupyter模式运行代码，默认为True。

    Returns:
        id: response from jupyter sandbox
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
        schema = get_json_schema(JupyterCI_new)
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
        return tool_schema

    async def execute(self, instance_id: str, tool_args: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        # try:
        response = await self.submit_python_jupyter(tool_args['code'], kwargs['ci_sandbox_psm'])
        if os.getenv('PRINT_JUPYTER_RESPONSE', '0') == '1':
            print('[doubao_code_interpreter] jupyter_response:', response)
        return response, 0, {}
        # except Exception as e:
        #     # Must handle this
        #     print(f"[CI] Error: {e}")
        #     return f"CI Error: {e}", 0, {}

    async def submit_python_jupyter(self, code, psm):
        headers = {"Content-Type": "application/json"}
        #data = {"cells": action_dict['code_blocks']}
        if not isinstance(code, list):
            code = [code]
        data = {
            "cells": code,
            'cell_timeout': self.jupyter_cell_timeout,
            'total_timeout': self.jupyter_cell_timeout * 3
        }

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

            except Exception as e:
                plugin_failed_message = str(e)
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


class JupyterCI_stateful(BaseTool):

    def __init__(self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None):
        if config is None:
            config = {}
        if tool_schema is None:
            tool_schema = self.get_openai_tool_schema()
        self.jupyter_cell_timeout = os.getenv("jupyter_cell_timeout", config.get("jupyter_cell_timeout", 30))

        self.jupyter_env_id = None
        default_stateful_ci_env_address = 'http://[2605:340:cd51:7700:31f:8248:8db1:6c3d]:9455'
        self.jupyter_w_state_env_manager = os.getenv('JUPYTER_W_STATE_ENV_MANAGER', None)
        if self.jupyter_w_state_env_manager is None:
            self.jupyter_w_state_env_manager = default_stateful_ci_env_address
            server_psm = "data.aml.python_sandbox_stable.service.wlby"
            for i in range(10):
                try:
                    new_address = servicediscovery.lookup(name=server_psm, timeout=3, cachetime=10, address_family="v6")
                    self.jupyter_w_state_env_manager = f'http://[{new_address[0]["Host"]}]:{new_address[0]["Port"]}'
                    break
                except:
                    continue

        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        from transformers.utils import get_json_schema
        schema = get_json_schema(jupyter_ci)
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
        return tool_schema

    async def execute(self, instance_id: str, tool_args: dict[str, Any], initial_files,
                      **kwargs) -> Tuple[str, float, dict]:
        _start_time = time.time()
        # try:
        if self.jupyter_env_id is None:
            await self.start_up_jupyter_w_state(initial_files)
        response = await self.submit_python_jupyter_w_state({'code_blocks': tool_args['code']},
                                                            env_id=self.jupyter_env_id)
        if "⚠️ Jupyter Sandbox Restarted" in response:
            self.jupyter_env_id = None

        if os.getenv('PRINT_JUPYTER_RESPONSE', '0') == '1':
            print('[doubao_code_interpreter] jupyter_response:', response)

        _step_time = time.time() - _start_time
        if _step_time > 60:
            print(
                f'[doubao_code_interpreter] WARNING: The current search step runs for {_step_time:.2f} seconds with result {response}'
            )
        return response, 0, {}
        # except Exception as e:
        #     # Must handle this
        #     print(f"[CI] Error: {e}")
        #     return f"CI Error: {e}", 0, {}

    async def start_up_jupyter_w_state(self, initial_files):
        max_total_retries = 3
        for total_attempt in range(1, max_total_retries + 1):
            # try:
            url = self.jupyter_w_state_env_manager + '/get_instance'
            max_retries = 3
            timeout_seconds = 5
            # import pdb
            # pdb.set_trace()
            for attempt in range(1, max_retries + 1):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with async_timeout.timeout(timeout_seconds):
                            async with session.get(url) as response:
                                status = response.status
                                text = await response.json()
                                print(f"Attempt {attempt}: Status Code {status}")
                                print(f"Response Body:\n{text}")
                                break
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(2**attempt)  # exponential backoff
                    else:
                        print("All retry attempts failed.")
            self.jupyter_w_state_endpoint = f"http://[{text['ip']}]:{text['port']}"
            url = self.jupyter_w_state_endpoint + '/start_jupyter'
            headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
            max_retries = 3
            timeout_seconds = 10
            # import pdb
            # pdb.set_trace()
            for attempt in range(1, max_retries + 1):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with async_timeout.timeout(timeout_seconds):
                            async with session.post(url, headers=headers) as response:
                                status = response.status
                                text = await response.json()
                                print(f"Attempt start jupyter {attempt}: Status Code {status}")
                                print(f"Response Body:\n{text}")
                                break  # success, exit the function
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"Attempt start jupyter {attempt} failed: {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(2**attempt)  # exponential backoff
                    else:
                        print("All start jupyter retry attempts failed.")

            start_up_test_action_dict = {'code_blocks': ['print("sucess start up")']}
            start_up_test_response = await self.submit_python_jupyter_w_state(start_up_test_action_dict,
                                                                              env_id=text['env_id'],
                                                                              files=initial_files)
            if 'sucess start up' in start_up_test_response:
                self.jupyter_env_id = text['env_id']
                print('jupyter start up sucess')
                break
            else:
                print(f"Attempt start jupyter workflow {total_attempt} failed: {start_up_test_response}")
                await asyncio.sleep(2)
                continue

            # except Exception as e:
            #     print(f"Attempt full jupyter workflow {total_attempt} failed: {e}")
            #     await asyncio.sleep(2)
            #     continue

    async def submit_python_jupyter_w_state(self, action_dict, env_id, files={}):
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        data = {
            "cells": action_dict['code_blocks'],
            'cell_timeout': self.jupyter_cell_timeout,
            'total_timeout': self.jupyter_cell_timeout + 10,
            'kernel': 'python3',
            'env_id': env_id
        }

        # data = {"cells": action_dict['code_blocks'],\
        #         'cell_timeout': 1,
        #         'total_timeout': 2}

        data["files"] = files
        data = json.dumps(data)
        eval_response = None
        rsp_str = ''
        plugin_failed_message = ''
        for _ in range(3):
            # try:
            url = f"{self.jupyter_w_state_endpoint}/run_jupyter"
            async with aiohttp.ClientSession() as session:
                # request timeout should be smaller than sandbox timeout
                async with session.post(url,
                                        headers=headers,
                                        data=data.encode("utf-8"),
                                        timeout=self.jupyter_cell_timeout + 20) as response:
                    eval_response = await response.json()
            # print('eval_response',eval_response)
            # import pdb
            # pdb.set_trace()
            #if eval_response["cells"] is not None and len(eval_response["cells"]) > 0:
            if eval_response['status'] == 'Failed' and eval_response['driver']['cpu_time'] == 0.0:
                rsp_str = (
                    "⚠️ Jupyter Sandbox Restarted\n\n"
                    "The previous Python Jupyter sandbox has crashed and been terminated. "
                    "This was likely caused by a fatal error in the previous code block, such as a core dump or kernel failure.\n\n"
                    "🛑 Possible causes include:\n"
                    "- Setting an excessively high recursion limit, e.g., sys.setrecursionlimit(1_000_000)\n"
                    "- Creating infinite recursion or deep call stacks\n"
                    "- Using too much memory, such as loading massive datasets into RAM\n"
                    "- Executing unsafe low-level operations (e.g., ctypes, cffi, or certain numba calls)\n"
                    "- Unbounded loops that prevent kernel responsiveness\n"
                    "- Bugs in native extensions (e.g., compiled Python packages)\n\n"
                    "A new Jupyter sandbox has now been started.\n\n"
                    "✅ What to do next:\n"
                    "Please rewrite and rerun your code, ensuring that:\n"
                    "- You include all necessary imports and definitions\n"
                    "- You re-establish context, such as reloading variables or datasets\n"
                    "- You avoid risky operations unless necessary, and test incrementally if debugging\n\n"
                    "This is a fresh environment — nothing from the previous session is preserved.")
                break
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

            # except Exception as e:
            #     # if not ("Read timed out" in str(e)):
            #     #     if os.getenv('PRINT_JUPYTER_ERROR', '0') == '1':
            #     #         print("[doubao_code_interpreter] code response:", action_dict['code_blocks'])
            #     #         print("sandbox error: ", e)
            #     #         print("eval response:", eval_response)
            #     plugin_failed_message = str(e)
            #     # import pdb
            #     # pdb.set_trace()
            #     await asyncio.sleep(0.1)
            #     continue

        if rsp_str == "":
            rsp_str = "plugin_error:" + plugin_failed_message
        return rsp_str

    async def get_endpoint(self, psm):
        code_sandbox_psm = psm if psm else SANDBOX_PSM
        from bytedance import servicediscovery
        for i in range(30):
            try:
                # Run potentially blocking service discovery in executor
                loop = asyncio.get_event_loop()
                sd_result = await loop.run_in_executor(
                    None, functools.partial(servicediscovery.get_one, code_sandbox_psm, address_family="dual-stack"))

                host = f"[{sd_result['Host']}]" if ':' in sd_result['Host'] else sd_result['Host']
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
