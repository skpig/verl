import asyncio
import json
import aiohttp
import requests
from tenacity import retry, stop_after_attempt
from bytedance import servicediscovery

# CODE_PSM = "seed.alphaseed.ci_for_horizon.service.lq"
# @retry(stop=stop_after_attempt(40))
# def get_sandbox_endpoint():
#     sd_result = servicediscovery.get_one(CODE_PSM, address_family="dual-stack")
#     host = f"[{sd_result['Host']}]" if ':' in sd_result['Host'] else sd_result['Host']
#     port = sd_result["Port"]
#     endpoint = f"http://{host}:{port}"
#     rsp = requests.get(f"{endpoint}/v1/ping", timeout=5.0)
#     assert rsp.status_code == 200
#     assert rsp.text == '"pong"'
#     return endpoint

# endpoint = get_sandbox_endpoint()

# jupyter_enpoint = f"{endpoint}/run_jupyter"

# headers = {"Content-Type": "application/json"}

# jupyter_cell_timeout = 30

# async def submit_jupyter(code):
#     headers = {"Content-Type": "application/json"}
#     #data = {"cells": action_dict['code_blocks']}
#     data = {
#         "cells": [code],
#         'cell_timeout': jupyter_cell_timeout,
#         'total_timeout': jupyter_cell_timeout * 3
#     }
#     data = json.dumps(data)
#     async with aiohttp.ClientSession() as session:
#         # request timeout should be smaller than sandbox timeout
#         async with session.post(jupyter_enpoint, headers=headers, data=data.encode("utf-8"), timeout=jupyter_cell_timeout*4) as response:
#             eval_response = await response.json()

#     return eval_response

# response = asyncio.run(submit_jupyter('print("hello world")'))
# print(response)
# print("\nSTDOUT:\n" + response["cells"][-1]["stdout"] + "\n")

CODE_PSM = "data.aml.code_sandbox_arnold_celery.service.hl"


@retry(stop=stop_after_attempt(40))
def get_sandbox_endpoint():
    sd_result = servicediscovery.get_one(CODE_PSM, address_family="dual-stack")
    host = f"[{sd_result['Host']}]" if ':' in sd_result['Host'] else sd_result['Host']
    port = sd_result["Port"]
    endpoint = f"http://{host}:{port}"
    rsp = requests.get(f"{endpoint}/v1/ping", timeout=5.0)
    assert rsp.status_code == 200
    assert rsp.text == '"pong"'
    return endpoint


from sandbox_fusion import RunCodeRequest, run_code_async

req = RunCodeRequest(code="print(3 + 5)", language="python")
response = asyncio.run(run_code_async(req, get_sandbox_endpoint(), client_timeout=10))
print(response)
