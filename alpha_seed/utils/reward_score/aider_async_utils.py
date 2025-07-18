import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional
from bytedance import servicediscovery
from pydantic import BaseModel

CLIENT_TIMEOUT = 30
OJ_MAX_ATTEMPTS = 3


class AiderV2Result(BaseModel):
    accepted: bool
    extracted_code: str
    stdout: str
    stderr: str


async def get_sandbox_endpoint(aider_service_psm):
    sd_result = servicediscovery.get_one(aider_service_psm, address_family="dual-stack")
    host = f"[{sd_result['Host']}]" if ':' in sd_result['Host'] else sd_result['Host']
    port = sd_result["Port"]
    endpoint = f"http://{host}:{port}"
    return endpoint


async def before_retry_sleep(s):
    print(f'error requesting faas for {s.attempt_number} time(s), will retry... error: {s.outcome.exception()}')


async def on_retry_error(s):
    e = s.outcome.exception()
    raise e


async def evaluate(url, item: dict) -> AiderV2Result:
    """异步评估函数"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f'{url}/evaluate', json=item, timeout=180) as response:
                if response.status != 200:
                    raise Exception(
                        f'[Aider V2] responded with code {response.status}: {await response.text()}. request: {item}')
                return AiderV2Result(**await response.json())
        except aiohttp.ClientError as e:
            raise Exception(f'Network error: {str(e)}')


async def compute_score(solution_str, ground_truth, aider_service_psm, **argv) -> float:
    """异步计算得分函数"""
    result = AiderV2Result(accepted=False, extracted_code="", stdout="", stderr="")
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    for run in range(OJ_MAX_ATTEMPTS):
        try:
            endpoint = await get_sandbox_endpoint(aider_service_psm)
            req = {
                "language": ground_truth["language"],
                "name": ground_truth["name"],
                "completion": solution_str,
                "is_training": ground_truth.get("is_training", False)
            }

            result = await evaluate(endpoint, req)
            if result.accepted:
                return 1 if not argv.get('return_result', False) else result
            else:
                return -1 if not argv.get('return_result', False) else result
        except Exception as ex:
            print(f'sandbox fail with error: {ex}, retrying with {run+1}/{OJ_MAX_ATTEMPTS} attempts')
            if run < OJ_MAX_ATTEMPTS - 1:
                await asyncio.sleep(1)

    print(f'Finally aider sandbox fails')
    return -2 if not argv.get('return_result', False) else result
