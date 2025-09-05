import copy
import json
import logging
import random
import time

import requests
from bytedance import servicediscovery
from tenacity import retry, stop_after_attempt

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import VerifierFailed

logger = logging.getLogger()

oj_headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

OJ_RETRY_TIMES = 10


def get_oj_json_data(dataset):
    oj_json_data = {
        'id': 0,
        'completion': '',
        'config': {
            'language': 'python',
            'is_fewshot': False,
            "extra": {
                "append_flag": True
            },
            'dataset_type': 'AutoEvalV4Dataset'
        }
    }
    if dataset.startswith('humaneval_'):
        oj_json_data['config']['extra']['is_freeform'] = True
    return oj_json_data


@retry(stop=stop_after_attempt(40))
def get_sandbox_endpoint(code_sandbox_psm):
    sd_result = servicediscovery.get_one(code_sandbox_psm, address_family="dual-stack")
    host = f"[{sd_result['Host']}]" if ':' in sd_result['Host'] else sd_result['Host']
    port = sd_result["Port"]
    endpoint = f"http://{host}:{port}"
    rsp = requests.get(f"{endpoint}/v1/ping", timeout=5.0)
    assert rsp.status_code == 200
    assert rsp.text == '"pong"'
    return endpoint


class GeneralSandboxVerifier(BaseVerifier):

    def __init__(self, code_sandbox_service_psm) -> None:
        super().__init__()
        self.code_sandbox_service_psm = code_sandbox_service_psm or "data.aml.code_sandbox_arnold_celery.service.hl"
        self.code_sandbox_service_psm = self.code_sandbox_service_psm.replace('\\', '')
        self.sandbox_url = ''

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        data_source = verifier_feature_dict["data_source"]
        sandbox_dataset, sandbox_qid, sandbox_task = data_source.split("@")
        if True:
            assert not ('autoeval' in sandbox_dataset)
            assert not ('mbpp' in sandbox_dataset)
            assert not ('humaneval_python' in sandbox_dataset)

        if verifier_feature_dict["answer"]:
            oj_json_data_t = json.loads(verifier_feature_dict["answer"])
        else:
            oj_json_data_t = copy.deepcopy(get_oj_json_data(sandbox_dataset))
            oj_json_data_t['id'] = sandbox_qid
            oj_json_data_t['dataset'] = sandbox_dataset
        tag = sandbox_task

        oj_json_data_t['completion'] = response
        for i in range(OJ_RETRY_TIMES):
            try:
                if self.code_sandbox_service_psm:
                    # cn: data.aml.code_sandbox_arnold_online.service.hl
                    # i18n: data.aml.sandbox_arnold_for_qiying
                    endpoint = get_sandbox_endpoint(self.code_sandbox_service_psm)
                    eval_response = requests.post(f"{endpoint.rstrip('/')}/submit",
                                                  headers=oj_headers,
                                                  json=oj_json_data_t,
                                                  timeout=30).json()
                elif self.sandbox_url:
                    # cn: https://faas-code-sandbox.bytedance.net/online_judge/submit
                    # i18n: https://seed-sandbox.byteintl.net/faas/sandbox/online_judge/submit
                    eval_response = requests.post(self.sandbox_url, headers=oj_headers, json=oj_json_data_t,
                                                  timeout=30).json()
                else:
                    raise RuntimeError("should have sandbox_psm or sandbox_url")
                if eval_response['accepted'] and not ('exit(' in response):
                    score = 1.0
                    return VerifyResult(score=1.0, extracted_answer=response)
                else:
                    score = 0.0
                    return VerifyResult(score=0.0, extracted_answer=response)
            except Exception as e:
                import traceback
                logger.info(traceback.format_exc())
                logger.info(f'Got exception in compute_score via stem verifier_service: {e}')
                time.sleep(random.randint(60, 150))
                continue
        score = -2.0
        raise VerifierFailed
