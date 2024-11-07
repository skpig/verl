import json
import requests
import time
from sandbox_fusion import submit, SubmitRequest, TestConfig
from bytedance import servicediscovery


def compute_score(solution_str, ground_truth, code_sandbox_psm, **argv) -> float:
    if code_sandbox_psm != "":
        sd_result = servicediscovery.get_one(code_sandbox_psm)
        endpoint = f"http://{sd_result['Host']}:{sd_result['Port']}"
    else:
        endpoint = "https://faas-code-sandbox.bytedance.net/"
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    oj_features = ground_truth["oj_features"]
    oj_features["completion"] = solution_str
    req = SubmitRequest(dataset=oj_features["dataset"],
                        id=oj_features["id"],
                        completion=solution_str,
                        config=TestConfig(**oj_features["config"]))
    try:
        req_res = submit(req, endpoint=endpoint)
        if req_res.accepted:
            return 2
        if req_res.extracted_code != "" and req_res.tests[0].exec_info.status == "Finished":
            return 1
        if req_res.extracted_code != "":
            return -1
        return -2
    except Exception as ex:
        return -2
