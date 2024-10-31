import json
import requests
import time
from sandbox_fusion import set_sandbox_endpoint, set_dataset_endpoint, submit, SubmitRequest, TestConfig

set_sandbox_endpoint("https://faas-code-sandbox.bytedance.net/")
set_dataset_endpoint("https://faas-code-sandbox.bytedance.net/online_judge/")
# def compute_score(solution_str, ground_truth, **argv) -> float:
#     if isinstance(ground_truth, str):
#         ground_truth = json.loads(ground_truth)
#     oj_features = ground_truth["oj_features"]
#     oj_features["completion"] = solution_str
#     req = SubmitRequest(
#         dataset=oj_features["dataset"],
#         id=oj_features["id"],
#         completion=solution_str,
#         config=TestConfig(**oj_features["config"])
#     )
#     try:
#         req_res = submit(req)
#         if req_res.accepted:
#             return 2
#         if req_res.extracted_code != "" and req_res.tests[0].exec_info.status == "Finished":
#             return 1
#         if req_res.extracted_code != "":
#             return -1
#         return -2
#     except Exception as ex:
#         return -2
import json
import requests
import time

OJ_RETRY_TIMES = 5
oj_headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}


def compute_score(solution_str, ground_truth, **argv) -> float:
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    oj_features = ground_truth["oj_features"]
    oj_features["completion"] = solution_str
    for i in range(OJ_RETRY_TIMES):
        try:
            eval_response = requests.post('https://faas-code-sandbox.bytedance.net/online_judge/submit',
                                          headers=oj_headers,
                                          json=oj_features,
                                          timeout=30).json()
            if eval_response['accepted'] and not ('exit(' in solution_str):
                return 1
            else:
                return 0
        except Exception as ex:
            time.sleep(0.5)
            continue
    return -2
