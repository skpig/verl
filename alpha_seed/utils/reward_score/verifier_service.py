import os
import sys
import json
import ray
import requests

from bytedance import servicediscovery
from tenacity import retry, stop_after_attempt


@retry(stop=stop_after_attempt(40))
def get_endpoint(gaokao_verifier_psm):
    sd_result = servicediscovery.get_one(gaokao_verifier_psm, address_family="dual-stack")
    host = f"[{sd_result['Host']}]" if ':' in sd_result['Host'] else sd_result['Host']
    port = sd_result["Port"]
    endpoint = f"http://{host}:{port}"
    rsp = requests.get(f"{endpoint}/ping", timeout=5.0)
    assert rsp.status_code == 200
    assert rsp.text == '"pong"'
    return endpoint


def compute_score_client(solution_str, ground_truth, verifier_service_psm, data_uid, config, **argv) -> float:
    """Directly retrieve the scores from SandboxClient"""
    score = None
    if config.trainer.use_remote_verifier:
        # get the sandbox client endpoint
        handler = ray.get_actor('remote_client')
        # retrieve the score directly
        score = ray.get(handler.get_results.remote(data_uid))

    if score is None:
        score = compute_score(solution_str, ground_truth, verifier_service_psm, **argv)

    # optionally, compute the score with original code to compare the results
    # score_original = compute_score(solution_str, ground_truth, code_sandbox_psm, **argv)
    # assert score == score_original

    return score


def compute_score(solution_str, ground_truth, verifier_service_psm, **argv) -> float:
    endpoint = get_endpoint(verifier_service_psm)
    if solution_str.startswith("A conversation between user and assistant."):
        solution_str = solution_str[400:]
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    problem = ground_truth["problem"]
    reference_answer = ground_truth["reference_answer"]
    if isinstance(reference_answer, int):
        reference_answer = str(reference_answer)
    verify_type = ground_truth["verify_type"]
    data = {
        "problem": problem,
        "reference_answer": reference_answer,
        "verify_type": verify_type,
        "generated_response": solution_str
    }
    for i in range(3):
        try:
            response = requests.post(f"{endpoint}/verify", json=data, timeout=30)
            if response.status_code != 200:
                continue
            resp_json = response.json()
            return 1 if resp_json.get("is_correct") else -1
        except Exception as ex:
            print(f'[RETRY] Got exception in compute_score via verifier_service, ex: {ex}')
            continue
    print(f'Got exception in compute_score via verifier_service')
    return -2


if __name__ == "__main__":
    solution_str = "答案：\n2019\n<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"
    ground_truth = {
        "problem":
            " 设集合S是由平面上任意三点不共线的 4039 个点构成的集合，且其中 2019 个点为红色，2020 个点为蓝色；在平面上画出一组直线，可以将平面分成若干区域，若一组直线对于点集S满足下述两个条件，称这是一个 “好直线组”：\n (1) 这些直线不经过该点集S中的任何一个点；\n (2) 每个区域中均不会同时出现两种颜色的点.\n 求k的最小值，使得对于任意的点集S，均存在由k条直线构成的 “好直线组”.",
        "reference_answer":
            "2019",
        "verify_type":
            4
    }
    verifier_service_psm = "data.aml.online_verifier_yueyu.service.lq"
    print(compute_score(solution_str, ground_truth, verifier_service_psm))
