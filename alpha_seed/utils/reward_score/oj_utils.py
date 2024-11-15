import json
import requests
import time
from sandbox_fusion import submit, SubmitRequest, TestConfig
from bytedance import servicediscovery

OJ_MAX_ATTEMPTS = 3
CLIENT_TIMEOUT = 30


def compute_score(solution_str, ground_truth, code_sandbox_psm, **argv) -> float:
    if code_sandbox_psm != "":
        sd_result = servicediscovery.get_one(code_sandbox_psm, address_family="dual-stack")
        sd_result['Host'] = f"[{sd_result['Host']}]" if ':' in sd_result['Host'] else sd_result['Host']
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
        req_res = submit(req, endpoint=endpoint, max_attempts=OJ_MAX_ATTEMPTS, client_timeout=CLIENT_TIMEOUT)
        if req_res.accepted:
            return 2
        if req_res.extracted_code != "" and req_res.tests[0].exec_info.status == "Finished":
            return 1
        if req_res.extracted_code != "":
            return -1
        return -2
    except Exception as ex:
        print(f'sandbox fail with error: {ex}')
        return -3


def test_compute_score():
    print(
        compute_score(
            solution_str='''def is_sorted(items):
    for i in range(len(items) - 1):
        if items[i] > items[i + 1]:
            return False
    return True''',
            ground_truth=
            '{"oj_features": {"dataset": "mining_11697_v1", "id": 8015, "config": {"dataset_type": "PythonAutoDataset", "language": "python", "is_fewshot": false, "extra": {"append_flag": true}, "provided_data": {"content": "\\n输入一个列表, 判断这个列表中的所有元素是否按照升序排列. 用 python 定义函数 is_sorted(items) 解决这个问题.\\n", "test": "\\n\\ndef check(): \\n    assert str(is_sorted([])) == \'True\'\\n    assert str(is_sorted([1])) == \'True\'\\n    assert str(is_sorted([1, 2])) == \'True\'\\n    assert str(is_sorted([2, 1])) == \'False\'\\n    assert str(is_sorted([1, 2, 3])) == \'True\'\\n\\ncheck()", "labels": "{\\"tags\\": [\\"mining_v1\\"], \\"programming_language\\": \\"python\\", \\"execution_language\\": \\"python\\"}", "id": 8015}}, "completion": ""}}',
            code_sandbox_psm="seed.alpha.sandboxd1112.service.hl"))


def test_compute_score_timeout():
    print(
        compute_score(
            solution_str=r'''
import time
def is_sorted(items):
    time.sleep(40)
    for i in range(len(items) - 1):
        if items[i] > items[i + 1]:
            return False
    return True''',
            ground_truth=
            '{"oj_features": {"dataset": "mining_11697_v1", "id": 8015, "config": {"dataset_type": "PythonAutoDataset", "language": "python", "is_fewshot": false, "extra": {"append_flag": true}, "provided_data": {"content": "\\n输入一个列表, 判断这个列表中的所有元素是否按照升序排列. 用 python 定义函数 is_sorted(items) 解决这个问题.\\n", "test": "\\n\\ndef check(): \\n    assert str(is_sorted([])) == \'True\'\\n    assert str(is_sorted([1])) == \'True\'\\n    assert str(is_sorted([1, 2])) == \'True\'\\n    assert str(is_sorted([2, 1])) == \'False\'\\n    assert str(is_sorted([1, 2, 3])) == \'True\'\\n\\ncheck()", "labels": "{\\"tags\\": [\\"mining_v1\\"], \\"programming_language\\": \\"python\\", \\"execution_language\\": \\"python\\"}", "id": 8015}}, "completion": ""}}',
            code_sandbox_psm="seed.alphaseed.sandbox.service.hl"))


if __name__ == '__main__':
    pass
    test_compute_score()
    test_compute_score_timeout()
