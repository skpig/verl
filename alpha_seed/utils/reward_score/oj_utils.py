import ray
import json
import requests
import time
from sandbox_fusion import submit, SubmitRequest, TestConfig
from bytedance import servicediscovery
from tenacity import retry, stop_after_attempt
from alpha_seed.utils.reward_score.utils import Verifier

OJ_MAX_ATTEMPTS = 3
CLIENT_TIMEOUT = 30


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


class OJVerifier(Verifier, reward_style="code-sandbox"):

    def is_remote(self):
        return True

    def preprocess(self, *args, **kwargs):
        solution_str, ground_truth = super().preprocess(*args, **kwargs)
        return solution_str, ground_truth, self.config.trainer.code_sandbox_psm

    @staticmethod
    def compute_score(solution_str, ground_truth, code_sandbox_psm) -> float:
        return compute_score(solution_str, ground_truth, code_sandbox_psm)


def parse_sandbox_error_msg(req_res):
    ret_code, error_msg, stdout, stderr = "Unknown", None, "", ""
    if req_res.accepted == True:
        return "Accepted", error_msg
    for test in req_res.tests:
        # ignore AC testcase
        if test.passed:
            continue
        # CE
        if test.exec_info.compile_result:  # For Non-Compile Language is None
            if test.exec_info.compile_result.stderr:
                return "Compile Error", test.exec_info.compile_result.stderr[:2048]
        # TLE + WA ret code
        d = {}
        if test.exec_info.run_result:
            if test.exec_info.run_result.status == "TimeLimitExceeded":
                ret_code = "Time Limit Exceeded"
            elif test.exec_info.run_result.status == "Finished":
                ret_code = "Wrong Answer (Finished)"
                stdout = test.exec_info.run_result.stdout
            elif test.exec_info.run_result.status == "Failed":
                ret_code = "Wrong Answer (Failed)"
                stdout = test.exec_info.run_result.stdout
            # extra info maybe in test.exec_info.run_result.stdout
            if test.exec_info.run_result.stderr:
                stderr = test.exec_info.run_result.stderr
            d.update({"stdout": stdout, "stderr": stderr})  # simple message; will be overridden if test_info given
        # WA extra info
        if test.test_info:  # test.test_info is dict
            d.update({
                "input": test.test_info["input"]["stdin"],
                "correct output": test.test_info["output"]["stdout"],
            })
            if len(stdout) > 0:
                d["your code output"] = stdout
            else:
                if ret_code == "Time Limit Exceeded":
                    d["your code output"] = "Time ran out; no result output"
                else:
                    d["your code output"] = "Your code output was not same as correct output"

            for k, v in d.items():
                if len(v) > 1024:
                    d[k] = v[:500] + "...(truncated)..." + v[-500:]

        if ret_code != "Unknown":
            error_msg = f"Not Passed TestCase Display as followed:\n" + json.dumps(d)
            # already got reason, return.
            return ret_code, error_msg
        else:
            return "Unknown status", test.exec_info.run_result.status
    # print("Unknown Error", "No TestCase Ran", req_res)
    return "Unknown (No Tests)", error_msg


def compute_score(solution_str, ground_truth, code_sandbox_psm, **argv) -> float:
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    oj_features = ground_truth["oj_features"]
    oj_features["completion"] = solution_str
    client_timeout = 30
    for run in range(OJ_MAX_ATTEMPTS):
        if code_sandbox_psm != "":
            endpoint = get_sandbox_endpoint(code_sandbox_psm)
        else:
            endpoint = "https://faas-code-sandbox.bytedance.net/"
        req = SubmitRequest(dataset=oj_features["dataset"],
                            id=oj_features["id"],
                            completion=solution_str,
                            config=TestConfig(**oj_features["config"]))
        try:
            req_res = submit(req, endpoint=endpoint, max_attempts=1, client_timeout=client_timeout)
            if req_res.accepted:
                return {"score": 1, "msg": ""}
            else:
                ret_code, error_msg = parse_sandbox_error_msg(req_res)
                return {"score": -1, "msg": f"{ret_code}\n{error_msg}"}
        except Exception as ex:
            print(f'sandbox fail with error: {ex}, retrying with {run+1}/{OJ_MAX_ATTEMPTS} attempts')
        client_timeout += 30
    print(f'Finally sandbox fails')
    return {"score": -2, "msg": ""}


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
