import os
import sys
import json
import ray
import requests

from bytedance import servicediscovery
from tenacity import retry, stop_after_attempt
from .utils import Verifier


@retry(stop=stop_after_attempt(40))
def get_endpoint(gaokao_verifier_service_psm):
    sd_result = servicediscovery.get_one(gaokao_verifier_service_psm, address_family="dual-stack")
    host = f"[{sd_result['Host']}]" if ':' in sd_result['Host'] else sd_result['Host']
    port = sd_result["Port"]
    endpoint = f"http://{host}:{port}"
    rsp = requests.get(f"{endpoint}/ping", timeout=5.0)
    assert rsp.status_code == 200
    assert rsp.text == '"pong"'
    return endpoint


class GaokaoVerifier(Verifier, reward_style="gaokao_verifier_service"):

    def is_remote(self):
        return self.config.trainer.use_remote_verifier

    def preprocess(self, input_ids, ground_truth):
        solution_str, ground_truth = super().preprocess(input_ids, ground_truth)
        return solution_str, ground_truth, self.config.trainer.gaokao_verifier_service_psm

    @staticmethod
    def compute_score(solution_str, ground_truth, gaokao_verifier_service_psm, **kwargs) -> float:
        return compute_score(solution_str, ground_truth, gaokao_verifier_service_psm)


VERIFIER_PROMPT = """
请以官方阅卷教师的身份，对下方给出的【题目】【参考答案】【评分细则】【考生答案】进行评分。

首先，阅读【评分细则】，确认本题满分及每一得分点所占分值
然后，按照下列步骤给出评分结果与评语，逐条列出评分细则中的每一“得分点”，例如：第 i 项：实际得分 / 该项满分
接着，详细概述考生答案的主要优缺点，供质检或二评参考
最后，给出本题总得分，用\\boxed{{}}包裹住一个数字X：\\boxed{{X}}

# 评分细则的扣分原则
- **答案正确 + 过程正确 →** 该得分点满分  
- **答案正确 + 过程有误 →** 该得分点按照细则标注的“过程错误”扣分  
- **答案错误 + 过程正确 →** 若细则说明过程可得分，则给相应过程分  
- **答案、过程均错误或缺失 →** 该得分点 0 分  
- 若考生出现与题设、物理常识或数学常识相矛盾的“致命错误”，按细则一次性扣除相应分值  
- 同一错误不重复扣分；格式、书写等非知识性问题严格按照细则执行

！！注意：选择题、填空题不得给过程分！！

# 输入
【题目】
{question}

【本题总分（考生本题得分不得超过此分）】
{full_score}分

【正确答案与评分细则】
{rubric}

【考生答案】
{prediction}
""".strip()


def compute_score(solution_str, ground_truth, gaokao_verifier_service_psm, **argv) -> float:

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    endpoint = get_endpoint(gaokao_verifier_service_psm)
    solution_str = solution_str.split('</think>')[-1]
    problem = ground_truth["problem"]
    reference_answer = ground_truth["reference_answer"]
    verify_principles = ground_truth.get("verify_principles", "")
    full_score = float(ground_truth.get("full_score", 1.0))
    env = {
        "arnold_trial_id": os.environ.get("ARNOLD_TRIAL_ID", "0"),
        "arnold_trial_owner": os.environ.get("ARNOLD_TRIAL_OWNER", "0")
    }

    if verify_principles == "":
        verify_principles = f"""【正确答案】
{reference_answer}

做对得1分，做错得0分。"""
        full_score = 1
    prompt = VERIFIER_PROMPT.format(question=problem,
                                    full_score=full_score,
                                    rubric=verify_principles,
                                    prediction=solution_str)

    for i in range(3):
        try:
            data = {"prompt": prompt, "env": env}
            response = requests.post(f"{endpoint}/gaokao_verify", json=data)
            if response.status_code != 200:
                continue
            response = response.json()
            score = 2 * float(response['score']) / full_score - 1.0
            score = min(max(score, -1), 1)
            return score
        except Exception as ex:
            print(ex)
    print(f'Got exception in compute_score via gaokao_verifier_service')
    return -2
