import random
import time

from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name
from alpha_seed.utils.reward_score.vlm_verifiers.extra_reward import match_visual_cot_format
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, ExtractAnswerFailed
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import VerifierFailed
from alpha_seed.utils.reward_score.vlm_verifiers.utils import check_language_correctness, get_vlm_client_and_endpoint

ANSWER_JUDGE_PROMPT = '''你是一个超强的判题专家。给定一道题目的参考答案和一段学生的回答。你的任务是判断该回答是否符合参考答案。若符合，回答是，否则回答否。请注意你的回答要么是是，要么是否，不能含有其他内容。
<问题>
{}
<学生的回答>
{}
<参考答案>
{}
'''

GEO_CITY_JUDGE_PROMPT = '''你是一个超强的判题专家。给定一道预测城市的地理题目的参考答案和一段学生的回答。你的任务是判断该回答是否符合参考答案，同一个城市的不同名称、拼写、缩写也算符合，但学生的回答必须是一个准确的城市名称，模糊的描述或者不确定的答案都算不符合。若符合，回答是，否则回答否。请注意你的回答要么是是，要么是否，不能含有其他内容。
<问题>
{}
<学生的回答>
{}
<参考答案>
{}
'''


def answer_judge(query, answer, ref_answer, client, endpoint, verify_prompt_type):
    max_retry_num = 5
    tb = ''
    if verify_prompt_type == 'geo_city':
        prompt = GEO_CITY_JUDGE_PROMPT.format(query, answer, ref_answer)
    else:
        prompt = ANSWER_JUDGE_PROMPT.format(query, answer, ref_answer)

    for _ in range(max_retry_num):
        try:
            completion = client.chat.completions.create(
                model=endpoint,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                temperature=0.1,
                timeout=120,
            )
            matched_results_raw = completion.choices[0].message.content
            matched_results_raw = matched_results_raw.strip()
            assert matched_results_raw in ["是", "否"], f"matched_results_raw {matched_results_raw}"
            if matched_results_raw == "是":
                score = 1.0
            else:
                score = 0.0
            return score
        except Exception:
            import sys
            import traceback
            tb = "".join(traceback.format_exception(*sys.exc_info()))
            sleep_time = random.random() * 90 + 30
            time.sleep(sleep_time)
            continue
    raise VerifierFailed(message=tb)


class VisualCoTVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        # 检查FC格式错误
        if not match_visual_cot_format(response, verifier_feature=verifier_feature_dict):
            raise ExtractAnswerFailed

        query = verifier_feature_dict['query']
        answer = verifier_feature_dict['answer']
        verify_prompt_type = verifier_feature_dict.get('verify_prompt_type', 'visual_cot')

        client, endpoint = get_vlm_client_and_endpoint()

        try:
            eot_token = get_special_tokens_dict_or_name("think_end_token")
            eos_token = get_special_tokens_dict_or_name("eos")
            final_answer: str = response.split(eot_token)[-1].split(eos_token)[0]
        except Exception:
            raise ExtractAnswerFailed

        if not final_answer.strip():
            raise ExtractAnswerFailed

        check_language_correctness(query, response, verifier_feature_dict)

        score = answer_judge(query, final_answer, answer, client, endpoint, verify_prompt_type)
        return VerifyResult(score=score, extracted_answer=final_answer)
