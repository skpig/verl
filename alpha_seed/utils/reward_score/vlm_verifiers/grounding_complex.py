import logging
import os
import random
import time

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifierFailed, VerifyResult

logger = logging.getLogger()

# VERIFY_TEMPLATE = """你是一位资深的大模型阅卷专家，你的核心任务是依据给定的考试题目、对应的参考答案，以及学生的作答内容，精准判断学生回答是否正确。
# 执行任务时，请严格遵循以下准则：
# 1. 如果参考答案中存在多个正确答案，只要学生的回答与其中一个正确答案相符，就认为该回答符合参考答案。
# 2. 如果学生的答案不在参考答案中，但是不与参考答案冲突，且满足题目的要求，也认为该回答是正确的。
# 3. 当参考答案与学生作答在形式上有所不同时，比如参考答案仅为漫画，而学生作答完整写出该漫画的名字；或是参考答案为分数表达式子，而学生答案是等价的四舍五入的小数形式，这类情况均应判定为回答正确。
# 5. 若学生回答与问题类型不匹配时，如问题类型是一道计数题，但学生给出了一串坐标信息，且不包含最终的计数答案，应判定为回答错误。
# # 考试题目
# {problem}
# # 参考答案
# {reference_answer}
# # 学生作答
# {model_response}
# 仅输出"是/否"，不允许包含其他多余内容。"""
VERIFY_TEMPLATE = '''你是一个超强的判题专家。给定一道题目的参考答案和一段学生的回答。你的任务是判断该回答是否符合参考答案。若符合，回答是，否则回答否。
**请注意：**
1. 学生的回答不一定要和参考答案完全一致，要结合问题的要求判断学生的答案是否和参考答案一样满足题意。
2. 如果参考答案中存在多个正确答案，只要学生的回答与其中一个正确答案相符，就认为该回答符合参考答案。
3. 如果参考答案中存在多个得分点，学生的回答必须同时满足所有得分点，才认为该回答符合参考答案。
4. 如果学生的回答不完整，有被截断的现象，则认为该回答不符合参考答案。
5. 你的回答要么是是，要么是否，不能含有其他内容。
<问题>
{problem}
<学生的回答>
{model_response}
<参考答案>
{reference_answer}
'''


class ModelBasedGroundingComplexVerifierVolc(BaseVerifier):

    def __init__(self, volc_ark_key: str, volc_model_name: str) -> None:
        super().__init__()
        if not volc_ark_key:
            raise ValueError('volc_ark_key is not set')
        if not volc_model_name:
            raise ValueError('volc_model_name is not set')
        base_url = os.environ.get('VOLC_ARK_BASE_URL', "https://ark-cn-beijing.bytedance.net/api/v3")

        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=volc_ark_key, timeout=1800)
        self.model = volc_model_name

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        problem = verifier_feature_dict['problem']

        for i in range(10):
            try:
                prompt = VERIFY_TEMPLATE.format(problem=problem, reference_answer=answer, model_response=response)

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        },
                    ],
                    timeout=120,
                    temperature=0.1,
                    max_tokens=100,
                )
                judge_response = completion.choices[0].message.content
                if judge_response.startswith("是") or judge_response.lower().startswith("yes"):
                    return VerifyResult(score=1, extracted_answer=response)
                return VerifyResult(score=0, extracted_answer=response)

            except Exception as ex:
                import traceback
                logger.info(traceback.format_exc())
                time.sleep(random.choice(list(range(60, 150))))
                continue
        raise VerifierFailed
