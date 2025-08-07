import logging
import os
import random
import time

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifierFailed, VerifyResult

logger = logging.getLogger()

VERIFY_TEMPLATE = """你是一个超强的判题专家。给定一道视觉谜题的参考答案和一段学生的回答，你的任务是判断该回答是否符合参考答案。若符合，回答是，否则回答否。
**请注意：**
1. 学生的回答不一定要和参考答案完全一致，要结合谜题的要求判断学生的答案是否和参考答案一样满足题意。
2. 如果参考答案中存在多个正确答案，只要学生的回答与其中一个正确答案相符，就认为该回答符合参考答案。
3. 如果参考答案中存在多个得分点，学生的回答必须同时满足所有得分点，才认为该回答符合参考答案。
4. 如果学生的回答不完整，有被截断的现象，则认为该回答不符合参考答案。
5. 你的回答要么是“是”，要么是“否”，不能含有其他内容。
<视觉谜题>
{problem}
<学生的回答>
{reference_answer}
<参考答案>
{model_response}"""


def get_text_after_think_tag(text):
    """
    查找文本中"<Think>"之后的内容。
    参数:
        text (str): 输入的文本。
    返回:
        str 或 None: 返回Think后的内容，或如果未找到则返回None。
    """
    search_phrase = "</Think>".lower()
    lower_text = text.lower()
    start_index = lower_text.rfind(search_phrase)
    if start_index == -1:
        return None
    content_start = start_index + len(search_phrase)
    content = text[content_start:].strip()
    for eos in ["<|endoftext|>", "<|im_end|>", "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"]:
        if eos in content:
            content = content.split(eos)[0].strip()
            break
    return content if content else None


class ModelBasedPuzzleVerifierVolc(BaseVerifier):

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

        for i in range(3):
            try:
                candidate_answer = get_text_after_think_tag(response)
                if candidate_answer is None:
                    extracted_response = response[-200:]
                else:
                    extracted_response = candidate_answer

                prompt = VERIFY_TEMPLATE.format(problem=problem,
                                                reference_answer=answer,
                                                model_response=extracted_response)

                completion = self.client.chat.completions.create(model=self.model,
                                                                 messages=[
                                                                     {
                                                                         "role": "user",
                                                                         "content": prompt
                                                                     },
                                                                 ],
                                                                 timeout=120)
                judge_response = completion.choices[0].message.content
                if judge_response.startswith("是") or judge_response.lower().startswith("yes"):
                    return VerifyResult(score=1, extracted_answer=response)
                return VerifyResult(score=0, extracted_answer=response)

            except Exception as ex:
                import traceback
                logger.info(traceback.format_exc())
                time.sleep(random.choice(list(range(10, 25))))
                continue
        raise VerifierFailed
