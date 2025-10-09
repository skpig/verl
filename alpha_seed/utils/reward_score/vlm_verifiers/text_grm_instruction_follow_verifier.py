import logging
import os
import json
import random
import time

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifierFailed, VerifyResult

logger = logging.getLogger()

SYSTEM_PROMPT = """从现在开始你的角色是一名严谨的指令遵循判卷老师，你的任务是根据评分标准对学生答案进行判卷评分。我会给你一段对话，包括该对话的场景设定（在<对话场景设定>和</对话场景设定>之间）、对话历史（在<对话历史>和</对话历史>之间）、当前轮次用户问题（在<当前user问题>和</当前user问题>之间）和当前assistant的回答（在<当前assistant回答>和</当前assistant回答>之间），以及当前对话的校验准则（在<校验准则>和</校验准则>之间）。你的评分档位一共为2档，分别为0分和1分。0分表示学生答案未满足评分标准中的所有要求，评分标准中的每一个要求都同等重要，学生答案中只要有不满足评分标准要求的地方，直接判为0分；1分表示学生答案完全满足评分标准中的所有要求。请一步一步的分析和思考题目，并按照以下步骤进行： 
- 首先，请仔细阅读并理解评分标准中的每一个要求。
- 其次，请分析学生答案中的内容是否完全遵循了评分标准中的所有要求，把学生答案与评分标准中的每一个要求进行逐条对照分析，如果题目合理且assistant拒答，按0分处理。
你的输出格式为： 
如果assistant遵循指令得1分，则输出：{"check_valid": true}，如果不遵循指令得0分，则输出：{"check_valid": false}

【示例】： 
<对话场景设定>无</对话场景设定>
<对话历史></对话历史>
<当前user问题>跳绳是有氧还是无氧，在跳绳后加上emoji表情</当前user问题>
<校验准则>1.是否在“跳绳”后加上emoji表情？2.是否提到跳绳是一项有氧运动？</校验准则>
<当前assistant回答>跳绳是一项可以有效消耗热量的有氧运动，它可以帮助你达到减肥的目的。但是，跳绳减肥需要长期坚持，并且需要结合合理的饮食和其他运动方式。如果你想通过跳绳减肥，建议你每天跳绳 30 分钟以上，并且逐渐增加跳绳的难度和强度。同时，你还需要注意饮食的搭配，控制热量的摄入，避免食用高热量、高脂肪、高糖分的食物。</当前assistant回答>
分析：1.学生答案中的“跳绳”一词后没有加emoji，不满足要求。2.学生答案中提到跳绳时一项有氧运动，满足要求。第一项不满足要求，最终为0分。
输出：
{"check_valid": false}
"""


def check_cot_llm_as_judge(raw_str):
    # TODO. make sure tokenizer is the same
    if raw_str == "" or not isinstance(raw_str, str):
        return 0.0
    valid_score = 1.0
    non_valid_score = 0.0
    if "{\"check_valid\": true}" in raw_str and "{\"check_valid\": false}" not in raw_str:
        return valid_score
    elif "{\"check_valid\": false}" in raw_str and "{\"check_valid\": true}" not in raw_str:
        return non_valid_score
    try:
        info_doc = json.loads(raw_str)
        if info_doc["check_valid"]:
            return valid_score
        elif not info_doc["check_valid"]:
            return non_valid_score
        else:
            return 0.0
    except Exception as e:
        logging.info("llm as judge parse fail:" + str(raw_str))
        print("llm as judge parse fail:" + str(raw_str))
    return 0.0


class TextGRMInstructionFollowVerifierVolc(BaseVerifier):

    def __init__(self, volc_ark_key: str, volc_model_name: str) -> None:
        super().__init__()
        if not volc_ark_key:
            raise ValueError('volc_ark_key is not set')
        if not volc_model_name:
            raise ValueError('volc_model_name is not set')
        base_url = os.environ.get('VOLC_ARK_BASE_URL', "https://ark-cn-beijing.bytedance.net/api/v3")

        if os.environ.get("VOLC_ARC_GPT_OSS_KEY", None):
            volc_ark_key = os.environ.get("VOLC_ARC_GPT_OSS_KEY")
        if os.environ.get("VOLC_ARK_GPT_OSS_ENDPOINT", None):
            volc_model_name = os.environ.get("VOLC_ARK_GPT_OSS_ENDPOINT")

        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=volc_ark_key, timeout=1800)
        self.model = volc_model_name

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        llm_as_judge_before_response_text = verifier_feature_dict['llm_as_judge_before_response_text']
        llm_as_judge_after_response_text = verifier_feature_dict['llm_as_judge_after_response_text']

        for i in range(10):
            try:
                prompt = llm_as_judge_before_response_text + response + llm_as_judge_after_response_text

                completion = self.client.chat.completions.create(model=self.model,
                                                                 messages=[
                                                                     {
                                                                         "role": "system",
                                                                         "content": SYSTEM_PROMPT
                                                                     },
                                                                     {
                                                                         "role": "user",
                                                                         "content": prompt
                                                                     },
                                                                 ],
                                                                 timeout=120)
                judge_response = completion.choices[0].message.content
                score = check_cot_llm_as_judge(judge_response)
                return VerifyResult(score=score, extracted_answer=response)

            except Exception as ex:
                import traceback
                logger.info(traceback.format_exc())
                time.sleep(random.choice(list(range(120, 300))))
                continue
        raise VerifierFailed
