import itertools
import json
import logging
import os
import random
import time

import numpy as np

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifierFailed, VerifyResult

logger = logging.getLogger()


def compute_score(gpt_response):
    try:
        score_json = gpt_response.split('【JSON】：')[-1].replace('```json', '').replace('```', '')
        score = json.loads(score_json)['score']
        score = min(1, score)
        score = max(0, score)
    except:
        score = 0
    return score


judge_system_prompt = """现在你的角色是一名严格的判卷老师，你的任务是以【标准答案】为唯一参考，对学生针对图表的作答进行审核和评分。整个评分过程中，你需要熟知以下关键点（本提示词适用于柱状图、折线图、直方图、散点图、饼图、堆叠图、箱线图、热力图、以及双坐标轴图等常见可视化场景）： 
- 评分只用参考学生给出的【最终答案】来判定对错，不考察中间推理或读图过程是否正确。 
- 请先从学生的解答中“提取最终答案”（见下文提取规则），展示在分析结果中，然后再判断正误。 
- 本题为**单题单问**，只需要给出该问的一个分数（0或1）。 
- 你的输出格式必须严格遵循文末的固定模板。

【图表答案等价与容差规则】（仅用于判等，不用于反推“更优解”）：
1) 数值容差：若标准答案为数值/百分比，允许 ±1 个小刻度或 ±2%（相对误差）中较大者的偏差。例：标准 40%，学生答 39% 或 41%判等价；若轴无细刻度，只用 ±2% 规则。  
2) 单位等价：数值与单位一致即可；“40%”与“0.40”在百分比语境下等价；“40（百分比）”与“40%”等价。  
3) 文本等价：同义表达等价（如“上升”“增加”“增长”）；“约/大约/≈/around/about/近似”为可接受近似，只要落在容差内。  
4) 区间与阈值：题目给出范围或阈值时，若标准答案是单点而学生给出包含该点的最小合理区间（如“约 40–45%”且 40% 在其内），判等价；若题干明确要求“最小/最大/首次/拐点”等，则学生答案必须对应正确位置（首次满足条件的点、极值点、或拐点所在区间/刻度）。  
5) 分类与标签：读图得到的是类别/月份/分组/图例项等时，需严格匹配名称（中英文别名或常用缩写视为等价，如“US”“USA”“United States”）。  
6) 双坐标轴：若标准答案依赖“左轴/右轴（LHS/RHS）”，学生答案只要落在正确轴的读数容差内即判对。  
7) 离散刻度（直方/分箱）：以箱（bin）标签为准；回答“40%”等价于选择“40%这一档/刻度”。  
8) 特殊标注：如“>100%/Over 100%/N/A/无数据”等，语义一致视为等价。

【最终答案提取规则】（从学生作答中抽取用于评分的值/项）：  
- 若出现“最终答案/Answer/结论/所以/因此/选/选项为”等提示词，则以其后最近的明确答案为准；若多处给出冲突答案，以**最后出现**的明确答案为准。  
- 若学生只写了说明没有明确值，则视为未作答。  
- 多选题（若题型为选择题）仅看最终选择项集合是否与标准答案完全一致（有漏选/错选则 0 分）。  
- 如果学生给出多个候选（如“40%或45%”），仅当两者都等价于标准答案时判 1 分，否则 0 分。

【评分执行】：  
- 学生最终答案与标准答案在上述等价与容差规则下“等价”判 1 分，否则判 0 分。  
- 不得使用图表以外的外部知识推断答案；不对图表本身正确性做价值判断。  
- 对于比较/排序/找极值/判断趋势/是否超过阈值等问题，按题意结合等价与容差规则判定，并在评分依据中说明是否满足“首次/最大值/超过阈值”等语义约束。

【输出格式要求】（严格照此输出）： 
【评分依据】：  
【总分】：X分  
【JSON】：  
{ 
    "score": [Score] 
}  
其中 Score 只能是 1 或 0（本题为单问，不使用嵌套列表）。  

【分数档位】（单问 1 分制）：  
- 1 分：学生最终答案与标准答案在数值/单位/语义上等价（满足容差与规则）。  
- 0 分：存在漏选/错选/数值超出容差/单位或语义不一致/未给出明确答案。  

请严格遵守以上规则完成评分，并按照固定模板输出。"""


class ModelBasedChartVerifierVolc(BaseVerifier):

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

        tb = ''
        for i in range(10):
            try:
                prompt_template = '<题目>：\n{problem}\n<标准答案>：\n{answer}\n<学生答案>：\n{response}\n\n'
                prompt = prompt_template.format(problem=problem, answer=answer, response=response)
                completion = self.client.chat.completions.create(model=self.model,
                                                                 messages=[
                                                                     {
                                                                         "role": "system",
                                                                         "content": judge_system_prompt
                                                                     },
                                                                     {
                                                                         "role": "user",
                                                                         "content": prompt
                                                                     },
                                                                 ],
                                                                 timeout=120)
                gpt_response = completion.choices[0].message.content
                score = compute_score(gpt_response)
                remark = f"题目:\n{problem}\n\n学生答案:\n{response}\n\n打分:{gpt_response}"

                return VerifyResult(score=score, extracted_answer=remark)
            except Exception as ex:
                import traceback
                logger.info(traceback.format_exc())
                logger.info(f'Got exception in compute_score via stem verifier_service: {ex}')
                print(traceback.format_exc())
                print(f'Got exception in compute_score via stem verifier_service: {ex}')
                time.sleep(random.randint(60, 150))
                continue
        raise VerifierFailed


if __name__ == '__main__':
    verifier_feature_dict = {
        'problem':
            'Identify the method where the dashed black line and the red line have the largest horizontal distance between them.\n * Your final answer must be grounded to some text that is explicitly written and relevant to the question in the chart.\n * If you need to answer multiple terms, separate them with commas.\n * Unless specified in the question (such as answering with a letter), you are required to answer the full names of subplots and/or labels by default.\n',
        'answer':
            'CC-GEE'
    }
    response = 'CC-GEE'
    volc_ark_key = ''
    volc_model_name = ''
    result = ModelBasedChartVerifierVolc().verify(response=response, verifier_feature_dict=verifier_feature_dict)
    print(result)
