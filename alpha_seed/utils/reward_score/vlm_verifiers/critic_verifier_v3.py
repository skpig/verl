import ast
import json
import logging
import os
import random
import re
import time
from collections import defaultdict

import openai
from openai import OpenAI

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult

MATCHING_PROMPT = '''你是一个语言专家。给定一个用户的问题和一条来自助理的回答，以及两个记录来自助理的回答中存在的**缺陷**的集合。其中一个是模型预测出来的缺陷集合，称为**预测集合**；另一个是真实的缺陷集合，称为**真实集合**。预测集合中的缺陷被称为**预测缺陷**，真实集合中的缺陷被称为**真实缺陷**。这两个集合采用JSON格式的List of Dict记录，其中每个Dict都是一条**缺陷**。例如：[{{"context": [缺陷的上下文], "content": [缺陷的内容], "pred_id" 或 "oracle_id": [缺陷的ID]}}, ...]。 其中context用来表示该条缺陷的**上下文**，即所处的原始回答片段。content用来表示缺陷的**内容**，包括缺陷的原因和具体信息等。pred_id和oracle_id在每个集合里面是独一无二的，用来表示缺陷的**序号**。

你的任务是对预测集合中的每条缺陷，依次检测真实集合中的缺陷，返回与预测集合的缺陷最为匹配的一条来自真实集合的缺陷。如果真实集合的所有缺陷均不能与之匹配，则返回None。

**请注意**：逐个检查预测集合中的每条缺陷，每条缺陷必须单独考虑，不得遗漏任何一条缺陷。对每条预测缺陷，逐个检查真实集合中的每条缺陷，每条缺陷必须单独考虑，不得遗漏任何一条缺陷。成功的匹配需要**同时全部满足**下面的规则：

1. 预测集合中的缺陷的内容应该是详细且具体的，如果缺陷的内容中没有详细指出错误的具体信息，而是宽泛地表达存在某种类型的错误，则该缺陷不能和真实集合中的缺陷匹配。

2. 预测集合中的缺陷的内容应该是明确的，如果缺陷的内容没有明确指出错误的对象，而是模棱两可地表达错误可能存在于其中，则该缺陷不能和真实集合中的缺陷匹配。

3. 预测集合中的缺陷的上下文应该只包含原始回答的内容，如果缺陷的上下文出现了其他内容，则该缺陷不能和真实集合中的缺陷匹配。

4. 预测集合中的缺陷的上下文应该和真实集合中的缺陷的上下文相匹配，如果预测缺陷的上下文范围比真实缺陷的上下文范围大很多、或者小很多、或者二者不存在交集，则该预测缺陷不能和该真实缺陷匹配。

5. 预测集合中的缺陷的内容不能只简单复述缺陷的上下文，如果缺陷的内容只是对缺陷的上下文的简单复述或片段截取，则该缺陷不能和真实集合中的缺陷匹配。

6. 在上述规则都满足的情况下，结合缺陷的上下文和内容，判断两条缺陷是否传达相同的意思。即两条缺陷是否具有相近的上下文，是否是出于同样的原因，其发生的对象是否为同一对象。如果是，则可以匹配。如果不是，则不能匹配。

记录所有匹配的缺陷（即两集合的交集），你最终只需要使用JSON格式输出匹配结果，例如
```json
[{{"pred_flaw_id": [预测缺陷的id], "matched_oracle_flaw_id": [对应的真实缺陷的id]}}, ...]
```
对于键名 "matched_oracle_flaw_id"，其值应是与当前预测缺陷匹配的真实缺陷的ID。如果当前预测缺陷无法与任何真实缺陷匹配，将 "matched_oracle_flaw_id" 设置为 "None"。

<Start of user's question>
{}
<End of user's question>
<Start of assistant's response>
{}
<End of assistant's response>
<Start of Predicted Set>
{}
<End of Predicted Set>
<Start of Oracle Set>
{}
<End of Oracle Set>
'''

logger = logging.getLogger()


def message_creator_v2(prompt, base64_img, sys_prompt="You are a helpful assistant.", detail='high'):
    system_msg = {"role": "system", "content": sys_prompt}
    user_content = [{"type": "text", "text": prompt}]
    user_msg = {"role": "user", "content": user_content}

    if base64_img:
        # if isinstance(image_path, str):
        #     image = Image.open(image_path).convert('RGB')
        #     base64_img = pil2base64(image)
        # else:
        #     image = Image.open(io.BytesIO(image_path)).convert("RGB")
        #     base64_img = pil2base64(image)
        new_msg = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": detail}}
        user_content.append(new_msg)

    return [system_msg, user_msg]


def run_gpt4v_v2(client,
                 model_name,
                 prompt,
                 image_path='',
                 temperature=1.0,
                 sys_prompt="You are a helpful assistant.",
                 n=1,
                 detail='high'):
    msgs = message_creator_v2(prompt, image_path, sys_prompt=sys_prompt, detail=detail)

    completion = client.chat.completions.create(
        # model="gptv",
        model=model_name,
        messages=msgs,
        max_tokens=2048,
        temperature=temperature,
        # response_format={"type": "json_object"},
        n=n,
    )
    results = [choice.message.content for choice in completion.choices]

    return results, completion


def format_json_input(input_list):
    res = '```json\n[\n'
    for d in input_list:
        res += '  {},\n'.format(json.dumps(d, ensure_ascii=False))
    res += ']\n```'
    return res


def extract_json_blocks(text):
    # Pattern to match content between ```json and ```
    pattern = r"```json\s*([\s\S]*?)\s*```"

    # Find all matches
    matches = re.findall(pattern, text)

    if matches:
        return matches[-1]
    else:
        return text


def compute_jaccard_similarity(pred_set, oracle_set, match_results):
    pred_matched = {}
    oracle_matched = {}
    for p in pred_set:
        pred_matched[p["pred_id"]] = 0
    for o in oracle_set:
        oracle_matched[o["oracle_id"]] = 0

    for match_result in match_results:
        if not ("pred_flaw_id" in match_result and "matched_oracle_flaw_id" in match_result):
            continue
        pred_id = match_result['pred_flaw_id'] if not match_result['pred_flaw_id'] else str(
            match_result['pred_flaw_id'])
        oracle_id = match_result['matched_oracle_flaw_id'] if not match_result['matched_oracle_flaw_id'] else str(
            match_result['matched_oracle_flaw_id'])
        if not oracle_id or oracle_id in ["None", "Null", "NULL", "none", "null"]:
            pass
        elif pred_id not in pred_matched or oracle_id not in oracle_matched:
            pass
        elif oracle_matched[oracle_id] == 0 and pred_matched[pred_id] == 0:
            # 如果一个oracle_claim匹配了多个pred_claim，除第一个外的pred_claim皆匹配None
            pred_matched[pred_id] = 1
            oracle_matched[oracle_id] = 1

    assert sum(list(pred_matched.values())) == sum(list(oracle_matched.values()))
    matched_num = sum(list(pred_matched.values()))
    non_matched_pred = len(pred_matched) - sum(list(pred_matched.values()))
    non_matched_oracle = len(oracle_matched) - sum(list(oracle_matched.values()))

    return float(matched_num) / float(matched_num + non_matched_pred + non_matched_oracle + 1e-5)


def compute_f_beta_score(pred_set, oracle_set, match_results, beta=1.0):
    pred_matched = {}
    oracle_matched = {}
    for p in pred_set:
        pred_matched[p["pred_id"]] = 0
    for o in oracle_set:
        oracle_matched[o["oracle_id"]] = 0

    for match_result in match_results:
        if not ("pred_flaw_id" in match_result and "matched_oracle_flaw_id" in match_result):
            continue
        pred_id = match_result['pred_flaw_id'] if not match_result['pred_flaw_id'] else str(
            match_result['pred_flaw_id'])
        oracle_id = match_result['matched_oracle_flaw_id'] if not match_result['matched_oracle_flaw_id'] else str(
            match_result['matched_oracle_flaw_id'])
        if not oracle_id or oracle_id in ["None", "Null", "NULL", "none", "null"]:
            pass
        elif pred_id not in pred_matched or oracle_id not in oracle_matched:
            pass
        elif oracle_matched[oracle_id] == 0 and pred_matched[pred_id] == 0:
            # 如果一个oracle_claim匹配了多个pred_claim，除第一个外的pred_claim皆匹配None
            pred_matched[pred_id] = 1
            oracle_matched[oracle_id] = 1

    assert sum(list(pred_matched.values())) == sum(list(oracle_matched.values()))
    matched_num = sum(list(pred_matched.values()))
    non_matched_pred = len(pred_matched) - sum(list(pred_matched.values()))
    non_matched_oracle = len(oracle_matched) - sum(list(oracle_matched.values()))
    precision = float(matched_num) / float(matched_num + non_matched_pred + 1e-5)
    if matched_num + non_matched_pred == 0:
        precision = 1.0
    recall = float(matched_num) / float(matched_num + non_matched_oracle + 1e-5)
    if matched_num + non_matched_oracle == 0:
        recall = 1.0
    f_beta_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-5)
    logger.info("mark debug > f_beta: {}".format(beta))

    return f_beta_score


def compute_jaccard_similarity_old(pred_set, oracle_sets, match_results):
    mappings = defaultdict(list)

    num_not_matched = 0
    for match_result in match_results:
        if match_result['matched_oracle_flaw_id'] is None or match_result['matched_oracle_flaw_id'] == 'None':
            num_not_matched += 1
        else:
            mappings[match_result['matched_oracle_flaw_id']].append(match_result['pred_flaw_id'])

    numerator = len(mappings)
    denominator = num_not_matched + len(oracle_sets)

    return numerator / (denominator + 1e-5)


class PointwiseCriticVerifier(BaseVerifier):

    def __init__(
        self,
    ):
        self.gpt_api_keys = os.environ.get('AZURE_OPENAI_API_KEY', "").split(',')
        if self.gpt_api_keys:
            azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT',
                                            'https://search.bytedance.net/gpt/openapi/online/v2/crawl')
            self.key_2_clients = {}
            self.key_2_model_names = {}
            for api_key in self.gpt_api_keys:
                self.key_2_clients[api_key] = openai.AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_version="2024-11-01-preview",
                    api_key=api_key,
                    timeout=30,
                )
                self.key_2_model_names[api_key] = "gpt-4o-2024-08-06"

        self.r1_api_key = os.environ.get("VLM_ARC_KEY", None)
        base_url = os.environ.get('VOLC_ARK_BASE_URL', "https://ark-cn-beijing.bytedance.net/api/v3")

        if self.r1_api_key:
            self.r1_client = OpenAI(
                api_key=self.r1_api_key,
                base_url=base_url,
                timeout=1800,
            )
            self.r1_endpoint = os.environ.get("VLM_ARC_ENDPOINT", None)
            if not self.r1_endpoint:
                raise ValueError("VLM_ARC_ENDPOINT is not set")

    def verify(self, response: str, verifier_feature_dict: dict, f_beta: float = 1.0) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        pred = extract_json_blocks(response)
        if pred == "":
            raise ExtractAnswerFailed

        # Prepare Predicted Critic
        try:
            predicted_critics = json.loads(pred)
        except:
            try:
                predicted_critics = ast.literal_eval(pred)
                if type(predicted_critics) is not list:
                    raise ExtractAnswerFailed
            except:
                raise ExtractAnswerFailed

        try:
            if len(predicted_critics) == 0:
                # raise ExtractAnswerFailed
                pass
            else:
                pred_id_set = set()
                for pc in predicted_critics:
                    if type(pc) is not dict:
                        raise ExtractAnswerFailed
                    if "context" not in pc:
                        raise ExtractAnswerFailed
                    if "pred_id" not in pc:
                        raise ExtractAnswerFailed
                    if "content" not in pc:
                        raise ExtractAnswerFailed
                    if "type" not in pc:
                        raise ExtractAnswerFailed
                    if "critical" not in pc:
                        raise ExtractAnswerFailed
                    if pc["pred_id"] in pred_id_set:
                        raise ExtractAnswerFailed
                    pred_id_set.add(pc["pred_id"])
        except:
            raise ExtractAnswerFailed

        # predicted_critics_with_ids = [{"pred_id": idx+1, "flaw": flaw} for idx, flaw in enumerate(predicted_critics.values())]
        predicted_critics_new = []
        for idx, pc in enumerate(predicted_critics):
            pc_new = {
                "pred_id": str(pc["pred_id"]),
                "context": pc["context"],
                "content": pc["content"],
            }
            predicted_critics_new.append(pc_new)
        predicted_critics_with_ids_str = format_json_input(predicted_critics_new)

        # Prepare Oracle Critic
        # oracle_critics_with_ids = [{"oracle_id": idx+1, "flaw": flaw} for idx, flaw in enumerate(oracle_critics)]
        for oc in answer:
            oc["oracle_id"] = str(oc["oracle_id"])
        oracle_critics_with_ids_str = format_json_input(answer)

        verify_prompt = MATCHING_PROMPT.format(verifier_feature_dict["raw_question"],
                                               verifier_feature_dict["raw_response"], predicted_critics_with_ids_str,
                                               oracle_critics_with_ids_str)

        OJ_RETRY_TIMES = 50
        for i in range(OJ_RETRY_TIMES):
            try:
                # use r1
                if self.r1_api_key:
                    completion = self.r1_client.chat.completions.create(
                        model = self.r1_endpoint,  # your model endpoint ID
                        messages = [
                            {"role": "system", "content": "你是人工智能助手"},
                            {"role": "user", "content": verify_prompt},
                        ],
                        temperature=0.1,
                    )
                    matched_results_raw = completion.choices[0].message.content
                elif self.gpt_api_keys:
                    # use 4o
                    api_key = random.choice(self.gpt_api_keys)
                    client = self.key_2_clients[api_key]
                    model_name = self.key_2_model_names[api_key]
                    results, _ = run_gpt4v_v2(
                        client=client,
                        model_name=model_name,
                        prompt=verify_prompt,
                        image_path=None,
                        temperature=0.1,
                        n=1,
                    )
                    matched_results_raw = results[0]
                else:
                    raise ValueError("No API Key!")

                matched_results = extract_json_blocks(matched_results_raw)
                logger.info(f"[GPT Verification Results]: {matched_results_raw}")
                assert matched_results
                matched_json = json.loads(matched_results)
                score = compute_f_beta_score(predicted_critics_new, answer, matched_json, f_beta)
                return VerifyResult(score=score, extracted_answer=matched_results_raw)
            except Exception as e:
                import sys
                import traceback
                tb = "".join(traceback.format_exception(*sys.exc_info()))
                logger.info("-" * 40 + " GPT Verification Error!!! " + "-" * 40 + f"\n{tb}\n" + "-" * 80)
                sleep_time = random.random() * 120 + 60
                time.sleep(sleep_time)
                continue
        return VerifyResult(score=0, extracted_answer="GPT Verification Server Error")
