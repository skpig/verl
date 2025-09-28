import base64
import io
import json
import os
import random
import re
import time

import openai
from PIL import Image
from tenacity import (retry, stop_after_attempt, wait_exponential_jitter, retry_if_not_exception_type)

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, VerifierFailed

JUDGE_PROMPT_TEMPLATE = """题目:
{prompt}

标准答案:
{response_reference}

学生答案:
{response}"""

JUDGE_SYSTEM_PROMPT = """现在你的角色是一名阅卷老师，你的任务是以标准答案参考，对学生的答案进行审核和评分。整个评分过程中，你需要熟知以下关键点：
-评分只用参考学生得到的最终答案来评判正确性，不需要考察中间的解题步骤是否正确。
-请先从学生的解答中提取出最终的答案，展示在分析结果中，然后再对答案是否正确进行评判。
-根据你的分析结果进行评分，在表述评分依据时，你应根据分析的逻辑进行分段描述。评分依据的总结应置于最后，可以采用如下格式：“综上，学生的答案应得x分”（x代表学生的具体得分）。
-请保持结果简短，输出在200字以内。
-根据你的分析来给出评分并用代码块以“JSON”格式展示。
你的输出格式为：
【评分依据】：
【评分】：x分
【JSON】：
{"answer_score": [[score]]}

以下是判卷评分标准：
【分数档位】：
参考标准答案评判最终答案，一共分为2档，由高到低分别为1分、0分（最低即为0分，如出现0分仍需扣分的情况，则给0分即可）。
【档位细则】：
1分：
学生得出的最终答案和标准答案一致，给1分。
如果题目有多个小问，需要每一问都回答正确才能给1分。
学生的答案和标准答案在数学上等价，例如学生的答案是1+1/2x, 标准答案为1+0.5x，由于1/2=0.5，所以这种情况也给1分。

0分：
学生得出的最终答案和标准答案不一致的给0分。

【示例1】：
<题目>：小明决定举办一场聚会聚会。如果他有5张桌子，每张桌子搭配有10把椅子。那么他的椅子比桌子多多少？
<标准答案>：45
<学生答案>：他一共有$5$张桌子，每张桌子配有$10$把椅子，那么他一共有$5\times10=50$把椅子。
椅子的数量减去桌子的数量可得：$50-5=45$（把）
所以，他的椅子比桌子多$45$把。

【评分依据】：学生得到的最终结论是椅子比桌子多45把。
标准答案是45。
这两个答案是一致的，所以学生的答案得分为1分。
【评分】：1分
【JSON】：
{
  "answer_score": [[1]]
}

【示例2】：
<题目>：甲筐有橙子42个，乙筐有橙子18个，如果甲、乙两筐倒出个数相等的橙子，剩下的橙子个数恰好甲筐是乙筐的9倍。那么甲筐剩下橙子多少个？乙筐剩下橙子多少个？
<标准答案>：27，3
<学生答案>：设甲筐和乙筐倒出相等的橙子各为x个。
根据题意，甲筐剩下的橙子数是乙筐剩下橙子数的9倍，可得方程：
甲筐剩下的橙子数 = 乙筐剩下的橙子数 * 9
(42 - x) = (18 - x) * 9
展开等式：
42 - x = 162 - 9x
添加9x到等式的两端：
9x - x = 162 - 42
合并同类项得到：
8x = 120
再次解方程得到：
x = 120 / 8
x = 15
甲、乙筐各自倒出了15个橙子。
现在我们可以计算剩下的橙子数：
甲筐剩下的橙子数 = 42 - 15 = 27个
乙筐剩下的橙子数 = 18 - 15 = 3个
现在检查是否符合甲筐剩下的橙子数是乙筐剩下的9倍：
27 = 3 * 9
27 = 27
答案符合题意。因此，甲筐剩下27个橙子，乙筐剩下3个橙子。
【评分依据】：学生的答案是甲筐剩下27个橙子，乙筐剩下3个橙子。
标准答案是27和3。
这两个答案是一致的，所以学生的答案得分为1分。
【评分】：1分
【JSON】：
{
  "answer_score": [[1]]
}

【示例3】：
<题目>：\(\sin105^{\circ}\)的值为（）
<标准答案>：\\frac{\\sqrt2+\\sqrt6}{4}
<学生答案>：要计算 sin(105°)，我们可以利用三角恒等式和90°以上角度的三角函数关系。105°可以表示为45°和60°的和，即105° = 45° + 60°。
三角函数相加的和差公式为：
sin(A + B) = sin(A)cos(B) + cos(A)sin(B)
因此，sin(105°)可以通过下列方式计算：
sin(105°) = sin(45° + 60°)
 = sin(45°)cos(60°) + cos(45°)sin(60°)
我们知道：
sin(45°) = cos(45°) = √2/2
sin(60°) = √3/2
cos(60°) = 1/2
代入这些值，计算 sin(105°) 为：
sin(105°) = (√2/2)(1/2) + (√2/2)(√3/2)
 = √2/4 + √6/4
 = (√2 + √6) / 4
所以 sin(105°) 的精确值是 (√2 + √6) / 4。

【评分依据】：
学生得到的最终答案是(√2 + √6) / 4。
标准答案是\\frac{\\sqrt6+\\sqrt2}{4}。答案是用latex表示的，我们知道\\frac{\\sqrt6+\\sqrt2}{4}=(√6 + √2)/4=(√2 + √6)/4。
这和标准答案(√2 + √6) / 4是一致的。因此所以学生的答案得分为1分。
【评分】：1分
【JSON】：
{
  "answer_score": [[1]]
}"""


def pil2base64(pil_img):
    buf = io.BytesIO()
    # Save the image as a PNG to the buffer
    pil_img.save(buf, format='jpeg')
    # Retrieve the byte data
    image_bytes = buf.getvalue()
    # Encode as base64
    image_base64 = base64.b64encode(image_bytes)
    # Convert bytes to string
    image_base64_str = image_base64.decode('utf-8')
    return image_base64_str


def base642pil(base64_str):
    # Convert base64 string to bytes
    image_bytes = base64.b64decode(base64_str)
    # Create a BytesIO object from the image bytes
    image_io = io.BytesIO(image_bytes)
    # Open the image using PIL
    image = Image.open(image_io)
    return image


def message_creator_v2(prompt, image_path, sys_prompt="You are a helpful assistant.", detail='high'):
    system_msg = {"role": "system", "content": sys_prompt}
    user_content = [{"type": "text", "text": prompt}]
    user_msg = {"role": "user", "content": user_content}

    if image_path:
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
            base64_img = pil2base64(image)
        else:
            image = Image.open(io.BytesIO(image_path)).convert("RGB")
            base64_img = pil2base64(image)
        new_msg = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}", "detail": detail}}
        user_content.append(new_msg)

    return [system_msg, user_msg]


def run_gpt4v_v2(client,
                 prompt,
                 image_path='',
                 temperature=1.0,
                 sys_prompt="You are a helpful assistant.",
                 n=1,
                 detail='high'):
    msgs = message_creator_v2(prompt, image_path, sys_prompt=sys_prompt, detail=detail)

    @retry(retry=retry_if_not_exception_type(openai.BadRequestError),
           wait=wait_exponential_jitter(jitter=120, max=180),
           stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)

    completion = completion_with_backoff(
        # model="gptv",
        model="gpt-4o-2024-11-20",
        messages=msgs,
        max_tokens=2048,
        temperature=temperature,
        # response_format={"type": "json_object"},
        n=n,
    )
    results = [choice.message.content for choice in completion.choices]

    return results, completion


def is_chat_prompt(prompt) -> bool:
    return isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt)


def populate_grading_inputs(response, answer):
    curr_query = "T1:\nResponse 1: {}\nGround Truth 1: {}\n\n" \
        .format(response, answer)
    return curr_query


def text_prompt_to_chat_prompt(prompt: str, role: str = "system"):
    assert isinstance(prompt, str), f"Expected a text prompt, got {prompt}"
    return [
        {
            "role": role,
            "content": prompt
        },
    ]


def llm_as_metric(question, answer, predict, client):
    gpt_version = "gpt-4o-2024-11-20"
    prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=question, response_reference=answer, response=predict)
    score = -100000
    gpt_response = ""
    tb = ""
    for _ in range(5):
        tb = ""
        try:
            if 'gpt' in gpt_version:
                result, _ = run_gpt4v_v2(client=client,
                                         prompt=prompt,
                                         temperature=0.1,
                                         n=1,
                                         sys_prompt=JUDGE_SYSTEM_PROMPT)
            else:
                raise NotImplementedError
            output_text = result[0]
            if '```' in output_text:
                score_pred = re.findall('```(json)?(.+)```', output_text, re.DOTALL)
                score_pred = json.loads(score_pred[0][1])
            else:
                score_pred = json.loads(output_text)
            score = score_pred['answer_score'][0][0]
            gpt_response = output_text
            break

        except Exception:
            import traceback
            import sys
            tb = "".join(traceback.format_exception(*sys.exc_info()))
            score = -100000
            gpt_response = ""
            sleep_time = random.random() * 60 + 60
            time.sleep(sleep_time)
            continue
    if tb:
        print(f'[VLM VERIFIER ERROR (V*)] {repr(tb)}')
    return score, gpt_response


class VstarVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        query = verifier_feature_dict['query']
        answer = verifier_feature_dict['answer']

        if not os.environ.get("GPT_KEYS", None):
            raise VerifierFailed("There is no GPT_KEYS in env!")

        api_keys = os.environ.get("GPT_KEYS", None).split(",")
        client = openai.AzureOpenAI(
            azure_endpoint="https://search.bytedance.net/gpt/openapi/online/multimodal/crawl",
            api_version="2023-07-01-preview",
            api_key=random.choice(api_keys),
        )

        # Our peak GPT quota is very limited. Wait a while to smooth the traffic.
        time.sleep(random.random() * 300)

        score, gpt_response = llm_as_metric(query, answer, response, client)
        return VerifyResult(score=score, extracted_answer=response)
