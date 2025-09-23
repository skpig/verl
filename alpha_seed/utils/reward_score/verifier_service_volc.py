import os
import json
import ray
from typing import Optional
import random
import time

from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name


def last_boxed_only_string_v2(string: str) -> Optional[str]:
    """
    find last \\boxed{...}
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    left = "\\boxed{"

    assert s[:len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"

    return s[len(left):-1]


def get_text_after_last_final_answer(text):
    """
    查找文本中最后一个“Final Answer”之后的内容。

    参数:
        text (str): 输入的文本。

    返回:
        str 或 None: 返回最后一个“Final Answer”后的内容，或如果未找到则返回None。
    """
    search_phrase = "final answer"
    lower_text = text.lower()
    index = lower_text.rfind(search_phrase)

    if index == -1:
        return None

    # 计算“Final Answer”后面的起始位置
    content_start = index + len(search_phrase)
    content = text[content_start:].strip()
    if content.startswith(":"):
        content = content[1:].strip()
    return content if content else None


def get_text_after_answer_tag(text):
    """
    查找文本中第一个"<Answer>"之后的内容。

    参数:
        text (str): 输入的文本。

    返回:
        str 或 None: 返回最后一个“Final Answer”后的内容，或如果未找到则返回None。
    """
    if text.startswith("A conversation between user and assistant."):
        text = text[400:]
    search_phrase = "<Answer>".lower()
    lower_text = text.lower()
    start_index = lower_text.rfind(search_phrase)

    if start_index == -1:
        return None

    # 计算“Final Answer”后面的起始位置
    content_start = start_index + len(search_phrase)
    content = text[content_start:].strip()

    search_phrase = "</Answer>".lower()
    lower_content = content.lower()
    end_index = lower_content.find(search_phrase)

    if end_index == -1:
        return None

    content = content[:end_index].strip()
    return content if content else None


def get_text_after_think_tag(text):
    """
    查找文本中"think start token"之后的内容。

    参数:
        text (str): 输入的文本。

    返回:
        str 或 None: 返回Think后的内容，或如果未找到则返回None。
    """
    eot_token = get_special_tokens_dict_or_name("think_end_token")
    search_phrase = eot_token.lower()
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


class LLMClient:

    def __init__(self, psm, idc="maliva", cluster="default", model_name=""):
        from bytedagi.chat_models import ChatUltramanLLMServer
        self.llm = ChatUltramanLLMServer(
            psm=psm,
            idc=idc,
            cluster=cluster,
            model_name=model_name,
            timeout=120,
        )

    def streaming(self, text):
        for chunk in self.llm.stream(text):
            yield chunk

    async def async_stream(self, text):
        async for chunk in self.llm.astream(text, top_p=0.7):
            yield chunk

    def chat(self, prompts, system="", temperature=1.0, top_p=0.7, max_tokens=4096):
        from langchain.schema.messages import (AIMessage, HumanMessage, SystemMessage)

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(system, str):
            if len(system) <= 0:
                system = len(prompts) * [None]
            else:
                system = len(prompts) * [system]
        elif isinstance(system, list):
            if len(system) != len(prompts):
                system = len(prompts) * [None]
                print(f"system prompt length :`{len(system)}` is not equal to prompts length: `{len(prompts)}`")
        else:
            if system is not None:
                print(f"system prompt `{system}` is not supported, convert to default: None")
            system = len(prompts) * [None]

        messages = []
        for sp, p in zip(system, prompts):
            sub_messages = []
            if isinstance(p, str):
                p = [p]
            if not isinstance(p, list):
                print(f"prompts type `{type(p)}` and prompt `{p}` are not supported")
                continue
            if sp is not None:
                sub_messages.append(SystemMessage(content=sp))
            for idx, item in enumerate(p):
                if idx % 2 == 0:
                    sub_messages.append(HumanMessage(content=item))
                else:
                    sub_messages.append(AIMessage(content=item))
            messages.append(sub_messages)
        try:
            results = self.llm.batch(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            results = [result.content for result in results]
            return results
        except Exception as e:
            print(e)
            return None


VERIFY_TEMPLATE = """你是一位资深的大模型阅卷专家，你的核心任务是依据给定的考试题目、对应的参考答案，以及学生的作答内容，精准判断学生回答是否正确。

执行任务时，请严格遵循以下准则：
1. 部分题目存在多个合理的参考答案，面对这类题目，无需关注参考答案，直接依据学生作答内容判断其正确性。
2. 若原始题目存在一定程度的不完整性，例如选择题缺失具体选项，同样忽略参考答案，仅根据学生的回答做出判断。
3. 若学生在回答中呈现多个答案，应以其最后一个答案为准来判断对错，多答案情况不视为正确回答。
4. 当参考答案与学生作答在形式上有所不同时，比如参考答案仅标明选项 D，而学生作答完整写出了 D 选项的具体内容；或是在数学作答中，学生给出的数学表达式虽然元素顺序与参考答案不完全一致，但通过运算规则、数学原理可证明二者本质等价，表达的数学含义相同；又或者学生给出了分数表达式，而参考答案是等价的四舍五入的小数形式，这类情况均应判定为回答正确。
5. 若学生回答与问题类型不匹配时，如问题类型是一道数学题或者选择题，但学生给出了一段代码实现，且不包含最终的答案，应判定为回答错误。

# 考试题目
{problem}

# 参考答案
{reference_answer}

# 学生作答
{model_response}

仅输出"是/否"，不允许包含其他多余内容。"""

from .utils import Verifier


class VerifierServiceVolc(Verifier, reward_style="verifier_service_volc"):

    def is_remote(self):
        return self.config.trainer.use_remote_verifier

    def preprocess(self, input_ids, ground_truth):
        solution_str, ground_truth = super().preprocess(input_ids, ground_truth)
        return solution_str, ground_truth, self.config.trainer.volc_ark_key, self.config.trainer.volc_model_name

    @staticmethod
    def compute_score(solution_str, ground_truth, volc_ark_key, volc_model_name) -> float:
        return compute_score(solution_str, ground_truth, volc_ark_key, volc_model_name)


def compute_score(solution_str, ground_truth, volc_ark_key, volc_model_name, **argv) -> float:
    correct_reward = 1
    wrong_reward = -1

    for i in range(10):
        try:
            if solution_str.startswith("A conversation between user and assistant."):
                solution_str = solution_str[400:]
            if isinstance(ground_truth, str):
                ground_truth = json.loads(ground_truth)
            problem = ground_truth["problem"]
            reference_answer = ground_truth["reference_answer"]
            if isinstance(reference_answer, int):
                reference_answer = str(reference_answer)
            verify_type = ground_truth["verify_type"]
            if verify_type == 0:
                extracted_response = solution_str[-100:]
            elif verify_type == 1:
                last_boxed_string = last_boxed_only_string_v2(solution_str)
                if last_boxed_string is None:
                    return wrong_reward
                else:
                    extracted_response = remove_boxed(last_boxed_string)
            elif verify_type == 2:
                candidate_answer = get_text_after_last_final_answer(solution_str)
                if candidate_answer is None:
                    return wrong_reward
                else:
                    extracted_response = candidate_answer
            elif verify_type == 3:
                candidate_answer = get_text_after_answer_tag(solution_str)
                if candidate_answer is None:
                    return wrong_reward
                else:
                    extracted_response = candidate_answer
            elif verify_type == 4:
                candidate_answer = get_text_after_think_tag(solution_str)
                if candidate_answer is None:
                    extracted_response = solution_str[-200:]
                else:
                    extracted_response = candidate_answer
            else:
                raise NotImplementedError

            prompt = VERIFY_TEMPLATE.format(problem=problem,
                                            reference_answer=reference_answer,
                                            model_response=extracted_response)

            from openai import OpenAI

            client = OpenAI(
                base_url="https://ark-cn-beijing.bytedance.net/api/v3",
                api_key=volc_ark_key,
            )
            model = volc_model_name

            completion = client.chat.completions.create(model=model,
                                                        messages=[
                                                            {
                                                                "role": "user",
                                                                "content": prompt
                                                            },
                                                        ],
                                                        timeout=120)
            response = completion.choices[0].message.content
            if response.startswith("是") or response.lower().startswith("yes"):
                return correct_reward
            return wrong_reward
        except Exception as ex:
            import traceback
            print(traceback.format_exc())
            time.sleep(random.randint(60, 150))
            continue
    print(f'Got exception in compute_score via verifier_service:')
    return -2
