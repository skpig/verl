import json
import os
import random
import time

from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name
from alpha_seed.utils.reward_score.vlm_verifiers.extra_reward import VLM_ARC_CLIENT
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import ExtractAnswerFailed
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import VerifierFailed


def get_vlm_client_and_endpoint():
    if not os.environ.get("VLM_ARC_KEY", None):
        raise VerifierFailed(message="There is no VLM_ARC_KEY in env!")
    if VLM_ARC_CLIENT is None:
        raise VerifierFailed(message="No VLM_ARC_CLIENT, likely because there is no VLM_ARC_KEY in env!")

    if not os.environ.get("VLM_ARC_ENDPOINT", None):
        raise VerifierFailed(message="There is no VLM_ARC_ENDPOINT in env!")
    endpoint = os.environ.get("VLM_ARC_ENDPOINT", None)
    return VLM_ARC_CLIENT, endpoint


LANG_DETECT_PROMPT = '''You are a precise language detection assistant with strict output rules.

# Detection Guidelines
- Detect the primary language of the input text
- Output ONLY one of these exact options: "English", "Chinese", or "Uncertain"
- No additional text, punctuation, or explanation
- Determine language based on majority of text content

# Detailed Criteria
- English: Predominantly English words, grammar, and sentence structure
- Chinese: Predominantly Chinese characters, grammar, and sentence structure
- Uncertain:
  * Mixed languages with no clear majority
  * Mostly symbols, numbers, mathematical notation
  * Insufficient linguistic context to determine language
  * Highly technical or specialized text with minimal natural language

# Examples:

Input: 你好！虽然这段文本可能涉及一些用非中文的专业名词，比如机器学习（Machine Learning），但因为这段文本以中文为主，所以……
Output: Chinese

Input: Hello! This is an English text. English means 英文 in Chinese.
Output: English

Input: $\boxed{53}$
Output: Uncertain

Input: こんにちは。これは日本語のテキストです。
Output: Uncertain

# Current Task:
Input: <input>
Output:'''


def is_language_consistent(answer: str, expected_response_lang: str, client, endpoint: str) -> bool:
    assert expected_response_lang in ('zh', 'en')
    max_retry_num = 3
    tb = ''
    prompt = LANG_DETECT_PROMPT.replace('<input>', answer)
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
            result: str = completion.choices[0].message.content
            result = result.strip().lower()
            if result == 'english':
                return expected_response_lang == 'en'
            elif result == 'chinese':
                return expected_response_lang == 'zh'
            elif result == 'uncertain':
                return True
            else:
                return True
        except Exception:
            import sys
            import traceback
            tb = "".join(traceback.format_exception(*sys.exc_info()))
            sleep_time = random.random() * 90 + 30
            time.sleep(sleep_time)
            continue
    if tb:
        print(f'[VLM VERIFIER ERROR (LANG)] {repr(tb)}')
    return True


def check_language_correctness(query: str, response: str, verifier_feature: dict) -> None:
    client, endpoint = get_vlm_client_and_endpoint()
    response_language: str = verifier_feature.get('response_language')
    if response_language:
        # Check the answer:
        sot_token = get_special_tokens_dict_or_name("think_start_token")
        eot_token = get_special_tokens_dict_or_name("think_end_token")
        final_answer: str = response.split(eot_token)[-1].strip()
        if final_answer:
            if not is_language_consistent(
                    answer=final_answer, expected_response_lang=response_language, client=client, endpoint=endpoint):
                print(f'[VLM VERIFIER LANG SWITCH DETECTED (ANS)] {repr(query)} {repr(final_answer)}')
                raise ExtractAnswerFailed
        # Check the first CoT:
        first_cot = response.find(eot_token)
        if first_cot > 0:
            first_cot = response[:first_cot].split(sot_token)[-1].strip()
            if first_cot:
                if not is_language_consistent(
                        answer=first_cot, expected_response_lang=response_language, client=client, endpoint=endpoint):
                    print(f'[VLM VERIFIER LANG SWITCH DETECTED (COT)] {repr(query)} {repr(first_cot)}')
                    raise ExtractAnswerFailed


def get_valid_visual_tool_calls(full_rollout_response: str, num_input_images: int):
    last_call_valid = False
    last_call_str = ''
    # It maps `imgidx` to the tool call that generates the corresponding image, or to an empty dict if the image is given in the prompt:
    valid_tool_calls: list[dict] = [{}] * num_input_images
    eos = get_special_tokens_dict_or_name("eos")
    bos = get_special_tokens_dict_or_name("bos")
    soi = get_special_tokens_dict_or_name("soi")
    eoi = get_special_tokens_dict_or_name("eoi")
    convs = ("assistant\n" + full_rollout_response).split(f'{eos}{bos}')
    for idx, conv in enumerate(convs):
        if '<|FunctionCallBegin|>' in conv:
            last_call_str = conv.split("<|FunctionCallBegin|>")[1].split("<|FunctionCallEnd|>")[0]
            if (idx + 1 < len(convs)) and ((f'{soi}{eoi}' in convs[idx + 1]) or
                                           (f'{soi}<ImageHere>{eoi}' in convs[idx + 1])):
                last_call_valid = True
                valid_tool_calls.append(json.loads(last_call_str)[0])
            else:
                last_call_valid = False
    return last_call_valid, last_call_str, valid_tool_calls
