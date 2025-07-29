import collections
import re
from typing import Optional
import warnings
import json
from transformers import PreTrainedTokenizer

from alpha_seed.utils.reward_score.remote_verify import OpenAIConfig, ThreadPoolOpenAIClient, judge_template_v1, judge_template_v2, judge_template_v2_1


def _get_last_ans(text: str) -> Optional[str]:
    matches = re.findall(r'<ans>(.*?)</ans>', text, re.DOTALL)
    return matches[-1] if matches else None


def count_turns_v1(text: str, bos_token: str, eos_token: str) -> int:
    pattern = re.escape(bos_token) + r'(.*?)' + re.escape(eos_token)
    # pattern = r'(.*?)' + re.escape(eos_token)
    matches = re.findall(pattern, text, re.DOTALL)
    return len(matches)


def count_turns_v2(text: str, bos_token: str, eos_token: str) -> int:
    # 1. 匹配第一条（不需要 <bos>，只要以 <eos> 结尾）
    pattern_first = r'^(.*?)' + re.escape(eos_token)
    first_match = re.match(pattern_first, text, re.DOTALL)
    count = 1 if first_match else 0
    # 2. 匹配后面的 <bos> ... <eos>
    pattern_rest = re.escape(bos_token) + r'(.*?)' + re.escape(eos_token)
    rest_matches = re.findall(pattern_rest, text, re.DOTALL)
    count += len(rest_matches)
    return count


def strip_format(response):
    idx = response.find("```json")
    if idx != -1:
        response = response[idx + len("```json"):]
    # find the end ``` and remove it
    idx = response.find("```")
    if idx != -1:
        response = response[:idx]
    response = response.strip().strip('`').strip()
    return response


def safe_json_loads(json_string: str,
                    verbose: bool = False,
                    default_on_failure: any = None,
                    res_key: str = "is_correct"):
    """
    Safely loads a JSON string, attempting to fix common escape issues if initial parsing fails.

    This function is particularly helpful for parsing JSON strings from LLMs
    that might contain improperly escaped characters, leading to "Invalid \escape" errors.

    Args:
        json_string (str): The JSON string to parse.
        verbose (bool): If True, prints error messages to stdout if parsing fails
                        at any stage.
        default_on_failure (any): The value to return if parsing fails even after
                                  attempting fixes. Defaults to None.
        res_key (str): The key for final results.

    Returns:
        any: The parsed Python object (dict, list, etc.), or `default_on_failure`
             if all parsing attempts fail.
    """
    if not isinstance(json_string, str):
        if verbose:
            print(f"Error: Input is not a string (type: {type(json_string)}). Cannot parse JSON.")
        return default_on_failure

    original_string_for_retry = json_string

    try:
        try:
            # First attempt: direct parsing
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(json_string)
            if verbose:
                print(f"Initial JSON parsing failed: {e}")
                print(f"Problematic section (around char {e.pos}): '{json_string[max(0, e.pos - 20):e.pos + 20]}'")
                print("Attempting to fix common escape characters and re-parse...")

            fixed_string = json_string

            # 1. 修复路径中的单斜杠 (e.g., "C:\Users" -> "C:\\Users")
            fixed_string = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', fixed_string)

            # 2. 修复未转义的双引号 (e.g., '{"key": "value"with"quotes"}' -> '{"key": "value\"with\"quotes"}')
            # 注意：这只会修复不在引号对中的双引号
            in_quotes = False
            result = []
            for i, char in enumerate(fixed_string):
                if char == '"':
                    # 检查是否在转义字符后
                    if i > 0 and fixed_string[i - 1] == '\\':
                        result.append(char)
                        continue
                    in_quotes = not in_quotes
                if char == '"' and not in_quotes:
                    result.append('\\"')
                else:
                    result.append(char)
            fixed_string = ''.join(result)

            # 3. 处理非转义控制字符 (e.g., \x00-\x1F except \t \n \r)
            fixed_string = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', fixed_string)

            # 4. 修复缺少的引号 (e.g., {key: "value"} -> {"key": "value"})
            # 注意：这是一个简化的实现，可能不适用于所有情况
            fixed_string = re.sub(r'([{,])\s*(\w+)\s*:', r'\1 "\2":', fixed_string)

            # 5. 修复尾随逗号 (e.g., {"key": "value",} -> {"key": "value"})
            fixed_string = re.sub(r',\s*([}\]])', r'\1', fixed_string)

            if verbose:
                if fixed_string != original_string_for_retry:
                    print("Modifications made by escape fixer.")
                    # Showing only a part of potentially very long strings
                    print(
                        f"Snippet of original (around char {e.pos}): '{original_string_for_retry[max(0, e.pos - 50):e.pos + 50]}'"
                    )
                    print(f"Snippet of fixed (around char {e.pos}): '{fixed_string[max(0, e.pos - 50):e.pos + 50]}'")
                else:
                    print("No changes made by escape fixing regex. Re-parsing original (likely to fail again).")

            return json.loads(fixed_string)

    except Exception as ex_initial:  # Catch any other unexpected errors during initial parsing
        print(
            f"An unexpected critical error occurred during initial JSON parsing: {ex_initial} and automatically fix unsuccessfully. Try to manually extract the judgement result."
        )
        # NOTE: 尝试使用规则方法提取判断，如果提取不出来，则为错误
        if "explanation" in json_string and res_key in json_string:
            explanation = json_string.split("explanation:")[-1].split(f"{res_key}:")[0].strip()
            judgement = json_string.split(f"{res_key}:")[-1].strip()
            if 'true' in judgement.lower() and 'false' not in judgement.lower():
                print(f"Manually extract True from {json_string}")
                return {res_key: True, "explanation": explanation}
            elif 'false' in judgement.lower() and 'true' not in judgement.lower():
                print(f"Manually extract False from {json_string}")
                return {res_key: False, "explanation": explanation}
            else:
                raise ex_initial
        else:
            raise ex_initial


def postprocess_fn_v1(result):
    return result.strip()


def postprocess_fn_v2(result):
    try:
        res = safe_json_loads(strip_format(result))["is_correct"]
        if res is True:
            return "CORRECT"
    except Exception as ex:
        print(f"Error processing result for result string: {result}")
        print(f"Exception: {ex}")
        return ""
    return "INCORRECT"


judge_template_mapping = {
    "v1": (judge_template_v1, postprocess_fn_v1),
    "v2": (judge_template_v2, postprocess_fn_v2),
    "v2.1": (judge_template_v2_1, postprocess_fn_v2)
}


def validate_format(s: str, tokenizer: PreTrainedTokenizer):
    rounds = s.split("assistant\n")

    for turn in rounds:
        format_reward = validate_format_single_turn(turn, tokenizer)
        if format_reward is False:
            return 0.
    return 1.


def validate_format_single_turn(s: str,
                                tokenizer: PreTrainedTokenizer,
                                ans_begin_str: str = "<ans>",
                                ans_end_str: str = "</ans>") -> bool:
    """
    校验字符串格式：
      1. 必须包含 <think>任意内容</think>
      2. </think> 之后紧跟 <|FunctionCallBegin|> 或 <ans>
      3. <|FunctionCallEnd|> 之后紧跟 EOS，且字符串以此结束
    """
    # 1. 检查 <think>…</think> 存在
    if not re.search(r'<think>.*?</think>', s, flags=re.DOTALL):
        return False

    # 2. 检查 </think> 后面紧跟指定标记
    close_idx = s.find('</think>')
    if close_idx == -1:
        return False
    after_think = s[close_idx + len('</think>'):]
    if not (after_think.strip().startswith('<|FunctionCallBegin|>') or after_think.strip().startswith(ans_begin_str)):
        return False

    # 3. 检查最后的 <|FunctionCallEnd|> 后紧跟 EOS 并且以此结束
    end_tag = '<|FunctionCallEnd|>'
    end_idx = s.rfind(end_tag)
    if end_idx == -1:
        end_tag = ans_end_str
        end_idx = s.rfind(end_tag)
    if end_idx == -1:
        return False
    # 从 end_tag 末尾开始，应当直接是 EOS
    if not s[end_idx + len(end_tag):].strip().startswith(tokenizer.eos_token):
        return False

    return True


def process_single_request_result(result, postprocess_fn, question, ground_truth, answer):
    real_result = ""
    if not isinstance(result, dict):
        raise Exception(f"Unexpected result type {type(result)}: {result}")
    if result["success"]:
        try:
            real_result = result['data']['choices'][0]['message']['content']
            real_result = postprocess_fn(real_result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(
                f"[DEBUG 遭遇问题: {e}]: {question}\n正确答案：{ground_truth}\n回答：{answer}\n判定结果：{real_result}\n远程验证回复：{result}")
    else:
        print(
            f"[DEBUG 遭遇问题: Unsuccess]: {question}\n正确答案：{ground_truth}\n回答：{answer}\n判定结果：{real_result}\n远程验证回复：{result}"
        )
    return real_result


def agent_env_score(**kwargs) -> tuple[int, float]:
    """
    注意，需要kwargs["solution_str"]是包含bos、eos token的字符串（skip_special_tokens=False），才能正确计算有几轮

    使用到的输入参数：
        use_remote_verify
        openai_config
        solution_str
        env_question
        ground_truth
        max_try
        step_penalty_coef
        tokenizer
    """
    use_remote_rm: bool = kwargs.get("use_remote_verify", False)
    openai_config: OpenAIConfig = kwargs.get("openai_config", None)
    judge_prompt_ver: str = kwargs.get("judge_prompt_ver", "v1")
    judge_template, judge_postprocess = judge_template_mapping.get(judge_prompt_ver,
                                                                   (judge_template_v1, postprocess_fn_v1))
    enable_think_format: bool = kwargs.get("enable_think_format", False)
    is_validation: bool = kwargs.get("is_validation", False)
    if use_remote_rm and openai_config is not None:
        try:
            answer = _get_last_ans(kwargs["solution_str"])
            if answer is None:
                print(f"[DEBUG] 没有找到答案")
                return 0, 0
            max_try = kwargs.get("max_try", 1)
            step_penalty_coef = kwargs.get("step_penalty_coef", 0.)
            tokenizer = kwargs['tokenizer']
            turn_cnt = count_turns_v2(kwargs["solution_str"], tokenizer.bos_token, tokenizer.eos_token)
            if turn_cnt == 0:
                warnings.warn(f"[DEBUG]回答只有0轮（没有检测到eos token），需要留意有没有地方出错了，检查。response为：\n{kwargs['solution_str']}")
            judge_prompt = judge_template.format(question=kwargs['env_question'],
                                                 gold_answer=kwargs["ground_truth"],
                                                 answer=answer)
            messages_list = [{"role": "user", "content": judge_prompt}]
            real_result = ""
            try_count = 0
            client = ThreadPoolOpenAIClient(config=openai_config)
            while real_result not in ['CORRECT', 'INCORRECT'] and try_count < max_try:
                # results = ThreadPoolOpenAIClient._make_request(messages=messages_list, config=openai_config)
                results = client.make_multiple_request(messages_list, openai_config)
                # if results["success"]:
                #     try:
                #         real_result = results['data']['choices'][0]['message']['content']
                #         real_result = judge_postprocess(real_result)
                #     except Exception as e:
                #         import traceback
                #         traceback.print_exc()
                #         print(
                #             f"[DEBUG 遭遇问题: {e}]: {kwargs['env_question']}\n正确答案：{kwargs['ground_truth']}\n回答：{answer}\n判定结果：{real_result}\n远程验证回复：{results}")
                #         real_result = ""
                # else:
                #     real_result = ""
                votes = collections.Counter()
                for result in results:
                    real_result = process_single_request_result(result, judge_postprocess, kwargs['env_question'],
                                                                kwargs['ground_truth'], answer)
                    votes[real_result] += 1
                try_count += 1
                real_result = votes.most_common(1)[0][0]
            reward = max(int(real_result == 'CORRECT') - step_penalty_coef * turn_cnt, 0.)
            acc = real_result == 'CORRECT'
            print(f"[DEBUG 远程验证结果]\n问题：{kwargs['env_question']}\n正确答案：{kwargs['ground_truth']}\n回答：{answer}\n"
                  f"判定结果：{real_result}\n奖励：{reward}\n")

            if enable_think_format and not is_validation:
                format_reward = validate_format(kwargs["solution_str"] + kwargs["tokenizer"].eos_token,
                                                kwargs["tokenizer"])
                # if format_reward == 0:
                #     if reward > 0:
                #         print(f"[DEBUG 远程验证结果]\nBad Format: {kwargs['solution_str']}")
                reward = min(format_reward, reward)

            return acc, reward
        except Exception as e:
            raise RuntimeError(f"Failed to get remote reward score: {e}")
    raise RuntimeError("No remote verify method is available.")
