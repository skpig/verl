import collections
import re
import warnings

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from alpha_seed.utils.reward_score.remote_verify import OpenAIConfig, make_multiple_requests
from alpha_seed.utils.reward_score.websearch_verifier import judge_template_mapping, process_single_request_result, \
    validate_format_single_turn
try:
    from groot.state import BaseState
except:
    BaseState = object
from verl import DataProto


def validate_format_state(state: BaseState, tokenizer, ans_begin_str: str = "<ans>", ans_end_str: str = "</ans>"):
    for turn in state.history:
        if turn["role"] in ["user", "system", "tool"]:
            continue
        format_reward = validate_format_single_turn(turn["content"] + tokenizer.eos_token, tokenizer, ans_begin_str,
                                                    ans_end_str)
        if format_reward is False:
            return 0.
    return 1.


def get_last_ans(text: str):
    if '<ans>' in text and '</ans>' in text:
        matches = re.findall(r'<ans>(.*?)</ans>', text, re.DOTALL)
        return matches[-1] if matches else None
    elif '<answer>' in text and '</answer>' in text:
        matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return matches[-1] if matches else None
    else:
        return None


async def judge_answer(plugin_config: DictConfig, item: DataProto, state: BaseState, tokenizer: PreTrainedTokenizer,
                       is_training: bool):
    remote_verify_config = plugin_config.remote_verify_config
    judge_prompt_ver = getattr(remote_verify_config, "judge_prompt_ver", "v1")
    enable_think_format = getattr(remote_verify_config, "enable_think_format", False)
    enable_special_judge = getattr(remote_verify_config, "enable_special_judge", False)
    ans_begin_str = getattr(remote_verify_config, "ans_begin_str", "<ans>")
    ans_end_str = getattr(remote_verify_config, "ans_end_str", "</ans>")

    question = item.non_tensor_batch['reward_model'][0].get('question', None)
    ground_truth = item.non_tensor_batch['reward_model'][0]['ground_truth']
    judge_template, judge_postprocess = judge_template_mapping.get(judge_prompt_ver, "v1")

    if state.history[0]["role"] == "system":
        turn_cnt = len(state.history) - 2
    elif state.history[0]["role"] == "user":
        turn_cnt = len(state.history) - 1
    else:
        raise ValueError(state.history[0]["role"])

    if state.history[-1]["role"] != "assistant":
        print(f"[DEBUG] 最后一轮非 [assistant] - Got {state.history[-1]['role']}")
        return 0, 0

    model_answer = get_last_ans(state.history[-1]["content"])
    if model_answer is None:
        print(f"[DEBUG] 最后一轮回答没有找到答案 - Got {state.history[-1]['content']}")
        return 0, 0

    base_openai_config = OpenAIConfig(
        model=getattr(remote_verify_config, "model", "Qwen2.5-32B-Instruct"),
        max_concurrency=getattr(remote_verify_config, "max_concurrency", 16),
        base_url=getattr(remote_verify_config, "base_url", None),
        timeout=getattr(remote_verify_config, "timeout", 300),
        api_key=getattr(remote_verify_config, "api_key", None),
    )

    max_try = getattr(remote_verify_config, "max_try", 1)
    step_penalty_coef = getattr(plugin_config, "step_penalty_coef", 0.)

    if len(state.history) <= 2:
        warnings.warn(f"[DEBUG]回答只有0轮（没有检测到eos token），需要留意有没有地方出错了，检查。response为：\n{state.history[-1]['content']}")
    judge_prompt = judge_template.format(question=question, gold_answer=ground_truth, answer=model_answer)
    messages_list = [{"role": "user", "content": judge_prompt}]

    results = await make_multiple_requests(base_openai_config, max_try, messages_list)

    votes = collections.Counter()
    for result in results:
        real_result = process_single_request_result(result, judge_postprocess, question, ground_truth, model_answer)
        votes[real_result] += 1
    real_result = votes.most_common(1)[0][0]

    reward = max(int(real_result == 'CORRECT') - step_penalty_coef * turn_cnt, 0.)
    acc = real_result == 'CORRECT'
    print(f"[DEBUG 远程验证结果]\n问题：{question}\n正确答案：{ground_truth}\n回答：{model_answer}\n"
          f"判定结果：{real_result}\n奖励：{reward}\n")

    if enable_think_format and is_training:
        format_reward = validate_format_state(state, tokenizer, ans_begin_str, ans_end_str)
        if getattr(remote_verify_config, "think_format_coef", 0) > 0:
            reward = format_reward * remote_verify_config.think_format_coef + (
                1 - remote_verify_config.think_format_coef) * reward
        else:
            reward = min(format_reward, reward)

    return acc, reward
