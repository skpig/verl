import re
from collections import Counter

from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name


def find_single_turn_duplicate(response_text, enable_resp_para=False):
    # 输入单条 response，返回：是否重复，是否极端重复，标记出重复片段的 response
    resp_para_list = [
        para.strip()
        for para in response_text.split("\n")
        if para.strip() != "" and 'align' not in para and re.match(r"[\w*\d*]", para) is not None
    ]
    # 极端重复情况
    resp_para_counter = Counter([para for para in resp_para_list])
    # 段落重复出现超过 10 次
    if len(resp_para_counter) and resp_para_counter.most_common(1)[0][1] > 10:
        lcs_str = resp_para_counter.most_common(1)[0][0]
        return True, lcs_str

    if enable_resp_para:
        # 检测句子级重复
        resp_para_list = [
            para.strip()
            for para in re.split(r'[.,。，！!；;？?]', response_text)
            if para.strip() != "" and 'align' not in para and re.match(r"[\w*\d*]", para) is not None
        ]

        # TODO: Minimize the impact by only penalizing words in a blacklist.
        # blacklist = ['wait', '不对', '...']
        # resp_para_list = [x for x in resp_para_list if x in blacklist]

        # 极端重复情况
        resp_para_counter = Counter([para for para in resp_para_list])

        # 句子重复出现超过 15 次
        if len(resp_para_counter) and resp_para_counter.most_common(1)[0][1] > 15:
            lcs_str = resp_para_counter.most_common(1)[0][0]
            return True, lcs_str
    return False, response_text


def is_final_answer_lengthy(response_ids: list[int], tokenizer, max_ans_tokens: int = 4000) -> bool:
    num_total_tokens = len(response_ids)

    bos_token, = tokenizer.encode('<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>')
    k = len(response_ids) - 1
    while (k >= 0) and (response_ids[k] != bos_token):
        k -= 1
    if k >= 0:  # If there is BOS, indicating the start of the final turn:
        response_ids = response_ids[k + 1:]  # Only check the last turn when doing multi-turn RL.

    think_end = get_special_tokens_dict_or_name("think_end_token")
    end_of_think_token, = tokenizer.encode(think_end)
    k = len(response_ids) - 1
    while (k >= 0) and (response_ids[k] != end_of_think_token):
        k -= 1
    if k >= 0:  # If there is think_end_token, indicating the end of the CoT:
        response_ids = response_ids[k + 1:]  # Remove the CoT part.

    num_answer_tokens = len(response_ids)
    is_lengthy = num_answer_tokens > max_ans_tokens
    if is_lengthy:
        print(f"[LENGTHY ANSWER DETECTED] {num_total_tokens} total tokens, {num_answer_tokens} answer tokens.")
    return is_lengthy
