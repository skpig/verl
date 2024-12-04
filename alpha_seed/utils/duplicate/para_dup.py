import re
from collections import Counter


def find_single_turn_duplicate(response_text):
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
    return False, response_text
