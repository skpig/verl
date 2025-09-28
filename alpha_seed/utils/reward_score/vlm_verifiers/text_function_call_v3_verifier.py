import re
import json

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult


def check_function_call_tags(text):
    """
    检查文本中 <|FunctionCallBegin|>…<|FunctionCallEnd|> 标签与 JSON：
        1. 若全文没有任何这两个标签，返回0。
        2. 若存在标签，则它们必须成对、无交叠，且标签之间的内容必须是合法 JSON，才返回 1；否则返回0。
    """
    # 找出所有 <|FunctionCallBegin|> 或 <|FunctionCallEnd|> 及其索引
    tags = [(m.group(), m.start(), m.end()) for m in re.finditer(r"<\|FunctionCall(?:Begin|End)\|>", text)]

    # 如果全文没有标签，则再检查是否含有任何可解析的 JSON
    if not tags:
        return 0
        #stripped = text.strip()
        ## 尝试把全文当作 JSON 解析
        #try:
        #    json.loads(stripped)
        #    return 0
        #except:
        #    pass
        ## 若全文不是纯 JSON，再尝试找第一个 {...} 或 [...] 或 (...)，如果存在则返回0分
        ## TODO: 优化上述规则，目前的规则过于严格
        #m = re.search(r"(\{.*?\}|\[.*?\]|\(.*?\))", text)
        #if m:
        #    return 0
        #    #try:
        #    #    json.loads(m.group(1))
        #    #    return 0
        #    #except:
        #    #    pass
        ## 当且仅当返回的是纯文本的时候，才考虑给分
        #return 1

    stack = []
    pairs = []  # 存储 (begin_start, begin_end, end_start, end_end)

    # 遍历所有标签按顺序配对
    for tag, s, e in tags:
        if "Begin" in tag:
            stack.append((s, e))
        else:  # End
            if not stack:
                return 0
            b_s, b_e = stack.pop()
            pairs.append((b_s, b_e, s, e))

    # 还有多余的 Begin
    if stack:
        return 0

    # 检查这些配对在原文中是否有交叠：按 begin_start 排序后，相邻的 current.begin_start >= prev.end_end
    pairs.sort(key=lambda x: x[0])
    for i in range(1, len(pairs)):
        prev_end = pairs[i - 1][3]
        curr_begin = pairs[i][0]
        if curr_begin < prev_end:
            return 0

    # 检查每对标签之间的内容是否为合法 JSON
    for b_s, b_e, e_s, e_e in pairs:
        fragment = text[b_e:e_s].strip()
        if not fragment:
            return 0
        try:
            parsed_calls = json.loads(fragment)
            if not isinstance(parsed_calls, list) or len(parsed_calls) == 0:
                return 0
            for call in parsed_calls:
                if not isinstance(call, dict):
                    return 0
                if "name" not in call:
                    return 0
                # 必须有parameters字段（不能是arguments）
                if "parameters" not in call:
                    return 0
                # 如果有arguments字段说明格式错误
                if "arguments" in call:
                    return 0
                # # parameters必须是字典
                # if not isinstance(call["parameters"], dict):
                #     return 0
        except:
            return 0
    return 1


class FunctionCallv3Verifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        score = check_function_call_tags(response)
        return VerifyResult(score=score, extracted_answer=response)
