import re

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult


def extract_order(response: str) -> str | None:
    """
    Extracts an ordering like "1 2 3 4 5" from strings such as:
      - "Final Answer:(1 2 3 4 5)"
      - "Final Answer: 1   2  3 4    5"
      - "最终答案：(1 2 3 4 5)"
      - "最终答案: 1 2 3 4 5"
    Returns the numbers separated by a single space, or None if no match.
    """
    pattern = r'''
        (?:Final\s*Answer|最终答案)  # match "Final Answer" or Chinese
        \s*[:：]\s*                  # colon (English or Chinese) with optional spaces
        \(?                          # optional opening parenthesis
        \s*                          # optional spaces
        ([\d]+(?:\s+[\d]+)*)         # capture one or more digits separated by spaces
        \s*                          # optional spaces
        \)?                          # optional closing parenthesis
    '''
    match = re.search(pattern, response, flags=re.IGNORECASE | re.VERBOSE)
    if not match:
        return None
    # Normalize whitespace: collapse any run of spaces into a single space
    nums = match.group(1)
    nums = ' '.join(nums.split())
    return nums


def ranking_reward(answer_order, predict_order):
    answer_order = list(map(int, answer_order.split()))
    predict_order = list(map(int, predict_order.split()))
    if len(answer_order) != len(predict_order):
        return 0
    if sorted(answer_order) != sorted(predict_order):
        return 0
    n = len(answer_order)
    # 1. 构建元素 → 正确位置的映射
    pos_map = {val: idx for idx, val in enumerate(answer_order)}
    # 2. 把 predict_order 映射为在 answer_order 中的排名序列
    mapped = [pos_map[val] for val in predict_order]
    # 3. 暴力计算逆序对
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if mapped[i] > mapped[j]:
                inv_count += 1
    # 4. 归一化得分
    max_inv = n * (n - 1) // 2
    score = 1.0 - (inv_count / max_inv) if max_inv > 0 else 1.0
    return score


def ranking_reward_strict(answer_order, predict_order):
    answer_order = list(map(int, answer_order.split()))
    predict_order = list(map(int, predict_order.split()))
    if len(answer_order) != len(predict_order):
        return 0
    if sorted(answer_order) != sorted(predict_order):
        return 0
    n = len(answer_order)
    if answer_order == predict_order:
        return 1
    return 0


class VideoShuffleVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        if response == "":
            raise ExtractAnswerFailed
        predict_order = extract_order(response)
        if predict_order is None:
            raise ExtractAnswerFailed

        score = ranking_reward(answer, predict_order)
        return VerifyResult(score=score, extracted_answer=response)


class VideoShuffleVerifierStrict(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        if response == "":
            raise ExtractAnswerFailed
        predict_order = extract_order(response)
        if predict_order is None:
            raise ExtractAnswerFailed

        score = ranking_reward_strict(answer, predict_order)
        return VerifyResult(score=score, extracted_answer=response)
