from functools import partial

from alpha_seed.utils.reward_score.extra_reward import filter_thinking_part, extract_answer_failed_reward
from alpha_seed.utils.reward_score.vlm_verifiers.tools import extract_and_convert_number
import json
from .utils import Verifier


class CountVerifier(Verifier, reward_style="verifier_count"):

    @staticmethod
    def compute_score(*args, **kwargs) -> float:
        return partial(compute_score, delta=0.8)(*args, **kwargs)


def compute_score(solution_str, ground_truth, delta, **kwargs):
    solution_str, success = filter_thinking_part(solution_str, kwargs['config'])
    if solution_str == "":
        # ExtractAnswerFailed
        return extract_answer_failed_reward()

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    answer = ground_truth['answer']
    predict_num, truncated = extract_and_convert_number(solution_str)
    count_answer = int(answer)
    if predict_num == count_answer:
        score = 1.0
    else:
        score = min(max(1 - abs(count_answer - predict_num) / count_answer, 0), delta)

    return float(score)
