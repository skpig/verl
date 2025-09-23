import math
from .utils import Verifier


class FnCallPenalize(Verifier, reward_style="verifier_penalize_fc_times"):

    @staticmethod
    def compute_score(*args, **kwargs) -> float:
        return compute_score(*args, **kwargs)


def compute_score(solution_str, ground_truth, **kwargs) -> float:
    cnt_l = solution_str.count('<|FunctionCallBegin|>')
    cnt_r = solution_str.count('<|FunctionCallEnd|>')
    if cnt_l != cnt_r:
        score = 0.0
    else:
        score = math.exp(-(cnt_l + cnt_r) / 10.0)
    return score
