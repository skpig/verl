import re
import signal
from typing import Optional

from alpha_seed.utils.reward_score.math_v2 import last_boxed_only_string_v2, remove_boxed

from .utils import Verifier


class GPQAVerifier(Verifier, reward_style="rule-boxed_gpqa"):

    @staticmethod
    def compute_score(*args, **kwargs) -> float:
        return compute_score(*args, **kwargs)


def compute_score(solution_str, ground_truth, **argv) -> float:
    pred = last_boxed_only_string_v2(solution_str[-100:])
    if pred is None:
        return -1
    pred = remove_boxed(pred)
    if pred.upper() == ground_truth:
        return 1
    return -1


if __name__ == "__main__":

    pass
