import re
import signal
from typing import Optional

from alpha_seed.utils.reward_score.math_v2 import last_boxed_only_string_v2, remove_boxed


def compute_score(solution_str, ground_truth, **argv) -> float:
    pred = last_boxed_only_string_v2(solution_str[-100:])
    if pred is None:
        return 0
    pred = remove_boxed(pred)
    if pred.upper() == ground_truth:
        return 1
    return 0


if __name__ == "__main__":

    pass
