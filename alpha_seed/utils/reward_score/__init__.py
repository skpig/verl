"""
This sub-package should contain all the rule-based reward score
"""
import os
from functools import partial

from alpha_seed.workers.agents import load_external_module


def _select_rm_score_fn(reward_style, external_lib: str = None):

    external_module = load_external_module(package_name=reward_style,
                                           external_lib=external_lib,
                                           external_path=os.environ.get('EXTERNAL_REWARD_FN_PATH', None))
    if external_module is not None:
        compute_score_fn = getattr(external_module, 'compute_score')
        assert hasattr(compute_score_fn, '__call__'), f"{reward_style}.compute_score is not a function"
        return compute_score_fn

    if reward_style == "model-raw_score":
        from . import model_score_fn
        return model_score_fn.raw_score
    elif reward_style == "model-raw_score_reflection_penalty":
        from . import model_score_fn
        return model_score_fn.raw_score_reflection_penalty
    elif reward_style == "code-sandbox":
        from . import oj_utils
        return oj_utils.compute_score_client
    elif reward_style == "code-localexec":
        from . import code_local_verifier
        return code_local_verifier.compute_score
    elif reward_style == 'rule-openai/gsm8k':
        from . import gsm8k
        return gsm8k.compute_score
    elif reward_style in ('rule-lighteval/MATH', 'lighteval/MATH'):
        from . import math_v1
        return math_v1.compute_score
    elif reward_style == 'rule-lighteval/MATH_v2':
        from . import math_v2
        return math_v2.compute_score
    elif reward_style == 'rule/deepscale':
        from . import math_deepscale
        return math_deepscale.deepscaler_reward_fn
    elif reward_style == "rule-math_verifier":
        from . import math_verifier
        return math_verifier.compute_score
    elif reward_style == "rule-boxed_gpqa":
        from . import gpqa_verifier
        return gpqa_verifier.compute_score
    elif reward_style == "verifier_service":
        from . import verifier_service
        return verifier_service.compute_score_client
    elif reward_style == "verifier_math":
        from . import math_verifier_v2
        return math_verifier_v2.compute_score
    elif reward_style == "verifier_gui":
        from . import gui_verifier
        return gui_verifier.compute_score
    elif reward_style == 'verifier_boxed_str':
        from . import string_verifier  # import BoxStrVerifier
        return string_verifier.compute_score
    elif reward_style == 'verifier_penalize_fc_times':
        from . import fncall_verifier  # import FnCallPenalize
        return fncall_verifier.compute_score
    elif reward_style == "verifier_count":
        from . import count_verifier
        return partial(count_verifier.compute_score, delta=0.8)
    elif reward_style == "verifier_pointing":
        from . import point_verifier
        return point_verifier.compute_score
    elif reward_style == "verifier_bbox":
        from . import bbox_verifier
        return bbox_verifier.compute_score
    elif reward_style == "verifier_countbypoint":
        from . import cotcount_verifier
        return partial(cotcount_verifier.compute_score, delta=0.6)
    elif reward_style == 'verifier_plain_str':
        from . import plain_str_verifier
        return plain_str_verifier.compute_score
    elif reward_style == "verifier_findiff":
        from . import finddiff_verifier
        return finddiff_verifier.compute_score
    elif reward_style == "verifier_maze":
        from . import maze_verifier
        return maze_verifier.compute_score
    elif reward_style == "verifier_matching_game":
        from . import matching_game_verifier
        return matching_game_verifier.compute_score
    elif reward_style.startswith("rule-logic_puzzle"):
        from . import logic_puzzle
        return logic_puzzle.compute_score
    else:
        raise NotImplementedError
