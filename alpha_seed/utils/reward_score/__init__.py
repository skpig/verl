"""
This sub-package should contain all the rule-based reward score
"""
import os
from functools import partial

from alpha_seed.utils.reward_score import model_score_fn, oj_utils, code_local_verifier, gsm8k, math_v1, math_v2, \
    math_deepscale, math_verifier, gpqa_verifier, verifier_service, logic_puzzle
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
        return model_score_fn.raw_score
    elif reward_style == "model-raw_score_reflection_penalty":
        return model_score_fn.raw_score_reflection_penalty
    elif reward_style == "code-sandbox":
        return oj_utils.compute_score_client
    elif reward_style == "code-localexec":
        return code_local_verifier.compute_score
    elif reward_style == 'rule-openai/gsm8k':
        return gsm8k.compute_score
    elif reward_style == 'rule-lighteval/MATH':
        return math_v1.compute_score
    elif reward_style == 'rule-lighteval/MATH_v2':
        return math_v2.compute_score
    elif reward_style == 'rule/deepscale':
        return math_deepscale.deepscaler_reward_fn
    elif reward_style == "rule-math_verifier":
        return math_verifier.compute_score
    elif reward_style == "rule-boxed_gpqa":
        return gpqa_verifier.compute_score
    elif reward_style == "verifier_service":
        return verifier_service.compute_score_client
    elif reward_style == "verifier_math":
        from alpha_seed.utils.reward_score import math_verifier_v2
        return math_verifier_v2.compute_score
    elif reward_style == "verifier_gui":
        from alpha_seed.utils.reward_score import gui_verifier
        return gui_verifier.compute_score
    elif reward_style == 'verifier_boxed_str':
        from alpha_seed.utils.reward_score import string_verifier  # import BoxStrVerifier
        return string_verifier.compute_score
    elif reward_style == 'verifier_penalize_fc_times':
        from alpha_seed.utils.reward_score import fncall_verifier  # import FnCallPenalize
        return fncall_verifier.compute_score
    elif reward_style == "verifier_count":
        from alpha_seed.utils.reward_score import count_verifier
        return partial(count_verifier.compute_score, delta=0.8)
    elif reward_style == "verifier_pointing":
        from alpha_seed.utils.reward_score import point_verifier
        return point_verifier.compute_score
    elif reward_style == "verifier_bbox":
        from alpha_seed.utils.reward_score import bbox_verifier
        return bbox_verifier.compute_score
    elif reward_style == "verifier_countbypoint":
        from alpha_seed.utils.reward_score import cotcount_verifier
        return partial(cotcount_verifier.compute_score, delta=0.6)
    elif reward_style == 'verifier_plain_str':
        from alpha_seed.utils.reward_score import plain_str_verifier
        return plain_str_verifier.compute_score
    elif reward_style == "verifier_findiff":
        from alpha_seed.utils.reward_score import finddiff_verifier
        return finddiff_verifier.compute_score
    elif reward_style == "verifier_maze":
        from alpha_seed.utils.reward_score import maze_verifier
        return maze_verifier.compute_score
    elif reward_style == "verifier_matching_game":
        from alpha_seed.utils.reward_score import matching_game_verifier
        return matching_game_verifier.compute_score
    elif reward_style.startswith("rule-logic_puzzle"):
        return logic_puzzle.compute_score
    else:
        raise NotImplementedError
