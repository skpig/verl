import re
from typing import Optional
import json
import logging

from alpha_seed.utils.reward_score.vlm_verifiers.parser import extract_answer
from alpha_seed.utils.reward_score.vlm_verifiers.grader import math_equal
from alpha_seed.utils.reward_score.extra_reward import filter_thinking_part
import ray

logger = logging.getLogger(__file__)


def compute_score(solution_str, ground_truth, **kwargs) -> float:
    think_template = kwargs['config'].data.think_template \
        if hasattr(kwargs['config'].data, 'think_template') else 'v2'
    try:
        ref = submit_verifier.remote(solution_str, ground_truth, think_template=think_template)
        score, tag = ray.get(ref, timeout=30)
    except ray.exceptions.GetTimeoutError:
        # timeout
        score = -2
        logger.warning(f"compute_score_client got timeout error")
        ray.cancel(ref, force=True, recursive=True)
    except Exception as e:
        # other error
        logger.warning(f"compute_score_client got unknown error: {e}")
        score = 0
    return score


@ray.remote(num_cpus=1)
def submit_verifier(decoded_text, verifier_feature, think_template: str = None):
    tag = 'none'
    if not decoded_text:
        return -0.1, tag

    if verifier_feature == "":
        raise ValueError(f"must provide verifier feature")

    ## check format if invalid, return 0
    extracted_response, success = filter_thinking_part(decoded_text, think_template=think_template)
    if not success:
        return 0, tag
    response = extracted_response

    pred = extract_answer(response, is_choice=False, use_last_number=False, use_box_only=True)

    feature = json.loads(verifier_feature)
    assert 'answer' in feature, feature
    answer = feature['answer']

    if pred == "":
        tag = "parsing_fail"
        status = json.dumps({'tag': tag, 'pred': pred, 'answer': answer, 'score': -1.0})
        logger.info(f"[VERIFIER LOG] {status}")
        return -0.1, status

    try:
        tag = "verified"
        score = float(math_equal(pred, answer, timeout=False))
        status = json.dumps({'tag': tag, 'pred': pred, 'answer': answer, 'score': score})
        logger.info(f"[VERIFIER LOG] {status}")
        return score, status
    except Exception as e:
        status = json.dumps({'tag': f"verification error: {e}", 'pred': pred, 'answer': answer, 'score': -1.0})
        logger.info(f"[VERIFIER LOG] {status}")
        return -0.1, status
