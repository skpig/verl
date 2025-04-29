from alpha_seed.utils.reward_score.verifier.parser import extract_answer, strip_string
from alpha_seed.utils.reward_score.verifier.grader import math_equal
from alpha_seed.utils.reward_score.extra_reward import filter_thinking_part
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut
import logging
import json

logger = logging.getLogger(__file__)


def compute_score(solution_str, ground_truth, **kwargs) -> float:
    tokenizer = kwargs['tokenizer']
    try:
        score, tag = submit_verifier(solution_str, ground_truth, tokenizer.eos_token, tokenizer.bos_token)
    except FunctionTimedOut:
        logger.info(f"timeout when compute score for {solution_str} and {ground_truth}")
        score = -2
    return score


@func_set_timeout(30)
def submit_verifier(decoded_text, verifier_feature, eos_token, bos_token):
    tag = 'none'
    if not decoded_text:
        return -0.1, tag

    if verifier_feature == "":
        raise ValueError(f"must provide verifier feature")

    ## check format if invalid, return 0
    extracted_response, success = filter_thinking_part(decoded_text)
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
        score = float(strip_string(answer).lower() == pred.lower())
        status = json.dumps({'tag': tag, 'pred': pred, 'answer': answer, 'score': score})
        logger.info(f"[VERIFIER LOG] {status}")
        return score, status
    except Exception as e:
        status = json.dumps({'tag': f"verification error: {e}", 'pred': pred, 'answer': answer, 'score': -1.0})
        logger.info(f"[VERIFIER LOG] {status}")
        return -0.1, status
