from alpha_seed.utils.reward_score.extra_reward import filter_thinking_part, extract_answer_failed_reward
import json


def compute_score(solution_str, ground_truth, **kwargs):
    solution_str, success = filter_thinking_part(solution_str)
    if solution_str == "":
        # ExtractAnswerFailed
        return extract_answer_failed_reward()
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    answer = ground_truth['answer']
    pred = solution_str
    score = float(pred.strip() == answer)
    return score
