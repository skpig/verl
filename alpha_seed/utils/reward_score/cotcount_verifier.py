from alpha_seed.utils.reward_score.verifier.tools import extract_and_convert_number
from alpha_seed.utils.reward_score import point_verifier
from alpha_seed.utils.reward_score import bbox_verifier
import json


def compute_score(solution_str, ground_truth, delta, **kwargs):
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    answers = ground_truth['answer']
    if isinstance(answers, str):
        answers = json.loads(answers)
    else:
        answers = answers

    point_answer, count_answer = answers[:-1], answers[-1]
    count_answer = int(count_answer)

    if "<point>" in solution_str:
        ground_truth['answer'] = point_answer
        point_score = point_verifier.compute_score(solution_str, ground_truth, **kwargs)
    elif "<bbox>" in solution_str:
        bbox_answer = []
        for point in point_answer:
            x1, y1 = point[0][0], point[0][1]
            x2, y2 = point[3][0], point[3][1]
            bbox_answer.append([x1, y1, x2, y2])
        ground_truth['answer'] = bbox_answer
        point_score = bbox_verifier.compute_score(solution_str, ground_truth, **kwargs)
    else:
        point_score = 0 if count_answer != 0 else 1

    predict_num, truncated = extract_and_convert_number(solution_str)
    if truncated:
        score = point_score * 0.5
    else:
        if predict_num == count_answer:
            count_score = 1
        else:
            count_score = min(max(1 - abs(count_answer - predict_num) / count_answer, 0), delta)
        score = 0.5 * count_score + 0.5 * point_score

    return float(score)
