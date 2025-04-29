from alpha_seed.utils.reward_score.extra_reward import filter_thinking_part, extract_answer_failed_reward
from alpha_seed.utils.reward_score.verifier.tools import calculate_iou
import re
import copy
import json


def compute_score(solution_str, ground_truth, **kwargs):
    solution_str, success = filter_thinking_part(solution_str)
    if solution_str == "":
        # ExtractAnswerFailed
        return extract_answer_failed_reward()

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    answers = ground_truth['answer']
    if isinstance(answers, str):
        answers = json.loads(answers)
    else:
        answers = answers

    bbox_pattern = re.compile(r'<bbox>(.*?)</bbox>')

    bbox_contents = bbox_pattern.findall(solution_str)
    predict_bboxes = []
    iou_threshold = 0.5
    for bboxes in bbox_contents:
        try:
            bboxes = bboxes.split(" ")
            predict_bboxes.append([int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])])
        except:
            continue

    correct_predictions = 0
    for predict_bbox in predict_bboxes:
        x1, y1, x2, y2 = predict_bbox
        if x2 > x1 and y2 > y1:
            x1, y1 = max(min(int(x1), 1000), 0), max(min(int(y1), 1000), 0)
            x2, y2 = max(min(int(x2), 1000), 0), max(min(int(y2), 1000), 0)

        predict_point = [x1, y1, x2, y2]
        gts = copy.deepcopy(answers)
        matched_bbox_idx = []
        for idx, gt_box in enumerate(gts):
            # 判断点是否在矩形内，并且该矩形没被命中过
            if calculate_iou(predict_point, gt_box) >= iou_threshold and idx not in matched_bbox_idx:
                correct_predictions += 1
                matched_bbox_idx.append(idx)
                break
    total_predictions = len(predict_bboxes)
    total_gts = len(answers)
    if total_gts == 0 and total_predictions == 0:
        return 1
    precision = correct_predictions / total_predictions if total_predictions > 0 else 0
    recall = correct_predictions / total_gts if total_gts > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1
