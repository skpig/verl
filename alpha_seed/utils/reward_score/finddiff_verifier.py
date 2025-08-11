from alpha_seed.utils.reward_score.extra_reward import filter_thinking_part, extract_answer_failed_reward
import numpy as np
from scipy.optimize import linear_sum_assignment
import re
import copy
import json


def calculate_iou(box1, box2):
    '''计算两个框 box1 和 box2 的 IoU。每一个框的格式是 [x1, y1, x2, y2]。'''
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / (box1_area + box2_area - intersection)
    return iou


def extract_predict_bboxes(pred_str):

    # Find content between begin_of_solution and end_of_solution
    # solution_pattern = r'<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>'
    # solution_match = re.search(solution_pattern, pred_str, re.DOTALL)
    # if not solution_match:
    #     # try to direct get bbox
    #     solution_text = pred_str
    # else:
    #     solution_text = solution_match.group(1)

    bbox_pattern = re.compile(r'<bbox>(.*?)</bbox>')
    bbox_contents = bbox_pattern.findall(pred_str)

    if bbox_contents is None:
        return []

    predict_bboxes = []
    for bboxes in bbox_contents:
        try:
            bboxes = bboxes.split(" ")
            p_x1, p_y1, p_x2, p_y2 = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])

            # valid bbox
            if ((p_x2 > p_x1) and (p_y2 > p_y1)):
                predict_bboxes.append([p_x1, p_y1, p_x2, p_y2])
        except:
            continue

    return predict_bboxes


def get_bbox_metrics(predict_bboxes, gt_ans, iou_rwei=1.0, PN=20):
    # judge whether output following tlx1 tly1 brx1 bry1
    gts = copy.deepcopy(gt_ans)

    N = len(predict_bboxes)
    M = len(gts)
    # Calculate centers of boxes
    pred_centers = np.array([[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0] for box in predict_bboxes])
    gt_centers = np.array([[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0] for box in gts])
    # Calculate pairwise distances between all centers
    dist_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_matrix[i, j] = np.sqrt(np.sum((pred_centers[i] - gt_centers[j])**2))

    # Use Hungarian algorithm to find optimal matching
    pred_indices, gt_indices = linear_sum_assignment(dist_matrix)

    # Calculate total distance for matched boxes

    total_reward = 0
    MAX_DIST = np.sqrt(1000**2 + 1000**2)
    reward_l = []
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if pred_idx < N and gt_idx < M:
            dist = dist_matrix[pred_idx, gt_idx]
            dist_reward = 1.0 - dist / MAX_DIST
            iou_reward = calculate_iou(predict_bboxes[pred_idx], gt_ans[gt_idx])
            reward_l.append({
                'match_predbox': predict_bboxes[pred_idx],
                'match_gtbox': gt_ans[gt_idx],
                'dist': dist,
                'dist_reward': dist_reward,
                "iou": iou_reward
            })

            # p_reward = (dist_reward + iou_rwei*iou_reward)/(1+iou_rwei)
            p_reward = iou_reward
            total_reward = total_reward + 1.0 / M * p_reward

    # do penaly
    penalty_ = 1.0 / (PN * M)
    penalty_reward = 0
    if N > M:
        penalty_reward = (N - M) * penalty_

    final_reward = max(0, total_reward - penalty_reward)

    reward_info = {
        'matched_rinfo': json.dumps(reward_l),
        'total_pred': len(predict_bboxes),
        'total_gts': len(gt_ans),
        'penalty_reward': penalty_reward,
        'penalty_': penalty_
    }

    return final_reward, reward_info


def compute_score(solution_str, ground_truth, **kwargs):
    solution_str, success = filter_thinking_part(solution_str, kwargs['config'])
    if solution_str == "":
        # ExtractAnswerFailed
        return extract_answer_failed_reward()
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    answer = ground_truth['answer']
    if isinstance(answer, str):
        answers = json.loads(answer)
    else:
        answers = answer

    pred_boxes = extract_predict_bboxes(solution_str)
    if len(pred_boxes) < 1:
        # ExtractAnswerFailed
        return extract_answer_failed_reward()

    score, sinfo = get_bbox_metrics(pred_boxes, answers)
    pred_sinfo = json.dumps(sinfo)

    return score
