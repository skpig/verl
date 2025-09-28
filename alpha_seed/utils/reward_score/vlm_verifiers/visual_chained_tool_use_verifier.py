import json
import math
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name
from alpha_seed.utils.reward_score.vlm_verifiers.extra_reward import match_visual_cot_format
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, ExtractAnswerFailed
from alpha_seed.utils.reward_score.vlm_verifiers.utils import get_valid_visual_tool_calls


class VisualChainedToolUseVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        if not match_visual_cot_format(response, verifier_feature=verifier_feature_dict):
            raise ExtractAnswerFailed

        n_remain_vertices: int = verifier_feature_dict['n_remain_vertices']
        gt_tool_calls: list[dict] = verifier_feature_dict['tool_calls']

        score = 0.0

        # Check if the model predicts the correct final answer. Accounts for 20% of the total reward.
        eot_token = get_special_tokens_dict_or_name("think_end_token")
        eos_token = get_special_tokens_dict_or_name("eos")
        final_answer: str = response.split(eot_token)[-1].split(eos_token)[0]
        if (final_answer.count('boxed') == 1) and (f'\\boxed{{{n_remain_vertices}}}' in final_answer):
            score += 0.2

        # Extract the tool calls from the model's response.
        num_input_images = 1
        _, _, valid_tool_calls = get_valid_visual_tool_calls(response, num_input_images=num_input_images)
        assert len(valid_tool_calls) > 0
        valid_tool_calls = valid_tool_calls[num_input_images:]  # remove the dummy one

        # Penalize it if the model predicts more tool calls than the ground truth.
        if len(valid_tool_calls) > len(gt_tool_calls):
            score -= 0.1 * (len(valid_tool_calls) - len(gt_tool_calls))
            valid_tool_calls = valid_tool_calls[:len(gt_tool_calls)]
        else:
            valid_tool_calls += [{}] * (len(gt_tool_calls) - len(valid_tool_calls))
        assert len(valid_tool_calls) == len(gt_tool_calls)

        # The last operation is a challenging one. Use soft reward for it. Accounts for 30% of the total reward.
        if n_remain_vertices > 0:
            pred = valid_tool_calls[-1]
            gold = gt_tool_calls[-1]
            gt_tool_calls = gt_tool_calls[:-1]
            valid_tool_calls = valid_tool_calls[:-1]
            if pred and all([
                    pred['name'] == gold['name'],
                    pred['parameters']['imgidx'] == gold['parameters']['imgidx'],
                    pred['parameters'].get('crop', False) == gold['parameters'].get('crop', False),
                    pred['parameters'].get('draw_line', False) == gold['parameters'].get('draw_line', False),
            ]):
                score += 0.1
                if pred['name'] == 'POINT':
                    score += 0.2 * _pointset_similarity(pred['parameters']['points'], gold['parameters']['points'])
                elif pred['name'] == 'GROUNDING':
                    score += 0.2 * _bbox_similarity(pred['parameters']['bbox_str'], gold['parameters']['bbox_str'])
                else:
                    raise NotImplementedError

        # The first few operations are easy. Use hard reward for them. Accounts for 50% of the total reward.
        for gt_call, pred_call in zip(gt_tool_calls, valid_tool_calls):
            if json.dumps(gt_call, sort_keys=True) == json.dumps(pred_call, sort_keys=True):
                score += 0.5 / len(gt_tool_calls)

        return VerifyResult(score=max(score, 0.0), extracted_answer=str(n_remain_vertices))


def _point_similarity(point_a: str, point_b: str) -> float:
    # Regex to extract integers from <point>x y</point>
    pattern = r"<point>(\d+)\s+(\d+)</point>"

    match_a = re.fullmatch(pattern, point_a.strip())
    match_b = re.fullmatch(pattern, point_b.strip())

    if not match_a or not match_b:
        return 0.0  # Invalid format

    try:
        x1, y1 = int(match_a.group(1)), int(match_a.group(2))
        x2, y2 = int(match_b.group(1)), int(match_b.group(2))
    except ValueError:
        return 0.0

    # Validate bounds
    for val in (x1, y1, x2, y2):
        if not (0 <= val <= 999):
            return 0.0

    # Compute Euclidean distance
    dist = math.hypot(x2 - x1, y2 - y1)

    # Max possible distance in 1000x1000 grid (corner to corner)
    max_dist = math.hypot(999, 999)

    # Similarity: closer distance → higher score
    similarity = 1.0 - (dist / max_dist)

    # Clamp into [0.0, 1.0]
    return max(0.0, min(1.0, similarity))


def _bbox_similarity(bbox_a: str, bbox_b: str) -> float:
    # Regex to extract integers from <bbox>x1 y1 x2 y2</bbox>
    pattern = r"<bbox>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)</bbox>"

    match_a = re.fullmatch(pattern, bbox_a.strip())
    match_b = re.fullmatch(pattern, bbox_b.strip())

    if not match_a or not match_b:
        return 0.0  # Invalid format

    try:
        ax1, ay1, ax2, ay2 = map(int, match_a.groups())
        bx1, by1, bx2, by2 = map(int, match_b.groups())
    except ValueError:
        return 0.0

    # Validate bounds
    for val in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        if not (0 <= val <= 999):
            return 0.0

    # Ensure (x1,y1) is top-left and (x2,y2) is bottom-right
    if ax1 >= ax2 or ay1 >= ay2 or bx1 >= bx2 or by1 >= by2:
        return 0.0

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    # Areas
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0  # Degenerate boxes

    iou = inter_area / union_area

    # Clamp to [0,1]
    return max(0.0, min(1.0, iou))


def _pointset_similarity(set_a: str, set_b: str) -> float:
    pattern = r"<point>\d+\s+\d+</point>"
    points_a = re.findall(pattern, set_a)
    points_b = re.findall(pattern, set_b)

    if len(points_a) != len(points_b):
        return 0.0
    if not points_a:  # both empty
        return 1.0

    n = len(points_a)
    sim_matrix = np.zeros((n, n))

    for i, pa in enumerate(points_a):
        for j, pb in enumerate(points_b):
            sim_matrix[i, j] = _point_similarity(pa, pb)

    # Hungarian algorithm to maximize similarity
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    avg_sim = sim_matrix[row_ind, col_ind].mean()
    return float(avg_sim)
