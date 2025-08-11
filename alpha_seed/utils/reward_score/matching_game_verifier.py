import json

from alpha_seed.utils.reward_score.extra_reward import filter_thinking_part, extract_answer_failed_reward
import os
import logging
import re


def transfer_points_to_xy(response, row, col):
    """
    Convert relative coordinates to the coordinates of small blocks in the image.
    """
    x_sep = 1000 // col
    y_sep = 1000 // row
    result = []
    for group in response:
        result.append([[y // y_sep, x // x_sep] for x, y in group])
    return result


def sort_segment(segment):
    """
    Determine whether a line segment follows the rules of the Linking Game (must be horizontal or vertical).
    If it meets the criteria, return the result sorted in natural order and with duplicates removed; otherwise, return None.
    """
    if not segment:
        return None
    segment = list(set(tuple(coord) for coord in segment))

    if all(coord[0] == segment[0][0] for coord in segment):
        return sorted(segment, key=lambda coord: coord[1])
    elif all(coord[1] == segment[0][1] for coord in segment):
        return sorted(segment, key=lambda coord: coord[0])
    else:
        return None


def is_contiguous_subsequence(subseq, seq):
    """
    Check whether subseq is a contiguous subsequence of seq (order must be preserved).
    """
    n = len(subseq)
    for i in range(len(seq) - n + 1):
        if seq[i:i + n] == subseq:
            return True
    return False


def calculate_points_score(response, response_xy, row, col):
    x_sep = 1000 // col
    y_sep = 1000 // row
    total_reward = 0.0
    count = 0

    for group_idx, group in enumerate(response):
        for point_idx, point in enumerate(group):
            cell_y, cell_x = response_xy[group_idx][point_idx]
            center_y = cell_y * y_sep + y_sep / 2
            center_x = cell_x * x_sep + x_sep / 2
            point_transformed = [point[1], point[0]]

            d = ((point_transformed[0] - center_y)**2 + (point_transformed[1] - center_x)**2)**0.5
            max_d = (((x_sep)**2 + (y_sep)**2)**0.5) / 2
            threshold = max_d / 2

            if d <= threshold:
                reward = 1 - 0.3 * (d / threshold)
            elif d <= max_d:
                reward = 0.7 * (max_d - d) / (max_d - threshold)
            else:
                reward = 0.0
            total_reward += reward
            count += 1

    mean_reward = total_reward / count if count else 0.0
    return mean_reward


def calculate_matching_score(label_segments, predicted_segments):
    total_score = 0.0
    processed_pred = []
    for seg in predicted_segments:
        sorted_pred = sort_segment(seg)
        if sorted_pred is not None and len(sorted_pred) >= 2:
            processed_pred.append(sorted_pred)
    used_preds = set()

    for seg in label_segments:
        sorted_seg = sort_segment(seg)
        best_match_score = 0.0
        best_match_pred = None
        for pred in processed_pred:
            pred_tuple = tuple(pred)
            if pred_tuple in used_preds:
                continue

            if sorted_seg == pred:
                best_match_score = 1.0
                best_match_pred = pred_tuple
                break
            elif is_contiguous_subsequence(pred, sorted_seg):
                score = len(pred) / len(sorted_seg)
                if score > best_match_score:
                    best_match_score = score
                    best_match_pred = pred_tuple

        if best_match_pred:
            used_preds.add(best_match_pred)

        total_score += best_match_score
    return total_score / max(len(label_segments), len(predicted_segments))


def compute_score(solution_str, ground_truth, **kwargs):
    solution_str, success = filter_thinking_part(solution_str, kwargs['config'])
    if solution_str == "":
        # ExtractAnswerFailed
        return extract_answer_failed_reward()
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    groups = [group.strip() for group in solution_str.split(';') if group.strip()]
    predictions = []
    for group in groups:
        points = re.findall(r'<point>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*</point>', group)
        points = [[float(x), float(y)] for x, y in points]
        if len(points) > 0:
            predictions.append(points)

    if len(predictions) == 0:
        # ExtractAnswerFailed
        return extract_answer_failed_reward()

    response_xy = transfer_points_to_xy(predictions, ground_truth['grid_shape'][0], ground_truth['grid_shape'][1])
    label_xy = [group["positions"] for group in ground_truth["label"]]

    points_score = calculate_points_score(predictions, response_xy, ground_truth['grid_shape'][0],
                                          ground_truth['grid_shape'][1])
    matching_score = calculate_matching_score(label_xy, response_xy)
    score = points_score * 0.2 + matching_score * 0.8

    return score
