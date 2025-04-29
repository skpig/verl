import json

from alpha_seed.utils.reward_score.extra_reward import filter_thinking_part, extract_answer_failed_reward
from shapely import Polygon, Point
import re


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
    point_pattern = re.compile(r'<point>(.*?)</point>')
    point_contents = point_pattern.findall(solution_str)
    predict_points = []
    for points in point_contents:
        try:
            predict_points.append([int(points.split(' ')[0]), int(points.split(' ')[1])])
        except:
            continue
    # predict_points.append([int(point_contents[0].split(' '))[0], int(point_contents[0].split(' '))[1]])
    gt_polygons = [Polygon(answer) for answer in answers]
    correct_predictions = 0
    use_polygons = set()
    for predict_point in predict_points:
        x, y = predict_point[0], predict_point[1]
        x, y = max(min(int(x), 1000), 0), max(min(int(y), 1000), 0)
        predict_point = [x, y]
        point = Point(predict_point)
        for idx, polygon in enumerate(gt_polygons):
            # 判断点是否在多边形内，并且该多边形没被命中过
            if point.within(polygon) and idx not in use_polygons:
                correct_predictions += 1
                use_polygons.add(idx)
                break
    total_predictions = len(predict_points)
    total_gts = len(gt_polygons)
    if total_gts == 0 and total_predictions == 0:
        return 1
    precision = correct_predictions / total_predictions if total_predictions > 0 else 0
    recall = correct_predictions / total_gts if total_gts > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1
