import base64
import io
import re
from typing import List

import numpy as np

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult


class MazeVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        feature = verifier_feature_dict
        assert 'starting_point' in feature, feature
        assert 'ending_point' in feature, feature
        pred = extract_maze_answer(response)
        if not pred:
            raise ExtractAnswerFailed
        distances_compressed = feature['distances']
        score = verifier_maze(np.load(io.BytesIO(base64.b64decode(distances_compressed)))['distances'],
                              start_point=feature['starting_point'],
                              end_point=feature['ending_point'],
                              path=pred)
        return VerifyResult(score=score, extracted_answer=str(pred))


def l2_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def extract_maze_answer(pred_str):
    line_pattern = re.compile(r'<point>(.*?)</point>', re.DOTALL)
    line_contents = line_pattern.findall(pred_str)
    try:
        line_contents = [(int(_.split(' ')[0]), int(_.split(' ')[1])) for _ in line_contents]
    except:
        return []
    return line_contents


def cross_wall(distances, p1, p2):
    step = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
    for j in range(step):
        t = j / step
        x = int(p1[0] * (1 - t) + p2[0] * t)
        y = int(p1[1] * (1 - t) + p2[1] * t)
        if distances[x, y] == 0xFFFFFF:
            return True
    return False


def verifier_maze(distances, start_point, end_point, path: List[List[int]], threshold=9.0):
    assert len(start_point) == 2
    rows, cols = distances.shape
    # Convert path (ranging from 0-999, (col, row) format), to coordinates system in distances (row, col) format.
    converted_path = [(int(p[1] * rows / 1000.0), int(p[0] * cols / 1000.0)) for p in path]
    # converted_path = path

    # 检查路径是否从起始点附近开始
    if distances[converted_path[0][0], converted_path[0][1]] == 0xFFFFFF:
        return 0.0
    if l2_distance(converted_path[0], start_point) > threshold:
        return 0.1 * (threshold / l2_distance(converted_path[0], start_point))

    # 遍历path中所有的相邻点对，对于连线上每一个点进行遍历扫描，如果处于distances[i][j] == 0xFFFFFF (表示无穷大的点)，则退出循环
    last_point = None
    penalty_factor = 1.0
    best_relative_distance = 1.0
    straight_steps = 0
    bonus_points = 0
    for i in range(len(converted_path) - 1):
        p1 = converted_path[i]
        p2 = converted_path[i + 1]

        if cross_wall(distances, p1, p2):
            penalty_factor = 0.8
            break
        else:
            last_point = p2
        if i >= 1 and not cross_wall(distances, converted_path[i - 1], p2):
            straight_steps += 1
        if i >= 1:
            is_bonus = True
            for j in range(0, i):
                if not cross_wall(distances, converted_path[j], p2):
                    is_bonus = False
                    break
            if is_bonus:
                bonus_points += 1
        best_relative_distance = min(best_relative_distance, distances[p2[0], p2[1]] / distances[tuple(start_point)])
    if last_point is None:
        return 0.1
    # print(last_point, distances[last_point], end_point, distances[end_point])
    r2 = max(0, 1 - best_relative_distance - 0.02 * straight_steps)
    r3 = max(0, 1 - distances[last_point] / distances[tuple(start_point)] - 0.02 * straight_steps)
    r4 = min(1, 0.05 * bonus_points)
    ret = 0.1 + max(0.4 * r4 + 0.2 * r2 + 0.3 * r3, 0.4 * r2 + 0.5 * r3)
    if ret > 0.9:
        ret *= penalty_factor
    assert 0.0 <= ret <= 1.0001, ret
    return ret
