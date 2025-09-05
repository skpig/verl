import math
import re
import sys

import numpy as np

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, ExtractAnswerFailed
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import VerifierFailed
from alpha_seed.utils.reward_score.vlm_verifiers.utils import check_language_correctness


def parse_draw_point(points_str, figsize):
    points = points_str.replace('<point>', ' ').replace('</point>', ' ').split()
    w, h = figsize
    point_list = [float(p) for p in points]
    if len(point_list) % 2 == 1:
        return None
    abs_points = []
    for i in range(0, len(point_list), 2):
        abs_points.append([point_list[i] / 1000. * w, point_list[i + 1] / 1000. * h])
    return abs_points


def get_abs_line(line, fig_size):
    w, h = fig_size
    return [[line[0][0] / 1000. * w, line[0][1] / 1000. * h], [line[1][0] / 1000. * w, line[1][1] / 1000. * h]]


def calculate_distance(point1, point2):
    """
    计算二维平面上两点之间的欧几里得距离
    
    参数:
    point1: 第一个点的坐标，格式为 (x1, y1)
    point2: 第二个点的坐标，格式为 (x2, y2)
    
    返回:
    float: 两点之间的距离
    """
    # 提取坐标
    x1, y1 = point1
    x2, y2 = point2

    # 计算距离
    dx = (x2 - x1)
    dy = (y2 - y1)
    distance = math.sqrt(dx**2 + dy**2)

    return distance


def calculate_score_by_dist(input_value, min_value, max_value, min_score, max_score):
    """
    计算从min_value到max_value之间的加速递减值
    
    参数:
    input_value (float): 当前输入值
    min_value (float): 最小值，对应1.0
    max_value (float): 最大值，对应0.1
    
    返回:
    float: 计算结果，范围在0.1到1.0之间，且越靠近min_value值越大且减少得越快
    """
    # 边界检查
    if input_value <= min_value:
        return max_score
    if input_value >= max_value:
        return min_score
    # 归一化输入值到[0,1]区间，注意这里是反向的（min_value对应1.0，max_value对应0.0）
    normalized = 1.0 - (input_value - min_value) / (max_value - min_value)

    # 使用指数函数实现加速递减效果
    accelerated = normalized**5

    # 调整范围从[0,1]到[0.1,1.0]
    return min_score + accelerated * (max_score - min_score)


class AuxlinePointVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        query = verifier_feature_dict['query']
        groundtruth_points = verifier_feature_dict['gt_lines']  # groundtruth points[[x1, y1], [x2, y2]]格式
        fig_size = verifier_feature_dict['fig_size']  # 图片的长宽，用于转换成绝对位置坐标，[w, h]格式
        w, h = fig_size

        check_language_correctness(query, response, verifier_feature_dict)

        SCORE_INVALID_FORMAT = 0  # 格式错误、没有打点、打点数不对
        SCORE_VALID_FORMAT = 0.1  # 格式正确，打点数对，但距离太远
        SCORE_RIGHT = 1.0  # 格式正确，打点正确

        try:
            point_pattern = re.compile(r'<point>(.*?)</point>')
            if response == "":
                raise ExtractAnswerFailed
            point_contents = point_pattern.findall(response)
            draw_points = []
            for points in point_contents:
                try:
                    draw_points.append([int(points.split(' ')[0]) / 1000. * w, int(points.split(' ')[1]) / 1000. * h])
                except:
                    continue

            if len(draw_points) != len(groundtruth_points):
                return VerifyResult(score=SCORE_INVALID_FORMAT, extracted_answer=response)

            # 计算点距离
            valid_point_cnt = 0
            w, h = fig_size
            dist_thres = min(math.sqrt(w * h) * 0.02, 15)
            score_list = []
            for p in draw_points:
                min_dist = 10000
                for gt_p in groundtruth_points:
                    min_dist = min(min_dist, calculate_distance(p, gt_p))
                score_list.append(
                    calculate_score_by_dist(min_dist, dist_thres, dist_thres * 5, SCORE_VALID_FORMAT, SCORE_RIGHT))
            return VerifyResult(score=float(np.mean(score_list)), extracted_answer=response)
        except Exception:
            import traceback
            tb = "".join(traceback.format_exception(*sys.exc_info()))  # noqa
            raise VerifierFailed(message=tb)


def _test():
    verifier = AuxlinePointVerifier()
    response = """- 点 \( A \) 的坐标：<point>401 254</point> - 点 \( B \) 的坐标：<point>54 356</point> - 点 \( D \) 的坐标：<point>282 345</point>"""
    verifier_feature_dict = {
        "problem": "请用POINT工具在图中标出点A、点D、点B，请忽略图片中的题目，只需标记出点即可。",
        "reference_answer": "",
        "verify_type": 4,
        "answer": "",
        "verifier_name": "point_rule_verifier",
        "gt_lines": [[258.6868402219287, 564.4287471846038], [202.5319382611444, 763.1307079689175],
                     [29.7476245356543, 819.2856099297018]],
        "query": "请用POINT工具在图中标出点A、点D、点B，请忽略图片中的题目，只需标记出点即可。",
        "fig_size": [670, 2203]
    }
    print(verifier.verify(response, verifier_feature_dict))


if __name__ == '__main__':
    _test()
