import json
import math
import sys

import numpy as np

from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name
from alpha_seed.utils.reward_score.vlm_verifiers.extra_reward import match_visual_cot_format
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, ExtractAnswerFailed
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import VerifierFailed
from alpha_seed.utils.reward_score.vlm_verifiers.utils import check_language_correctness


def parse_point(points_str):
    points = points_str.replace('<point>', ' ').replace('</point>', ' ').split()
    point_list = [float(p) for p in points]
    if len(point_list) < 4 or len(point_list) % 2 == 1:
        return None
    if len(point_list) == 4:
        return [[point_list[0], point_list[1]], [point_list[2], point_list[3]]]
    else:
        # 多个点连线，需要保证多个点连线的角度不能差太多，这样可以看成是一条线
        for i in range(0, len(point_list), 2):
            if i + 5 >= len(point_list):
                break
            line1 = [[point_list[i], point_list[i + 1]], [point_list[i + 2], point_list[i + 3]]]
            line2 = [[point_list[i + 2], point_list[i + 3]], [point_list[i + 4], point_list[i + 5]]]
            angle = calculate_angle(line1, line2)
            if angle >= 5 and angle <= 175:
                return None
        points = []
        for i in range(0, len(point_list), 2):
            points.append([point_list[i], point_list[i + 1]])
        return points


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


def calculate_angle(line1, line2):
    """
    计算两条线段之间的夹角（单位：度）
    
    参数:
    line1: 第一条线段的端点坐标，格式为 [(x1, y1), (x2, y2)]
    line2: 第二条线段的端点坐标，格式为 [(x3, y3), (x4, y4)]
    
    返回:
    float: 两条线段之间的夹角，范围从0到180度
    """
    # 提取线段端点
    (x1, y1), (x2, y2) = line1[0], line1[-1]
    (x3, y3), (x4, y4) = line2[0], line2[-1]

    # 计算线段的向量
    vector1_x = x2 - x1
    vector1_y = y2 - y1
    vector2_x = x4 - x3
    vector2_y = y4 - y3

    # 计算向量点积
    dot_product = vector1_x * vector2_x + vector1_y * vector2_y

    # 计算向量的模
    magnitude1 = math.sqrt(vector1_x**2 + vector1_y**2)
    magnitude2 = math.sqrt(vector2_x**2 + vector2_y**2)

    # 处理线段长度为0的情况
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # 计算夹角的余弦值
    cos_theta = dot_product / (magnitude1 * magnitude2)

    # 确保余弦值在有效范围内（由于浮点数误差，可能会出现微小偏差）
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    # 计算夹角（弧度）
    theta_rad = math.acos(cos_theta)

    # 转换为角度
    theta_deg = math.degrees(theta_rad)

    return theta_deg


def merge_same_lines(gt_lines):
    dedup_gt_lines = []
    dedup_gt_lines_set = set()
    for line in gt_lines:
        key = str(line)
        if key in dedup_gt_lines_set:
            continue
        dedup_gt_lines_set.add(key)
        dedup_gt_lines.append(line)
    return dedup_gt_lines


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
    print('!!', input_value)
    # 归一化输入值到[0,1]区间，注意这里是反向的（min_value对应1.0，max_value对应0.0）
    normalized = 1.0 - (input_value - min_value) / (max_value - min_value)

    # 使用指数函数实现加速递减效果
    # 底数选择2，你可以调整这个值来改变递减速率
    accelerated = normalized**3

    # 调整范围从[0,1]到[0.1,1.0]
    return min_score + accelerated * (max_score - min_score)


class AuxlineVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        # 检查FC格式错误
        if not match_visual_cot_format(response, verifier_feature=verifier_feature_dict):
            raise ExtractAnswerFailed

        query = verifier_feature_dict['query']
        groundtruth_lines = merge_same_lines(
            verifier_feature_dict['gt_lines'])  # groundtruth lines，绝对位置坐标，[[x1, y1], [x2, y2]]格式
        fig_size = verifier_feature_dict['fig_size']  # 图片的长宽，用于转换成绝对位置坐标，[w, h]格式

        check_language_correctness(query, response, verifier_feature_dict)

        SCORE_INVALID_FORMAT = 0  # 格式错误、没有画线、画线条数不对
        SCORE_VALID_FORMAT = 0.1  # 格式正确，画线条数对，但画的线不对
        SCORE_RIGHT = 1.0  # 格式正确，画的线正确

        try:
            # 取最后一次调用

            convs = (
                "assistant\n" +
                response).split(f'{get_special_tokens_dict_or_name("eos")}{get_special_tokens_dict_or_name("bos")}')
            tool_str = ''
            valid_tool_call = [{}]
            eoi = get_special_tokens_dict_or_name("eoi")
            soi = get_special_tokens_dict_or_name("soi")
            # 选最后一次工具调用
            for idx, conv in enumerate(convs):
                content = conv
                if '<|FunctionCallBegin|>' in content:
                    tool_str = conv.split("<|FunctionCallBegin|>")[1].split("<|FunctionCallEnd|>")[0]
                    cur_tools = json.loads(tool_str)
                    cur_tool_param = cur_tools[0]["parameters"]
                    cur_tool_name = cur_tools[0]["name"]
                    if idx + 1 < len(convs) and f'{soi}{eoi}' in convs[idx + 1]:
                        valid_tool_call.append(cur_tools[0])
            if not tool_str:
                return VerifyResult(score=SCORE_INVALID_FORMAT, extracted_answer=response)
            try:
                tools = json.loads(tool_str)
            except:
                return VerifyResult(score=SCORE_INVALID_FORMAT, extracted_answer=response)
            tool_param = tools[0]["parameters"]
            if not tool_param.get('draw_line', False):
                return VerifyResult(score=SCORE_INVALID_FORMAT, extracted_answer=response)
            final_points = []
            cur_tool = tools[0]
            # 根据imgidx累计之前的图片画的线
            recur_cnt = 0
            while cur_tool:
                recur_cnt += 1
                # 避免死循环
                if recur_cnt > 20:
                    break
                if cur_tool['name'] == 'GROUNDING':
                    # 不能用GROUNDING工具画线
                    return VerifyResult(score=SCORE_INVALID_FORMAT, extracted_answer=response)
                if cur_tool['name'] == 'POINT' and cur_tool['parameters'].get('draw_line', False):
                    final_points.extend(cur_tool['parameters']['points'].split('\n'))
                if 'imgidx' not in cur_tool['parameters']:
                    return VerifyResult(score=SCORE_INVALID_FORMAT, extracted_answer=response)
                if cur_tool['parameters']['imgidx'] < 0 or cur_tool['parameters']['imgidx'] >= len(valid_tool_call):
                    break
                cur_tool = valid_tool_call[cur_tool['parameters']['imgidx']]
            draw_lines = []
            for points in final_points:
                line = parse_point(points)
                if line is not None:
                    abs_line = get_abs_line(line, fig_size)
                    draw_lines.append(abs_line)

            # 判断画的线条数是否正确
            if len(draw_lines) != len(groundtruth_lines):
                return VerifyResult(score=SCORE_INVALID_FORMAT, extracted_answer=response)

            # 计算与ground truth线的角度
            valid_line_cnt = 0
            for i in range(len(draw_lines)):
                is_valid = False
                for j in range(len(groundtruth_lines)):
                    angle = calculate_angle(draw_lines[i], groundtruth_lines[j])
                    if angle < 5 or angle > 175:
                        is_valid = True
                        break
                valid_line_cnt += int(is_valid)
            if valid_line_cnt < len(draw_lines):
                return VerifyResult(score=SCORE_VALID_FORMAT, extracted_answer=response)

            # 计算线的端点与ground truth线的端点的距离，要求画的线至少一个点与ground truth的线不能距离太远
            valid_line_cnt = 0
            w, h = fig_size
            dist_thres = min(math.sqrt(w * h) * 0.02, 15)
            dist_list = []
            for line in draw_lines:
                min_dist = 10000
                for gt_line in groundtruth_lines:
                    for p in line:
                        for gt_p in gt_line:
                            min_dist = min(min_dist, calculate_distance(p, gt_p))
                if min_dist <= dist_thres * 2:
                    valid_line_cnt += 1
                    dist_list.append(min_dist)
            if valid_line_cnt < len(draw_lines):
                return VerifyResult(score=SCORE_VALID_FORMAT, extracted_answer=response)
            scores = []
            for dist in dist_list:
                scores.append(calculate_score_by_dist(dist, dist_thres, dist_thres * 2, SCORE_VALID_FORMAT,
                                                      SCORE_RIGHT))
            return VerifyResult(score=float(np.mean(scores)), extracted_answer=response)
        except Exception:
            import traceback
            tb = "".join(traceback.format_exception(*sys.exc_info()))  # noqa
            raise VerifierFailed(message=tb)
