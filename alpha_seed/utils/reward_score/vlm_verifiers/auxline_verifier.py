import json
import traceback
import re
import sys
import os
import random
import time
import math
import numpy as np

from alpha_seed.utils.reward_score.extra_reward import match_visual_cot_format
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


def check_tool_call_valid(name, params):
    # 检测工具调用是否合法
    if 'imgidx' not in params:
        return False
    try:
        if name == 'POINTS':
            pattern_line = r'^(<point>\s*(\d{1,3})\s+(\d{1,3})\s*</point>\s*)+$'
            if not re.match(pattern_line, params.get('points', '')):
                return False
        if name == 'GROUNDING':
            pattern = r'(<bbox>\s*(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s*</bbox>\s*)+'
            if not re.fullmatch(pattern, params.get('bbox_str', '')):
                return False
    except:
        #    traceback.print_exc()
        return False
    return True


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
            convs = ("assistant\n" + response).split(
                "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"
            )
            tool_str = ''
            valid_tool_call = [{}]
            # 选最后一次工具调用
            for conv in convs:
                content = conv
                if '<|FunctionCallBegin|>' in content:
                    tool_str = conv.split("<|FunctionCallBegin|>")[1].split("<|FunctionCallEnd|>")[0]
                    cur_tools = json.loads(tool_str)
                    cur_tool_param = cur_tools[0]["parameters"]
                    cur_tool_name = cur_tools[0]["name"]
                    if check_tool_call_valid(cur_tool_name, cur_tool_param):
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
            return VerifyResult(score=np.mean(scores), extracted_answer=response)
        except Exception:
            import traceback
            tb = "".join(traceback.format_exception(*sys.exc_info()))
            raise VerifierFailed(message=tb)


def _test():
    verifier = AuxlineVerifier()
    response = """<think>Got it, let's see. The problem is to draw a line FG parallel to BC, starting from point F and intersecting AC at G. First, I need to identify the coordinates of the points. Looking at the diagram, point F is on AB, and we need to draw FG parallel to BC. So first, find point F, then draw a line from F that's parallel to BC (which is the base from B to C to D). So the steps would be: use the POINT tool to mark F, then G on AC, then draw the line between them. Wait, the POINT tool can draw lines if draw_line is true. So first, find the coordinates. Let's assume the coordinates: F is somewhere on AB, G is on AC. So let's mark F as <point>195 300</point> (approximate), G as <point>226 450</point> (on AC), then connect them with a line. So the function call would be POINT with imgidx 0, points as <point>195 300</point><point>226 450</point>, and draw_line true.</think><|FunctionCallBegin|>[{"name": "POINT", "parameters": {"imgidx": 0, "points": "<point>196 301</point><point>225 451</point>", "draw_line": true}}]<|FunctionCallEnd|><[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>tool name=plugin
[SOI]<ImageHere>[EOI]<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant
<think>Wait, the previous POINT tool call might not have been accurate. Let me re-examine the diagram. Point F is on AB, and we need to draw FG parallel to BC, intersecting AC at G. Let's find the correct coordinates. Let's assume the triangle has vertices: A at the top, B at the bottom left, C at the bottom right, D extended from C. F is on AB with AF/BF = 1/2. So FG should be parallel to BC, so it should be a horizontal line (if BC is horizontal) from F to AC. Let's mark F more accurately. Let's say F is at <point>198 300</point>, and G is on AC at <point>228 450</point>. Wait, maybe the coordinates need adjustment. Alternatively, use the GROUNDING tool to first locate F and then draw the line. Wait, the POINT tool with draw_line true should work. Let me try again with more precise points. Let's look at the diagram: AB is from A (top) to B (bottom left), F is on AB. AC is from A to C (bottom right). So FG parallel to BC: BC is from B to C to D, so horizontal. So FG should be horizontal from F to AC. So F's coordinates: let's say <point>195 305</point>, G on AC: <point>226 455</point>. So call POINT with these two points and draw_line true.</think><|FunctionCallBegin|>[{"name": "POINT", "parameters": {"imgidx": 1, "points": "<point>195 305</point><point>226 455</point>", "draw_line": true}}]<|FunctionCallEnd|><[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>tool name=plugin
[SOI]<ImageHere>[EOI]<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant
<think>Hmm, maybe the issue is that the initial image index was wrong. Let's check the image index. The first image was imgidx 0, then after the first tool call, it became imgidx 1. Wait, the user provided the first image, then after the first tool, the second image. So now, to draw FG, let's use the correct imgidx. Let's look at the diagram again. Point F is on AB, let's find its coordinates. Let's say F is at <point>786 250</point> (since in the diagram, F is on AB, upper part). Then G is on AC, so <point>804 350</point>? Wait, maybe I need to zoom in. Let's use the ZOOM tool to zoom the image to see better. So call ZOOM with imgidx 1, scale 2, to zoom the whole image.</think><|FunctionCallBegin|>[{"name": "ZOOM", "parameters": {"imgidx": 1, "scale": 2.0}}]<|FunctionCallEnd|><[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>tool name=plugin
[SOI]<ImageHere>[EOI]<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant
<think>After zooming, I can see the diagram more clearly. Point F is on AB, and we need to draw FG parallel to BC, intersecting AC at G. Let's identify the coordinates: F is at <point>785 280</point>, and G should be on AC. AC goes from A (top) to C (bottom right). So drawing a line from F parallel to BC (which is horizontal from B to C to D) would be a horizontal line from F to AC. So G would be at <point>803 380</point>? Wait, maybe better to use the POINT tool with the zoomed image (imgidx 2 now, since we zoomed imgidx 1). Wait, the ZOOM tool returns a new image with imgidx incremented. So after zooming imgidx 1, the new image is imgidx 2. So now, let's use POINT on imgidx 2. Let's mark F at <point>786 285</point> and G at <point>804 385</point>, then draw the line between them.</think><|FunctionCallBegin|>[{"name": "POINT", "parameters": {"imgidx": 2, "points": "<point>786 285</point><point>804 385</point>", "draw_line": true}}]<|FunctionCallEnd|><[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>tool name=plugin
[SOI]<ImageHere>[EOI]<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant
<think>Looking at the current diagram, the line FG has been drawn from F to G on AC, parallel to BC. This should satisfy the requirement of drawing the auxiliary line. The previous steps involved zooming to clarify the coordinates and then using the POINT tool to draw the line, which now appears correctly in the diagram. So the task of drawing FG parallel to BC is complete.</think>The auxiliary line \( FG \parallel BC \) (with \( G \) on \( AC \)) has been successfully drawn using the `POINT` tool to connect \( F \) (on \( AB \)) and \( G \) (on \( AC \)) with a line, as shown in the diagram.<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"""
    verifier_feature_dict = {
        "problem": "过点F作FG[线-2]平行于BC，交AC于点G[点-6] 不需要解答图中的题目，你只需要图中按要求画出辅助线即可。",
        "reference_answer": "",
        "verify_type": 4,
        "answer": "",
        "verifier_name": "auxline_rule_verifier",
        "gt_lines": [[[535.9325184409086, 68.9012856011804], [566.2957053230083, 66.0755588351011]]],
        "query": "过点F作FG[线-2]平行于BC，交AC于点G[点-6] 不需要解答图中的题目，你只需要图中按要求画出辅助线即可。",
        "fig_size": [670, 207]
    }

    print(verifier.verify(response, verifier_feature_dict))


if __name__ == '__main__':
    _test()
