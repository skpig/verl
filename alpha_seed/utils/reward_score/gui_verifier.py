import re
import ast
from rouge_chinese import Rouge
import jieba
import numpy as np


def get_truth_action_type_value(content):
    pattern = r"Action: (\w+)\((.*)\)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        action_type = match.group(1)  # 提取 action type (click)
        action_value = match.group(2)
        if action_type == 'type':
            content_pattern = r"content='(.*?)'"
            bbox_pattern = r"start[_ ]box='<\|box_start\|\>(.*?)<\|box_end\|\>'"

            # 使用正则表达式提取内容
            content_match = re.search(content_pattern, action_value)
            bbox_match = re.search(bbox_pattern, action_value)
            # 获取匹配的内容
            line_content = content_match.group(1) if content_match else None
            box_content = bbox_match.group(1) if bbox_match else None
            if box_content is None:
                bbox = []
            else:
                box_tuples = [tuple(map(int, coord.strip('()').split(','))) for coord in box_content.split('),(')]
                if len(box_tuples) == 1:
                    bbox = [
                        box_tuples[0][0] / 1000, box_tuples[0][1] / 1000, box_tuples[0][0] / 1000,
                        box_tuples[0][1] / 1000
                    ]
                elif len(box_tuples) == 2:
                    bbox = [
                        box_tuples[0][0] / 1000, box_tuples[0][1] / 1000, box_tuples[1][0] / 1000,
                        box_tuples[1][1] / 1000
                    ]
            return action_type, (line_content, bbox)
        elif action_type == 'drag':
            pattern = r"start[_ ]box='<\|box_start\|\>(.*?)<\|box_end\|\>', end[_ ]box='<\|box_start\|\>(.*?)<\|box_end\|\>'"
            # 提取坐标
            bbox_match = re.search(pattern, action_value)

            if bbox_match:
                start_box_content = bbox_match.group(1)
                start_box_tuples = [
                    tuple(map(int,
                              coord.strip('()').split(','))) for coord in start_box_content.split('),(')
                ]
                if len(start_box_tuples) == 1:
                    start_bbox = [
                        start_box_tuples[0][0] / 1000, start_box_tuples[0][1] / 1000, start_box_tuples[0][0] / 1000,
                        start_box_tuples[0][1] / 1000
                    ]
                elif len(start_box_tuples) == 2:
                    start_bbox = [
                        start_box_tuples[0][0] / 1000, start_box_tuples[0][1] / 1000, start_box_tuples[1][0] / 1000,
                        start_box_tuples[1][1] / 1000
                    ]
                end_box_content = bbox_match.group(2)
                end_box_tuples = [
                    tuple(map(int,
                              coord.strip('()').split(','))) for coord in end_box_content.split('),(')
                ]
                if len(end_box_tuples) == 1:
                    end_bbox = [
                        end_box_tuples[0][0] / 1000, end_box_tuples[0][1] / 1000, end_box_tuples[0][0] / 1000,
                        end_box_tuples[0][1] / 1000
                    ]
                elif len(end_box_tuples) == 2:
                    end_bbox = [
                        end_box_tuples[0][0] / 1000, end_box_tuples[0][1] / 1000, end_box_tuples[1][0] / 1000,
                        end_box_tuples[1][1] / 1000
                    ]

            else:
                start_bbox, end_bbox = [], []
            return action_type, (start_bbox, end_bbox)
        elif action_type == 'scroll':
            direction_pattern = r"direction='(.*?)'"
            bbox_pattern = r"start[_ ]box='<\|box_start\|\>(.*?)<\|box_end\|\>'"

            # 使用正则表达式提取内容
            direction_match = re.search(direction_pattern, action_value)
            bbox_match = re.search(bbox_pattern, action_value)
            # 获取匹配的内容
            direction = direction_match.group(1) if direction_match else None
            box_content = bbox_match.group(1) if bbox_match else None
            if box_content is None:
                bbox = []
            else:
                box_tuples = [tuple(map(int, coord.strip('()').split(','))) for coord in box_content.split('),(')]
                if len(box_tuples) == 1:
                    bbox = [
                        box_tuples[0][0] / 1000, box_tuples[0][1] / 1000, box_tuples[0][0] / 1000,
                        box_tuples[0][1] / 1000
                    ]
                elif len(box_tuples) == 2:
                    bbox = [
                        box_tuples[0][0] / 1000, box_tuples[0][1] / 1000, box_tuples[1][0] / 1000,
                        box_tuples[1][1] / 1000
                    ]
            return action_type, (direction, bbox)
        else:
            value_pattern = r"<\|box_start\|\>(.*?)<\|box_end\|\>"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                box_content = value_match.group(1)
                box_tuples = [tuple(map(int, coord.strip('()').split(','))) for coord in box_content.split('),(')]
                if len(box_tuples) == 1:
                    bbox = [
                        box_tuples[0][0] / 1000, box_tuples[0][1] / 1000, box_tuples[0][0] / 1000,
                        box_tuples[0][1] / 1000
                    ]
                elif len(box_tuples) == 2:
                    bbox = [
                        box_tuples[0][0] / 1000, box_tuples[0][1] / 1000, box_tuples[1][0] / 1000,
                        box_tuples[1][1] / 1000
                    ]
                return action_type, bbox
            else:
                value_pattern = r"'(.*)'"
                value_match = re.search(value_pattern, action_value)
                if value_match:
                    quoted_value = value_match.group(1)  # 提取引号中的内容
                    return action_type, quoted_value
                else:
                    # print(f"truth doesn't found match value:{content}")  #finished wait
                    return action_type, action_value

    else:
        try:
            final_value = ast.literal_eval(content)
            final_value = [final_value[0], final_value[1], final_value[0], final_value[1]]
            return "click", final_value
        except Exception as e:
            print(content)
            return None, None


def get_pred_action_type_value(content):
    pattern = r"Action: (\w+)\((.*)\)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        action_type = match.group(1)  # 提取 action type (click)
        action_value = match.group(2)
        if action_type == 'type':
            content_pattern = r"content='(.*?)'"
            bbox_pattern = r"start[_ ]box='\((\d+\.\d+|\d+),(\d+\.\d+|\d+)\)'"

            # 使用正则表达式提取内容
            content_match = re.search(content_pattern, action_value, re.DOTALL)
            bbox_match = re.search(bbox_pattern, action_value, re.DOTALL)
            # 获取匹配的内容
            line_content = content_match.group(1) if content_match else None
            tuple1 = (int(bbox_match.group(1)), int(bbox_match.group(2))) if bbox_match else None  # 第一个元组
            if tuple1 is None:
                bbox = []
            else:
                bbox = [tuple1[0] / 1000, tuple1[1] / 1000, tuple1[0] / 1000, tuple1[1] / 1000]
            return action_type, (line_content, bbox)
        elif action_type == 'drag':
            pattern = r"start[_ ]box='\((\d+\.\d+|\d+),(\d+\.\d+|\d+)\)', end[_ ]box='\((\d+\.\d+|\d+),(\d+\.\d+|\d+)\)'"
            # 提取坐标
            match = re.search(pattern, action_value)

            if match:
                start_coords = [int(match.group(1)), int(match.group(2))]
                end_coords = [int(match.group(3)), int(match.group(4))]
                start_bbox = [
                    start_coords[0] / 1000, start_coords[1] / 1000, start_coords[0] / 1000, start_coords[1] / 1000
                ]
                end_bbox = [end_coords[0] / 1000, end_coords[1] / 1000, end_coords[0] / 1000, end_coords[1] / 1000]

            else:
                start_bbox, end_bbox = [], []
            return action_type, (start_bbox, end_bbox)
        elif action_type == 'scroll':
            direction_pattern = r"direction='(.*?)'"
            bbox_pattern = r"start[_ ]box='\((\d+\.\d+|\d+),(\d+\.\d+|\d+)\)'"

            # 使用正则表达式提取内容
            direction_match = re.search(direction_pattern, action_value, re.DOTALL)
            bbox_match = re.search(bbox_pattern, action_value)
            # 获取匹配的内容
            direction = direction_match.group(1) if direction_match else None
            tuple1 = (int(bbox_match.group(1)), int(bbox_match.group(2))) if bbox_match else None  # 第一个元组
            if tuple1 is None:
                bbox = []
            else:
                bbox = [tuple1[0] / 1000, tuple1[1] / 1000, tuple1[0] / 1000, tuple1[1] / 1000]

            return action_type, (direction, bbox)
        else:
            value_pattern = r"start[_ ]box='\((\d+\.\d+|\d+),(\d+\.\d+|\d+)\)'"

            value_match = re.search(value_pattern, action_value)
            if value_match:
                tuple1 = (int(value_match.group(1)), int(value_match.group(2)))  # 第一个元组
                bbox = [tuple1[0] / 1000, tuple1[1] / 1000, tuple1[0] / 1000, tuple1[1] / 1000]
                return action_type, bbox
            else:
                value_pattern = r"'(.*)'"
                value_match = re.search(value_pattern, action_value)
                if value_match:
                    quoted_value = value_match.group(1)  # 提取引号中的内容
                    return action_type, quoted_value
                else:
                    # print(f"truth doesn't found match value:{content}")  #finished wait
                    return action_type, action_value
    else:
        # print(f"pred doesn't have match action:{content}")
        return None, None


def cal_distance(golden_cord, pred_cord):
    golden_point = [(golden_cord[0] + golden_cord[2]) / 2, (golden_cord[1] + golden_cord[3]) / 2]
    pred_point = [(pred_cord[0] + pred_cord[2]) / 2, (pred_cord[1] + pred_cord[3]) / 2]
    distance = ((golden_point[0] - pred_point[0])**2 + (golden_point[1] - pred_point[1])**2)**0.5
    return distance


def metric_distance(gt_bbox, pred_bbox):
    try:
        dis = cal_distance(gt_bbox, pred_bbox)
        dis_value_error = False
    except Exception as e:
        # print(f"cal dis error:{e}, gt_value:{gt_bbox}, pred_value:{pred_bbox}, pred_ori:{pred_line}")
        dis_value_error = True
        dis = 100
    return dis, dis_value_error


def get_ori_truth_action_type_value(content):
    pattern = r"Action: (\w+)\((.*)\)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        action_type = match.group(1)  # 提取 action type (click)
        action_value = match.group(2)
        if action_type == 'type':
            content_pattern = r"content='(.*?)'"
            bbox_pattern = r"start[_ ]box='\[(.*?)\]'"

            # 使用正则表达式提取内容
            content_match = re.search(content_pattern, action_value)
            bbox_match = re.search(bbox_pattern, action_value)
            # 获取匹配的内容
            line_content = content_match.group(1) if content_match else None
            box_content = bbox_match.group(1) if bbox_match else None
            if box_content is None:
                bbox = []
            else:
                box_tuples = list(map(float, box_content.split(',')))
                if len(box_tuples) == 2:
                    bbox = [box_tuples[0], box_tuples[1], box_tuples[0], box_tuples[1]]
                elif len(box_tuples) == 4:
                    bbox = [(box_tuples[0] + box_tuples[2]) / 2, (box_tuples[1] + box_tuples[3]) / 2,
                            (box_tuples[0] + box_tuples[2]) / 2, (box_tuples[1] + box_tuples[3]) / 2]
            return action_type, (line_content, bbox)
        elif action_type == 'drag':
            pattern = r"start[_ ]box='\[(.*?)\]', end[_ ]box='\[(.*?)\]'"
            # 提取坐标
            bbox_match = re.search(pattern, action_value)

            if bbox_match:
                start_box_content = bbox_match.group(1)
                start_box_tuples = list(map(float, start_box_content.split(',')))
                if len(start_box_tuples) == 2:
                    start_bbox = [start_box_tuples[0], start_box_tuples[1], start_box_tuples[0], start_box_tuples[1]]
                elif len(start_box_tuples) == 4:
                    start_bbox = [(start_box_tuples[0] + start_box_tuples[2]) / 2,
                                  (start_box_tuples[1] + start_box_tuples[3]) / 2,
                                  (start_box_tuples[0] + start_box_tuples[2]) / 2,
                                  (start_box_tuples[1] + start_box_tuples[3]) / 2]
                end_box_content = bbox_match.group(2)
                end_box_tuples = list(map(float, end_box_content.split(',')))
                if len(end_box_tuples) == 2:
                    end_bbox = [end_box_tuples[0], end_box_tuples[1], end_box_tuples[0], end_box_tuples[1]]
                elif len(end_box_tuples) == 4:
                    end_bbox = [
                        (end_box_tuples[0] + end_box_tuples[2]) / 2, (end_box_tuples[1] + end_box_tuples[3]) / 2,
                        (end_box_tuples[0] + end_box_tuples[2]) / 2, (end_box_tuples[1] + end_box_tuples[3]) / 2
                    ]

            else:
                start_bbox, end_bbox = [], []
            return action_type, (start_bbox, end_bbox)
        elif action_type == 'scroll':
            direction_pattern = r"direction='(.*?)'"
            bbox_pattern = r"start[_ ]box='\[(.*?)\]'"

            # 使用正则表达式提取内容
            direction_match = re.search(direction_pattern, action_value)
            bbox_match = re.search(bbox_pattern, action_value)
            # 获取匹配的内容
            direction = direction_match.group(1) if direction_match else None
            box_content = bbox_match.group(1) if bbox_match else None
            if box_content is None:
                bbox = []
            else:
                box_content = bbox_match.group(1)
                box_tuples = list(map(float, box_content.split(',')))
                if len(box_tuples) == 2:
                    bbox = [box_tuples[0], box_tuples[1], box_tuples[0], box_tuples[1]]
                elif len(box_tuples) == 4:
                    bbox = [(box_tuples[0] + box_tuples[2]) / 2, (box_tuples[1] + box_tuples[3]) / 2,
                            (box_tuples[0] + box_tuples[2]) / 2, (box_tuples[1] + box_tuples[3]) / 2]
            return action_type, (direction, bbox)
        else:
            value_pattern = r"start[_ ]box='\[(.*?)\]'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                box_content = value_match.group(1)
                box_tuples = list(map(float, box_content.split(',')))
                if len(box_tuples) == 2:
                    bbox = [box_tuples[0], box_tuples[1], box_tuples[0], box_tuples[1]]
                elif len(box_tuples) == 4:
                    bbox = [(box_tuples[0] + box_tuples[2]) / 2, (box_tuples[1] + box_tuples[3]) / 2,
                            (box_tuples[0] + box_tuples[2]) / 2, (box_tuples[1] + box_tuples[3]) / 2]
                return action_type, bbox
            else:
                value_pattern = r"'(.*)'"
                value_match = re.search(value_pattern, action_value)
                if value_match:
                    quoted_value = value_match.group(1)  # 提取引号中的内容
                    return action_type, quoted_value
                else:
                    # print(f"truth doesn't found match value:{content}")  #finished wait
                    return action_type, action_value

    else:
        try:
            final_value = ast.literal_eval(content)
            final_value = [final_value[0], final_value[1], final_value[0], final_value[1]]
            return "click", final_value
        except Exception as e:
            print(content)


import re
from collections import Counter


def rule_for_action(gt_content, pred_content):
    scorer = Rouge()
    dis_threshold = 0.05
    item_rouge_score = None
    item_dis = None
    item_type_correct = False
    item_value_correct = False
    try:
        gt_type, gt_value = get_ori_truth_action_type_value(gt_content)
    except:
        return False
    try:
        pred_type, pred_value = get_truth_action_type_value(pred_content)
    except:
        return False
    if pred_type is None:
        return False
    if pred_type == "left_single":
        pred_type = "click"

    if gt_type == pred_type and gt_type is not None:
        item_type_correct = True
        if gt_type.lower() in ['click', 'select', 'hover', 'right_single', 'left_double', 'left_single']:
            gt_bbox = gt_value
            pred_bbox = pred_value
            item_dis, dis_value_error = metric_distance(gt_bbox, pred_bbox)
            if item_dis < dis_threshold:
                item_value_correct = True

        elif gt_type.lower() == 'scroll':
            gt_direction = gt_value[0]
            pred_direction = pred_value[0]
            gt_bbox = gt_value[1]
            pred_bbox = pred_value[1]
            if gt_bbox != []:
                item_dis, dis_value_error = metric_distance(gt_bbox, pred_bbox)
                if item_dis < dis_threshold and gt_direction == pred_direction:
                    item_value_correct = True
            else:
                if gt_direction == pred_direction:
                    item_value_correct = True

        elif gt_type.lower() == 'drag':
            gt_start_bbox = gt_value[0]
            pred_start_bbox = pred_value[0]
            gt_end_bbox = gt_value[1]
            pred_end_bbox = pred_value[1]
            gt_bbox = [gt_start_bbox, gt_end_bbox]
            pred_bbox = [pred_start_bbox, pred_end_bbox]
            dis_start, dis_start_value_error = metric_distance(gt_start_bbox, pred_start_bbox)
            dis_end, dis_end_value_error = metric_distance(gt_end_bbox, pred_end_bbox)
            item_dis = (dis_start + dis_end) / 2
            dis_value_error = dis_start_value_error or dis_end_value_error
            if dis_start < dis_threshold and dis_end < dis_threshold:
                item_value_correct = True

        elif gt_type.lower() in ['navigate_back', 'navigate_home', 'enter', 'wait', 'finished']:
            item_value_correct = True

        elif gt_type.lower() == 'type':
            gt_content = gt_value[0]
            gt_bbox = gt_value[1]
            pred_content = pred_value[0]
            pred_bbox = pred_value[1]
            gt_content_seg = ' '.join(jieba.cut(str(gt_content)))
            pred_content_seg = ' '.join(jieba.cut(str(pred_content)))
            if gt_content_seg == '\n':
                gt_content_seg = '\\n'
            if pred_content_seg == '\n':
                pred_content_seg = '\\n'
            try:
                scores = scorer.get_scores(gt_content_seg, pred_content_seg)
                item_rouge_score = scores[0]['rouge-l']['f']
            except:
                item_rouge_score = 0

            if item_rouge_score >= 0.5:
                item_value_correct = True
            else:
                item_value_correct = False

        else:
            if gt_value == pred_value:
                item_value_correct = True
    return item_value_correct


def compute_score(solution_str, ground_truth, **argv):
    if rule_for_action(ground_truth, solution_str):
        score = 1.0
    else:
        score = -1.0

    return score
