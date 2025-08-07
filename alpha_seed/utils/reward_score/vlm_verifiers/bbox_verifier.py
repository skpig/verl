import json
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult
from alpha_seed.utils.reward_score.vlm_verifiers.tools import calculate_iou


class BBoxVerifier(BaseVerifier):

    def find_best_match(self, pred_boxes, gt_boxes):
        """
        根据 IoU 计算预测框和真实框的最优匹配。

        参数:
            pred_boxes: [[x11, y11, x12, y12], ..., [xn1, yn1, xn2, yn2]] 表示框的列表
            gt_boxes: [[gx11, gy11, gx12, gy12], ..., [gxm1, gym1, gxm2, gym2]] 表示真实框的列表

        返回:
            list: 最优匹配 [(pred_idx, true_idx)]
        """
        # 初始化 IoU 矩阵
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))

        # 计算 IoU 矩阵
        for i in range(len(pred_boxes)):
            for j in range(len(gt_boxes)):
                iou_matrix[i, j] = calculate_iou(pred_boxes[i], gt_boxes[j])

        # 转换成成本矩阵（1 - IoU），因为匈牙利算法是最小化总成本
        cost_matrix = 1 - iou_matrix

        # 使用匈牙利算法找到最优匹配
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        # 返回最优匹配
        matches = list(zip(pred_indices, gt_indices))
        return matches, iou_matrix

    def verify(self, response: str, verifier_feature_dict: dict, PN=20) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        if isinstance(answer, str):
            answers = json.loads(answer)
        else:
            answers = answer
        bbox_pattern = re.compile(r'<bbox>(.*?)</bbox>')
        if response == "":
            raise ExtractAnswerFailed
        bbox_contents = bbox_pattern.findall(response)
        predict_bboxes = []

        for bboxes in bbox_contents:
            try:
                bboxes = bboxes.split(" ")
                predict_bboxes.append([int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])])
            except:
                continue

        # correct_predictions = 0
        filter_bboxes = []
        for predict_bbox in predict_bboxes:
            x1, y1, x2, y2 = predict_bbox
            if x2 > x1 and y2 > y1:
                x1, y1 = max(min(int(x1), 1000), 0), max(min(int(y1), 1000), 0)
                x2, y2 = max(min(int(x2), 1000), 0), max(min(int(y2), 1000), 0)
            else:
                continue
            filter_bboxes.append([x1, y1, x2, y2])

        total_predictions = len(filter_bboxes)
        total_gts = len(answers)
        if total_gts == 0 and total_predictions == 0:
            return 1

        matches, iou_matrix = self.find_best_match(filter_bboxes, answers)
        ious = []
        iou_threshold = 0.5
        correct_predictions = 0
        for (predict_index, gt_index) in matches:
            ious.append(iou_matrix[predict_index][gt_index])

            cur_iou = iou_matrix[predict_index][gt_index]
            if cur_iou >= iou_threshold:
                correct_predictions += 1

        avg_ious = sum(ious) / total_gts

        # calculate

        precision = correct_predictions / total_predictions if total_predictions > 0 else 0
        recall = correct_predictions / total_gts if total_gts > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        final_reward = 0.5 * avg_ious + 0.5 * f1

        score_info = {
            'response': response,
            'total_predictions': total_predictions,
            'total_gts': total_gts,
            'correct_predictions': correct_predictions,
            'avg_ious': avg_ious,
            'f1': f1,
            'final_reward': final_reward
        }
        out_info = json.dumps(score_info)

        return VerifyResult(score=final_reward, extracted_answer=out_info)
