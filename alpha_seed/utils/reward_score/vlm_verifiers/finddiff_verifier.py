import copy
import json
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult


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


def calculate_iou_l(predbox, gt_box_l):
    iou_re = []
    for gbox in gt_box_l:
        iou_re.append(calculate_iou(predbox, gbox))
    return max(iou_re)


class FindDiffVerifier(BaseVerifier):

    def is_valid_point(self, p_x1, p_y1, p_x2, p_y2):
        # Check if all coordinates are within [0,1000] range
        if (0 <= p_x1 < 1000 and 0 <= p_y1 < 1000 and 0 <= p_x2 < 1000 and 0 <= p_y2 < 1000):
            return True
        else:
            return False

    def extract_predict_bboxes(self, pred_str):

        bbox_pattern = re.compile(r'<bbox>(.*?)</bbox>')
        bbox_contents = bbox_pattern.findall(pred_str)

        if bbox_contents is None:
            return []

        predict_bboxes = []
        added_bboxes = {}
        for bboxes in bbox_contents:
            try:
                bboxes = bboxes.split(" ")
                p_x1, p_y1, p_x2, p_y2 = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])

                # valid bbox
                if (self.is_valid_point(p_x1, p_y1, p_x2, p_y2)) and ((p_x2 > p_x1) and (p_y2 > p_y1)):
                    # check whether the bbox already added
                    box_str = '_'.join([str(e) for e in [p_x1, p_y1, p_x2, p_y2]])
                    if box_str not in added_bboxes:
                        predict_bboxes.append([p_x1, p_y1, p_x2, p_y2])
                        added_bboxes[box_str] = 1

            except Exception as e:
                continue

        return predict_bboxes

    def get_bbox_metrics(self, predict_bboxes, gt_ans, iou_rwei=1.0, PN=20):
        # judge whether output following tlx1 tly1 brx1 bry1
        gts = copy.deepcopy(gt_ans)

        N = len(predict_bboxes)
        M = len(gts)
        # Calculate centers of boxes
        pred_centers = np.array([[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0] for box in predict_bboxes])
        gt_centers = np.array(
            [[(box_pair[0][0] + box_pair[0][2]) / 2.0, (box_pair[0][1] + box_pair[0][3]) / 2.0] for box_pair in gts])
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
                iou_reward = calculate_iou_l(predict_bboxes[pred_idx], gt_ans[gt_idx])
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

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:

        # get answer
        answer = verifier_feature_dict['answer']
        if isinstance(answer, str):
            answer = json.loads(answer)
        else:
            answer = answer

        pred_boxes = self.extract_predict_bboxes(response)

        if len(pred_boxes) < 1:
            raise ExtractAnswerFailed

        score, sinfo = self.get_bbox_metrics(pred_boxes, answer)
        pred_sinfo = json.dumps(sinfo)

        return VerifyResult(score=score, extracted_answer=pred_sinfo)


class FindDiffReflectVerifier(FindDiffVerifier):

    def get_prev_score(self, prev_preds, answer):

        prev_pred_boxes = self.extract_predict_bboxes(prev_preds)

        if len(prev_pred_boxes) < 1:
            return 0

        # calcuate current score
        prev_score, prev_sinfo = self.get_bbox_metrics(prev_pred_boxes, answer)

        return prev_score

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:

        # get answer
        answer = verifier_feature_dict['answer']
        if isinstance(answer, str):
            answer = json.loads(answer)
        else:
            answer = answer

        pred_boxes = self.extract_predict_bboxes(response)

        if len(pred_boxes) < 1:
            raise ExtractAnswerFailed

        # calcuate current score
        score, sinfo = self.get_bbox_metrics(pred_boxes, answer)

        # calcuate score of prev resp
        prev_preds = verifier_feature_dict.get('prev_solution', '')
        prev_score = self.get_prev_score(prev_preds, answer)
        sinfo['prev_score'] = prev_score

        final_score = max(score - prev_score, -0.05)
        sinfo['final_score'] = final_score

        pred_sinfo = json.dumps(sinfo)
        return VerifyResult(score=final_score, extracted_answer=pred_sinfo)
