import json
import re

from shapely import Polygon, Point

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult


class PointVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        if isinstance(answer, str):
            answers = json.loads(answer)
        else:
            answers = answer
        point_pattern = re.compile(r'<point>(.*?)</point>')
        if response == "":
            raise ExtractAnswerFailed
        point_contents = point_pattern.findall(response)
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
        return VerifyResult(score=f1, extracted_answer=response)
