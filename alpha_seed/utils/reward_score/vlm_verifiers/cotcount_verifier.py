import json

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult
from alpha_seed.utils.reward_score.vlm_verifiers.bbox_verifier import BBoxVerifier
from alpha_seed.utils.reward_score.vlm_verifiers.point_verifier import PointVerifier
from alpha_seed.utils.reward_score.vlm_verifiers.tools import extract_and_convert_number


class CoTCountVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict, delta: float) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        if isinstance(answer, str):
            answers = json.loads(answer)
        else:
            answers = answer
        point_answer, count_answer = answers[:-1], answers[-1]
        count_answer = int(count_answer)
        if response == "":
            raise ExtractAnswerFailed
        if "<point>" in response:
            verifier_feature_dict['answer'] = point_answer
            point_result = PointVerifier().verify(response=response, verifier_feature_dict=verifier_feature_dict)
            point_score = point_result.score
        elif "<bbox>" in response:
            bbox_answer = []
            for point in point_answer:
                x1, y1 = point[0][0], point[0][1]
                x2, y2 = point[3][0], point[3][1]
                bbox_answer.append([x1, y1, x2, y2])
            verifier_feature_dict['answer'] = bbox_answer
            bbox_result = BBoxVerifier().verify(response=response, verifier_feature_dict=verifier_feature_dict)
            point_score = bbox_result.score
        else:
            point_score = 0 if count_answer != 0 else 1

        predict_num, truncated = extract_and_convert_number(response)
        # 出现截断问题，则输出不符合要求，只获得point_score
        if truncated:
            score = 0
        else:
            if predict_num == count_answer:
                count_score = 1
            else:
                count_score = min(max(1 - abs(count_answer - predict_num) / count_answer, 0), delta)
            score = 0.5 * count_score + 0.5 * point_score
        return VerifyResult(score=score, extracted_answer=response)
