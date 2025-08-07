from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult
from alpha_seed.utils.reward_score.vlm_verifiers.tools import extract_boxed_number


class CountVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict, delta: float) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        if response == "":
            raise ExtractAnswerFailed
        predict_num = extract_boxed_number(response)
        try:
            predict_num = int(predict_num)
        except TypeError:
            raise ExtractAnswerFailed
        count_answer = int(answer)
        if predict_num == count_answer:
            score = 1
        else:
            # score = 1 if predict_num == count_answer else 0
            score = min(max(1 - abs(count_answer - predict_num) / count_answer, 0), delta)
        return VerifyResult(score=score, extracted_answer=response)
