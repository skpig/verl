from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult
from alpha_seed.utils.reward_score.vlm_verifiers.parser import extract_answer, strip_string


class BoxStrVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        pred = extract_answer(
            response,
            is_choice=False,
            use_last_number=False,
            use_box_only=True  # only extract answers from \boxed{}
        )
        if pred == "":
            raise ExtractAnswerFailed
        score = float(strip_string(answer) == pred)
        return VerifyResult(score=score, extracted_answer=pred)


class PlainStrVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        pred = response
        score = float(pred.strip() == answer)
        return VerifyResult(score=score, extracted_answer=pred)
