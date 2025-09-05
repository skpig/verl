from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name
from alpha_seed.utils.reward_score.verifier_service_volc import compute_score
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult, \
    VerifierFailed
from alpha_seed.utils.reward_score.vlm_verifiers.grader import math_equal
from alpha_seed.utils.reward_score.vlm_verifiers.parser import extract_answer


class MathVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        pred = extract_answer(response, is_choice=False, use_last_number=False, use_box_only=True)
        if pred == "":
            raise ExtractAnswerFailed
        score = float(math_equal(pred, answer, timeout=False))
        return VerifyResult(score=score, extracted_answer=pred)


class ModelBasedMathVerifierVolc(BaseVerifier):

    def __init__(self, volc_ark_key: str, volc_model_name: str) -> None:
        super().__init__()
        self.volc_ark_key = volc_ark_key
        self.volc_model_name = volc_model_name

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        if 'reference_answer' in verifier_feature_dict:
            reference_answer = verifier_feature_dict['reference_answer']
        else:
            reference_answer = verifier_feature_dict['answer']
        ground_truth = {
            "problem": verifier_feature_dict['problem'],
            "reference_answer": reference_answer,
            "verify_type": verifier_feature_dict.get("verify_type", 4),
        }
        solution_str = response
        if ground_truth["verify_type"] == 4:
            sot_token = get_special_tokens_dict_or_name("think_start_token")
            eot_token = get_special_tokens_dict_or_name("think_end_token")
            if eot_token not in response:
                solution_str = f"{sot_token}dummy{eot_token}" + solution_str  # required by verifier_type=4
        else:
            raise NotImplementedError
        score = compute_score(solution_str=solution_str,
                              ground_truth=ground_truth,
                              volc_ark_key=self.volc_ark_key,
                              volc_model_name=self.volc_model_name)
        if score == 1:
            return VerifyResult(score=1.0, extracted_answer=solution_str)
        elif score == -1:
            return VerifyResult(score=0.0, extracted_answer=solution_str)
        elif score == -2:
            raise VerifierFailed
        else:
            raise NotImplementedError
