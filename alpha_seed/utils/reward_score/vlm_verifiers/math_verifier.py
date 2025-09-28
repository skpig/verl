from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name
from alpha_seed.utils.reward_score.verifier_service_volc import compute_score
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult, \
    VerifierFailed
from alpha_seed.utils.reward_score.vlm_verifiers.grader import math_equal
from alpha_seed.utils.reward_score.vlm_verifiers.parser import extract_answer
from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name

from alpha_seed.utils.reward_score.vlm_verifiers.math_verify import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify.errors import TimeoutException


def last_boxed_only_string_v2(string: str):
    """
    find last \\boxed{...}
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        idx = string.rfind("boxed{")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
        if retval.startswith('boxed'):
            retval = '\\' + retval
    return retval


class MathVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        pred = extract_answer(response, is_choice=False, use_last_number=False, use_box_only=True)
        if pred == "":
            raise ExtractAnswerFailed
        score = float(math_equal(pred, answer, timeout=False))
        return VerifyResult(score=score, extracted_answer=pred)


class MathV2Verifier(BaseVerifier):

    def __init__(self, volc_ark_key: str, volc_model_name: str) -> None:
        super().__init__()
        self.volc_ark_key = volc_ark_key
        self.volc_model_name = volc_model_name

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        rule_success_flag = True
        pred = last_boxed_only_string_v2(response)
        if pred is None:
            rule_success_flag = False

        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        score = 0.0
        # Wrap the ground truth in \boxed{} format for verification
        if not answer.startswith("\\boxed"):
            answer_boxed = "\\boxed{" + answer + "}"
        else:
            answer_boxed = answer
        try:
            score, _ = verify_func([answer_boxed], [pred])
        except Exception as e:
            print("[DEBUG] MathV2Verifier failed", e)
            rule_success_flag = False
        except TimeoutException:
            print(f"[DEBUG] MathV2Verifier Timeout: {pred}, {answer_boxed}")
            rule_success_flag = False

        # If rule-based verifier failed or give negative score, call model verifier for recall
        if not rule_success_flag or score == 0.0:
            print(f"[DEBUG] call math model verifier")
            ground_truth = {
                "problem": verifier_feature_dict['problem'],
                "reference_answer": answer,
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
        assert score in [0.0, 1.0], f'math_v2 score {score}'
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
