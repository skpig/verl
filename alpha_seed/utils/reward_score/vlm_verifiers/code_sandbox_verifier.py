from alpha_seed.utils.reward_score.oj_utils import compute_score
from alpha_seed.utils.reward_score.response_post_proc import last_codeblock_postprocess
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifierFailed, VerifyResult


class CodeSandboxVerifier(BaseVerifier):

    def __init__(self, code_sandbox_service_psm) -> None:
        super().__init__()
        self.code_sandbox_service_psm = code_sandbox_service_psm or "data.aml.code_sandbox_arnold_celery.service.hl"
        self.code_sandbox_service_psm = self.code_sandbox_service_psm.replace('\\', '')

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        solution_str = last_codeblock_postprocess(input_text=response,
                                                  codeblock_seps=['python', 'cpp', 'java'],
                                                  last_response_strict=True)
        ground_truth = verifier_feature_dict['answer']
        result = compute_score(solution_str=solution_str,
                               ground_truth=ground_truth,
                               code_sandbox_psm=self.code_sandbox_service_psm)
        score = result['score']
        msg = result['msg']
        if score == 1:
            return VerifyResult(score=1.0, extracted_answer=solution_str)
        elif score == -1:
            return VerifyResult(score=0.0, extracted_answer=msg)
        elif score == -2:
            raise VerifierFailed
        else:
            raise NotImplementedError
