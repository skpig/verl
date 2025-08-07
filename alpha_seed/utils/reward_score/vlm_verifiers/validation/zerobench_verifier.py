import re

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult


class ZeroBenchVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        pattern = r"\[(.*?)\]"
        try:
            parsed_answer = re.findall(pattern, response.strip().lower())[-1]
        except Exception:
            return VerifyResult(score=0, extracted_answer=response)
        score = 1 if (parsed_answer[:len(answer)].lower() == answer.strip().lower() and
                      len(parsed_answer) == len(answer.strip())) else 0
        return VerifyResult(score=score, extracted_answer=parsed_answer)
