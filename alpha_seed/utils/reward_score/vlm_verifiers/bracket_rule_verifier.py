import json
import re

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, ExtractAnswerFailed


class BracketRuleVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        if isinstance(verifier_feature_dict, str):
            verifier_feature_dict = json.loads(verifier_feature_dict)
        answer = str(verifier_feature_dict['answer'])
        if verifier_feature_dict['bracket_type'] == 'curly':
            pattern = r"\{(.*?)\}"
        else:
            pattern = r"\[(.*?)\]"
        try:
            parsed_answer = re.findall(pattern, response.strip().lower())[-1]
        except Exception:
            raise ExtractAnswerFailed
        score = 1 if (parsed_answer[:len(answer)].lower() == answer.strip().lower() and
                      len(parsed_answer) == len(answer.strip())) else 0
        return VerifyResult(score=score, extracted_answer=parsed_answer)
