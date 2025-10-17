import json

from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult


class RuleBasedMaintaskVerifier(BaseVerifier):

    def __init__(self) -> None:
        super().__init__()

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        eot_token = get_special_tokens_dict_or_name("think_end_token")
        response = response.split(eot_token)[-1].strip()
        try:
            score_list = json.loads(response.strip())
        except:
            return VerifyResult(score=0.0, extracted_answer="error load score list")
        format_score = 0.05
        if len(score_list) != len(answer):
            return VerifyResult(score=0.0, extracted_answer="invalid score length")
        match_cnt = 0
        for pred, gt in zip(score_list, answer):
            if int(pred) == int(gt):
                match_cnt += 1
        score = max(format_score, float(match_cnt) / len(answer))
        return VerifyResult(score=score, extracted_answer=response)
