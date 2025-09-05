import json

from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name
from alpha_seed.utils.reward_score.vlm_verifiers.extra_reward import match_visual_cot_format
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, ExtractAnswerFailed


class RotateToolVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        if not match_visual_cot_format(response, verifier_feature=verifier_feature_dict):
            raise ExtractAnswerFailed

        gt_rotate_degree = int(verifier_feature_dict['answer']) % 360
        assert gt_rotate_degree >= 0

        if ('<|FunctionCallBegin|>' not in response) and ('<|FunctionCallEnd|>' not in response):
            return VerifyResult(score=float(0 == gt_rotate_degree), extracted_answer='0')

        last_call_valid = False
        valid_tool_calls: list[dict] = [
            {}
        ]  # It maps `imgidx` to the tool call generating the corresponding image, assuming a query has only 1 image.
        eos = get_special_tokens_dict_or_name("eos")
        bos = get_special_tokens_dict_or_name("bos")
        convs = ("assistant\n" + response).split(f'{eos}{bos}')
        for idx, conv in enumerate(convs):
            if '<|FunctionCallBegin|>' in conv:
                tool_call_str = conv.split("<|FunctionCallBegin|>")[1].split("<|FunctionCallEnd|>")[0]
                if (idx + 1 < len(convs)) and ('[SOI][EOI]' in convs[idx + 1]):
                    last_call_valid = True
                    valid_tool_calls.append(json.loads(tool_call_str)[0])
                else:
                    last_call_valid = False
        if not last_call_valid:
            return VerifyResult(score=0.0, extracted_answer='invalid_tool_call')

        pred_rotate_degree = 0
        cur_tool = valid_tool_calls[-1]
        while cur_tool:
            if cur_tool['name'] != 'ROTATE':
                return VerifyResult(score=0.0, extracted_answer=cur_tool['name'])
            pred_rotate_degree += int(cur_tool['parameters']['degree'])
            imgidx = int(cur_tool['parameters']['imgidx'])
            cur_tool = valid_tool_calls[imgidx]
        pred_rotate_degree %= 360
        assert pred_rotate_degree >= 0

        score = float(pred_rotate_degree == gt_rotate_degree)
        if gt_rotate_degree == 0:
            unnecessary_tool_call_times = len(valid_tool_calls) - 1.0
        else:
            unnecessary_tool_call_times = len(valid_tool_calls) - 2.0
        score = max(0.0, score - unnecessary_tool_call_times * 0.1)
        return VerifyResult(score=score, extracted_answer=str(pred_rotate_degree))
