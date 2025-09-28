import os
import tempfile
from hdfs_io import hcopy

from transformers import AutoTokenizer
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult

RULE_RM_SCORE_DEFAULT = -100
FLOAT_ROUND = 1e-10


def check_repeat_pattern(think_str, segment_length=20, min_repetitions=10):
    segments = {}
    if len(think_str) < segment_length:
        return False
    for i in range(len(think_str) - segment_length + 1):
        segment = think_str[i:i + segment_length]
        segments[segment] = segments.get(segment, 0) + 1
        if segments[segment] >= min_repetitions:
            return True
    return False


def remove_think_tag(input_resp, tag_start="<think>", tag_end="</think>", use_repeat_punish_rule=False):
    repeat_flag = False
    while True:
        think_start_pos = input_resp.find(tag_start)
        think_end_pos = input_resp.find(tag_end)
        if 0 <= think_start_pos < think_end_pos and think_end_pos >= 0:
            if use_repeat_punish_rule:
                repeat_flag |= check_repeat_pattern(input_resp[think_start_pos + len(tag_start):think_end_pos])
            input_resp = input_resp[:think_start_pos] + input_resp[think_end_pos + len(tag_end):]
        else:
            break
    return input_resp, repeat_flag


class SeedGRMVerifier(BaseVerifier):

    def __init__(self):
        self.hdfs_path = "hdfs://haruna/home/byte_data_seed/ssd_hldy/user/sunzewei.v/tokenizers/bbpe155k-v6.4.3-ml.pret_v5.2_20250519"
        self.tokenizer = None

    def _load_tokenizer_once(self):
        if self.tokenizer is None:
            with tempfile.TemporaryDirectory(prefix="tokenizer_") as temp_dir:
                tokenizer_path = os.path.join(temp_dir, "tokenizer")
                hcopy(self.hdfs_path, tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def verify(self,
               response: str,
               verifier_feature_dict: dict,
               rule_right_length_bonus=False,
               rule_wrong_length_punish=False) -> VerifyResult:
        if response == "":
            raise ExtractAnswerFailed

        self._load_tokenizer_once()

        raw_label = verifier_feature_dict['answer']
        rule_rm_score = RULE_RM_SCORE_DEFAULT
        extracted_rsp_text, check_repeat_flag = remove_think_tag(response)
        if raw_label == 0 or raw_label == 1:
            rule_rm_score = -2.0
            ans_part = extracted_rsp_text.replace(" ", "")
            if (not check_repeat_flag) and ("回答1对比回答2胜出" in ans_part or "回答1对比回答2落败" in ans_part):
                rule_rm_score = -1.0
                if "回答1对比回答2胜出" in ans_part and "回答1对比回答2落败" in ans_part:
                    rule_rm_score = -2.0
                elif "回答1对比回答2胜出" in ans_part and raw_label == 0:
                    rule_rm_score = 1.0
                elif "回答1对比回答2落败" in ans_part and raw_label == 1:
                    rule_rm_score = 1.0

        # Post-processing
        rule_len_threshold = 4000.
        response_length = len(self.tokenizer(response)['input_ids'])
        score = 0
        if abs(rule_rm_score - 1.0) < FLOAT_ROUND:
            if not rule_right_length_bonus:
                score = 0.8
            else:
                score = 0.8 + max(16384. - response_length, 0) * 0.2 / 16384.
        elif abs(rule_rm_score + 1.0) < FLOAT_ROUND:
            if not rule_wrong_length_punish:
                score = 0.0
            else:
                score = -max(response_length - rule_len_threshold, 0) * 0.2 / (16384. - rule_len_threshold)
        elif abs(rule_rm_score + 2.0) < FLOAT_ROUND:
            score = -0.2
        elif abs(rule_rm_score - RULE_RM_SCORE_DEFAULT) < FLOAT_ROUND:
            score = 0.0

        if score >= 0:
            return VerifyResult(score=score, extracted_answer=response)
        else:
            raise ExtractAnswerFailed(message=f"Fail to extract grm response, response: {response}")
