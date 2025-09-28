# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from word2number import w2n

from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, ExtractAnswerFailed
from alpha_seed.utils.reward_score.vlm_verifiers.tools import remove_bold, get_base_precision_recall


def text_normalization(text):
    text = remove_bold(text).lower()
    text = re.sub(r'\s+', ' ', text)
    words = []
    for word in text.split(" "):
        if word in w2n.american_number_system:
            word = str(w2n.word_to_num(word))
        words.append(word)
    return ' '.join(words).strip()


def extract_number(text: str):
    text = text_normalization(text)
    answer = re.findall(r'\\boxed\{(\d+)\}', text)
    if len(answer):
        return int(answer[-1])
    answer = re.findall(r'^(\d+)', text)
    if len(answer):
        return int(answer[0])

    # has English prefix
    answer = re.findall(r"answer(\ is|:|\ is:)\ ?(\d+)", text, re.DOTALL)
    if len(answer):
        return int(answer[0][-1])

    # has Chinese prefix
    answer = re.findall(r"(回答|答案)(是|：|是：)\ ?(\d+)", text, re.DOTALL)
    if len(answer):
        return int(answer[0][-1])

    # finally, try to get the first number
    answer = re.findall(r"\d+", answer)
    if len(answer):
        return int(answer[0])
    return


class VideoGroundingCountingVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict, is_validation: bool = False) -> VerifyResult:
        gt_number = verifier_feature_dict['answer']

        try:
            eot_token = get_special_tokens_dict_or_name("think_end_token")
            eos_token = get_special_tokens_dict_or_name("eos")
            final_answer: str = response.split(eot_token)[-1].split(eos_token)[0]
            pred_number = extract_number(final_answer)
            assert isinstance(pred_number, int)
        except:
            raise ExtractAnswerFailed(f"Failed to extract predict number from: {response}")

        detailed_answer = verifier_feature_dict['detailed_answer']
        gt_timestamps: list[tuple[float, float]] = list()
        for x in detailed_answer:
            gt_timestamps.append((x["start_time"], x["end_time"]))

        timestamp_pattern = re.compile(r'"timestamps":\s*"([^"]*)"')
        timestamp_contents = re.findall(timestamp_pattern, response)
        pred_timestamps: list[tuple[float, float]] = list()
        for timestamps in timestamp_contents:
            try:
                timestamps = timestamps.split(',')
                timestamps = [m.strip() for m in timestamps]
                for timestamp in timestamps:
                    pattern = r'\d+(?:\.\d+)?'
                    timestamp = re.findall(pattern, timestamp)
                    timestamp = [float(m) for m in timestamp]
                    start_sec, end_sec = timestamp
                    if start_sec > end_sec:
                        continue
                    pred_timestamps.append((start_sec, end_sec))
            except:
                continue

        grounding_score = 0.0
        if len(pred_timestamps) > 0:
            pr_metrics = get_base_precision_recall(gt_timestamps, pred_timestamps)
            p = pr_metrics['precision']
            r = pr_metrics['recall']
            k = 1.5
            if p + r > 0:
                p_focused = p**k
                r_focused = r**k
                if p_focused + r_focused > 0:
                    grounding_score = 2 * (p_focused * r_focused) / (p_focused + r_focused)

        if gt_number == pred_number:
            score = 1.0
        else:
            score = 0.0
        sre = 1 - abs(gt_number - pred_number) / (gt_number + pred_number)

        if not is_validation:
            if len(pred_timestamps) > 0:
                final_score = 0.5 * (grounding_score * sre)**0.5 + 0.5 * score
            else:
                final_score = score
        else:
            final_score = score
        return VerifyResult(score=final_score, extracted_answer=pred_number)
