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
import torch
from .utils import Verifier


def extract_rm_score(batch_info, rm_scores):
    prompt_length = batch_info['prompts'].shape[-1]
    eos_mask_idx = torch.clamp(torch.sum(batch_info['attention_mask'][prompt_length:]) - 1, min=0)
    return rm_scores[eos_mask_idx]


class RawScore(Verifier, reward_style="model-raw_score"):

    @staticmethod
    def compute_score(*args, **kwargs) -> float:
        return raw_score(*args, **kwargs)


def raw_score(batch_info, **argv):
    return extract_rm_score(batch_info, batch_info['rm_scores']).item()


def count_subsequences(sequence, subsequence):
    count = 0
    sub_len = len(subsequence)
    i = 0
    while i < len(sequence) - sub_len + 1:
        if sequence[i] != subsequence[0]:
            i += 1
            continue
        elif sequence[i + 1] != subsequence[1]:
            i += 2
            continue
        elif sequence[i + 2] != subsequence[2]:
            i += 3
            continue
        elif sequence[i + 3] != subsequence[3]:
            i += 4
            continue
        else:
            i += 4
            count += 1
    return count


class RawScoreReflectionPenalty(Verifier, reward_style="model-raw_score_reflection_penalty"):

    def compute_score(self, *args, **kwargs) -> float:
        return raw_score_reflection_penalty(*args, **kwargs)


def raw_score_reflection_penalty(batch_info, tokenizer, **argv):
    reflection_start_token = tokenizer.encode('<reflection>', add_special_tokens=False)
    reflection_end_token = tokenizer.encode('</reflection>', add_special_tokens=False)
    response_ids = batch_info["responses"].tolist()
    rm_score = extract_rm_score(batch_info, batch_info['rm_scores'])
    refl_start_count = count_subsequences(response_ids, reflection_start_token)
    refl_end_count = count_subsequences(response_ids, reflection_end_token)
    refl_count_i = min(refl_start_count, refl_end_count)

    if rm_score >= 1.5 and refl_count_i > 1:
        new_score = rm_score + 1 + 0.05 * (refl_count_i - 2)
    elif rm_score < 0 and refl_count_i <= 1:
        new_score = rm_score - (2 - refl_count_i)
    else:
        new_score = rm_score
    new_score = torch.tensor(new_score).to(dtype=rm_score.dtype, device=rm_score.device)
    return new_score.item()
