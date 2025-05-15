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
"""
Metrics related to the PPO trainer.
"""

import ray
from collections import defaultdict
import torch
from typing import Any, Dict, List
import numpy as np
from traitlets import default
from verl import DataProto
import re
from langdetect import detect_langs
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance
import multiprocessing


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

@ray.remote
def compute_rollout_metrics(batch: DataProto, tokenizer) -> Dict[str, Any]:
    return dict()
    # rollouts = [(index, tokenizer.decode(token_ids, skip_special_tokens=True)) for index, token_ids in zip(batch.batch['index'], batch.batch['responses'])]
    # # sort by index
    # rollouts = sorted(rollouts, key=lambda x: x[0])
    index2rollout = defaultdict(list)
    for index, token_ids in zip(batch.batch['index'], batch.batch['responses']):
        # decode
        rollout = tokenizer.decode(token_ids, skip_special_tokens=True)
        index2rollout[index.item()].append(rollout)
    rollouts = [r for rollouts in index2rollout.values() for r in rollouts]
    if len(rollouts) == 0:
        raise ValueError("No rollouts found in the batch.")
        return {
            'rollout/language_mixing_count': 0,
            'rollout/language_mixing_ratio': 0,
            'rollout/self_bleu': 0,
            'rollout/edit_distance': 0,
        }
    print(f"avg rollout num: {len(rollouts) / len(index2rollout)}")


    """Frequency of language mixing"""
    def contains_language_mixing(text: str) -> bool:
        try:
            detected_langs = detect_langs(text)
        except Exception:
            return False
        # 只统计概率超过threshold的语言
        languages = {lang.lang for lang in detected_langs if lang.prob > 1e-4}
        return len(languages) > 1

    mix_count = sum(contains_language_mixing(rollout) for rollout in rollouts)
    metrics = {
        'rollout/language_mixing_count': mix_count,
        'rollout/language_mixing_ratio': mix_count / len(rollouts) if rollouts else 0.0,
    }

    """Diversity in all responses to one prompt"""
    @ray.remote
    def compute_self_bleu_and_edit_distance(responses):
        """
        计算一组文本的 Self-BLEU 和编辑距离均值。
        
        :param responses: list[str], 包含多条生成文本
        :return: (avg_self_bleu, avg_edit_distance)
        """
        bleu_scores = []
        edit_distances = []
        # 计算每对文本之间的 Self-BLEU 和编辑距离
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                # 计算编辑距离
                edit_dist = edit_distance(responses[i], responses[j])
                # 计算 Self-BLEU
                # 这里使用 nltk 的 sentence_bleu 函数计算 Self-BLEU
                # 需要将文本分词
                reference = responses[j].split()
                candidate = responses[i].split()
                bleu_score = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)

                bleu_scores.append(bleu_score)
                edit_distances.append(edit_dist)
        # 计算平均值
        avg_self_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_edit_distance = np.mean(edit_distances) if edit_distances else 0.0
        return avg_self_bleu, avg_edit_distance

        

    # bleu_edit_lst = [compute_self_bleu_and_edit_distance(responses) for responses in index2rollout.values()]
    # 使用多进程计算 Self-BLEU 和编辑距离
    bleu_edit_tasks = [compute_self_bleu_and_edit_distance.remote(responses) for responses in index2rollout.values()]
    # 等待所有任务完成
    bleu_edit_results = ray.get(bleu_edit_tasks)
    # 计算平均值
    bleu_edit_lst = [result for result in bleu_edit_results if result is not None]

    metrics.update({
        'rollout/self_bleu': np.mean([bleu for bleu, _ in bleu_edit_lst]),
        'rollout/edit_distance': np.mean([edit for _, edit in bleu_edit_lst]),
    })

    """ TODO: Add more pattern matching metrics here """

    return metrics


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }
