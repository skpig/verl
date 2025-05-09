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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from alpha_seed.logging import refine_log

refine_log()

import time
import warnings
import contextlib
import json
from datetime import datetime
from multiprocessing import Process
from collections import Counter
# rule-based reward score
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import os

import ray
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.utils.fs import copy_local_path_from_hdfs
import torch
import wandb
import pandas as pd
import hdfs_io
try:
    from bytedance.trainingmetrics.rl_metrics_client_context_manager import \
        RLMetricsClientContextManager as MegavisionMetricsCtx
except ImportError:
    MegavisionMetricsCtx = None

# rule-based reward score
from alpha_seed.utils.reward_score.extra_reward import add_length_reward, punish_format_return_positions
from alpha_seed.utils.reward_score import verifier_service, oj_utils, response_post_proc, _select_rm_score_fn
from alpha_seed.utils.duplicate import para_dup
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from alpha_seed.workers.streaming_service.streaming_rollout import RemoteAsyncXPerfGPTRollout
from alpha_seed.workers.actors.critic_worker import CriticWorker
from alpha_seed.utils.alarm.lark_util import send_message_to_employee
from alpha_seed.utils.server_client import validate_client_config, KVStore, ServerHealthCheck, TaskRunner, ClientTaskRunner, check_all_workers_alive, recreate_actor

user_email = os.getenv('ARNOLD_LARK_RECEIVER', '')
task_url = os.getenv('ARNOLD_ORIGIN_PLATFORM_URL', '')
ARNOLD_REGION = os.getenv("ARNOLD_REGION", "CN")
ENABLE_REDIS_TRITON_CACHE = int(os.getenv("ENABLE_REDIS_TRITON_CACHE", '1'))


def post_process_solution_str(config, solution_str, eos_token):
    solution_str = solution_str.rsplit(eos_token, 1)[0]
    if config.reward_model.use_last_response == 'summarize':
        solution_str_post_proc = response_post_proc.summary_postprocess(
            solution_str,
            last_response_sep=config.reward_model.last_response_sep,
            last_response_strict=config.reward_model.last_response_strict)
    elif config.reward_model.use_last_response == 'lastcodeblock':
        solution_str_post_proc = response_post_proc.last_codeblock_postprocess(
            solution_str,
            codeblock_seps=config.reward_model.last_response_sep,
            last_response_strict=config.reward_model.last_response_strict)
    else:
        solution_str_post_proc = solution_str
    return solution_str_post_proc


@ray.remote(num_cpus=1)
class RemoteClient:
    """
    A centralized remote client that pipelines any function with generation at [EOS] 
    """

    def __init__(self, config, tokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.results = {}

        self.call_oj = ray.remote(num_cpus=1)(oj_utils.compute_score)
        self.verifier_service = ray.remote(num_cpus=1)(verifier_service.compute_score)

    def clear(self):
        # for some cases, the results won't be claimed. So we need to clear the results.
        self.results = {}

    def get_num_pending_outputs(self):
        """Return the number of outputs, whose result is not claimed"""
        return len(self.results)

    async def add_requests(self, req_id, input_ids, ground_truth, reward_style):
        solution_str = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        solution_str_post_proc = post_process_solution_str(self.config,
                                                           solution_str,
                                                           eos_token=self.tokenizer.eos_token)

        if reward_style == 'code-sandbox':
            result_future = self.call_oj.remote(solution_str_post_proc, ground_truth,
                                                self.config.trainer.code_sandbox_psm)
        elif reward_style == 'verifier_service':
            result_future = self.verifier_service.remote(solution_str_post_proc, ground_truth,
                                                         self.config.trainer.verifier_service_psm)
        else:
            raise NotImplementedError(f'Unsupported reward_style {reward_style}')

        assert req_id not in self.results, f"{req_id} already exists"
        self.results[req_id] = result_future

    async def get_results(self, req_id):
        if req_id not in self.results:
            return None

        assert req_id in self.results, f"{req_id} not found"
        result_future = self.results.pop(req_id)
        return await result_future


try:
    from nltk.util import ngrams
except ImportError:
    ngrams = None
    warnings.warn('nltk not installed, please install nltk. Disable diversity metrics.')

import math


class RewardManager():

    def __init__(self, tokenizer, config, logger: Tracking, rm_name="train") -> None:
        self.tokenizer = tokenizer
        self.logger = logger
        self.log_table = []
        self.rm_name = rm_name
        self.config = config
        self.case_study_dir = config.trainer.default_hdfs_dir + "/cases/"
        self.rm_req_executor = ThreadPoolExecutor(
            max_workers=int(self.config.reward_model.get('reward_executor_maxnum', 128)))
        self.mean = self.config.reward_model.mean
        self.std = self.config.reward_model.std
        self.need_punish_duplicate = self.config.reward_model.get('need_punish_duplicate', False)
        self.punish_score = self.config.reward_model.get('punish_score', 'rule-lighteval/MATH_v2:-1,code-sandbox:0')
        self.punish_score = dict(map(lambda x: (x.split(':')[0], float(x.split(':')[1])), self.punish_score.split(',')))
        self.need_punish_trunc = self.config.reward_model.get('need_punish_trunc', False)
        self.trunc_punish_score = self.config.reward_model.get('trunc_punish_score', -5)
        self.log_image = self.config.reward_model.get('log_image', True)
        self.len_ema_without_overlong = self.config.reward_model.get(
            'len_ema_without_overlong', False)  # 在计算平均长度时不考虑超长的，这部分反正会被打压（配合trunc_punish_score一起用）
        self.length_ema_method = self.config.reward_model.get('length_ema_method', 'mean')
        assert not self.len_ema_without_overlong or self.need_punish_trunc or self.config.algorithm.mask_overlong or self.config.algorithm.overlong_punish != 'v0', "len_ema_without_overlong is True, so self.need_punish_trunc or mask_overlong must be true."
        self.len_ema_lambda = self.config.reward_model.get('len_ema_lambda', 1)
        self.len_ema = {}
        self.len_ema_json = self.config.reward_model.get('len_ema_json', None)
        if self.len_ema_json is not None and len(self.len_ema) == 0:
            hdfs_io.hcopy(self.len_ema_json, 'len_ema.json')
            with open('len_ema.json') as f:
                self.len_ema = json.load(f)
            for k, v in self.len_ema.items():
                self.len_ema[k] = torch.tensor(v, dtype=torch.float32)

        if self.config.reward_model.add_int_verify:
            warnings.warn(
                "int_verify is deprecated and needs attention. It selects the last integer and judges its correctness, which could lead to unexpected behaviour. Robust verification like \\boxed{} is recommended."
            )

    def update_len_ema(self, data: DataProto):
        index = data.non_tensor_batch['index']
        lengths = data.batch['attention_mask'][:, self.config.data.max_prompt_length:].sum(-1)

        len_lst = {}
        for idx, length in enumerate(lengths):
            if index[idx] not in len_lst:
                len_lst[index[idx]] = []
            if self.len_ema_without_overlong is True:
                if 'max_new_tokens' in data.non_tensor_batch:
                    if length >= data.non_tensor_batch['max_new_tokens'][idx]:
                        continue
                if length >= self.config.data.max_response_length:
                    continue
            len_lst[index[idx]].append(length)

        for idx in len_lst:
            lst = len_lst[idx]
            if not lst:  # 全是超长，用之前的 EMA 或默认值 maxlen
                default_value = torch.tensor(self.config.data.max_response_length)
                # if 'max_new_tokens' in data.non_tensor_batch:
                #     default_value = torch.tensor(data.non_tensor_batch['max_new_tokens'][idx])
                cur_stat = self.len_ema.get(idx, default_value)
            else:
                if self.length_ema_method == 'mean':
                    cur_stat = sum(lst) / len(lst)
                elif self.length_ema_method == 'median':
                    cur_stat = sorted(lst)[len(lst) // 2]
                else:
                    raise ValueError(f"Unknown length_ema_method: {self.length_ema_method}")
            if idx not in self.len_ema:
                self.len_ema[idx] = cur_stat
            else:
                self.len_ema[idx] = (1 - self.len_ema_lambda) * self.len_ema[idx] + self.len_ema_lambda * cur_stat

        mean_len_per_prompt = [self.len_ema[idx].item() for idx in index]
        return mean_len_per_prompt

    def __call__(self, data: DataProto, global_step=None, need_norm=True, is_validation=False):
        """We will expand this function gradually based on the available datasets"""
        response_ids = data.batch['input_ids'][:, self.config.data.max_prompt_length:]

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        raw_scores = torch.zeros_like(response_ids, dtype=torch.float32)
        format_scores = torch.zeros_like(response_ids, dtype=torch.float32)
        len_scores = torch.zeros_like(response_ids, dtype=torch.float32)
        idx_tensor = torch.zeros(response_ids.shape[0], dtype=torch.int64, device=response_ids.device)
        already_print_data_sources = {}
        save_to_hdfs = []
        rm_res_future_list = []
        if global_step is not None and global_step % self.config.trainer.logger_step_interval == 0:
            self.log_table = []  # 清空self.log_table

        mean_len_per_prompt = self.update_len_ema(data)
        current_mean_len = data.batch['attention_mask'][:, self.config.data.max_prompt_length:].sum(
            -1).float().mean().item()

        def get_rm_score(idx):
            """
            只判断correctness的score，其他的score放到外面，方便logging
            """
            data_item = data[idx]  # DataProtoItem

            prompt_ids = data_item.batch['input_ids'][:self.config.data.max_prompt_length]
            response_ids = data_item.batch['input_ids'][self.config.data.max_prompt_length:]

            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum().item()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # the image placeholder in input_ids is negative
            valid_prompt_ids = valid_prompt_ids[valid_prompt_ids >= 0]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            solution_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            solution_str_post_proc = post_process_solution_str(config=self.config,
                                                               solution_str=solution_str,
                                                               eos_token=self.tokenizer.eos_token)

            format_reward = 0  # 默认是0
            pause_tokens_index = None
            thinking_len = 0
            if self.rm_name == 'train':
                if self.config.reward_model.punish_format:
                    format_reward, pause_tokens_index = punish_format_return_positions(
                        solution_str_post_proc, self.config)
                    if pause_tokens_index is not None:
                        thinking_len = len(
                            self.tokenizer(
                                solution_str_post_proc[pause_tokens_index[0]:pause_tokens_index[1]]).input_ids)

            # get prompt uuid
            data_uid = data_item.non_tensor_batch['uid']

            # select rm_score
            reward_style = data_item.non_tensor_batch['reward_model']['style']
            compute_score_fn = _select_rm_score_fn(reward_style)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            score_fn_inputs = {
                "batch_info": data_item.batch,
                "tokenizer": self.tokenizer,
                "solution_str": solution_str_post_proc,
                "ground_truth": ground_truth,
                "config": self.config,
                'data_uid': data_uid,
                "solution_len": valid_response_length,
                "solution_ids": valid_response_ids,
                'rm_name': self.rm_name,
                'pause_tokens_index': pause_tokens_index
            }
            if reward_style == "code-sandbox":
                score_fn_inputs["code_sandbox_psm"] = self.config.trainer.code_sandbox_psm
            if reward_style == "verifier_service":
                score_fn_inputs["verifier_service_psm"] = self.config.trainer.verifier_service_psm
            env_state_bytes = data_item.non_tensor_batch.get('env_states', None)
            if env_state_bytes is not None:
                score_fn_inputs['env_state_bytes'] = env_state_bytes

            if self.config.data.image_key is not None and format_reward != 0:
                score = 0
            else:
                score = compute_score_fn(**score_fn_inputs)

            is_para_dup = para_dup.find_single_turn_duplicate(solution_str)[0]
            is_trunc = (response_length == valid_response_length) and score == -1

            ngram = list(ngrams(valid_response_ids.tolist(), 2)) if ngrams is not None else []

            return_dict = {
                "prompt_str": prompt_str,
                "solution_str": solution_str,
                "ground_truth": ground_truth,
                "reward_style": reward_style,
                "valid_response_length": valid_response_length,
                "score": score,
                "is_para_dup": is_para_dup,
                "is_trunc": is_trunc,
                "idx": idx,
                "solution_str_post_proc": solution_str_post_proc,
                "ngram": ngram,
                "format_reward": format_reward,
                "pause_tokens_index": pause_tokens_index,
                "thinking_len": thinking_len,
                'global_index': data_item.non_tensor_batch['index']
            }

            return return_dict

        for i in range(len(data)):
            rm_res_future_list.append(self.rm_req_executor.submit(get_rm_score, i))
        oj_fail_cnt = 0
        verifier_fail_cnt = 0
        oj_total_cnt = 0
        verifier_total_cnt = 0
        dup_cnt = 0
        dup_lens = []
        timeout_cnt = 0
        not_dup_lens = []
        from tqdm import tqdm
        all_ngram = []

        all_raw_scores = []
        counter_raw_scores = []
        all_final_scores = []
        counter_final_scores = []
        all_format_scores = []
        counter_format_scores = []
        all_length_rewards = []
        counter_length_rewards = []
        all_overlong_rewards = []
        counter_overlong_rewards = []
        all_thinking_len = []
        all_dup_punish_scores = []

        all_final_scores_to_lens = defaultdict(list)
        for res in tqdm(as_completed(rm_res_future_list), total=len(data), desc="get_rm_score"):
            output_dict = res.result()
            prompt_str = output_dict["prompt_str"]
            solution_str = output_dict["solution_str"]
            ground_truth = output_dict["ground_truth"]
            reward_style = output_dict['reward_style']
            valid_response_length = output_dict['valid_response_length']
            score = output_dict['score']
            is_para_dup = output_dict['is_para_dup']
            is_trunc = output_dict['is_trunc']
            idx = output_dict['idx']
            solution_str_post_proc = output_dict['solution_str_post_proc']
            ngram = output_dict['ngram']
            thinking_len = output_dict['thinking_len']
            pause_tokens_index = output_dict['pause_tokens_index']
            format_reward = output_dict['format_reward']
            global_index = output_dict['global_index']

            all_thinking_len.append(thinking_len)

            all_ngram.extend(ngram)
            if reward_style == "code-sandbox":
                oj_total_cnt += 1
                # 访问失败的score现在设置成-2，用来计数，但是训练的时候还是当做没做对来处理
                if score == -2:
                    score = -1
                    oj_fail_cnt += 1
            if reward_style == "verifier_service":
                verifier_total_cnt += 1
                if score == -2:
                    score = -1
                    verifier_fail_cnt += 1
            if reward_style == "verifier_math":
                if score == -2:
                    timeout_cnt += 1
                    score = -0.1
            # train的时候做这个norm，但是打点的时候恢复，打原始值
            # eval的时候不做这个norm
            if need_norm:
                score = (score - self.mean) / self.std
            raw_scores[idx, valid_response_length - 1] = score
            raw_reward = score
            all_raw_scores.append(raw_reward)

            format_scores[idx, valid_response_length - 1] = format_reward

            length_reward, overlong_reward = 0, 0

            # 对score做额外的条件处理，例如length、dup、trunc、format等
            # 没有format punish的时候才加length reward
            if self.rm_name == 'train' and format_reward == 0:
                # length reward有不同版本，by default不加length reward
                thinking_len = valid_response_length if thinking_len == 0 else thinking_len
                length_reward, overlong_reward = add_length_reward(thinking_len,
                                                                   score,
                                                                   self.config,
                                                                   current_mean_len=mean_len_per_prompt[idx])
                score = score + length_reward + overlong_reward
                len_scores[idx, valid_response_length - 1] = score

            dup_punish_reward = 0
            if is_para_dup:
                dup_cnt += 1
                dup_lens.append(valid_response_length)
                if self.need_punish_duplicate and not is_validation:
                    dup_punish_reward = self.punish_score.get(reward_style, -1)
                score += dup_punish_reward
            else:
                not_dup_lens.append(valid_response_length)
            if self.need_punish_trunc and is_trunc and not is_validation:
                score = self.trunc_punish_score
            if format_reward != 0:
                score += format_reward
            reward_tensor[idx, valid_response_length - 1] = score
            idx_tensor[idx] = valid_response_length - 1

            all_final_scores.append(round(score, 1))
            all_final_scores_to_lens[round(score, 1)].append(valid_response_length)

            all_format_scores.append(format_reward)
            all_length_rewards.append(length_reward)
            all_overlong_rewards.append(overlong_reward)
            all_dup_punish_scores.append(dup_punish_reward)

            if reward_style not in already_print_data_sources:
                already_print_data_sources[reward_style] = 0

            if already_print_data_sources[reward_style] < self.config.trainer.num_cases_to_wandb:
                already_print_data_sources[reward_style] += 1
                if reward_style == "code-sandbox":
                    ground_truth = ''  # 对于OJ问题，ground_truth会比较大，扛不住
                if self.log_image:
                    from xperf_gpt.multi_models.preprocess.data_decoder import BytesDecoder
                    if 'raw_image' in data[idx].non_tensor_batch and len(data[idx].non_tensor_batch['raw_image']) > 0:
                        img = wandb.Image(BytesDecoder()(data[idx].non_tensor_batch['raw_image'][0]))
                    else:
                        img = None
                else:
                    img = None
                if self.config.data.image_key is not None:
                    solution_str_save = solution_str_post_proc.split("boxed{")[-1][-80:]
                else:
                    solution_str_save = solution_str_post_proc[-32:]

                self.log_table.append([
                    global_index, global_step, img, prompt_str, solution_str, ground_truth, score, solution_str_save,
                    is_para_dup, is_trunc, valid_response_length
                ])
            save_to_hdfs.append([
                global_index, idx, global_step, prompt_str, solution_str, ground_truth, score, solution_str_save,
                is_para_dup, is_trunc, valid_response_length
            ])

        raw_counter = Counter(counter_raw_scores)
        final_counter = Counter(counter_final_scores)
        format_counter = Counter(counter_format_scores)
        length_counter = Counter(counter_length_rewards)
        overlong_counter = Counter(counter_overlong_rewards)
        dup_punish_counter = Counter(all_dup_punish_scores)

        all_final_scores_to_lens = {key: sum(value) / len(value) for key, value in all_final_scores_to_lens.items()}
        counter = Counter(all_final_scores)
        prefix = "" if not is_validation else "val/"
        log_data = {
            prefix + "oj/fail_rate": oj_fail_cnt / oj_total_cnt if oj_total_cnt > 0 else -1,
            prefix + "verifier/fail_rate": verifier_fail_cnt / verifier_total_cnt if verifier_total_cnt > 0 else -1,
            prefix + "dup/para_dup": dup_cnt / len(data),
            prefix + "dup/dup_response_len": sum(dup_lens) / max(1, len(dup_lens)),
            prefix + "dup/not_dup_response_len": sum(not_dup_lens) / max(1, len(not_dup_lens)),
            prefix + 'unique_2gram': len(set(all_ngram)) / (len(all_ngram) + 1),
            prefix + 'current_mean_len': current_mean_len,
            prefix + 'timeout_cnt': timeout_cnt,
        }
        log_counter = {prefix + f"score_counter/raw_{key}": value for key, value in raw_counter.items()}
        log_counter.update({prefix + f"score_counter/final_{key}": value for key, value in final_counter.items()})
        log_counter.update({prefix + f"score_counter/format_{key}": value for key, value in format_counter.items()})
        log_counter.update({prefix + f"score_counter/length_{key}": value for key, value in length_counter.items()})
        log_counter.update({prefix + f"score_counter/overlong_{key}": value for key, value in overlong_counter.items()})
        log_counter.update({
            prefix + f"score_counter/dup_punish_{key}": value for key, value in dup_punish_counter.items()
        })
        log_counter.update({prefix + f"score_counter/{key}": value for key, value in counter.items()})

        log_score_to_lens = {prefix + f"score_to_lens/{key}": value for key, value in all_final_scores_to_lens.items()}
        log_score = {
            prefix + f"score/raw": sum(all_raw_scores) / max(1, len(all_raw_scores)),
            prefix + f"score/format": sum(all_format_scores) / max(1, len(all_format_scores)),
            prefix + f"score/length": sum(all_length_rewards) / max(1, len(all_length_rewards)),
            prefix + f"score/overlong": sum(all_overlong_rewards) / max(1, len(all_overlong_rewards)),
            prefix + f"score/final": sum(all_final_scores) / max(1, len(all_final_scores)),
            prefix + f"score/dup_punish": sum(all_dup_punish_scores) / max(1, len(all_dup_punish_scores)),
        }
        log_data = {**log_data, **log_counter, **log_score_to_lens, **log_score}
        self.logger.log(data=log_data, step=global_step)

        if oj_total_cnt > 0 and oj_fail_cnt / oj_total_cnt >= 0.01:
            send_message_to_employee("alpha seed任务oj失败率过高",
                                     f"任务链接: {task_url}, 失败率: {round(oj_fail_cnt / oj_total_cnt * 100.0, 2)}",
                                     user_email)

        if verifier_total_cnt > 0 and verifier_fail_cnt / verifier_total_cnt >= 0.01:
            send_message_to_employee(
                "alpha seed任务verifier失败率过高",
                f"任务链接: {task_url}, 失败率: {round(verifier_fail_cnt / verifier_total_cnt * 100.0, 2)}", user_email)
        log_table = None
        if self.config.trainer.num_cases_to_wandb > 0:
            log_table = {
                f"gen&score_{self.rm_name}_{global_step}":
                    wandb.Table(columns=[
                        "Index", "Step", "Image", "Prompt", "Gen Sequence", "GroundTruth", "Score",
                        "Gen Sequence PostProc", "Is_Dup", "Is_Trunc", "Len"
                    ],
                                data=self.log_table)
            }
            if (not is_validation and global_step % self.config.trainer.logger_step_interval == 0) or global_step == 1:
                # logger_step = global_step - global_step % self.config.trainer.logger_step_interval
                self.logger.log(log_table, step=global_step, backend='tracking')

        if self.config.trainer.save_cases_to_hdfs:
            print(f"reward_fn begin hput: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            dir_name = self.case_study_dir
            file_name = f"{self.rm_name}.{str(global_step)}.parquet"

            def async_hput(save_to_hdfs, dir_name, file_name):
                df = pd.DataFrame(columns=[
                    "global_index", "idx", "step", "prompt", "gen", "groundtruth", "score", "gen_postproc", "is_dup",
                    "is_trunc", 'len'
                ],
                                  data=save_to_hdfs)
                df.to_parquet(f"{dir_name}{file_name}")

            p = Process(target=async_hput, args=(save_to_hdfs, dir_name, file_name))
            p.start()
            print(f"reward_fn end hput: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not is_validation:
            return reward_tensor, raw_scores, len_scores, idx_tensor
        else:
            return reward_tensor, log_table


import ray
import hydra
from hydra.core.hydra_config import HydraConfig
import omegaconf
from omegaconf import DictConfig

from alpha_seed.trainer.ppo import RayPPOTrainer


def override(config: DictConfig, overrides: DictConfig, skips: DictConfig, paths=None):
    """
    Override config with overrides.
    """
    paths = [] if paths is None else paths
    for name, value in overrides.items():
        if name not in config:
            config[name] = value
            continue
        if isinstance(value, DictConfig):
            assert isinstance(config[name], DictConfig)
            override(config[name], value, skips.get(name, {}), paths + [name])
            continue
        else:
            assert name in config, f"{config}"
            if name in skips:
                print(
                    f"found {'.'.join(paths+[name])}={skips[name]} specified in program entry, skip overridding it by recipe"
                )
            else:
                config[name] = value


def insert_nested(cfg_dict, key, value):
    """Recursively inserts a value into a nested dictionary based on a dot-separated key."""
    keys = key.split(".")
    for k in keys[:-1]:
        cfg_dict = cfg_dict.setdefault(k, {})
    cfg_dict[keys[-1]] = value


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    metric_collection_context = MegavisionMetricsCtx().collect_init_ray_cluster_duration() \
        if MegavisionMetricsCtx else contextlib.nullcontext()

    if config.recipe:
        skips = {}
        for kv in HydraConfig.get().overrides.task:
            key = kv.split("=")[0]
            insert_nested(skips, key, omegaconf.OmegaConf.select(config, key))
        # auto recipe with runtime profiling
        if config.recipe == "auto":
            from alpha_seed.tuner.auto_tuner import auto_tune_task
            init_ray(config)
            config.recipe = ray.get(
                auto_tune_task.remote(config, config.trainer.n_gpus_per_node, config.trainer.nnodes,
                                      config.recipe_hub))[0]
            print(f"get auto-tuned recipe at {config.recipe}")
        filepath = copy_local_path_from_hdfs(config.recipe, always_recopy=True)
        recipe = omegaconf.OmegaConf.load(filepath)
        print(f"recipe found: {config.recipe}, overriding with config: {recipe}")
        override(config, recipe, skips)

    with metric_collection_context:
        if config.server_client.role == "client":
            init_ray(config)
            config_yaml_dir = os.path.join(os.path.dirname(__file__), "config")
            ref_server_client_common_config = omegaconf.OmegaConf.load(
                os.path.join(config_yaml_dir, "ppo_trainer_server_client_common.yaml"))
            ref_server_config = omegaconf.OmegaConf.load(os.path.join(config_yaml_dir, "ppo_trainer_server.yaml"))
            config = validate_client_config(config, ref_server_config, ref_server_client_common_config)
        else:
            init_ray(config)
            check_arnold_resources(config=config)

    if config.server_client.role == "server":
        main_task(config=config)
    else:
        if config.server_client.role == "client":
            # Use a detached runner to prevent client scripts to run simultaneously
            runner = recreate_actor(ClientTaskRunner, name=ClientTaskRunner.name)
        else:
            runner = TaskRunner.remote()
        ray.get(runner.main.remote(main_task, config=config))


def get_total_gpus_in_ray_cluster():
    total_gpus = 0
    for node in ray.nodes():
        gpus = node['Resources'].get('GPU', 0)
        total_gpus += gpus
    return total_gpus


def wait_till_nodes_ready(total_required_gpus: int, try_time=100):
    while True:
        try:
            available_gpus = ray.available_resources().get('GPU', 0)
            print(f"Checking nodes ready, {available_gpus} GPUs available, {total_required_gpus} GPUs expected")
            if available_gpus >= total_required_gpus:
                break
            time.sleep(15)
            try_time -= 1
        except Exception as e:
            print(f"ray nodes not ready yet, {e}")
            continue


def check_arnold_resources(config):
    """Check the arnold resources before running"""
    num_gpu_nodes = int(os.getenv('ARNOLD_WORKER_NUM', '0'))
    num_gpus_per_node = int(os.getenv('ARNOLD_WORKER_GPU', '0'))

    total_gpus = num_gpu_nodes * num_gpus_per_node
    if total_gpus <= 0:
        # maybe not on arnold environment? skip the check
        return

    total_required_gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node + \
                          config.streaming_rollout.nnodes * config.streaming_rollout.n_gpus_per_node + \
                          config.streaming_validator.nnodes * config.streaming_validator.n_gpus_per_node

    assert total_required_gpus <= total_gpus, f'Require {total_required_gpus} GPUs, but only have {total_gpus} GPUs'

    if total_required_gpus < total_gpus:
        warnings.warn(
            f'The total gpus {total_gpus} is larger than total required GPUs {total_required_gpus}. There might be a waste.'
        )

    # wait for all the GPUs to be ready before training
    wait_till_nodes_ready(total_required_gpus)


def init_ray(config: DictConfig):
    if not ray.is_initialized():
        # this is for local ray cluster
        remote_cache_env = {
            'TRITON_CACHE_MANAGER': 'triton.runtime.cache:RemoteCacheManager',
            'TRITON_REMOTE_CACHE_BACKEND': 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend'
        }
        runtime_env = {
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'BPEX_NO_WARN_ON_UNTUNED_CASE': '1',
                'WANDB_IGNORE_STEP_ORDER': '1',
                "THINK_TEMPLATE": os.getenv("THINK_TEMPLATE", "v2"),
                # 'NCCL_DEBUG': 'WARN'
            }
        }
        if ENABLE_REDIS_TRITON_CACHE:
            runtime_env['env_vars'].update(remote_cache_env)

        is_client = config.server_client.role == "client"
        address = None
        if is_client:
            address = config.server_client.ray_address
            import yaml
            runtime_env_file = os.environ.get("BYTED_RAY_JOB_RUNTIME_PATH", "tasks/runtime_env/runtime_env.yaml")
            with open(runtime_env_file) as fin:
                extra_runtine_env = yaml.safe_load(fin)
                runtime_env.update(extra_runtine_env)

        ray.init(namespace="alphaseed", runtime_env=runtime_env, address=address)


def validate_config(config):
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    # data
    real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.num_bon
    assert real_train_batch_size % n_gpus == 0

    # rollout
    # assert real_train_batch_size % config.actor_rollout_ref.rollout.micro_batch_size == 0
    complete_ratio = config.actor_rollout_ref.rollout.get("complete_ratio", 1.0)
    if config.streaming_rollout.nnodes == 0 and config.rollout_server.nnodes == 0:
        assert complete_ratio == 1.0, f'When streaming rollout (server) is not enabled, complete_ratio must be 1. Got {complete_ratio}'
    else:
        assert complete_ratio < 1.0, f'When streaming rollout (server) is enabled, complete_ratio must be smaller than 1. Got {complete_ratio}.'

    # actor
    assert real_train_batch_size % config.actor_rollout_ref.actor.ppo_mini_batch_size == 0, f"{real_train_batch_size=} vs. {config.actor_rollout_ref.actor.ppo_mini_batch_size=}"
    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        ulysses = config.actor_rollout_ref.actor.ulysses_sequence_parallel_size
        assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
        assert config.actor_rollout_ref.actor.ppo_micro_batch_size * ulysses >= n_gpus
    assert not (config.actor_rollout_ref.actor.scale_pg_by_kl and config.actor_rollout_ref.actor.scale_pg_by_local_kl)

    if config.actor_rollout_ref.actor.kl_loss_weight == 0 and config.algorithm.kl_ctrl.kl_coef == 0:
        assert not config.actor_rollout_ref.actor.scale_pg_by_kl
        config.actor_rollout_ref.ref.fsdp_config.param_offload = True

    # critic
    assert real_train_batch_size % config.critic.ppo_mini_batch_size == 0
    if not config.critic.use_dynamic_bsz:
        ulysses = config.critic.ulysses_sequence_parallel_size
        assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
        assert config.critic.ppo_micro_batch_size * ulysses >= n_gpus

    min_required_seq_len = config.data.max_prompt_length + config.data.max_response_length
    if config.actor_rollout_ref.actor.use_dynamic_bsz:
        if min_required_seq_len > config.actor_rollout_ref.actor.ppo_max_token_len:
            config.actor_rollout_ref.actor.ppo_max_token_len = min_required_seq_len
            print(
                f"Warning: config.actor_rollout_ref.actor.ppo_max_token_len is set to {config.actor_rollout_ref.actor.ppo_max_token_len}"
            )
    if config.actor_rollout_ref.ref.use_dynamic_bsz:
        if min_required_seq_len > config.actor_rollout_ref.ref.max_token_len:
            config.actor_rollout_ref.ref.max_token_len = min_required_seq_len
            print(
                f"Warning: config.actor_rollout_ref.ref.max_token_len is set to {config.actor_rollout_ref.ref.max_token_len}"
            )
    if config.actor_rollout_ref.rollout.use_dynamic_bsz:
        if min_required_seq_len > config.actor_rollout_ref.rollout.max_token_len:
            config.actor_rollout_ref.rollout.max_token_len = min_required_seq_len
            print(
                f"Warning: config.actor_rollout_ref.rollout.max_token_len is set to {config.actor_rollout_ref.rollout.max_token_len}"
            )
    if config.actor_rollout_ref.use_cuda_timer or config.critic.use_cuda_timer or config.reward_model.use_cuda_timer:
        print(
            "Warning: actor_rollout_ref.use_cuda_timer or critic.use_cuda_timer or reward_model.use_cuda_timer is set to True, but not supported yet. "+\
            "Use ALPHASEED_USE_NDTIMELINE=1 instead."
        )

    if config.critic.use_dynamic_bsz:
        if min_required_seq_len > config.critic.ppo_max_token_len:
            config.critic.ppo_max_token_len = min_required_seq_len
            print(f"Warning: config.critic.ppo_max_token_len is set to {config.critic.ppo_max_token_len}")
    if config.reward_model.use_dynamic_bsz:
        if min_required_seq_len > config.reward_model.max_token_len:
            config.reward_model.max_token_len = min_required_seq_len
            print(f"Warning: config.reward_model.max_token_len is set to {config.reward_model.max_token_len}")

    assert config.actor_rollout_ref.actor.strategy in ['fsdp', 'megatron', 'vescale-fsdp2'
                                                      ], f'Got {config.actor_rollout_ref.actor.strategy}'
    assert config.actor_rollout_ref.ref.strategy in ['fsdp', 'megatron',
                                                     'vescale-fsdp2'], f'Got {config.actor_rollout_ref.ref.strategy}'
    assert config.critic.strategy in ['fsdp', 'megatron', 'vescale-fsdp2'], f'Got {config.critic.strategy}'

    # override each role mariana config with global mariana config
    config.actor_rollout_ref.mariana = config.mariana
    config.critic.mariana = config.mariana


def config_to_trainer_kwargs(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer, AutoProcessor

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    if config.data.get('chat_template', None) == 'seed':
        from verl.utils.seed import CHAT_TEMPLATE
        tokenizer.chat_template = CHAT_TEMPLATE
    if config.data.get('chat_template', None) == 'raw':
        raw_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
        tokenizer.chat_template = raw_template

    if config.data.image_key:
        processor = AutoProcessor.from_pretrained(local_path)
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<ImageHere>"]})

    # define worker classes
    if config.actor_rollout_ref.actor.strategy in ['fsdp', 'vescale-fsdp2', 'megatron']:
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    from alpha_seed.trainer.ppo import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRolloutRef: AsyncActorRolloutRefWorker,
        Role.Critic: CriticWorker,
        Role.Rollout: AsyncActorRolloutRefWorker,
        Role.Validator: AsyncActorRolloutRefWorker,
        Role.RolloutServer: RemoteAsyncXPerfGPTRollout
    }

    # in server client, the pool id should follow the format of f"{RoleNameInMerlin}_pool"
    global_pool_id = 'hybrid_pool'
    standalone_pool_id = 'rollout_pool'
    validation_pool_id = 'validator_pool'
    rollout_server_pool_id = 'server_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        standalone_pool_id: [config.streaming_rollout.n_gpus_per_node] * config.streaming_rollout.nnodes,
        validation_pool_id: [config.streaming_validator.n_gpus_per_node] * config.streaming_validator.nnodes,
        rollout_server_pool_id: [config.rollout_server.n_gpus_per_node] * config.rollout_server.nnodes,
    }
    mapping = {
        Role.ActorRolloutRef: global_pool_id,
        Role.Critic: global_pool_id,
        Role.Rollout: standalone_pool_id,
        Role.Validator: validation_pool_id,
        Role.RolloutServer: rollout_server_pool_id
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        from alpha_seed.workers.actors.reward_worker import RewardModelWorker
        role_worker_mapping[Role.RewardModel] = RewardModelWorker
        mapping[Role.RewardModel] = global_pool_id

    server_client_split = config.server_client.role in ["server", "client"]
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec,
                                                mapping=mapping,
                                                server_client_split=server_client_split)

    kwargs = {
        "config": config,
        "role_worker_mapping": role_worker_mapping,
        "resource_pool_manager": resource_pool_manager,
        "ray_worker_group_cls": ray_worker_group_cls,
        "tokenizer": tokenizer,
        "remote_client": None,
    }
    if config.data.image_key:
        kwargs['processor'] = processor

    trainer_config_actor = None
    if config.server_client.role == "server":
        trainer_config_actor = recreate_actor(KVStore, name=KVStore.name)
    elif config.server_client.role == "client":
        trainer_config_actor = ray.get_actor(name=KVStore.name)

    if config.server_client.role == "server":
        trainer_config_actor.set_key_val.remote("server_client", True)
        for k, v in kwargs.items():
            print(f"setting {k}")
            trainer_config_actor.set_key_val.remote(k, v)
        logger = Tracking(project_name=config.trainer.project_name,
                          experiment_name=config.trainer.experiment_name,
                          default_backend=['console'],
                          config=OmegaConf.to_container(config, resolve=True))
    else:
        # the following parameters are only used in fit() and _validate() so skip them in server_only mode
        logger = Tracking(project_name=config.trainer.project_name,
                          experiment_name=config.trainer.experiment_name,
                          default_backend=config.trainer.logger,
                          config=OmegaConf.to_container(config, resolve=True))

        reward_fn = RewardManager(tokenizer=tokenizer, config=config, logger=logger, rm_name="train")
        # Note that we always use function-based RM for validation
        val_reward_fn = RewardManager(tokenizer=tokenizer, config=config, logger=logger, rm_name="val")

        # we will always start a remote client
        kwargs['remote_client'] = RemoteClient.options(name='remote_client').remote(config=config, tokenizer=tokenizer)

        kwargs['tokenizer'] = tokenizer
        kwargs['logger'] = logger
        kwargs['reward_fn'] = reward_fn
        kwargs['val_reward_fn'] = val_reward_fn

    return kwargs, trainer_config_actor


def check_all_workers_alive(workers):
    from ray.experimental.state.api import get_actor
    for worker in workers:
        worker_state_dict = get_actor(worker._actor_id.hex())
        if worker_state_dict is None:
            return False
        if worker_state_dict.get("state", "undefined") != "ALIVE":
            return False
    return True


def main_task(config):

    metric_collection_context = MegavisionMetricsCtx().collect_setup_trainer_duration() \
        if MegavisionMetricsCtx else contextlib.nullcontext()

    with metric_collection_context:
        validate_config(config)
        trainer_kwargs, kv_store = config_to_trainer_kwargs(config)
        trainer = RayPPOTrainer(**trainer_kwargs)

    global_step, resume_folder = trainer.get_resume_checkpoint_info()

    metric_collection_context = MegavisionMetricsCtx().collect_init_worker_duration() \
        if MegavisionMetricsCtx else contextlib.nullcontext()
    with metric_collection_context:
        trainer.init_workers(kv_store, from_step=global_step, resume_folder=resume_folder)

    if config.server_client.role == "server":
        send_message_to_employee("alpha seed server启动", f"任务链接: {task_url}", user_email)
        health_check = recreate_actor(ServerHealthCheck, name=ServerHealthCheck.name)
        print("============== server started ==============")
        while True:
            if not check_all_workers_alive(trainer.workers):
                ray.get(health_check.set_ready.remote(ready=False))
                raise RuntimeError(f"found worker dead, exiting")
            ray.get(health_check.set_ready.remote(ready=True))
            time.sleep(60 * 1)
    elif config.convert_ckpt_to_omnistore_task.enable:
        trainer.convert_ckpt_to_omnistore()
        send_message_to_employee("alpha seed任务转换ckpt到omnistore完成，任务结束", f"任务链接: {task_url}", user_email)
    else:
        if config.server_client.role == "client":
            server_health_check = ray.get_actor(ServerHealthCheck.name)
            assert ray.get(server_health_check.is_ready.remote()) == True, "server not ready"

        send_message_to_employee("alpha seed任务开始训练", f"任务链接: {task_url}", user_email)
        trainer.fit()


if __name__ == '__main__':
    main()
