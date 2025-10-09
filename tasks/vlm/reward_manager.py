import math
import json

from alpha_seed.logging import refine_log

refine_log()

import warnings
from datetime import datetime
from multiprocessing import Process
from collections import Counter
# rule-based reward score
from concurrent.futures import as_completed
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from alpha_seed.utils.reward_score.grm_service import RM_INVALID_SCORE
import random
import gc
import os
import time
import traceback
import ray
from verl import DataProto
import torch
import wandb
import pandas as pd

try:
    from bytedance.trainingmetrics.rl_metrics_client_context_manager import \
        RLMetricsClientContextManager as MegavisionMetricsCtx
except ImportError:
    MegavisionMetricsCtx = None

# rule-based reward score
from alpha_seed.utils.reward_score.vlm_verifiers.extra_reward import add_length_reward, punish_format_return_positions, check_general_response_format, is_final_answer_lengthy
from alpha_seed.utils.reward_score import response_post_proc
from alpha_seed.utils.reward_score.utils import Verifier
from alpha_seed.utils.duplicate import para_dup
from alpha_seed.utils.alarm.lark_util import send_message_to_employee
from alpha_seed.utils.tracking_utils import async_save_cases_to_hdfs
from tasks.main_ppo import RewardManager, make_static_omegaconf
import hdfs_io
from hdfs_io import makedirs
try:
    from nltk.util import ngrams
except ImportError:
    ngrams = None
    warnings.warn('nltk not installed, please install nltk. Disable diversity metrics.')

user_email = os.getenv('ARNOLD_LARK_RECEIVER', '')
task_url = os.getenv('ARNOLD_ORIGIN_PLATFORM_URL', '')
ARNOLD_REGION = os.getenv("ARNOLD_REGION", "CN")
ENABLE_REDIS_TRITON_CACHE = int(os.getenv("ENABLE_REDIS_TRITON_CACHE", '1'))


def post_process_solution_str(config, solution_str, eos_token):
    last_eos_idx = solution_str.rfind(eos_token)
    if last_eos_idx >= 0:
        eos_then_assistant = f'{eos_token}<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant\n'
        # When use_remote_verifier=True, the solution_str here hasn't been forcibly appended EOS yet.
        # This if-statement is for avoiding accidentally rsplit the EOS **before** the assistant response:
        if solution_str[last_eos_idx:last_eos_idx + len(eos_then_assistant)] != eos_then_assistant:
            solution_str = solution_str.rsplit(eos_token, 1)[0]  # Remove the EOS **after** the assistant response.

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


def is_divisible_by_0_point_1(score):
    return math.isclose(score * 10, round(score * 10))


class VLMRewardManager(RewardManager):

    def __init__(self, tokenizer, config, logger, rm_name="train", single_batch=False):
        self.tokenizer = tokenizer
        self.logger = logger
        self.log_table = []
        self.rm_name = rm_name
        self.config = config
        self.case_study_dir = config.trainer.default_hdfs_dir + "/cases/"
        # === save_cases_to_hdfs optmization ===
        if self.config.trainer.save_cases_to_hdfs:
            makedirs(self.case_study_dir, exist_ok=True)
        self.async_case_pool = ProcessPoolExecutor(max_workers=8)
        self.async_case_running_tasks = set()
        # === save_cases_to_hdfs optmization ===
        self.rm_req_executor = None
        if not single_batch:
            self.rm_req_executor = ThreadPoolExecutor(
                max_workers=int(self.config.reward_model.get('reward_executor_maxnum', 128)))
        self.mean = self.config.reward_model.mean
        self.std = self.config.reward_model.std
        self.need_punish_duplicate = self.config.reward_model.get('need_punish_duplicate', False)
        self.score_merger = self.config.reward_model.grm.get('score_merger', 'v1')
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
        think_template = self.config.data.think_template if self.config.data.think_template is not None else 'v2'
        os.environ["THINK_TEMPLATE"] = think_template

        self.ability_keys = ["unknown"] + self.config.data.get(
            'ability_list',
            "creation,math,comprehension,rewrite,other,communication,qa,security,code,translation").split(",")
        if self.config.data.get('ability_kl_weights', ''):
            self.ability_kl_weights = [1.0] + [float(t) for t in self.config.data.ability_kl_weights.split(",")]
        else:
            self.ability_kl_weights = [1.0 for _ in range(len(self.ability_keys))]
        assert len(self.ability_kl_weights) == len(self.ability_keys)
        self.ability_dict = dict(map(lambda x: (x[0], x[1]), enumerate(self.ability_keys)))  # {idx: ability_name}

        self.verify_index_set = set()
        self.verify_fusion_rule_dict = defaultdict(set)
        for i, ability_key in self.ability_dict.items():
            verifier_keywords = ["verifier_", "verifiable_", "collie", "instrruler", "sandbox_code"]
            if any([x in ability_key for x in verifier_keywords]):
                self.verify_index_set.add(i)

            if "instrruler" in ability_key:
                self.verify_fusion_rule_dict["instrruler"].add(i)
            elif "function_call" in ability_key:
                self.verify_fusion_rule_dict["function_call"].add(i)
            elif any([
                    key == ability_key
                    for key in ["verifier_vlm_visionarena", "verifier_lmsys_arena", "verifier_lmsys_multi_lang"]
            ]):
                self.verify_fusion_rule_dict["code_switch"].add(i)

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

    def __call__(self, data: DataProto, global_step=None, need_norm=True, is_validation=False, val_only=False):
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

            # Sanity check end of think token
            END_THINK_ID = 17
            if self.config.data.check_template:
                expected_tokens = ['</think>', '</think_never_used_51bce0c785ca2f68081bfa7d91973934>']
                actual_token = self.tokenizer.convert_ids_to_tokens(END_THINK_ID)
                error_msg = f"END_THINK_ID token does not match expected token, got {actual_token}"
                assert actual_token in expected_tokens, error_msg

            valid_response_ids_list = valid_response_ids.tolist()
            if END_THINK_ID in valid_response_ids_list:
                valid_think_length = len(valid_response_ids_list) - 1 - valid_response_ids_list[::-1].index(
                    END_THINK_ID)
            else:
                valid_think_length = 0
            valid_answer_length = valid_response_length - valid_think_length

            # decode
            if 'prompt' in data_item.non_tensor_batch:
                # for vlm self.tokenizer.decode throws OverflowError: out of range integral type conversion attempted,
                # use prompt instead
                prompt_str = data_item.non_tensor_batch['prompt']
            else:
                valid_prompt_ids = valid_prompt_ids[valid_prompt_ids >= 0]
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            # remove potential special tokens(-100)
            valid_response_ids = valid_response_ids[valid_response_ids != -100]
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
            if 'ability_idx' in data_item.batch:
                ab_idx = data_item.batch['ability_idx'].squeeze().item()
                no_thinking_required = data_item.batch["no_thinking_required"].squeeze().item() == 1
            else:
                ab_idx = -1
                no_thinking_required = False
            is_overlong = valid_response_length == self.config.data.max_response_length

            # select rm_score
            reward_style = data_item.non_tensor_batch['reward_model']['style']
            verifier = None
            if reward_style != 'remote_service':
                verifier = Verifier.get_verifier(reward_style, tokenizer=self.tokenizer, config=self.config)
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
                'pause_tokens_index': pause_tokens_index,
                'no_thinking_required': no_thinking_required,
            }
            env_state_bytes = data_item.non_tensor_batch.get('env_states', None)
            if env_state_bytes is not None:
                score_fn_inputs['env_state_bytes'] = env_state_bytes

            extra_data = data_item.non_tensor_batch.get('extra_data', None)
            if isinstance(extra_data, dict) and ((env_state_bytes := extra_data.get('env_states', None)) is not None):
                import base64
                score_fn_inputs['env_state_bytes'] = base64.b64decode(env_state_bytes) if isinstance(
                    env_state_bytes, str) else env_state_bytes

            if format_reward == 0 or is_validation:
                verifier_score = 0
                if verifier is not None:
                    verifier_score = verifier.compute_score_client(**score_fn_inputs)
            else:
                verifier_score = -1

            score_fn_inputs["verifier_score"] = verifier_score
            rm_response, rm_score = "", 0
            waiting_time, rm_time_cost, rm_retry_cnt = 0, 0, 0

            is_valid_format = check_general_response_format(solution_str, no_thinking_required)
            score = verifier_score
            remote_rm_type = data_item.non_tensor_batch['reward_model'].get('rm_required_type', None)
            if (not is_validation) and remote_rm_type is not None and self.rm_name == 'train':
                if verifier is None:
                    verifier_score = -1  # for no verifier case, set verifier_score to -1
                remote_rm_type = self.config.trainer.remote_rm_type
                rm_verifier = Verifier.get_verifier(f"{remote_rm_type}_service",
                                                    tokenizer=self.tokenizer,
                                                    config=self.config)
                rm_response, rm_score, waiting_time, rm_time_cost, rm_retry_cnt = rm_verifier.compute_score_client(
                    data_uid)
                if waiting_time >= 0:  # it means the remote rm call is successful
                    rm_kwargs = {
                        "verify_index_set": self.verify_index_set,
                        "verify_fusion_rule_dict": self.verify_fusion_rule_dict,
                        "ab_idx": ab_idx,
                        "is_valid_format": is_valid_format
                    }
                    score = rm_verifier.merge_vlm_score(verifier_score, rm_score, self.score_merger, **rm_kwargs)

            is_para_dup = para_dup.find_single_turn_duplicate(
                solution_str, enable_resp_para=self.config.reward_model.enable_resp_para)[0]

            if self.config.reward_model.get('need_punish_lengthy_answer', False):
                # Maybe it's not a good coding style to reuse `is_para_dup` here. To be refactored if we have the time.
                is_para_dup = is_para_dup or is_final_answer_lengthy(response_ids=valid_response_ids.tolist(),
                                                                     tokenizer=self.tokenizer)
            is_trunc = (response_length == valid_response_length) and score == -1

            ngram = list(ngrams(valid_response_ids.tolist(), 2)) if ngrams is not None else []

            return_dict = {
                "prompt_str": prompt_str,
                "solution_str": solution_str,
                "ground_truth": ground_truth,
                "reward_style": reward_style,
                "valid_response_length": valid_response_length,
                "verifier_score": verifier_score,
                "score": score,
                "rm_score": rm_score,
                "rm_response": rm_response,
                "rm_wait_time": waiting_time,
                "rm_time_cost": rm_time_cost,
                "rm_retry_cnt": rm_retry_cnt,
                "is_para_dup": is_para_dup,
                "is_trunc": is_trunc,
                "idx": idx,
                "solution_str_post_proc": solution_str_post_proc,
                "ngram": ngram,
                "format_reward": format_reward,
                "pause_tokens_index": pause_tokens_index,
                "thinking_len": thinking_len,
                'global_index': data_item.non_tensor_batch['index'],
                "ability_idx": ab_idx,
                "no_thinking_required": no_thinking_required,
                'valid_think_length': valid_think_length,
                'valid_answer_length': valid_answer_length,
                'valid_prompt_length': valid_prompt_length,
                'is_valid_format': is_valid_format
            }

            return return_dict

        if self.rm_req_executor is None:
            for i in range(len(data)):
                rm_res_future_list.append(get_rm_score(i))
        else:
            for i in range(len(data)):
                rm_res_future_list.append(self.rm_req_executor.submit(get_rm_score, i))
        oj_fail_cnt = 0
        verifier_fail_cnt = 0
        oj_total_cnt = 0
        verifier_total_cnt = 0
        aider_total_cnt = 0
        aider_fail_cnt = 0
        rm_total_cnt = 0
        rm_fail_cnt = 0
        rm_retry_cnt_sum = 0
        rm_score_sum = 0
        rm_wait_time_sum = 0
        rm_time_cost_sum = 0
        dup_cnt = 0
        dup_lens = []
        timeout_cnt = 0
        not_dup_lens = []
        from tqdm import tqdm
        all_ngram = []

        all_raw_scores = []
        counter_raw_scores = []
        all_final_scores = []
        all_final_scores_nondiscretized = []
        counter_final_scores = []
        all_format_scores = []
        counter_format_scores = []
        all_length_rewards = []
        counter_length_rewards = []
        all_overlong_rewards = []
        counter_overlong_rewards = []
        all_thinking_len = []
        all_dup_punish_scores = []

        # For per ability log
        ability_infos = {
            "ability_idx": [],
            "rm_score": [],
            "verifier_score": [],
            "score": [],
            "no_thinking_required": [],
            "valid_think_length": [],
            "valid_answer_length": [],
            "valid_response_length": [],
            "valid_prompt_length": [],
            "is_valid_format": [],
            "indices": [],
        }

        all_final_scores_to_lens = defaultdict(list)
        static_conf = make_static_omegaconf(self.config)
        if self.rm_req_executor is not None:
            generator = tqdm(as_completed(rm_res_future_list), total=len(data), desc="get_rm_score")
        else:
            generator = iter(rm_res_future_list)
        for res in generator:
            if not isinstance(res, dict):
                output_dict = res.result()
            else:
                output_dict = res
            prompt_str = output_dict["prompt_str"]
            solution_str = output_dict["solution_str"]
            ground_truth = output_dict["ground_truth"]
            reward_style = output_dict['reward_style']
            valid_response_length = output_dict['valid_response_length']
            score = output_dict['score']
            score_msg = output_dict['score_msg'] if 'score_msg' in output_dict else 'None'
            verifier_score = output_dict['verifier_score']
            rm_score = output_dict['rm_score']
            rm_response = output_dict['rm_response']
            rm_wait_time = output_dict['rm_wait_time']
            rm_time_cost = output_dict['rm_time_cost']
            rm_retry_cnt = output_dict['rm_retry_cnt']
            is_para_dup = output_dict['is_para_dup']
            is_trunc = output_dict['is_trunc']
            idx = output_dict['idx']
            solution_str_post_proc = output_dict['solution_str_post_proc']
            ngram = output_dict['ngram']
            thinking_len = output_dict['thinking_len']
            pause_tokens_index = output_dict['pause_tokens_index']
            format_reward = output_dict['format_reward']
            global_index = output_dict['global_index']
            ability_idx = output_dict['ability_idx']
            no_thinking_required = output_dict['no_thinking_required']

            all_thinking_len.append(thinking_len)
            all_ngram.extend(ngram)
            if reward_style == "code-sandbox":
                oj_total_cnt += 1
                # 访问失败的score现在设置成-2，用来计数，但是训练的时候还是当做没做对来处理
                if score == -2:
                    score = -1
                    oj_fail_cnt += 1
            if reward_style in ("verifier_service", "verifier_service_volc", "gaokao_verifier_service"):
                verifier_total_cnt += 1
                if score == -2:
                    score = -1
                    verifier_fail_cnt += 1
            if reward_style == "vlm_verifier_router":
                verifier_total_cnt += 1
                if score == -2:
                    verifier_fail_cnt += 1
                    timeout_cnt += 1
                    score = -1
            if reward_style == "aider":
                aider_total_cnt += 1
                if score == -2:
                    score = -1
                    aider_fail_cnt += 1

            # train的时候做这个norm，但是打点的时候恢复，打原始值
            # eval的时候不做这个norm
            if need_norm:
                score = (score - self.mean) / self.std
            if self.config.trainer.use_remote_rm:
                if rm_score == 0:
                    # Skip this sample when rm_score == 0 (no RM service call).
                    pass
                else:
                    rm_total_cnt += 1
                    rm_retry_cnt_sum += rm_retry_cnt
                    rm_wait_time_sum += rm_wait_time
                    rm_time_cost_sum += rm_time_cost
                    if rm_score == RM_INVALID_SCORE:
                        rm_score = 0
                        rm_fail_cnt += 1
                    else:
                        rm_score_sum += rm_score

            raw_scores[idx, valid_response_length - 1] = score
            raw_reward = score
            all_raw_scores.append(score)

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

            ## add score list for logging
            if is_divisible_by_0_point_1(score):
                all_final_scores.append(score)
                all_final_scores_to_lens[score].append(valid_response_length)
            else:
                all_final_scores.append(-10)
                all_final_scores_to_lens[-10].append(valid_response_length)
            all_final_scores_nondiscretized.append(score)  # For correctly monitoring continuous final scores.
            all_format_scores.append(format_reward)
            all_length_rewards.append(length_reward)
            all_overlong_rewards.append(overlong_reward)
            all_dup_punish_scores.append(dup_punish_reward)

            ability_infos["ability_idx"].append(ability_idx)
            ability_infos["no_thinking_required"].append(no_thinking_required)
            ability_infos["rm_score"].append(rm_score)
            ability_infos["verifier_score"].append(verifier_score)
            ability_infos["score"].append(score)
            ability_infos["valid_response_length"].append(valid_response_length)
            ability_infos["valid_think_length"].append(output_dict["valid_think_length"])
            ability_infos["valid_answer_length"].append(output_dict["valid_answer_length"])
            ability_infos["valid_prompt_length"].append(output_dict["valid_prompt_length"])
            ability_infos["is_valid_format"].append(output_dict["is_valid_format"])
            ability_infos["indices"].append(idx)

            if reward_style not in already_print_data_sources:
                already_print_data_sources[reward_style] = 0

            if already_print_data_sources[reward_style] < self.config.trainer.num_cases_to_wandb and not val_only:
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
                self.log_table.append([
                    global_index, global_step, img, prompt_str, solution_str, ground_truth, score, verifier_score,
                    rm_score, rm_response,
                    solution_str_post_proc.split("boxed{")[-1][-80:], is_para_dup, is_trunc, valid_response_length
                ])
            save_to_hdfs.append([
                global_index, idx, global_step, prompt_str, solution_str, ground_truth, score, verifier_score, rm_score,
                rm_response, solution_str_post_proc[-32:], is_para_dup, is_trunc, valid_response_length
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
            prefix + "aider/fail_rate": aider_fail_cnt / aider_total_cnt if aider_total_cnt > 0 else -1,
            prefix + "oj/fail_rate": oj_fail_cnt / oj_total_cnt if oj_total_cnt > 0 else -1,
            prefix + "verifier/fail_rate": verifier_fail_cnt / verifier_total_cnt if verifier_total_cnt > 0 else -1,
            prefix + "dup/para_dup": dup_cnt / len(data),
            prefix + "dup/dup_response_len": sum(dup_lens) / max(1, len(dup_lens)),
            prefix + "dup/not_dup_response_len": sum(not_dup_lens) / max(1, len(not_dup_lens)),
            prefix + 'unique_2gram': len(set(all_ngram)) / (len(all_ngram) + 1),
            prefix + 'current_mean_len': current_mean_len,
            prefix + 'timeout_cnt': timeout_cnt,
        }

        if self.config.trainer.use_remote_rm:
            remote_rm_type = self.config.trainer.remote_rm_type
            log_data.update({
                prefix + f"remote_rm/{remote_rm_type}_call_cnt":
                    rm_total_cnt,
                prefix + f"remote_rm/{remote_rm_type}_fail_rate":
                    rm_fail_cnt / rm_total_cnt if rm_total_cnt > 0 else -1,
                prefix + f"remote_rm/{remote_rm_type}_mean_score":
                    rm_score_sum / (rm_total_cnt - rm_fail_cnt) if rm_total_cnt - rm_fail_cnt > 0 else -1,
                prefix + f"remote_rm/{remote_rm_type}_mean_wait_time":
                    rm_wait_time_sum / rm_total_cnt if rm_total_cnt > 0 else -1,
                prefix + f"remote_rm/{remote_rm_type}_mean_time":
                    rm_time_cost_sum / rm_total_cnt if rm_total_cnt > 0 else -1,
                prefix + f"remote_rm/{remote_rm_type}_mean_retry_cnt":
                    rm_retry_cnt_sum / rm_total_cnt if rm_total_cnt > 0 else -1,
            })

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
            prefix + f"score/raw":
                sum(all_raw_scores) / max(1, len(all_raw_scores)),
            prefix + f"score/format":
                sum(all_format_scores) / max(1, len(all_format_scores)),
            prefix + f"score/length":
                sum(all_length_rewards) / max(1, len(all_length_rewards)),
            prefix + f"score/overlong":
                sum(all_overlong_rewards) / max(1, len(all_overlong_rewards)),
            prefix + f"score/final":
                sum(all_final_scores_nondiscretized) / max(1, len(all_final_scores_nondiscretized)),
            prefix + f"score/dup_punish":
                sum(all_dup_punish_scores) / max(1, len(all_dup_punish_scores)),
        }
        # Ability Log (Only for Train)
        log_score_per_ability = {}
        log_score_overall = {}
        if not is_validation:
            valid_think_format_cnt, valid_non_think_format_cnt = 0, 0
            total_think_cnt, total_non_think_cnt = 0, 0
            for sample_idx in range(len(ability_infos["is_valid_format"])):
                if ability_infos["no_thinking_required"][sample_idx]:
                    if ability_infos["is_valid_format"][sample_idx]:
                        valid_non_think_format_cnt += 1
                    total_non_think_cnt += 1
                else:
                    if ability_infos["is_valid_format"][sample_idx]:
                        valid_think_format_cnt += 1
                    total_think_cnt += 1
            thinking_format_error = (total_think_cnt - valid_think_format_cnt) / (total_think_cnt + 1e-8)
            non_thinking_format_error = (total_non_think_cnt - valid_non_think_format_cnt) / (total_non_think_cnt +
                                                                                              1e-8)
            log_score_overall[prefix + f"ability_infos/thinking_format/thinking_format_error"] = thinking_format_error
            log_score_overall[prefix +
                              f"ability_infos/thinking_format/non_thinking_format_error"] = non_thinking_format_error

            max_response_length = self.config.data.max_response_length
            nothink_max_response_length = max_response_length

            def log_stats(log_dict, prefix, category, tensor_dict):
                """Log mean, max, min for each tensor in tensor_dict under a specific category."""
                for key, (tensor, need_max_min, default_value) in tensor_dict.items():
                    base_key = f"{prefix}{category}/{key}"
                    if len(tensor) > 0:
                        tensor = tensor.float()
                        log_dict[f"{base_key}"] = tensor.mean().item()
                        if need_max_min:
                            base_key_split = base_key.split('/')
                            base_key_prefix, base_key_suffix = '/'.join(base_key_split[:-1]), base_key_split[-1]
                            log_dict[f"{base_key_prefix}/max_{base_key_suffix}"] = tensor.max().item()
                            log_dict[f"{base_key_prefix}/min_{base_key_suffix}"] = tensor.min().item()
                    else:
                        log_dict[f"{base_key}"] = default_value
                        if need_max_min:
                            base_key_split = base_key.split('/')
                            base_key_prefix, base_key_suffix = '/'.join(base_key_split[:-1]), base_key_split[-1]
                            log_dict[f"{base_key_prefix}/max_{base_key_suffix}"] = default_value
                            log_dict[f"{base_key_prefix}/min_{base_key_suffix}"] = default_value

            # Overlong Calculation
            ability_infos["is_overlong"] = []
            for sample_idx in range(len(ability_infos["valid_response_length"])):
                if ability_infos["valid_response_length"][
                        sample_idx] == max_response_length and not ability_infos["no_thinking_required"][sample_idx]:
                    ability_infos["is_overlong"].append(1.0)
                elif ability_infos["valid_response_length"][
                        sample_idx] == nothink_max_response_length and ability_infos["no_thinking_required"][sample_idx]:
                    ability_infos["is_overlong"].append(1.0)
                else:
                    ability_infos["is_overlong"].append(0.0)

            all_log_tensors_per_ability = {}
            for ab_idx, ab_name in self.ability_dict.items():
                ab_mask = torch.LongTensor(ability_infos["ability_idx"]) == ab_idx
                nothink_ab_mask = torch.BoolTensor(ability_infos["no_thinking_required"]) & ab_mask

                all_log_tensors_per_ability[f"score/verifier_score_{ab_name}"] = (torch.tensor(
                    ability_infos["verifier_score"])[ab_mask], True, -1.0)
                all_log_tensors_per_ability[f"score/raw_rm_score_{ab_name}"] = (torch.tensor(
                    ability_infos["rm_score"])[ab_mask], True, 0.0)
                all_log_tensors_per_ability[f"score/rm_score_{ab_name}"] = (torch.tensor(
                    ability_infos["score"])[ab_mask], True, -1.0)
                all_log_tensors_per_ability[f"sample_generator/input_seq_len_{ab_name}"] = (torch.tensor(
                    ability_infos["valid_prompt_length"])[ab_mask], True, 0.0)
                all_log_tensors_per_ability[f"sample_generator/output_think_seq_len_{ab_name}"] = (torch.tensor(
                    ability_infos["valid_think_length"])[ab_mask], False, 0.0)
                all_log_tensors_per_ability[f"sample_generator/output_ans_seq_len_{ab_name}"] = (torch.tensor(
                    ability_infos["valid_answer_length"])[ab_mask], False, 0.0)
                all_log_tensors_per_ability[f"sample_generator/overlong_ratio_{ab_name}"] = (torch.tensor(
                    ability_infos["is_overlong"])[ab_mask], False, 0.0)
                all_log_tensors_per_ability[f"ability/seq_len_{ab_name}"] = (torch.tensor(
                    ability_infos["valid_response_length"])[ab_mask], False, 0.0)

                all_log_tensors_per_ability[f"score/nothink_verifier_score_{ab_name}"] = (torch.tensor(
                    ability_infos["verifier_score"])[nothink_ab_mask], True, -1.0)
                all_log_tensors_per_ability[f"score/nothink_raw_rm_score_{ab_name}"] = (torch.tensor(
                    ability_infos["rm_score"])[nothink_ab_mask], True, 0.0)
                all_log_tensors_per_ability[f"score/nothink_rm_score_{ab_name}"] = (torch.tensor(
                    ability_infos["score"])[nothink_ab_mask], True, -1.0)
                all_log_tensors_per_ability[f"sample_generator/nothink_input_seq_len_{ab_name}"] = (torch.tensor(
                    ability_infos["valid_prompt_length"])[nothink_ab_mask], False, 0.0)
                all_log_tensors_per_ability[f"sample_generator/nothink_overlong_ratio_{ab_name}"] = (torch.tensor(
                    ability_infos["is_overlong"])[nothink_ab_mask], False, 0.0)
                all_log_tensors_per_ability[f"ability/nothink_seq_len_{ab_name}"] = (torch.tensor(
                    ability_infos["valid_response_length"])[nothink_ab_mask], False, 0.0)

                # Log Data Fraction
                ab_frac = ab_mask.sum().item() / len(ab_mask)
                log_score_per_ability[prefix + f"ability_infos/training/{ab_name}_frac"] = ab_frac
            log_stats(log_score_per_ability, prefix, "ability_infos", all_log_tensors_per_ability)

            # Overall status
            all_verifier_score = torch.tensor(ability_infos["verifier_score"])
            all_raw_rm_score = torch.tensor(ability_infos["rm_score"])
            all_rm_score = torch.tensor(ability_infos["score"])
            all_input_len = torch.tensor(ability_infos["valid_prompt_length"])
            all_output_len = torch.tensor(ability_infos["valid_response_length"])
            mean_overlong_ratio = torch.tensor(ability_infos["is_overlong"]).mean().item()

            nothink_mask = torch.BoolTensor(ability_infos["no_thinking_required"])
            nothink_all_verifier_score = torch.tensor(ability_infos["verifier_score"])[nothink_mask]
            nothink_all_raw_rm_score = torch.tensor(ability_infos["rm_score"])[nothink_mask]
            nothink_all_rm_score = torch.tensor(ability_infos["score"])[nothink_mask]
            nothink_all_input_len = torch.tensor(ability_infos["valid_prompt_length"])[nothink_mask]
            nothink_all_output_len = torch.tensor(ability_infos["valid_response_length"])[nothink_mask]
            if nothink_mask.sum() > 0:
                nothink_mean_overlong_ratio = torch.tensor(ability_infos["is_overlong"])[nothink_mask].mean().item()
            else:
                nothink_mean_overlong_ratio = 0.0

            all_log_tensors = {
                "input_seq_len": (all_input_len, True, 0.0),
                "output_seq_len": (all_output_len, True, 0.0),
                "verifier_score": (all_verifier_score, False, -1.0),
                "raw_rm_score": (all_raw_rm_score, False, 0.0),
                "rm_score": (all_rm_score, False, -1.0),
                "nothink_input_seq_len": (nothink_all_input_len, True, 0.0),
                "nothink_output_seq_len": (nothink_all_output_len, True, 0.0),
                "nothink_verifier_score": (nothink_all_verifier_score, False, -1.0),
                "nothink_raw_rm_score": (nothink_all_raw_rm_score, False, 0.0),
                "nothink_rm_score": (nothink_all_rm_score, False, -1.0),
            }
            log_stats(log_score_overall, prefix, "ability_infos/training", all_log_tensors)
            log_score_overall[prefix + f"ability_infos/sample_generator/overlong_ratio_total"] = mean_overlong_ratio
            log_score_overall[
                prefix + f"ability_infos/sample_generator/nothink_overlong_ratio_total"] = nothink_mean_overlong_ratio
            log_score_overall[prefix + f"ability_infos/training/nothink_frac"] = torch.BoolTensor(
                ability_infos["no_thinking_required"]).float().mean().item()

            permutation_indices = torch.tensor(ability_infos["indices"])
            data.batch['raw_rm_score'] = torch.zeros_like(all_raw_rm_score).scatter(0, permutation_indices,
                                                                                    all_raw_rm_score)
            data.batch['raw_verifier_score'] = torch.zeros_like(all_verifier_score).scatter(
                0, permutation_indices, all_verifier_score)

        log_data = {
            **log_data,
            **log_counter,
            **log_score_to_lens,
            **log_score,
            **log_score_per_ability,
            **log_score_overall
        }
        if self.logger is not None:
            self.logger.log(data=log_data, step=global_step)

        if oj_total_cnt > 0 and oj_fail_cnt / oj_total_cnt >= 0.01:
            send_message_to_employee("alpha seed任务oj失败率过高",
                                     f"任务链接: {task_url}, 失败率: {round(oj_fail_cnt / oj_total_cnt * 100.0, 2)}",
                                     user_email)

        if verifier_total_cnt > 0 and verifier_fail_cnt / verifier_total_cnt >= 0.01:
            send_message_to_employee(
                "alpha seed任务verifier失败率过高",
                f"任务链接: {task_url}, 失败率: {round(verifier_fail_cnt / verifier_total_cnt * 100.0, 2)}", user_email)

        if aider_total_cnt > 0 and aider_fail_cnt / aider_total_cnt >= 0.01:
            send_message_to_employee("alpha seed任务aider失败率过高",
                                     f"任务链接: {task_url}, 失败率: {round(aider_fail_cnt / aider_total_cnt * 100.0, 2)}",
                                     user_email)
        if (not is_validation) and (self.config.trainer.remote_rm_type
                                    == "grm") and self.config.trainer.use_remote_rm and self.rm_name == 'train':
            if rm_total_cnt > 0 and rm_fail_cnt / rm_total_cnt >= 0.01:
                send_message_to_employee("alpha seed任务GRM失败率过高",
                                         f"任务链接: {task_url}, 失败率: {round(rm_fail_cnt / rm_total_cnt * 100.0, 2)}",
                                         user_email)

        log_table = None
        if self.config.trainer.num_cases_to_wandb > 0:
            log_table = {
                f"gen&score_{self.rm_name}_{global_step}":
                    wandb.Table(columns=[
                        "Index", "Step", "Image", "Prompt", "Gen Sequence", "GroundTruth", "Score", "Verifier Score",
                        "RM Score", "RM Response", "Gen Sequence PostProc", "Is_Dup", "Is_Trunc", "Len"
                    ],
                                data=self.log_table)
            }
            if (not is_validation and global_step % self.config.trainer.logger_step_interval == 0) or global_step == 1:
                # logger_step = global_step - global_step % self.config.trainer.logger_step_interval
                self.logger.log(log_table, step=global_step, backend='wandb')

        if self.config.trainer.save_cases_to_hdfs:
            print(f"[{time.ctime()}][save cases] reward_fn begin")
            # Clean up finished tasks
            finished = set()
            for task in self.async_case_running_tasks:
                if task.done():
                    try:
                        task.result()  # Collect result and any exceptions
                    except Exception as e:
                        print(f"[save cases] Error in async task: {e}")
                        traceback.print_exc()
                    finished.add(task)
            for task in finished:
                self.async_case_running_tasks.remove(task)
            print(f"[{time.ctime()}][save cases] Remaining async case tasks: {len(self.async_case_running_tasks)}")

            # Prepare data and submit new task
            file_name = f"{self.rm_name}.{str(global_step)}.parquet"
            print(f"[{time.ctime()}][save cases] Creating DataFrame and saving to local file: {file_name}")
            # Create DataFrame
            df = pd.DataFrame(columns=[
                "global_index", "idx", "step", "prompt", "gen", "groundtruth", "score", "verifier_score", "rm_score",
                "rm_response", "score_msg", "gen_postproc", "is_dup", "is_trunc", 'len'
            ],
                              data=save_to_hdfs)
            # Save to local file first
            df.to_parquet(file_name)

            # clear df and save_to_hdfs
            save_to_hdfs = []
            del df
            del save_to_hdfs
            gc.collect()

            # Submit async task
            task = self.async_case_pool.submit(
                async_save_cases_to_hdfs,
                file_name,
                self.case_study_dir,
            )
            self.async_case_running_tasks.add(task)
            print(f"[{time.ctime()}][save cases] reward_fn end")

        if not is_validation:
            return reward_tensor, raw_scores, len_scores, idx_tensor
        elif val_only:
            return reward_tensor, prompt_str, solution_str
        else:
            return reward_tensor, log_table
