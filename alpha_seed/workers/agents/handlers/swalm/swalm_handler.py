import torch
import logging

from alpha_seed.workers.streaming_service.streaming_utils import is_ipv6
from alpha_seed.workers.agents.handlers import register_handler
from transformers import PreTrainedTokenizer

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import TaskContext
from alpha_seed.workers.streaming_service.streaming_utils import DataPack
from alpha_seed.workers.agents.handlers.base import AsyncAgent, AsyncLLMInterface

from mono_rl import DataProto
import os
import json
import uuid
import copy
import random
import numpy as np
from typing import List
import importlib
import traceback
from unittest.mock import patch

try:
    from swalm.core.trace import InstanceTracker
    from swalm.core.agent.base import LLMConfig
    from swalm.core.trace.tracer import processor_context as processor_context

    enable_swalm_log = os.getenv("ENABLE_SWALM_LOG", False)
    if enable_swalm_log:
        from swalm.core.utils.log import setup_logging
        setup_logging(debug_file=True)
    else:
        logging.getLogger('swalm.core').setLevel(logging.WARNING)

    enable_swalm_fornax = os.getenv("ENABLE_SWALM_FORNAX", False)
    if enable_swalm_fornax:
        from swalm.core.trace.processors.fornax import FornaxSpanProcessor

    SUPPORTTED_AGNET_TYPES = [
        "swalm.core.agent.cline::ClineAgent", "swalm.core.agent.code_act::CodeActAgent",
        "swalm.core.agent.swe_agent::SWEAgent", "swalm.core.agent.swalm_math::SwalmMathAgent",
        "swalm.core.agent.super_doubao::SuperDoubaoAgent", "swalm.core.agent.mcp::MCPAgent",
        "swalm.core.agent.swe_trae_agent::SWETraeAgent"
    ]

    SWALM_ENV_FAIL_SCORE = -98.
except ImportError as e:
    logging.warning(
        f"Fail to import swalm module, if you want to use swalm agent, please install byted-swalm-core first.")


def _create_tokenized_batch(prompt_ids, response_ids, tokenizer, max_prompt_length, max_new_tokens):
    """创建经过分词和填充的批次数据"""
    # 左填充提示序列
    with patch.object(tokenizer, "padding_side", "left"):
        prompt_output = tokenizer.pad(dict(input_ids=[prompt_ids[-max_prompt_length:]]),
                                      padding="max_length",
                                      max_length=max_prompt_length,
                                      return_tensors="pt")

    # 右填充响应序列
    with patch.object(tokenizer, "padding_side", "right"):
        response_output = tokenizer.pad(dict(input_ids=[response_ids[:max_new_tokens]]),
                                        padding="max_length",
                                        max_length=max_new_tokens,
                                        return_tensors="pt")

    # 提取并转换张量
    prompt_tensor = prompt_output["input_ids"][:, -max_prompt_length:].to(torch.int32)
    prompt_mask = prompt_output["attention_mask"][:, -max_prompt_length:].to(torch.int8)
    response_tensor = response_output["input_ids"][:, :max_new_tokens].to(torch.int32)
    response_mask = response_output["attention_mask"][:, :max_new_tokens].to(torch.int8)

    # 合并输入序列和注意力掩码
    input_ids = torch.hstack((prompt_tensor, response_tensor))
    attention_mask = torch.hstack((prompt_mask, response_mask))

    return input_ids, attention_mask, response_tensor


def pack_to_dataproto_multi_turn(prompts,
                                 tokenizer,
                                 data_packs: List[DataPack],
                                 config,
                                 split_conversation_turn_in_train=True,
                                 use_decouple_critic=False,
                                 ground_truth_patch=None) -> DataProto:
    max_prompt_length = config.prompt_length
    max_new_tokens = prompts.meta_info.get('generation_kwargs').get('max_new_tokens', config.response_length)

    prompt_input_ids = []
    response_input_ids = []

    response_model_output_mask = []

    response_log_probs = []
    response_off_policy = []

    # 检查完成状态和截断状态
    is_finished = True
    is_truncated = False
    is_right_truncated = False

    tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True

    def _pad_to_max_len(cur_list, target_length, pad_token=-1):
        if len(cur_list) > target_length:
            padded_list = cur_list[:target_length]
        else:
            padded_list = cur_list + [pad_token] * (target_length - len(cur_list))
        return padded_list

    all_batches = []
    conversation_lens = []
    for idx, data_pack in enumerate(data_packs):
        if not data_pack.is_finished[0]:
            assert idx == (len(data_packs) - 1), "only the last data pack could be unfinished"
            is_finished = False

        if data_pack.extra_data[0].get('is_truncated', False):
            is_truncated = True

        response_length = len(data_pack.response_outputs[0])
        if idx == 0:
            prompt_input_ids = data_pack.extra_data[0]['input_ids']
            response_input_ids = data_pack.response_outputs[0]
            conversation_lens.append(len(prompt_input_ids))
            conversation_lens.append(len(response_input_ids))

            response_model_output_mask = data_pack.response_model_output_mask[0]
            response_log_probs = data_pack.response_log_probs[0]
            response_off_policy = data_pack.this_turn_off_policy_steps[0]
        else:
            last_prompt_bos_pos = len(data_pack.extra_data[0]['input_ids']) - 1
            bos_count = 0
            while last_prompt_bos_pos >= 0:
                if data_pack.extra_data[0]['input_ids'][last_prompt_bos_pos] == tokenizer.bos_token_id:
                    bos_count += 1
                if bos_count == 2:
                    break
                last_prompt_bos_pos -= 1

            if not is_truncated:
                cur_prompt = data_pack.extra_data[0]['input_ids'][last_prompt_bos_pos:]
                cur_response = data_pack.response_outputs[0]
                conversation_lens.append(len(cur_prompt))
                conversation_lens.append(len(cur_response))

            if (sum(conversation_lens) - conversation_lens[0]
                    >= max_new_tokens) and (not split_conversation_turn_in_train):
                is_right_truncated = True

            if (not is_truncated) and (not is_right_truncated):
                response_input_ids.extend((cur_prompt + cur_response))
                response_model_output_mask.extend(([False] * len(cur_prompt) + data_pack.response_model_output_mask[0]))
                response_log_probs.extend(([1.] * len(cur_prompt) + data_pack.response_log_probs[0]))
                response_off_policy.extend([-1] * len(cur_prompt) + data_pack.this_turn_off_policy_steps[0])
        if (is_truncated) or (split_conversation_turn_in_train):
            single_prompt_input_ids = data_pack.extra_data[0]['input_ids']
            single_response_input_ids = data_pack.response_outputs[0]
            single_input_ids, single_attention_mask, single_response_ids = _create_tokenized_batch(
                single_prompt_input_ids, single_response_input_ids, tokenizer, max_prompt_length, max_new_tokens)

            single_batch = {
                'input_ids':
                    single_input_ids.to(torch.int32),  # here input_ids become the whole sentences
                'attention_mask':
                    single_attention_mask.to(torch.int8),
                'rollout_behavior_log_probs':
                    torch.Tensor([_pad_to_max_len(data_pack.response_log_probs[0], max_new_tokens,
                                                  pad_token=1)]).to(torch.bfloat16),
                'off_policy_steps':
                    torch.Tensor(
                        [_pad_to_max_len(data_pack.this_turn_off_policy_steps[0], max_new_tokens,
                                         pad_token=-1)]).to(torch.int8),
                'model_output_mask':
                    torch.Tensor(
                        [_pad_to_max_len([True] * len(single_response_input_ids), max_new_tokens,
                                         pad_token=False)]).to(torch.int8),
                'is_finished':
                    torch.Tensor([is_finished]).to(torch.int8),
            }
            if use_decouple_critic:
                single_input_ids_critic = single_input_ids
                single_attention_mask_critic = single_attention_mask
                if ground_truth_patch is not None:
                    single_prompt_input_ids_critic = single_prompt_input_ids
                    single_prompt_input_ids_critic.extend(tokenizer.encode(ground_truth_patch))
                    single_input_ids_critic, single_attention_mask_critic, single_response_ids = _create_tokenized_batch(
                        single_prompt_input_ids_critic, single_response_input_ids, tokenizer, max_prompt_length,
                        max_new_tokens)
                single_batch.update({
                    'input_ids_critic': single_input_ids_critic.to(torch.int32),
                    'attention_mask_critic': single_attention_mask_critic.to(torch.int8),
                })
            all_batches.append(single_batch)

    if not split_conversation_turn_in_train:
        input_ids, attention_mask, _ = _create_tokenized_batch(prompt_input_ids, response_input_ids, tokenizer,
                                                               max_prompt_length, max_new_tokens)

        batch = {
            'input_ids':
                input_ids.to(torch.int32),  # here input_ids become the whole sentences
            'attention_mask':
                attention_mask.to(torch.int8),
            'rollout_behavior_log_probs':
                torch.Tensor([_pad_to_max_len(response_log_probs, max_new_tokens, pad_token=1)]).to(torch.bfloat16),
            'off_policy_steps':
                torch.Tensor([_pad_to_max_len(response_off_policy, max_new_tokens, pad_token=-1)]).to(torch.int8),
            'model_output_mask':
                torch.Tensor([_pad_to_max_len(response_model_output_mask, max_new_tokens,
                                              pad_token=False)]).to(torch.int8),
            'is_finished':
                torch.Tensor([is_finished]).to(torch.int8),
        }
        if use_decouple_critic:
            input_ids_critic = input_ids
            attention_mask_critic = attention_mask
            if ground_truth_patch is not None:
                prompt_input_ids_critic = prompt_input_ids
                prompt_input_ids_critic.extend(tokenizer.encode(ground_truth_patch))
                input_ids_critic, attention_mask_critic, response_ids = _create_tokenized_batch(
                    prompt_input_ids_critic, response_input_ids, tokenizer, max_prompt_length, max_new_tokens)
            batch.update({
                'input_ids_critic': input_ids_critic.to(torch.int32),
                'attention_mask_critic': attention_mask_critic.to(torch.int8),
            })
        all_batches.append(batch)

    outs = []
    for batch in all_batches:
        out = DataProto.from_dict(batch)
        data_pack.metrics["off_policy_steps"] = batch['off_policy_steps'].tolist()
        out.meta_info["xperf_metrics"] = data_pack.metrics
        out.meta_info["generation_kwargs"] = prompts.meta_info['generation_kwargs']
        out.non_tensor_batch = copy.deepcopy(prompts.non_tensor_batch)
        if data_pack.extra_data is not None:
            out.non_tensor_batch['extra_data'] = np.array(data_pack.extra_data, dtype=object)
        if data_pack.image_data_ref is not None and any(i is not None for i in data_pack.image_data_ref):
            out.non_tensor_batch['image_data_ref'] = np.fromiter(data_pack.image_data_ref, dtype=object)
        if data_pack.raw_output_ref is not None:
            out.non_tensor_batch['raw_output_ref'] = np.fromiter(data_pack.raw_output_ref, dtype=object)
        outs.append(out)
    return outs, is_truncated, is_right_truncated


"""
swalm_agent_score:
    -1 to 1, eval normally,
    -98, (i.e., SWALM_ENV_FAIL_SCORE), eval error,
    -99 (i.e., NON_AGENT_PLACE_HOLDER_SCORE), not swalm data
"""


@register_handler("agent/swalm_agent")
class SwalmAgent(AsyncAgent):

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        super().__init__(tokenizer, llm, **kwargs)

    def _get_agent_run_spec_args(self, host, port, meta_info, extra_info, task_uuid, tokenizer, dataset_id, instance_id,
                                 is_eval):
        # request_args
        request_args_template = {
            "url":
                f"http://{host}:{port}/chat/completions",
            "top_p":
                meta_info['generation_kwargs']['top_p'],
            "top_k":
                meta_info['generation_kwargs']['top_k'],
            "temperature":
                meta_info['generation_kwargs']['temperature'],
            "max_tokens":
                meta_info['generation_kwargs']['max_new_tokens'],
            "max_length":
                self.config.actor_rollout_ref.rollout.prompt_length +
                self.config.actor_rollout_ref.rollout.response_length,
            "meta_info":
                meta_info,
            'readTimeout':
                1500,
            'connTimeout':
                120,
        }
        request_args = request_args_template.copy()
        if extra_info.get("request_args", None) is not None:
            request_args.update(extra_info.get("request_args"))
        if is_eval:
            if self.config.data.get("agent_rollout_eval_max_new_tokens_in_turn", None):
                request_args['max_tokens'] = self.config.data.get("agent_rollout_eval_max_new_tokens_in_turn")
            if self.config.data.get("agent_rollout_eval_max_lengths_in_turn", None):
                request_args['max_length'] = self.config.data.get("agent_rollout_eval_max_lengths_in_turn")
        else:
            if self.config.data.get("agent_rollout_max_new_tokens_in_turn", None):
                request_args['max_tokens'] = self.config.data.get("agent_rollout_max_new_tokens_in_turn")
            if self.config.data.get("agent_rollout_max_lengths_in_turn", None):
                request_args['max_length'] = self.config.data.get("agent_rollout_max_lengths_in_turn")
        if self.config.trainer.get("agent_read_timeout", None):
            request_args['readTimeout'] = self.config.trainer.get("agent_read_timeout")
        if self.config.trainer.get("agent_conn_timeout", None):
            request_args['connTimeout'] = self.config.trainer.get("agent_conn_timeout")
        request_args['overlong_early_stop'] = self.config.trainer.get("overlong_early_stop", True)

        # agent_init_params
        agent_class = extra_info.get("agent_class")
        assert agent_class in SUPPORTTED_AGNET_TYPES, f"Unsupported task type: {agent_class}"
        if ("ClineAgent" in agent_class) or ("SwalmMathAgent" in agent_class):
            agent_init_params_template = {
                'remove_pattern': r'<think>[\s\S]*?</think>',
                'keep_removed_content': False,
            }
        elif ("CodeActAgent" in agent_class) or ("MCPAgent" in agent_class):
            think_token_type = self.config.trainer.get("think_token_type", "")
            if think_token_type == "code_think":
                remove_pattern = r'<code_think>[\s\S]*?</code_think>'
            elif think_token_type == "36b_think":
                remove_pattern = r'<think_never_used_51bce0c785ca2f68081bfa7d91973934>[\s\S]*?</think_never_used_51bce0c785ca2f68081bfa7d91973934>'
            elif think_token_type == "36b_oss_think":
                remove_pattern = r'<seed:think>[\s\S]*?</seed:think>'
            else:
                remove_pattern = r'<think>[\s\S]*?</think>'
            agent_init_params_template = {
                'remove_pattern': remove_pattern,
                'keep_removed_content': False,
                'observation_truncate_name': "openhands_truncate_content",
            }
        elif "SWEAgent" in agent_class:
            agent_init_params_template = {
                'config_type': "default",
            }
        elif ("SuperDoubaoAgent" in agent_class) or ("SWETraeAgent" in agent_class):
            agent_init_params_template = {}
        else:
            raise NotImplementedError
        agent_init_params = agent_init_params_template.copy()
        if extra_info.get("agent_init_params", None) is not None:
            agent_init_params.update(extra_info.get("agent_init_params"))
        if self.config.trainer.get("agent_keep_removed_thinking_content", None):
            agent_init_params['keep_removed_content'] = self.config.trainer.get("agent_keep_removed_thinking_content")
        if self.config.trainer.get("agent_observation_truncate_name", None):
            agent_init_params['observation_truncate_name'] = self.config.trainer.get("agent_observation_truncate_name")
        if self.config.trainer.get("agent_observation_max_chars", None):
            agent_init_params['observation_truncate_args'] = {
                "max_chars": self.config.trainer.get("observation_max_chars", 5000)
            }
        if self.config.trainer.get("swalm_shell_timeout", None):
            agent_init_params['shell_timeout'] = self.config.trainer.get("shell_timeout")

        # agent_run_params
        agent_run_params_template = {
            'max_iterations': 20,
        }
        agent_run_params = agent_run_params_template.copy()
        if extra_info.get("agent_run_params", None) is not None:
            agent_run_params.update(extra_info.get("agent_run_params"))
        if is_eval:
            if self.config.trainer.get("agent_eval_max_iterations", None):
                agent_run_params['max_iterations'] = self.config.trainer.get("agent_eval_max_iterations")
            elif self.config.trainer.get("agent_max_iterations", None):
                agent_run_params['max_iterations'] = self.config.trainer.get("agent_max_iterations")
        else:
            if self.config.trainer.get("agent_max_iterations", None):
                agent_run_params['max_iterations'] = self.config.trainer.get("agent_max_iterations")

        # eval_params
        if ("ClineAgent" in agent_class) or ("SwalmMathAgent" in agent_class) or ("CodeActAgent" in agent_class) or (
                "MCPAgent" in agent_class) or ("SWETraeAgent" in agent_class):
            eval_params_template = {
                "request_id": task_uuid,
                "eval_timeout": 900,
                "total_timeout": 1800,
                "env_url": 'https://swalm-em.bytedance.net/api/v1',
            }
        elif ("SuperDoubaoAgent" in agent_class):
            eval_params_template = {}
        else:
            raise NotImplementedError
        eval_params = eval_params_template.copy()
        if extra_info.get("eval_params", None) is not None:
            eval_params.update(extra_info.get("eval_params", {}))
        if self.config.trainer.get("agent_eval_timeout", None):
            eval_params['eval_timeout'] = self.config.trainer.get("agent_eval_timeout")
        if self.config.trainer.get("agent_total_timeout", None):
            eval_params['total_timeout'] = self.config.trainer.get("agent_total_timeout")
        if self.config.trainer.get("env_manager_url", None):
            eval_params['env_url'] = self.config.trainer.get("env_manager_url")
        if self.config.trainer.get("agent_eval_return_detail", None):
            eval_params['return_detail'] = self.config.trainer.get("agent_eval_return_detail")

        # agent_run_spec_kwargs
        agent_run_spec_kwargs = {
            "agent_class":
                self._get_swalm_module(agent_class),
            "llm_config":
                LLMConfig(
                    client_type='AlphaSeedStreaming',
                    client_args={
                        'default_headers': {
                            'x-tt-logid': task_uuid
                        },
                        'tokenizer': tokenizer,
                    },
                    request_args=request_args,
                ),
            "agent_init_params":
                agent_init_params,
            "agent_run_params":
                agent_run_params,
            "env_manager_token":
                None,
            "env_manager_url":
                self.config.trainer.get("env_manager_url", 'https://swalm-em.bytedance.net/api/v1'),
            "portal_version":
                self.config.trainer.get("portal_version", 'default'),
            "eval_params":
                eval_params,
        }
        if dataset_id:
            agent_run_spec_kwargs.update({"dataset_id": dataset_id})
        if instance_id:
            agent_run_spec_kwargs.update({"instance_id": instance_id})
        if extra_info.get("prompt", None) is not None:
            agent_run_spec_kwargs.update({"prompt": extra_info["prompt"].tolist()})
        if extra_info.get("reward_model", None) is not None:
            agent_run_spec_kwargs.update({"reward_model": extra_info["reward_model"]})
        if extra_info.get("ground_truth", None) is not None:
            agent_run_spec_kwargs.update({"ground_truth": extra_info["ground_truth"]})
        if self.config.trainer.get("enable_step_level_scores", False) and (not is_eval):
            eval_interval = self.config.trainer.get("step_level_eval_interval", None)
            eval_step_list = []
            if eval_interval is not None:
                eval_step_list = list(range(eval_interval, agent_run_params['max_iterations'], eval_interval))
                agent_run_spec_kwargs.update({"eval_step_list": eval_step_list})
            eval_on_change = self.config.trainer.get("step_level_eval_on_change", None)
            if eval_on_change is not None:
                agent_run_spec_kwargs.update({"eval_on_change": eval_on_change})
        if self.config.trainer.get("raise_on_agent_error", True):
            agent_run_spec_kwargs.update({"raise_on_agent_error": True})
        if extra_info.get("other_run_params", None):
            # hack for tbench
            str_to_json_keys = ["task_config", "test_files", "solution_files"]
            other_run_params = copy.deepcopy(extra_info['other_run_params'])
            for _key in str_to_json_keys:
                if _key in extra_info['other_run_params']:
                    other_run_params[_key] = json.loads(extra_info['other_run_params'][_key])
            agent_run_spec_kwargs.update(other_run_params)
        return agent_run_spec_kwargs

    def _get_step_level_eval_res(self, task_res, agent_max_iterations):
        steps_level_res = [0.] * agent_max_iterations
        if 'step_result' in task_res:
            for k, v in task_res['step_result'].items():
                steps_level_res[int(k) - 1] = float(v.accepted)
        return steps_level_res

    def _get_step_level_format_eval_res(self, task_res, agent_max_iterations):
        format_score_penalty = self.config.trainer.get("format_score_penalty", 0.1)
        steps_level_res = [0.] * agent_max_iterations
        if 'format_result' in task_res:
            for format_key in ["illegal_thinking", 'invalid_fn_call', 'repeat_fn_call', 'no_fn_call']:
                for k, v in enumerate(task_res['format_result'][format_key]):
                    steps_level_res[k] -= float(v) * format_score_penalty
        return steps_level_res

    def _post_process_step_level_eval_res(self, step_level_eval_res, final_score):
        step_level_score_penalty = self.config.trainer.get("step_level_score_penalty", 0)
        if step_level_score_penalty:
            for _idx, _st in enumerate(step_level_eval_res):
                step_level_eval_res[_idx] = -step_level_score_penalty * _st if step_level_score_penalty > 0 else 0.
        step_level_score_process_strategy = self.config.trainer.get("step_level_score_process_strategy", "first")
        true_step_idx = [_idx for _idx, _st in enumerate(step_level_eval_res) if _st > 0]
        new_step_level_eval_res = [0.] * (len(step_level_eval_res))
        is_success_to_fail = False
        if true_step_idx:
            if step_level_score_process_strategy == 'first':
                new_step_level_eval_res[true_step_idx[0]] = step_level_eval_res[true_step_idx[0]]
            elif step_level_score_process_strategy == 'random':
                select_step = random.choice(true_step_idx)
                new_step_level_eval_res[select_step] = step_level_eval_res[select_step]
            elif step_level_score_process_strategy == 'split':
                num_true_step = len(true_step_idx)
                split_step_reward = step_level_eval_res[true_step_idx[0]] / max(num_true_step, 1)
                for select_step in true_step_idx:
                    new_step_level_eval_res[select_step] = split_step_reward
            else:
                raise NotImplementedError
            if final_score == -1.:
                is_success_to_fail = True
        return new_step_level_eval_res, is_success_to_fail

    def _post_process_agent_res(self, task_uuid, instance_id, start_step, all_turns_sum, final_score, is_eval,
                                agent_max_iterations, is_early_stop, is_truncated, is_right_truncated, outs_len):
        swalm_prompt_overlong_train_stratagy = self.config.trainer.get(
            "swalm_prompt_overlong_train_stratagy", "drop_all")  # drop_all/learn_pos/learn_all/reweight
        swalm_response_overlong_train_stratagy = self.config.trainer.get(
            "swalm_response_overlong_train_stratagy", "drop_all")  # drop_all/learn_pos/learn_all/reweight
        swalm_max_turn_train_stratagy = self.config.trainer.get("swalm_max_turn_train_stratagy",
                                                                "learn_all")  # drop_all/learn_pos/learn_all/reweight

        if is_early_stop and (not is_eval) and ((swalm_prompt_overlong_train_stratagy == "drop_all") or
                                                ((swalm_prompt_overlong_train_stratagy == "learn_pos") and
                                                 (final_score == -1))):
            print(f"Stratagy occurred!!!!!!!!!!!!!!!!!! {task_uuid} drop early stop trajs")
            print(f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, straragy: drop early stop trajs")
            finish_reason = "stop_wtih_early_stop_drop"
            return finish_reason, final_score
        if is_truncated and (not is_eval) and ((swalm_prompt_overlong_train_stratagy == "drop_all") or
                                               ((swalm_prompt_overlong_train_stratagy == "learn_pos") and
                                                (final_score == -1))):
            print(f"Stratagy occurred!!!!!!!!!!!!!!!!!! {task_uuid} drop prompt truncated trajs")
            print(
                f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, straragy: drop prompt truncated trajs")
            finish_reason = "stop_wtih_prompt_truncated_drop"
            return finish_reason, final_score
        if is_right_truncated and (not is_eval) and ((swalm_response_overlong_train_stratagy == "drop_all") or
                                                     ((swalm_response_overlong_train_stratagy == "learn_pos") and
                                                      (final_score == -1))):
            print(
                f"Stratagy occurred!!!!!!!!!!!!!!!!!! {task_uuid} merged predicted messages is out of max response length"
            )
            print(
                f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, straragy: merged predicted messages is out of max response length"
            )
            finish_reason = "stop_wtih_response_truncated_drop"
            return finish_reason, final_score
        if (all_turns_sum
                == agent_max_iterations) and (not is_eval) and ((swalm_max_turn_train_stratagy == "drop_all") or
                                                                ((swalm_max_turn_train_stratagy == "learn_pos") and
                                                                 (final_score == -1))):
            print(f"Stratagy occurred!!!!!!!!!!!!!!!!!! {task_uuid} drop over max_turns trajs")
            print(f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, straragy: drop over max_turns trajs")
            finish_reason = "stop_wtih_max_turn_drop"
            return finish_reason, final_score
        if outs_len == 0:
            print(f"Stratagy occurred!!!!!!!!!!!!!!!!!! {task_uuid} no valid task response found in outs")
            print(
                f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, straragy: no valid task response found in outs"
            )
            finish_reason = "stop_with_no_valid_response"
            return finish_reason, final_score
        global_step = self.global_state.get_global_step()
        rollout_offpolicy_step_th = self.config.trainer.get("rollout_agent_offpolicy_step_th", -1)
        if rollout_offpolicy_step_th >= 0 and ((global_step - start_step) > rollout_offpolicy_step_th):
            print(
                f"Stratagy occurred!!!!!!!!!!!!!!!!!! {task_uuid} offpolicy rollout with start step {start_step} and end step {global_step}"
            )
            print(
                f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, straragy: offpolicy rollout with start step {start_step} and end step {global_step}"
            )
            finish_reason = "stop_with_offpolicy_drop"
            return finish_reason, final_score

        if is_early_stop:
            finish_reason = "finish_with_early_stop_learn"
        elif is_truncated:
            finish_reason = "finish_with_prompt_truncated_learn"
        elif is_right_truncated:
            finish_reason = "finish_with_response_truncated_learn"
        elif all_turns_sum == agent_max_iterations:
            finish_reason = "finish_with_max_turn_learn"
        else:
            finish_reason = "finish"

        if (is_early_stop or is_truncated) and (not is_eval) and (swalm_prompt_overlong_train_stratagy == "reweight"):
            pos_weight = self.config.trainer.get("swalm_prompt_overlong_pos_weight", 1.0)
            neg_weight = self.config.trainer.get("swalm_prompt_overlong_neg_weight", 1.0)
            if final_score >= 0:
                final_score = final_score * pos_weight
            else:
                final_score = final_score * neg_weight
        if (is_right_truncated) and (not is_eval) and (swalm_response_overlong_train_stratagy == "reweight"):
            pos_weight = self.config.trainer.get("swalm_response_overlong_pos_weight", 1.0)
            neg_weight = self.config.trainer.get("swalm_response_overlong_neg_weight", 1.0)
            if final_score >= 0:
                final_score = final_score * pos_weight
            else:
                final_score = final_score * neg_weight
        if (all_turns_sum == agent_max_iterations) and (not is_eval) and (swalm_max_turn_train_stratagy == "reweight"):
            overturns_penalty = self.config.trainer.get('swalm_overturns_penalty')
            overturns_start_step = self.config.trainer.get('swalm_overturns_start_step')
            penalty_scale = max(
                0., min((all_turns_sum - overturns_start_step) / (agent_max_iterations - overturns_start_step), 1.0))
            if final_score >= 0:
                final_score -= overturns_penalty * penalty_scale
        return finish_reason, final_score

    def _get_swalm_module(self, module_name):
        module_path, func_name = module_name.split("::", 1)
        module = importlib.import_module(module_path)
        swalm_module = getattr(module, func_name)
        return swalm_module

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        os.environ["no_proxy"] = ""
        tokenizer = context.tokenizer
        cur_step = context.global_step
        host = context.server_host
        port = context.server_port
        if is_ipv6(host):
            host = f'[{host}]'

        extra_info = copy.deepcopy(item.non_tensor_batch["extra_info"][0])
        is_eval = extra_info.get('is_eval', False)
        dataset_id = extra_info.get('dataset_id', '')
        instance_id = extra_info.get('instance_id', '')

        meta_info = copy.copy(item.meta_info)
        meta_info['uid'] = item.non_tensor_batch['uid'][0]
        meta_info['reward_model'] = item.non_tensor_batch['reward_model'][0]

        fake_data = copy.deepcopy(item)
        fake_data.batch['swalm_agent_score'] = torch.Tensor([SWALM_ENV_FAIL_SCORE]).to(torch.float32)
        fake_data.non_tensor_batch['agent_num_turns'] = np.array([SWALM_ENV_FAIL_SCORE])
        fake_data.non_tensor_batch['agent_num_tool_calls'] = np.array([SWALM_ENV_FAIL_SCORE])
        fake_data.meta_info['cur_step'] = cur_step

        task_uuid = meta_info['uid']
        print(f"[Task Start] === uid: {task_uuid}, instance_id: {instance_id}")
        # agent definations
        agent_run_spec = self._get_swalm_module(extra_info.get("task_spec_class"))
        agent_task_type_fn = self._get_swalm_module(extra_info.get("task_type"))
        agent_run_spec_kwargs = self._get_agent_run_spec_args(host, port, meta_info, extra_info, task_uuid, tokenizer,
                                                              dataset_id, instance_id, is_eval)

        agent_max_iterations = agent_run_spec_kwargs.get("agent_run_params", {}).get("max_iterations", 20)
        if self.config.trainer.get("enable_step_level_scores", False):
            fake_data.batch['step_level_scores'] = torch.Tensor([0.] * agent_max_iterations).to(
                torch.float32).unsqueeze(0).repeat(len(fake_data), 1)

        all_turns_sum = 0
        try:
            processors = []
            if enable_swalm_fornax:
                agent_fornax_ak = self.config.trainer.get("agent_fornax_ak", "dd7bcff57e734a9a94dd231e17b90dd3")
                agent_fornax_sk = self.config.trainer.get("agent_fornax_sk", "6b7c2bff23504d60bd010e772fe0c183")
                agent_fornax_space_id = self.config.trainer.get("agent_fornax_space_id", "7524328458281811970")
                if (agent_fornax_ak is not None) and (agent_fornax_sk is not None):
                    processors.append(FornaxSpanProcessor(agent_fornax_ak, agent_fornax_sk))
                else:
                    logging.warning(
                        f"ENABLE_SWALM_FORNAX is true, but without agent_fornax_ak and agent_fornax_sk, thus ignore it."
                    )
            with processor_context(processors):
                agent_run_spec_kwargs.update({"tracker": InstanceTracker(instance_id=task_uuid)})
                task_res = await agent_task_type_fn(agent_run_spec(**agent_run_spec_kwargs))

            if enable_swalm_fornax:
                fornax_urls = []
                for processor in processors:
                    if isinstance(processor, FornaxSpanProcessor):
                        for trace_id in processor.trace_ids:
                            fornax_urls.append(
                                f"https://fornax.bytedance.net/space/{agent_fornax_space_id}/analytics/trace/{trace_id}"
                            )
                        print(
                            f"[Task Fornax] === uid: {task_uuid}, instance_id: {instance_id}, fornax_urls: {fornax_urls}"
                        )

            enable_float_reward_score = self.config.trainer.get("enable_float_reward_score", False)
            if enable_float_reward_score:
                assert hasattr(task_res, "score")
                final_score = task_res.score
            else:
                if task_res.eval_result.accepted:
                    final_score = 1.
                else:
                    final_score = -1.

            is_success_to_fail = False
            if self.config.trainer.get("enable_step_level_scores", False):
                step_level_eval_res = self._get_step_level_eval_res(dict(task_res), agent_max_iterations)
                step_level_eval_res, is_success_to_fail = self._post_process_step_level_eval_res(
                    step_level_eval_res, final_score)
            if self.config.trainer.get("enable_step_level_format_score_penalty", False):
                step_level_split_scores = self._get_step_level_format_eval_res(dict(task_res), agent_max_iterations)

            outs = []
            is_truncated = False
            is_right_truncated = False
            is_early_stop = False
            for idx, completions in enumerate(task_res.trajectories):
                all_turn_pack = []
                for turn in completions:
                    if isinstance(turn['generation_detail']['output'], str) and (turn['generation_detail']['output']
                                                                                 == 'early_stop'):
                        is_early_stop = True
                        break
                    turn_pack = DataPack.create_from_completion_dict(
                        turn['generation_detail']['output']['choices'][0]['message'])
                    all_turns_sum += 1
                    all_turn_pack.append(turn_pack)
                if not all_turn_pack:
                    continue
                idx_outs, is_truncated, is_right_truncated = pack_to_dataproto_multi_turn(
                    item,
                    tokenizer,
                    all_turn_pack,
                    self.config.actor_rollout_ref.rollout,
                    split_conversation_turn_in_train=(is_eval or self.config.trainer.get("split_conversation_turn_in_train", True)),        # eval default use split_conversation_turn_in_train
                    use_decouple_critic=self.config.critic.get("use_decouple_critic", False),
                    ground_truth_patch=extra_info.get("ground_truth_patch", None),
                )
                for out in idx_outs:
                    outs.append(out)
            finish_reason, final_score = self._post_process_agent_res(task_uuid, instance_id, context.global_step,
                                                                      all_turns_sum, final_score, is_eval,
                                                                      agent_max_iterations, is_early_stop, is_truncated,
                                                                      is_right_truncated, len(outs))
            if "finish" not in finish_reason:
                fake_data.meta_info['agent_metrics'] = {"finish_reason": finish_reason, "all_turns_sum": all_turns_sum}
                return fake_data

            print(f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, success_turns: {all_turns_sum}")
            if is_eval:
                final_outs = outs[-1]
                final_outs.batch['swalm_agent_score'] = torch.Tensor([final_score
                                                                     ]).to(torch.float32).repeat(len(final_outs))
                final_outs.non_tensor_batch["extra_info"][0]['all_turns_sum'] = all_turns_sum
                final_outs.non_tensor_batch['agent_num_turns'] = np.array([all_turns_sum])
                final_outs.non_tensor_batch['agent_num_tool_calls'] = np.array([all_turns_sum])
                final_outs.meta_info['agent_metrics'] = {"finish_reason": finish_reason, "all_turns_sum": all_turns_sum}
                final_outs.non_tensor_batch["extra_info"][0]['all_turns_sum'] = all_turns_sum
                if enable_swalm_fornax:
                    final_outs.non_tensor_batch["extra_info"][0]['agent_traj_url'] = "\n".join(fornax_urls)
                return final_outs
            outs = DataProto.concat(outs)
            outs.meta_info['agent_metrics'] = {"finish_reason": finish_reason, "all_turns_sum": all_turns_sum}
            outs.meta_info['cur_step'] = cur_step
            for _idx in range(len(outs)):
                outs.non_tensor_batch["extra_info"][_idx]['all_turns_sum'] = all_turns_sum
                if enable_swalm_fornax:
                    outs.non_tensor_batch["extra_info"][_idx]['agent_traj_url'] = "\n".join(fornax_urls)
            outs.batch['swalm_agent_score'] = torch.Tensor([final_score]).to(torch.float32).repeat(len(outs))
            if self.config.trainer.get("enable_step_level_scores", False):
                step_level_eval_res = step_level_eval_res[:all_turns_sum]
                if len(step_level_eval_res) < agent_max_iterations:
                    step_level_eval_res += ([0.] * (agent_max_iterations - len(step_level_eval_res)))
                outs.batch['step_level_scores'] = torch.Tensor(step_level_eval_res).to(
                    torch.float32).unsqueeze(0).repeat(len(outs), 1)
                outs.non_tensor_batch["extra_info"][0]['is_success_to_fail'] = is_success_to_fail
            if self.config.trainer.get("enable_step_level_format_score_penalty", False):
                step_level_split_scores = step_level_split_scores[:all_turns_sum]
                if len(step_level_split_scores) < agent_max_iterations:
                    step_level_split_scores += ([0.] * (agent_max_iterations - len(step_level_split_scores)))
                outs.batch['step_level_split_scores'] = torch.Tensor(step_level_split_scores).to(
                    torch.float32).unsqueeze(0).repeat(len(outs), 1)
            for _idx in range(len(outs)):
                outs.non_tensor_batch["extra_info"][_idx]['all_turns_sum'] = all_turns_sum
            outs.non_tensor_batch['agent_num_turns'] = np.array([all_turns_sum] * len(outs))
            outs.non_tensor_batch['agent_num_tool_calls'] = np.array([all_turns_sum] * len(outs))
            return outs
        except Exception as e:
            full_traceback_str = traceback.format_exc()
            print(f"Error occurred!!!!!!!!!!!!!!!!!! {task_uuid}\n{full_traceback_str}")
            print(f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, error: {e}")
            fake_data.meta_info['agent_metrics'] = {
                "finish_reason": "stop_with_error_stop",
                "all_turns_sum": all_turns_sum
            }
            return fake_data
