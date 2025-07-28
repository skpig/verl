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
import uuid
import copy
import numpy as np
from typing import List
from omegaconf import OmegaConf
from unittest.mock import patch

try:
    from swalm.core.task.swe import run_swe_task, SWETaskSpec
    from swalm.core.task.math import run_math_task, MathTaskSpec
    from swalm.core.agent.cline import ClineAgent
    from swalm.core.agent.code_act import CodeActAgent
    from swalm.core.agent.swe_agent import SWEAgentConfig, SWEAgent
    from swalm.core.agent.swalm_math import SwalmMathAgent
    from swalm.core.agent.base import LLMConfig
    from swalm.core.utils.config import get_hydra_config

    enable_swalm_log = os.getenv("ENABLE_SWALM_LOG", False)
    if enable_swalm_log:
        from swalm.core.utils.log import setup_logging
        setup_logging(debug_file=True)
    else:
        logging.getLogger('swalm.core').setLevel(logging.WARNING)
    agent_config = get_hydra_config()

    AGENT_ID_MAPPING_CLASS = {
        "swalm_cline": ClineAgent,
        "swalm_codeact": CodeActAgent,
        "swalm_sweagent": SWEAgent,
        "swalm_math": SwalmMathAgent
    }
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
                                 think_tag_end=None) -> DataProto:
    max_prompt_length = config.prompt_length
    max_new_tokens = prompts.meta_info.get('generation_kwargs').get('max_new_tokens', config.response_length)

    prompt_input_ids = []
    response_input_ids = []

    response_model_output_mask = []

    response_log_probs = []
    response_off_policy = []

    # 检查完成状态和截断状态
    is_finished = True
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

            cur_prompt = data_pack.extra_data[0]['input_ids'][last_prompt_bos_pos:]
            cur_response = data_pack.response_outputs[0]
            conversation_lens.append(len(cur_prompt))
            conversation_lens.append(len(cur_response))

            if (sum(conversation_lens) - conversation_lens[0]
                    >= max_new_tokens) and (not split_conversation_turn_in_train):
                is_right_truncated = True
                return [], is_right_truncated

            response_input_ids.extend((cur_prompt + cur_response))
            response_model_output_mask.extend(([False] * len(cur_prompt) + data_pack.response_model_output_mask[0]))
            response_log_probs.extend(([-1.] * len(cur_prompt) + data_pack.response_log_probs[0]))
            response_off_policy.extend([-1] * len(cur_prompt) + data_pack.this_turn_off_policy_steps[0])
        if split_conversation_turn_in_train:
            single_prompt_input_ids = data_pack.input_ids[0]
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
                                                  pad_token=-1)]).to(torch.bfloat16),
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
                torch.Tensor([_pad_to_max_len(response_log_probs, max_new_tokens, pad_token=-1)]).to(torch.bfloat16),
            'off_policy_steps':
                torch.Tensor([_pad_to_max_len(response_off_policy, max_new_tokens, pad_token=-1)]).to(torch.int8),
            'model_output_mask':
                torch.Tensor([_pad_to_max_len(response_model_output_mask, max_new_tokens,
                                              pad_token=False)]).to(torch.int8),
            'is_finished':
                torch.Tensor([is_finished]).to(torch.int8),
        }
        all_batches.append(batch)

    outs = []
    for batch in all_batches:
        out = DataProto.from_dict(batch)
        data_pack.metrics["off_policy_steps"] = batch['off_policy_steps'].tolist()
        out.meta_info["xperf_metrics"] = data_pack.metrics
        out.meta_info["generation_kwargs"] = prompts.meta_info['generation_kwargs']
        out.non_tensor_batch = copy.deepcopy(prompts.non_tensor_batch)
        out.non_tensor_batch['extra_data'] = np.array(data_pack.extra_data, dtype=object)
        outs.append(out)
    return outs, is_right_truncated


"""
swalm_agent_score:
    1, eval success
    -1, eval failure
    -2, eval error,
    -99, no swalm data
"""


@register_handler("agent/swalm_agent")
class CodeAgent(AsyncAgent):

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        super().__init__(tokenizer, llm, **kwargs)

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        os.environ["no_proxy"] = ""
        config = context.config.actor_rollout_ref.rollout
        tokenizer = context.tokenizer
        host = context.server_host
        port = context.server_port
        if is_ipv6(host):
            host = f'[{host}]'

        agent_info = copy.deepcopy(item.non_tensor_batch["agent_info"][0])
        agent_id = agent_info['agent_id']
        dataset_id = agent_info['dataset_id']
        instance_id = agent_info['instance_id']
        is_eval = agent_info['is_eval']

        env_manager_url = context.config.trainer.get("env_manager_url", 'https://swalm-env.bytedance.net/api/v1')
        meta_info = copy.copy(item.meta_info)
        meta_info['uid'] = item.non_tensor_batch['uid'][0]
        meta_info['reward_model'] = item.non_tensor_batch['reward_model'][0]
        meta_info['agent_info'] = agent_info

        fake_data = copy.deepcopy(item)
        fake_data.batch['swalm_agent_score'] = torch.Tensor([-2]).to(torch.float32)

        task_uuid = meta_info['uid']
        print(f"[Task Start] === uid: {task_uuid}, instance_id: {instance_id}")

        assert agent_id in AGENT_ID_MAPPING_CLASS, f"do not supoort {agent_id} in classes: {AGENT_ID_MAPPING_CLASS.keys()}"
        agent_class = AGENT_ID_MAPPING_CLASS[agent_id]
        agent_think_tag_start = context.config.trainer.get("agent_think_tag_start", "<think>")
        agent_think_tag_end = context.config.trainer.get("agent_think_tag_end", "</think>")
        agent_max_iterations = context.config.trainer.get("agent_max_iterations", 20)
        if agent_id in ["swalm_cline", "swalm_codeact", "swalm_sweagent"]:
            if agent_id == "swalm_cline":
                agent_init_params = {
                    'remove_pattern': rf'{agent_think_tag_start}[\s\S]*?{agent_think_tag_end}',
                    'keep_removed_content': context.config.trainer.get("agent_removed_thinking_content", False),
                }
            elif agent_id == "swalm_codeact":
                agent_init_params = {
                    'remove_pattern':
                        rf'{agent_think_tag_start}[\s\S]*?{agent_think_tag_end}',
                    'keep_removed_content':
                        context.config.trainer.get("agent_removed_thinking_content", False),
                    'observation_truncate_name':
                        context.config.trainer.get("agent_robservation_truncate_name", "openhands_truncate_content"),
                }
            elif agent_id == "swalm_sweagent":
                swe_agent_fccall_type = agent_info.get("fc_call_type", "07_thought_action")
                # TODO: use 07_thought_action by default, will switch to 07_fcalling
                swe_agent_config = SWEAgentConfig(
                    **OmegaConf.to_container(agent_config['agent']['swe_agent'][swe_agent_fccall_type]['agent']))
                agent_init_params = {
                    'agent_config': swe_agent_config,
                }
            agent_run_spec = SWETaskSpec(
                dataset_id=dataset_id,
                instance_id=instance_id,
                agent_class=agent_class,
                llm_config=LLMConfig(client_type='AlphaSeedStreaming',
                                     client_args={
                                         'default_headers': {
                                             'x-tt-logid': task_uuid
                                         },
                                         'tokenizer': tokenizer,
                                     },
                                     request_args={
                                         "url":
                                             f"http://{host}:{port}/chat/completions",
                                         "top_p":
                                             meta_info['generation_kwargs']['top_p'],
                                         "top_k":
                                             meta_info['generation_kwargs']['top_k'],
                                         "temperature":
                                             meta_info['generation_kwargs']['temperature'],
                                         "max_tokens":
                                             context.config.data.get("agent_rollout_max_new_tokens_in_turn",
                                                                     meta_info['generation_kwargs']['max_new_tokens']),
                                         "max_length":
                                             context.config.data.get("agent_rollout_max_lengths_in_turn",
                                                                     config.prompt_length + config.response_length),
                                         "meta_info":
                                             meta_info,
                                         'readTimeout':
                                             context.config.trainer.get("agent_read_timeout", 1500),
                                         'connTimeout':
                                             context.config.trainer.get("agent_conn_timeout", 120),
                                     }),
                agent_init_params=agent_init_params,
                agent_run_params={
                    'max_iterations': agent_max_iterations,
                },
                env_manager_token=None,
                env_manager_url=env_manager_url,
                eval_params={
                    "request_id": task_uuid,
                    "eval_timeout": context.config.trainer.get("agent_eval_timeout", 900),
                    "total_timeout": context.config.trainer.get("agent_total_timeout", 1800),
                    "env_url": env_manager_url
                })
            agent_run_fn = run_swe_task
        elif agent_id in ["swalm_math"]:
            agent_init_params = {
                'remove_pattern': rf'{agent_think_tag_start}[\s\S]*?{agent_think_tag_end}',
                'keep_removed_content': context.config.trainer.get("agent_removed_thinking_content", False),
            }
            agent_run_spec = MathTaskSpec(
                dataset_id=dataset_id,
                instance_id=instance_id,
                agent_class=agent_class,
                agent_server_host_url=f"{os.getenv('RAY_IP', '::')}:{os.getenv('MATH_AGENT_SERVER_PORT', 50726)}",
                prompt=meta_info['agent_info'].pop('messages')[0]['content'],
                ground_truth=meta_info['agent_info'].pop('ground_truth'),
                llm_config=LLMConfig(client_type='AlphaSeedStreaming',
                                     client_args={
                                         'default_headers': {
                                             'x-tt-logid': task_uuid
                                         },
                                         'tokenizer': tokenizer,
                                     },
                                     request_args={
                                         "url":
                                             f"http://{host}:{port}/chat/completions",
                                         "top_p":
                                             meta_info['generation_kwargs']['top_p'],
                                         "top_k":
                                             meta_info['generation_kwargs']['top_k'],
                                         "temperature":
                                             meta_info['generation_kwargs']['temperature'],
                                         "max_tokens":
                                             context.config.data.get("agent_rollout_max_new_tokens_in_turn",
                                                                     meta_info['generation_kwargs']['max_new_tokens']),
                                         "max_length":
                                             context.config.data.get("agent_rollout_max_lengths_in_turn",
                                                                     config.prompt_length + config.response_length),
                                         "meta_info":
                                             meta_info,
                                         "readTimeout":
                                             context.config.trainer.get("agent_read_timeout", 1500),
                                         "connTimeout":
                                             context.config.trainer.get("agent_conn_timeout", 120),
                                     }),
                agent_init_params=agent_init_params,
                agent_run_params={
                    'max_iterations': agent_max_iterations,
                },
                eval_params={})
            agent_run_fn = run_math_task
        else:
            raise NotImplementedError

        all_turns_sum = 0
        try:
            task_res = await agent_run_fn(agent_run_spec)
            outs = []
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
                if is_early_stop:
                    continue
                if not all_turn_pack:
                    continue
                idx_outs, is_right_truncated = pack_to_dataproto_multi_turn(
                    item,
                    tokenizer,
                    all_turn_pack,
                    config,
                    split_conversation_turn_in_train=context.config.trainer.get("split_conversation_turn_in_train",
                                                                                True),
                    think_tag_end=agent_think_tag_end)
                if is_right_truncated and (not is_eval):
                    print(
                        f"Error occurred!!!!!!!!!!!!!!!!!! {task_uuid} merged predicted messages is out of max response length"
                    )
                    print(
                        f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, error: merged predicted messages is out of max response length"
                    )
                    fake_data.meta_info['agent_metrics'] = {
                        "finish_reason": "stop_wtih_response_truncated",
                        "all_turns_sum": all_turns_sum
                    }
                    return fake_data
                for out in idx_outs:
                    if not task_res.eval_result.accepted:
                        out.batch['swalm_agent_score'] = torch.Tensor([-1]).to(torch.float32)
                    else:
                        out.batch['swalm_agent_score'] = torch.Tensor([task_res.eval_result.accepted]).to(torch.float32)
                    outs.append(out)
            swalm_early_stop_train_stratagy = context.config.trainer.get("swalm_early_stop_train_stratagy", "drop_all")
            if is_early_stop and ((swalm_early_stop_train_stratagy == "drop_all") or
                                  ((swalm_early_stop_train_stratagy == "learn_pos") and
                                   (task_res.eval_result.accepted is False))):  # drop_all/learn_pos/learn_all
                print(f"Error occurred!!!!!!!!!!!!!!!!!! {task_uuid} drop early stop trajs")
                print(f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, error: drop early stop trajs")
                fake_data.meta_info['agent_metrics'] = {
                    "finish_reason": "early_stop_drop",
                    "all_turns_sum": all_turns_sum
                }
                return fake_data
            if len(outs) == 0:
                print(f"Error occurred!!!!!!!!!!!!!!!!!! {task_uuid} no valid task response found in outs")
                print(
                    f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, error: no valid task response found in outs"
                )
                fake_data.meta_info['agent_metrics'] = {
                    "finish_reason": "stop_wo_valid_response",
                    "all_turns_sum": all_turns_sum
                }
                return fake_data
            print(f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, success_turns: {all_turns_sum}")
            if is_eval:
                return outs[-1]
            outs = DataProto.concat(outs)
            if all_turns_sum == agent_max_iterations:
                finish_reason = "finish_with_max_iterations"
            else:
                finish_reason = "finish"
            outs.meta_info['agent_metrics'] = {"finish_reason": finish_reason, "all_turns_sum": all_turns_sum}
            return outs
        except Exception as e:
            print(f"Error occurred!!!!!!!!!!!!!!!!!! {task_uuid}", e)
            print(f"[Task Ended] === uid: {task_uuid}, instance_id: {instance_id}, error: {e}")
            fake_data.meta_info['agent_metrics'] = {"finish_reason": "error_stop", "all_turns_sum": all_turns_sum}
            return fake_data
