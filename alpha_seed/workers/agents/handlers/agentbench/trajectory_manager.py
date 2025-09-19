import os
import copy
import logging
import torch

import verl.utils.torch_functional as verl_F

from collections import defaultdict
from typing import List
from mono_rl import DataProto

from hdfs_io import hput

from alpha_seed.workers.agents.handlers.agentbench.proxy import Task

COMMON_TURN_KEYS = [
    "1_1_1_file_level_request_id", "1_1_2_file_level_irrelevant_request_id", "1_2_1_related_elements_request_id",
    '3_1_2_select_regression_request_id'
]
COMMON_TURN_LIST_KEYS = ['3_2_1_reproduction_test_samples_verified_list']


def get_turn_scores(request_id_model, score, extra, assign_test_reward):
    turn_scores = {}
    score = score if score == 1 else -1
    patch_score_max = score

    if extra is None or extra == {}:
        return turn_scores
    if not int(os.environ.get('AGENTLESS_EVALUATE_ALL_PATCHES', '0')):
        return turn_scores

    _n = int(os.environ['AGENTLESS_LOC_NUM_SAMPLES'])
    _m = int(os.environ['AGENTLESS_REPAIR_MAX_SAMPLES'])
    infer_error = False
    if _n != len(extra['1_3_1_edit_location_samples_request_id_list']):
        infer_error = True
    if _n * _m != len(extra['2_1_1_repair_sample_request_id_list']):
        infer_error = True
    if len(extra.get('2_1_1_repair_sample_score_list', [])) != len(extra['2_1_1_repair_sample_request_id_list']):
        infer_error = True

    if infer_error:
        return turn_scores

    patch_score_list = []
    for _patch_idx, (patch_score, repair_request_id) in enumerate(
            zip(extra['2_1_1_repair_sample_score_list'], extra['2_1_1_repair_sample_request_id_list'])):
        _locate_idx = _patch_idx // _m
        locate_request_id = extra['1_3_1_edit_location_samples_request_id_list'][_locate_idx]
        if patch_score <= 0:
            patch_score = -1
        if locate_request_id not in turn_scores:
            turn_scores[locate_request_id] = patch_score
        else:
            turn_scores[locate_request_id] = max(turn_scores[locate_request_id], patch_score)
        turn_scores[repair_request_id] = patch_score
        patch_score_max = max(patch_score_max, patch_score)
        patch_score_list.append(patch_score)

    random_select_score = (sum(patch_score_list) / len(patch_score_list)) if len(patch_score_list) > 0 else -1
    test_benefit = score - random_select_score
    turn_scores['test_benefit'] = test_benefit
    if assign_test_reward:
        if score != patch_score_max:
            test_reward = -1
            turn_scores['test_status'] = 'fail'
        else:
            if max(patch_score_list) == min(patch_score_list):  # all 0 or all 1
                test_reward = 0
                turn_scores['test_status'] = 'all{}'.format(max(patch_score_list))
            else:
                test_reward = 1
                turn_scores['test_status'] = 'success'
        assert len(
            extra['3_2_1_reproduction_test_samples_request_id_list']
        ) == 1, f"assign_test_reward only support one turn 7 {extra['3_2_1_reproduction_test_samples_request_id_list']}"
        for _request_id in extra['3_2_1_reproduction_test_samples_request_id_list'] + [
                extra['3_1_2_select_regression_request_id']
        ]:
            turn_scores[_request_id] = test_reward

    common_turn_ids = set()
    common_turn_ids |= set([extra[key] for key in COMMON_TURN_KEYS if key in extra])
    for key in COMMON_TURN_LIST_KEYS:
        if extra.get(key):
            common_turn_ids |= set(extra[key])
    for common_id in common_turn_ids:
        if common_id not in turn_scores:
            turn_scores[common_id] = patch_score_max

    return turn_scores


def agentless_training_samples(build_record_fn, task: Task, score, trajectory, config, **kwargs):
    fix_score = score if score == 1 else -1
    turn_scores = get_turn_scores(task.task_id, score, task.result.extra, config.get('assign_test_reward', False))
    train_samples = []
    for item in trajectory:
        turn_score = turn_scores.get(item.task_id, score if score == 1 else -1)
        dp = copy.deepcopy(item.response.payload)
        record = build_record_fn(dp,
                                 turn_score,
                                 original_agentbench_score=score,
                                 num_turns=len(trajectory),
                                 num_tool_calls=len(trajectory))
        train_samples.append(record)

    #  if train_samples:
    #  local_path = f'/opt/tiger/train_samples_{task.task_id}.pkl'
    #  torch.save(train_samples, local_path)
    #  hput(local_path, f'hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/jiangyun.jy/datasets/tmp/')
    #  logging.info(f"save train_samples to {local_path} and hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/jiangyun.jy/datasets/tmp/")

    return train_samples


def _create_tokenized_batch(prompt_ids, response_ids, max_prompt_length, max_response_length, pad_token_id, **kwargs):
    is_prompt_truncated = len(prompt_ids) > max_prompt_length
    is_response_truncated = len(response_ids) > max_response_length

    prompt_trunc = prompt_ids[-max_prompt_length:].unsqueeze(0)  # left truncation
    prompt_mask = torch.ones_like(prompt_trunc)
    prompt_output = verl_F.pad_sequence_to_length(prompt_trunc, max_prompt_length, pad_token_id, left_pad=True)
    prompt_mask = verl_F.pad_sequence_to_length(prompt_mask, max_prompt_length, 0, left_pad=True)

    response_trunc = torch.tensor([response_ids[:max_response_length]])  # right truncation
    response_mask = torch.ones_like(response_trunc)
    response_output = verl_F.pad_sequence_to_length(response_trunc, max_response_length, pad_token_id, left_pad=False)
    response_mask = verl_F.pad_sequence_to_length(response_mask, max_response_length, 0, left_pad=False)

    prompt_tensor = prompt_output.to(torch.int32)
    prompt_mask = prompt_mask.to(torch.int8)
    response_tensor = response_output.to(torch.int32)
    response_mask = response_mask.to(torch.int8)

    assert response_mask[0][
        0] == 1, f"[{kwargs.get(task_id, 'unkonwn_task')}] response_mask[0][0] must be 1, but {response_mask[0]}, \
        response length: {len(response_ids)}, response: {kwargs.get('tokenizer').decode(response_ids) if 'tokenizer' in kwargs else 'tokenizer not found'}"

    input_ids = torch.hstack((prompt_tensor, response_tensor))
    attention_mask = torch.hstack((prompt_mask, response_mask))

    return input_ids, attention_mask, is_prompt_truncated, is_response_truncated


def _pad_to_max_len(cur_list, target_length, pad_token=-1):
    if len(cur_list) > target_length:
        padded_list = cur_list[:target_length]
    else:
        padded_list = cur_list + [pad_token] * (target_length - len(cur_list))
    return padded_list


def openhands_training_samples(build_record_fn, task: Task, score, trajectory, config, **kwargs):
    tokenizer = kwargs.get('tokenizer')
    max_prompt_length = kwargs.get('max_prompt_length')
    max_response_length = kwargs.get('max_response_length')
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    assistant_idx = [bos_token_id] + tokenizer.encode('assistant')

    def find_eosid_after_next_to_last_assistant(prompt_ids):
        prompt_ids = prompt_ids.tolist()
        start_idx = -1
        vis_last = False
        for i in range(len(prompt_ids) - 1, -1, -1):
            if prompt_ids[i:i + len(assistant_idx)] == assistant_idx:
                start_idx = i
                if vis_last:
                    break
                vis_last = True
        if start_idx == -1:
            return -1
        for i in range(start_idx + len(assistant_idx), len(prompt_ids)):
            if prompt_ids[i] == eos_token_id:
                return i
        return -1

    if len(trajectory) == 0:
        print(f"ATTENTION: agentbench openhands_training_samples - trajectory is empty, task_id = {task.task_id}")
        return []
    original_agentbench_score = score
    score = score if score == 1 else -1
    train_samples = [item.response.payload for item in trajectory if not item.response.aborted]
    traj_samples = defaultdict(list)
    for sample in train_samples:
        tid = str(sample.non_tensor_batch['__AGENTBENCH_traj_id'][0])
        traj_samples[tid].append(sample)
    print(f"[task {task.task_id}] {traj_samples.keys()=}")

    out_samples = []
    for traj_id, samples in traj_samples.items():
        if 'condense' in traj_id:  # skip condense request
            continue
        _data_packs = [item.non_tensor_batch['__AGENTBENCH_data_pack'][0] for item in samples]
        _input_ids = [item.non_tensor_batch['__AGENTBENCH_input_ids'][0] for item in samples]
        first_sample = copy.deepcopy(samples[0])
        first_datapack = copy.deepcopy(_data_packs[0])
        response_length = first_sample.batch['rollout_behavior_log_probs'].shape[-1]
        real_prompt_length = first_sample.batch['input_ids'][0].shape[-1] - response_length

        prompt_ids = first_sample.batch['input_ids'][0][:real_prompt_length]
        response_ids = first_datapack.response_outputs[0]
        response_model_output_mask = first_datapack.response_model_output_mask[0]
        response_log_probs = first_datapack.response_log_probs[0]
        response_off_policy = first_datapack.this_turn_off_policy_steps[0]
        is_finished = first_datapack.is_finished[0]

        for message in zip(_data_packs[1:], _input_ids[1:]):
            data_pack = message[0]
            input_ids = message[1]
            last_response_idx = find_eosid_after_next_to_last_assistant(input_ids)
            cur_prompt = input_ids[last_response_idx + 1:].tolist()  # input_ids has no padding
            cur_response = data_pack.response_outputs[0]

            response_ids.extend(cur_prompt + cur_response)
            assert sum(data_pack.response_model_output_mask[0]) == len(
                cur_response), f"[{task.task_id}] sum(data_pack.response_model_output_mask[0]) != len(cur_response), \
                sum(data_pack.response_model_output_mask[0]) = {sum(data_pack.response_model_output_mask[0])}, len(cur_response) = {len(cur_response)}"

            response_model_output_mask.extend([0] * len(cur_prompt) + data_pack.response_model_output_mask[0])
            response_log_probs.extend([1] * len(cur_prompt) + data_pack.response_log_probs[0])
            response_off_policy.extend([-1] * len(cur_prompt) + data_pack.this_turn_off_policy_steps[0])
            is_finished = data_pack.is_finished[0]

        input_ids, attention_mask, is_prompt_truncated, is_response_truncated = _create_tokenized_batch(
            prompt_ids,
            response_ids,
            max_prompt_length,
            max_response_length,
            pad_token_id,
            task_id=task.task_id,
            tokenizer=tokenizer)

        if config.get('drop_truncated') and (is_response_truncated or is_prompt_truncated):
            print(
                f"ATTENTION: agentbench openhands_training_samples - response or prompt is truncated, task_id = {task.task_id}, {traj_id=}, {is_prompt_truncated=}, {is_response_truncated=}"
            )
            return []

        batch = {
            'input_ids':
                input_ids.to(torch.int32),
            'attention_mask':
                attention_mask.to(torch.int8),
            'model_output_mask':
                torch.Tensor([_pad_to_max_len(response_model_output_mask, max_response_length)]).to(torch.int8),
            'rollout_behavior_log_probs':
                torch.Tensor([_pad_to_max_len(response_log_probs, max_response_length)]).to(torch.bfloat16),
            'off_policy_steps':
                torch.Tensor([_pad_to_max_len(response_off_policy, max_response_length)]).to(torch.int8),
            'is_finished':
                torch.Tensor([is_finished]).to(torch.int8),
        }
        out = DataProto.from_dict(batch)
        dp = copy.deepcopy(first_sample)
        dp.meta_info["xperf_metrics"] = {
            **first_datapack.metrics,
            **{
                "off_policy_steps": batch['off_policy_steps'].tolist()
            }
        }
        dp.pop(batch_keys=batch.keys())
        dp.union(out)
        record = build_record_fn(dp,
                                 score,
                                 origin_agentbench_score=original_agentbench_score,
                                 num_turns=len(trajectory),
                                 num_tool_calls=len(trajectory),
                                 prompt_truncated=is_prompt_truncated,
                                 response_truncated=is_response_truncated,
                                 agent_interact_turn=len(trajectory))
        out_samples.append(record)

    return out_samples


BUILD_FN = {
    'agentless': agentless_training_samples,
    'openhands': openhands_training_samples,
}


def build_training_samples(build_record_fn, task: Task, score, trajectory, config, **kwargs):
    build_fn = BUILD_FN[task.task_args.framework]
    train_samples = build_fn(build_record_fn, task, score, trajectory, config, **kwargs)
    return train_samples
