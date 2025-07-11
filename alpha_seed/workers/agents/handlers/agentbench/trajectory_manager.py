import os
import logging
import torch

from typing import List
from mono_rl import DataProto
from unittest.mock import patch

from hdfs_io import hput
from alpha_seed.utils.dataset.rl_dataset import collate_fn

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


def agentless_training_samples(task: Task, score, trajectory, config, context):
    fix_score = score if score == 1 else -1
    turn_scores = get_turn_scores(task.task_id, score, task.result.extra, config.get('assign_test_reward', False))
    train_samples = []
    for item in trajectory:
        turn_score = turn_scores.get(item.task_id, fix_score)
        score_dp = DataProto.from_single_dict(
            collate_fn([{
                'original_agentbench_score': score,
                'agentbench_score': torch.Tensor([turn_score]).to(torch.bfloat16),
            }]))
        train_samples.append(item.response.payload.union(score_dp))

    #  if train_samples:
    #  local_path = f'/opt/tiger/train_samples_{task.task_id}.pkl'
    #  torch.save(train_samples, local_path)
    #  hput(local_path, f'hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/jiangyun.jy/datasets/tmp/')
    #  logging.info(f"save train_samples to {local_path} and hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/jiangyun.jy/datasets/tmp/")

    return train_samples


BUILD_FN = {
    'agentless': agentless_training_samples,
}


def build_training_samples(task: Task, score, trajectory, config, context):
    build_fn = BUILD_FN[task.task_args.framework]
    train_samples = build_fn(task, score, trajectory, config, context)
    return train_samples
