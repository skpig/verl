import json
from collections import defaultdict


def dummy_verify(solution_str, answer, meta):
    return -1


registried_tasks = {}

import os
import importlib

try:
    folder_path = '/opt/tiger/verifiable_tasks/verifiable_tasks/tasks'
    for task in os.listdir(folder_path):
        try:
            if os.path.isdir(os.path.join(folder_path, task)):
                module = importlib.import_module(f'verifiable_tasks.tasks.{task}.verifier')
                registried_tasks[task] = module.verify
        except:
            continue
except Exception:
    print('Unable to find verifiable_tasks')


def compute_score(solution_str, ground_truth, **argv) -> float:
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    meta = ground_truth["meta"]
    answer = ground_truth["answer"]
    task_name = ground_truth["task_name"]
    try:
        verify_fn = registried_tasks[task_name]
        score = verify_fn(solution_str, answer, meta)
        return score * 2 - 1  # 0, 1 -> -1, 1
    except Exception as ex:
        return dummy_verify(solution_str, answer, meta)
