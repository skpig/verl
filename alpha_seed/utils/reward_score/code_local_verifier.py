import json
import re
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from alpha_seed.utils.reward_score.code_local_execution import check_correctness

EXEC_POOL_WORKERS = 32
EXEC_POOL = None

IMPORT_HEADERS = {
    "python": [
        "import math", "import re", "import sys", "import copy", "import datetime", "import itertools",
        "import collections", "import heapq", "import functools", "import hashlib", "import string",
        "from typing import *", "from collections import *", "from functools import *"
    ],
}


def extract_python_code(generation: str):
    generation = generation.replace("[PYTHON]", '```python').replace("[/PYTHON]", '```')
    if '```python' in generation:
        p_code = re.compile(r'```python\n(.*?)\n```', flags=re.DOTALL)
        code_block_matches = p_code.findall(generation)
        return code_block_matches[0] if len(code_block_matches) > 0 else ""
    else:
        codelist = re.split("\ndef|\nclass|\nif|\n#|\nprint", generation)
        return codelist[0]


def compute_score(solution_str, ground_truth, **argv) -> float:
    """
    ground_truth schema:
    {
        'task_id': {name of the task, not necessary for computing the score}
        'lang': {python/cpp...}
        'test_str': {code to verify the results}
        'timeout': {timeout in seconds}
    }
    """
    ground_truth_dict = json.loads(ground_truth)
    task_id = ground_truth_dict["task_id"]
    lang = ground_truth_dict["lang"].lower()
    test_str = ground_truth_dict["test_str"]
    timeout = ground_truth_dict['timeout']

    assert lang == "python", f"only Python support for now, got lang={lang}"
    code = extract_python_code(solution_str)
    test_header = "\n".join(IMPORT_HEADERS["python"])
    test_code = test_header + "\n" + code + "\n" + test_str + "\n"

    sample = {'test_code': test_code}
    args = (task_id, sample, lang, timeout, None, None)
    try:
        result = check_correctness(*args)
        return 1 if result["passed"] else -1
    except Exception as ex:
        print(f"[warn]:{task_id} local exec failed with error: {ex}")
        return -2


def test_compute_score():
    test_str = r'''
assert remove_Occ("hello","l") == "heo"
assert remove_Occ("abcda","a") == "bcd"
assert remove_Occ("PHP","P") == "H"
'''
    prompt = r'''
Write a python function to remove first and last occurrence of a given character from the string.
'''
    output = r'''
'''
    ground_truth = {
        "task_id": "11",
        "test_str": test_str,
        "timeout": 3,
        "lang": "python",
    }
    print(compute_score(solution_str=output, ground_truth=json.dumps(ground_truth)))


def test_compute_score_timeout():
    ground_truth = {"task_id": "test-timeout", "test_str": "", "timeout": 3, "lang": "python"}
    print(
        compute_score(solution_str=r'''
x = 0
while True:
    x += 1
                        ''',
                      ground_truth=json.dumps(ground_truth)))


if __name__ == '__main__':
    test_compute_score()
    test_compute_score_timeout()
