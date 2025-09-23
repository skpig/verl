# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# copied from https://github.com/facebookresearch/swe-rl/blob/main/src/swerl/core/reward.py

import difflib
import re
import os
import json
import copy
import random
import time
import warnings
import functools
import requests
import pandas as pd
from typing import TypedDict

from unidiff import PatchedFile, PatchSet
from unidiff.errors import UnidiffParseError
from alpha_seed.utils.reward_score.tcc_v3 import TccV3
from .utils import Verifier

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<solution>"
ANSWER_END = "</solution>"

SEARCH_REPLACE_REGEX = r"```.*?\n### (.*)\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE\n```"

cfg = TccV3().get_agent_config()
registered_bench_hosts = cfg.get("swe", {}).get("bench_hosts", [])
registered_bench_repo2hosts = cfg.get("swe", {}).get("bench_repo2hosts", {})


class SWERepairVerifier(Verifier, reward_style="swe_repair_verifier"):

    def is_remote(self):
        return self.config.trainer.use_remote_swe_sandbox

    @staticmethod
    def compute_score(solution_str, ground_truth) -> float:
        return compute_score(solution_str, ground_truth)


def retry(max_retries=3, retry_delay=2):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.RequestException as e:
                    exception_message = f"{e}"
                    print(f"调用方法[{func.__name__}]时出现异常: {exception_message}")

                    retries += 1
                    if retries <= max_retries:
                        wait_time = retry_delay * retries
                        print(f"进行第{retries}次重试，等待{wait_time}秒...")
                        time.sleep(wait_time)
                    else:
                        print("达到最大重试次数，放弃重试。")
                        raise

                except Exception as e:
                    exception_message = f"{e}"
                    print(f"调用方法[{func.__name__}]时出现异常: {exception_message}")
                    raise

        return wrapper

    return decorator


class SWERemoteEnvClient:

    def __init__(self):
        random.seed(time.time_ns())

        self.bench_operation_long_timeout = int(os.getenv("SWE_REMOTE_BENCH_OPERATION_LONG_TIMEOUT", "1200"))
        self.bench_hosts = []
        if "SWE_REMOTE_BENCH_HOST_LIST" in os.environ:
            self.bench_hosts = os.getenv("SWE_REMOTE_BENCH_HOST_LIST").split(",")

        self.headers = {'Content-Type': 'application/json'}

    def bench_record2host(self, record):
        instance_id = record.get("dataset", {}).get("instance_id", "")
        repo = record.get("dataset", {}).get("repo", "")
        bench_hosts = []
        if len(registered_bench_hosts) > 0:
            bench_hosts = registered_bench_hosts
        if repo in registered_bench_repo2hosts:
            bench_hosts = registered_bench_repo2hosts[repo]

            by_instance_id_key = f'{repo}-by-instance-id'
            if by_instance_id_key in registered_bench_repo2hosts:
                # sample config: {"Project-MONAI__MONAI-4590": ["host1", "host2"], "Project-MONAI__MONAI-941": ["host3", "host4"]}
                sorted_key_value = sorted(registered_bench_repo2hosts.get(by_instance_id_key).items())
                for item in sorted_key_value:
                    if instance_id <= item[0]:
                        bench_hosts = item[1]
                        break

        if len(self.bench_hosts) > 0:
            bench_hosts = self.bench_hosts
        if len(bench_hosts) == 0:
            raise RuntimeError(f"candidate bench hosts not found for instance {instance_id}")
        bench_host = random.choice(bench_hosts)
        print(f"choose bench host {bench_host} for instance {instance_id}")
        return bench_host

    @retry()
    def bench_evaluate(self, record):
        bench_host = self.bench_record2host(record)

        record = copy.deepcopy(record)
        record.pop('dataset')
        payload = json.dumps({
            "record": record,
        })

        print(payload)
        resp = requests.request("POST",
                                f"{bench_host}/bench/evaluate",
                                headers=self.headers,
                                data=payload,
                                timeout=self.bench_operation_long_timeout)
        resp.raise_for_status()

        content = json.loads(resp.content)
        if content["code"] != 0:
            raise RuntimeError(str(content))

        context = content["data"]
        return context.get("report")


swe_verifier = SWERemoteEnvClient()


class FormatError(Exception):
    pass


def extract_code_blocks(text):
    pattern = r'\[start of (.+?\.py)\](.*?)\[end of \1\]'
    matches = re.findall(pattern, text, re.DOTALL)

    code_blocks = {}
    previous_text = ""
    last_end = 0

    for match in matches:
        filename, content = match
        start_index = text.find(f"[start of {filename}]")

        # 取[start of xxx.py]之前的部分作为值
        previous_text = text[last_end:start_index].strip()
        last_end = text.find(f"[end of {filename}]") + len(f"[end of {filename}]")

        code_blocks[filename] = content.strip()

    return code_blocks


def parse_git_patch(patch_content):
    """
    Parse a git patch string and return a dictionary with filenames as keys
    and corresponding patch content as values.
    """
    patch_dict = {}
    current_file = None
    current_content = []

    lines = patch_content.strip().split('\n')

    for line in lines:
        # Detect file header lines
        if line.startswith('diff --git'):
            # If we already have a file being processed, save it
            if current_file and current_content:
                patch_dict[current_file] = '\n'.join(current_content)
                current_content = []

            # Extract the filename from the diff line (take the b/ part)
            parts = line.split()
            current_file = parts[2][2:]  # Remove 'a/' prefix

        # Continue building the patch content
        if current_file:
            current_content.append(line)

    # Add the last file if there is one
    if current_file and current_content:
        patch_dict[current_file] = '\n'.join(current_content)

    return patch_dict


def extract_thought_solution(output: str) -> tuple[str, str]:
    """
    Extract the thought and solution from the output. It is expected to have the following format:
    <think>
    ...
    </think>
    <solution>
    ...
    </solution>
    """
    for tag in [THINK_START, THINK_END, ANSWER_START, ANSWER_END]:
        if output.count(tag) != 1:
            raise FormatError(f"count of {tag} is not 1")

    thought = output.split(THINK_START)[1].split(THINK_END)[0].strip()
    answer = output.split(ANSWER_START)[1].split(ANSWER_END)[0].strip()
    if len(thought) == 0:
        raise FormatError("Thought is empty")
    return thought, answer


def parse_search_replace(text: str) -> dict[str, list[tuple[str, str]]]:
    """
    Parse the search/replace blocks from the text.

    Returns:
        A dictionary where the key is the file path and the value is a list of search/replace pairs.
    """
    path_search_replaces: list[tuple[str, str, str]] = re.findall(SEARCH_REPLACE_REGEX, text)
    path_search_replace_dict = dict[str, list[tuple[str, str]]]()
    for path, search, replace in path_search_replaces:
        path_search_replace_dict.setdefault(path, []).append((search, replace))
    return path_search_replace_dict


def generate_unified_diff(
    old_code: str,
    new_code: str,
    old_file: str = 'old',
    new_file: str = 'new',
    return_file: bool = False,
    n_context: int = 3,
) -> str:
    """Generate a unified diff between two code.

    Args:
        old_code: The original code.
        new_code: The modified code.
        n_context: The number of context lines to show.

    Returns:
        A string representing the unified diff."""

    original_lines = old_code.splitlines()
    modified_lines = new_code.splitlines()

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{old_file}",
        tofile=f"b/{new_file}",
        lineterm="",
        n=n_context,
    )
    try:
        if not return_file:
            next(diff)
            next(diff)
        diff_code = "\n".join(diff)
        return diff_code
    except StopIteration:
        return ""


def apply_code_change(
    code_context: dict[str, str],
    search_replace_dict: dict[str, list[tuple[str, str]]],
    silent: bool = False,
) -> dict[str, str]:
    """
    Apply the search/replace edits to the code context.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        search_replace_dict: A dictionary mapping the file path to the search/replace edits.
        silent: Whether to suppress the error messages.

    Returns:
        A dictionary containing the file path and the new content of the code.
    """
    new_content_dict = dict[str, str]()
    for path, search_replaces in search_replace_dict.items():
        new_content = "\n" + code_context.get(path, "")
        for search, replace in search_replaces:
            # Ensure search block can be matched
            # "\n" + search to ensure the indentations are correct
            if not silent and len(search) == len(replace) and search == replace:
                raise FormatError("Search and replace blocks are identical")
            search = "\n" + search
            replace = "\n" + replace
            if not silent and search not in new_content:
                raise FormatError(f"Search block not found in the code: {search}")
            new_content = new_content.replace(search, replace)
        # Remove the leading "\n"
        new_content_dict[path] = new_content[1:]
    return new_content_dict


def get_normalized_patch(
    code_context: dict[str, str],
    new_content_dict: dict[str, str],
    return_str: bool = False,
) -> dict[str, str]:
    """
    According to the code context and new content, generate the normalized patch for each file.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        new_content_dict: A dictionary mapping the file path to the new content of the file.

    Returns:
        A dictionary containing the file path and the normalized patch.
    """
    patch_dict = dict[str, str]()
    for path, new_content in new_content_dict.items():
        old_content = code_context.get(path, "")
        patch = generate_unified_diff(old_content, new_content, path, path, return_file=True)
        # Only add the patch if it's not empty
        # NOTE: this should not happen due to the search == replace check in `apply_code_change`
        # but it can occur in general-purpose usages
        if patch:
            patch_dict[path] = patch

    if return_str:
        return "\n".join(patch_dict.values())
    return patch_dict


class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


def compute_change_similarities(
    pred_patch: dict[str, str],
    oracle_patch: dict[str, str],
) -> list[ChangeSimilarity]:
    all_file_paths = set(oracle_patch.keys()).union(set(pred_patch.keys()))
    similarities = list[ChangeSimilarity]()
    for path in all_file_paths:
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if oracle_change == "" or pred_change == "":
            # Both are empty changes, meaning search = replace. We should penalize this to avoid
            # the model predicting empty changes to hack the reward.
            # NOTE: this should not happen due to (1) the search == replace check in `apply_code_change`
            # and (2) the `if patch` check in `get_normalized_patch`.
            change_similarity = 0.0
        else:
            change_similarity = difflib.SequenceMatcher(
                None,
                pred_change,
                oracle_change,
                autojunk=False,
            ).ratio()
        similarities.append(
            ChangeSimilarity(
                path=path,
                pred_change=pred_change,
                oracle_change=oracle_change,
                similarity=change_similarity,
            ))
    return similarities


def submit_to_evaluate(
    repo: str,
    instance_id: str,
    pred_patch: str,
    model_name: str = 'seed-rl',
):
    record = {
        'instance_id': instance_id,
        'model_name_or_id': model_name,
        'model_patch': pred_patch,
        'dataset': {
            'instance_id': instance_id,
            'repo': repo,
        }
    }
    try:
        report = swe_verifier.bench_evaluate(record)
    except Exception as e:
        print(f"submit to evaluate failed, {e}")
        return -2
    is_pass = 2 * float(report['resolved_instances'] == 1) - 1
    return is_pass


def calculate_reward(
    code_context: dict[str, str],
    # oracle_new_content: dict[str, str],
    oracle_patch: dict[str, str],
    pred_new_content: dict[str, str],
    repo: str,
    instance_id: str,
) -> tuple[float, dict]:
    """
    Compute the SWE-RL reward given the code context, oracle patch, and the model output.
    Note that this function is a general version of the reward calculation, which can be used
    for code changes in any form, not just search/replace edits. For search/replace edits, use
    `calculate_search_replace_reward`.

    The return value is always within the range of [0, 1].

    Args:
        code_context: path -> original content of the file. It doesn't need to
            contain the entire codebase, only the files that are affected by the oracle patch.
        oracle_patch: path -> oracle new patch
        pred_new_content: path -> predicted new content of the file after change.

    Returns:
        A float value representing the reward, and a dictionary containing some metadata.
    """
    # Obtain a unified diff for each file, for both the predicted and the oracle patch
    # oracle_patch = get_normalized_patch(code_context, oracle_new_content)
    pred_patch = get_normalized_patch(code_context, pred_new_content, return_str=True)
    reward = submit_to_evaluate(repo, instance_id, pred_patch)
    return reward, {}


def _rm(text, key_pair):
    result_text = []  # 存储去除标签后的文本
    think_contents = []  # 存储标签内的内容
    start = 0
    text_length = len(text)
    while start < text_length:
        think_start = text.find(key_pair[0], start)
        if think_start == -1:
            result_text.append(text[start:])
            break

        think_end = text.find(key_pair[1], think_start)
        if think_end == -1:
            think_contents.append(text[start:])
            result_text.append("思考过程过长，被截断")
            break

        # 添加标签前的文本
        result_text.append(text[start:think_start])

        # 提取并存储标签内的内容 (去除<think>和</think>标签)
        content_start = think_start + len(key_pair[0])  # <think> 的长度是7
        think_contents.append(text[content_start:think_end])

        start = think_end + len(key_pair[1])  # </think> 的长度是8

    return ''.join(result_text), think_contents


def calculate_search_replace_reward(
        code_context: dict[str, str],
        # oracle_new_content: dict[str, str],
        oracle_patch: dict[str, str],
        output: str,
        repo: str,
        instance_id: str) -> tuple[float, dict]:
    """
    The search/replace version of the reward calculation. It expects the output to contain
    the thought and solution in the following format:
    <think>
    ...
    </think>
    <solution>
    ...
    </solution>

    Args:
        code_context: path -> original content of the file.
        oracle_patch: path -> oracle patch.
        output: The output from the model containing the thought and solution.

    Returns:
        A float value representing the reward, and a dictionary containing some metadata.
    """
    try:
        # Extract the thought and solution from the output
        answer = output
        # Parse the search/replace edits from the solution
        pred_search_replaces = parse_search_replace(answer)
        if len(pred_search_replaces) == 0:
            raise FormatError("No valid search blocks found")
        # Get the new content of each file after applying the search/replace edits
        pred_new_content = apply_code_change(code_context, pred_search_replaces)
        reward, metadata = calculate_reward(code_context, oracle_patch, pred_new_content, repo, instance_id)
        metadata["answer"] = answer
        return reward, metadata
    except FormatError as e:
        print(f"Format error: {e}")
        return -1.0, dict(error=str(e))


def compute_score(solution_str, ground_truth, **argv):

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    code_context = ground_truth['original_context']

    oracle_patch = ground_truth['oracle_patch']

    result_text, _ = _rm(solution_str, key_pair=('<think>', '</think>'))
    result_text, _ = _rm(result_text, key_pair=('<doubaothinking>', '</doubaothinking>'))

    instance_id = ground_truth['instance_id']
    repo = ground_truth['repo']

    if isinstance(code_context, str):
        code_context = json.loads(code_context)

    oracle_patch = parse_git_patch(oracle_patch)

    # # 因为parquet的特性，导致
    # code_context = {k: v for k, v in code_context.items() if v is not None}
    # oracle_patch = {k: v for k, v in oracle_patch.items() if v is not None}

    return calculate_search_replace_reward(code_context, oracle_patch, result_text, repo=repo,
                                           instance_id=instance_id)[0]


def test_record():
    # record = {"instance_id": "pandas-dev__pandas-55754", "model_name_or_id": "seed-rl", "model_patch": "--- a/pandas/io/formats/format.py\n+++ b/pandas/io/formats/format.py\n@@ -202,7 +202,7 @@\n         length: bool | str = True,\n         header: bool = True,\n         index: bool = True,\n-        na_rep: str = \"NaN\",\n+        na_rep: str = get_option(\"display.na_rep\"),\n         name: bool = False,\n         float_format: str | None = None,\n         dtype: bool = True,\n@@ -432,7 +432,7 @@\n         col_space: ColspaceArgType | None = None,\n         header: bool | SequenceNotStr[str] = True,\n         index: bool = True,\n-        na_rep: str = \"NaN\",\n+        na_rep: str = get_option(\"display.na_rep\"),\n         formatters: FormattersType | None = None,\n         justify: str | None = None,\n         float_format: FloatFormatType | None = None,\n@@ -1170,7 +1170,7 @@\n         values: ArrayLike,\n         digits: int = 7,\n         formatter: Callable | None = None,\n-        na_rep: str = \"NaN\",\n+        na_rep: str = get_option(\"display.na_rep\"),\n         space: str | int = 12,\n         float_format: FloatFormatType | None = None,\n         justify: str = \"right\","}
    # record['dataset'] = {
    #     'instance_id': 'pandas-dev__pandas-55754',
    #     'repo': 'pandas-dev/pandas',
    # }
    record = {
        "instance_id":
            "pandas-dev__pandas-50955",
        "model_name_or_id":
            "seed-rl",
        "model_patch":
            "--- a/pandas/core/frame.py\n+++ b/pandas/core/frame.py\n@@ -6957,8 +6957,8 @@\n \n         Parameters\n         ----------\n-        subset : list-like, optional\n-            Columns to use when counting unique combinations.\n+        subset : str, list-like, optional\n+            Columns to use when counting unique combinations. If a single string is provided, it is treated as a single-element list.\n         normalize : bool, default False\n             Return proportions rather than frequencies.\n         sort : bool, default True"
    }
    record['dataset'] = {
        'instance_id': 'pandas-dev__pandas-50955',
        'repo': 'pandas-dev/pandas',
    }

    try:
        report = swe_verifier.bench_evaluate(record)
    except Exception as e:
        print(f"submit to evaluate failed, {e}")
        return -2
    is_pass = float(report['resolved_instances'] == 1)
    return is_pass


if __name__ == '__main__':
    print(test_record())
    exit(0)
    path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/jiangyun.jy/datasets/rl_data/swe_gym_repair_0324_test.parquet'
    df = pd.read_parquet(path)
    output_strs = {
        'conan-io__conan-11799':
            """<think>?</think>\n\n<solution>```python\n### conan/tools/cmake/toolchain/blocks.py\n<<<<<<< SEARCH\n            'android_platform': self._conanfile.settings.os.api_level,\n=======\n            'android_platform': f\"android-{self._conanfile.settings.os.api_level}\",\n>>>>>>> REPLACE\n```\n\nThe bug is that older NDK versions (like r18b) require the Android platform to be specified in \"android-XX\" format rather than just the numeric API level. The fix adds the \"android-\" prefix to the platform version when generating the CMake toolchain context.\n\nThis matches the format expected by the NDK's android.toolchain.cmake file which was throwing the \"Invalid Android platform: 21\" error because it was receiving just \"21\" instead of \"android-21\".</solution>""",
        'conan-io__conan-10959':
            """<think>?</think>\n\n<solution>\n\n```python\n### conans/model/info.py\n<<<<<<< SEARCH\n    def recipe_revision_mode(self):\n        self._name = self._ref.name\n        self._version = self._ref.version\n        self._user = self._ref.user\n        self._channel = self._ref.channel\n        self._revision = self._ref.revision\n=======\n    def recipe_revision_mode(self):\n        self._name = self._ref.name\n        self._version = self._ref.version\n        self._user = self._ref.user\n        self._channel = self._ref.channel\n        self._revision = self._ref.revision\n\n    def unrelated_mode(self):\n        self._name = None\n        self._version = None\n        self._user = None\n        self._channel = None\n        self._revision = None\n>>>>>>> REPLACE\n```\n\nThe error occurs because the `PythonRequireInfo` class was missing the `unrelated_mode` method that's required when using `default_python_requires_id_mode = unrelated_mode`. The fix adds this missing method which properly nullifies all version components to ensure Python requires don't affect package IDs when using this mode</solution>""",
    }

    for item in df.iterrows():
        item = item[1]
        instance_id = item['extra_info']['index']
        output_str = output_strs[instance_id]
        gt = item['reward_model']['ground_truth']
        print(gt['original_context'].keys())
        res = compute_score(output_str, item['reward_model']['ground_truth'])
        print(res)
