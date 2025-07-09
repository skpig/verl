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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

from math_verify import parse
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.utils import timeout
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
import datasets
from numpy import argsort

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
import re


def extract_solution(solution_str):
    ground_truth_boxed = "\\boxed{" + solution_str + "}"
    gold_extraction_target=(LatexExtractionConfig(),)
    extracted_golds = parse(ground_truth_boxed, gold_extraction_target)
    return False if len(extracted_golds) == 0 else True

def format_question_to_prompt(question):
    # system_prompt = all_prompts[prompt_id]  # default system prompt
    user_prompt = question

    return [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "<think>\n"}
    ]

def process_numinamath_dataset():
    data_source = "PRIME-RL/Eurus-2-RL-Data"
    local_dir = os.path.basename(data_source)
    train_path = os.path.join(MY_DATA_DIR, local_dir, "train.parquet")
    test_path = os.path.join(MY_DATA_DIR, local_dir, "test.parquet")
    # 如果文件已存在且设置了恢复标志，则跳过处理
    if RESUME and os.path.exists(train_path):
        return

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]

    # gold_extraction_target=(LatexExtractionConfig(),)
    gold_extraction_target=(ExprExtractionConfig(),)
    def filter_fn(example):
        # golden_answer = '\\boxed{' + str(example['reward_model']['ground_truth']) + "}"
        golden_answer = example['reward_model']['ground_truth']
        extracted_golds = parse(golden_answer, gold_extraction_target, parsing_timeout=5)
        # 过滤掉code data
        return example['ability'] == "math" and len(extracted_golds) > 0
    train_dataset = train_dataset.filter(filter_fn)
    test_dataset = test_dataset.filter(filter_fn)
    print("Size of NuminaMath train dataset after filtering:", len(train_dataset))
    print("Size of NuminaMath test dataset after filtering:", len(test_dataset))
    test_dataset = test_dataset.shuffle(42).select(range(100)) # only select the first 100 samples for testing

    # 为每个数据项添加一个表示唯一ID的行
    def make_map_fn(split):
        def process_fn(example, idx):
            # 这里需要根据实际数据结构调整键名，假设数据结构与MATH-500类似
            question = example.pop("prompt")[-1]["content"]
            question = "\n\n".join(question.split("\n\n")[:-1]) # reformat the original prompt by removing "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
            example['data_source'] = "numinamath"
            example['prompt'] = format_question_to_prompt(question)
            example['reward_model']['ground_truth'] = str(example['reward_model']['ground_truth'])
            example['extra_info'] = {"split": split, "index": idx}
            return example

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print("Size of NuminaMath train dataset:", len(train_dataset))
    print("Size of NuminaMath test dataset:", len(test_dataset))

def process_math500_dataset():
    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    # data_source = "DigitalLearningGmbH/MATH-lighteval"
    data_source = "HuggingFaceH4/MATH-500"
    local_dir = os.path.basename(data_source)
    test_path = os.path.join(MY_DATA_DIR, local_dir, "test.parquet")
    # skip if the file already exists
    if RESUME and os.path.exists(test_path):
        return

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            answer = example.pop("answer")
            data = {
                "data_source": "math500",
                "prompt": format_question_to_prompt(question),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    test_dataset.to_parquet(test_path)
    print("Size of MATH-500 test dataset:", len(test_dataset))

def process_amc_dataset():
    # 数据源为 AI-MO/aimo-validation-amc
    data_source = "AI-MO/aimo-validation-amc"
    local_dir = os.path.basename(data_source)
    test_path = os.path.join(MY_DATA_DIR, local_dir, "test.parquet")
    # 如果文件已存在且设置了恢复标志，则跳过处理
    if RESUME and os.path.exists(test_path):
        return

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    test_dataset = dataset["train"]

    # 为每个数据项添加一个表示唯一ID的行
    def make_map_fn(split):
        def process_fn(example, idx):
            # 这里需要根据实际数据结构调整键名，假设与MATH-500类似
            question = example.pop("problem")
            answer = example.pop("answer")
            data = {
                "data_source": "amc12",
                "prompt": format_question_to_prompt(question),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(answer)},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    test_dataset.to_parquet(test_path)
    print("Size of AMC-12 test dataset:", len(test_dataset))


def process_math_dataset():
    data_source = "HuggingFaceH4/MATH"
    local_dir = os.path.basename(data_source)
    # test_path = os.path.join(MY_DATA_DIR, local_dir, "test.parquet")
    train_path = os.path.join(MY_DATA_DIR, local_dir, "train.parquet")
    filtered_dataset_path = os.path.join(MY_DATA_DIR, local_dir, "filtered_dataset.parquet")

    # Skip processing if resumed and file exists
    if RESUME and os.path.exists(train_path):
        return

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    # Load the entire dataset (assuming it is provided as a 'train' split)
    configs = datasets.get_dataset_config_names(data_source)
    all_ds = []
    for config in configs:
        print(config)
        ds = datasets.load_dataset(data_source, config, trust_remote_code=True, split="train")
        all_ds.append(ds)
    dataset = datasets.concatenate_datasets(all_ds)
    print("Size of MATH dataset before filtering:", len(dataset))
    exit(0)

    extraction_target = (ExprExtractionConfig(),)
    # Filtering function: ensure a non-empty ground truth and valid extraction
    def extract_solution(example):
        solution = example.get("solution", "")
        match = re.search(r'\\boxed\{(.+?)\}', solution)
        if match:
            return match.group(1)
        return None

    def filter_fn(example):
        if not example.get("ground_truth"):
            return False
        golden_answer = example["ground_truth"]
        if golden_answer is None:
            return False
        extracted = parse(golden_answer, extraction_target, parsing_timeout=5)
        return len(extracted) > 0

    if os.path.exists(filtered_dataset_path):
        print(f"Loading the filtered dataset from {filtered_dataset_path}...", flush=True)
        dataset = datasets.load_dataset("parquet", data_files=filtered_dataset_path)
    else:
        dataset = dataset.map(lambda example: {**example, "ground_truth": extract_solution(example)})
        dataset = dataset.filter(filter_fn)
        if not os.path.exists(os.path.dirname(filtered_dataset_path)):
            makedirs(os.path.dirname(filtered_dataset_path))
        dataset.to_parquet(filtered_dataset_path)
    print("Size of MATH dataset after filtering:", len(dataset))

    # Split the dataset into training and testing splits
    splits = dataset.train_test_split(test_size=1000, seed=42)
    train_dataset, test_dataset = splits["train"], splits["test"]
    test_dataset = test_dataset.shuffle(42).select(range(100))  # Select only the first 100 samples for testing

    # Create a mapping function to format each sample
    def make_map_fn(split):
        def process_fn(example, idx):
            # Assume each example contains 'problem' and 'answer'
            question = example.pop("problem")
            # For consistency, directly format the question into a prompt
            example["data_source"] = "math"
            example["prompt"] = format_question_to_prompt(question)
            example["ability"] = "math"
            answer = example.pop("answer")
            example["reward_model"] = {"style": "rule", "ground_truth": str(answer)}
            example["extra_info"] = {"split": split, "index": idx}
            return example
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print("Size of MATH train dataset:", len(train_dataset))
    print("Size of MATH test dataset:", len(test_dataset))

def process_dapomath_dataset():
    # 数据源为 dapomath/dapomath
    data_source = "BytedTsinghua-SIA/DAPO-Math-17k"
    local_dir = os.path.basename(data_source)
    test_path = os.path.join(MY_DATA_DIR, local_dir, "test.parquet")
    train_path = os.path.join(MY_DATA_DIR, local_dir, "train.parquet")
    filtered_dataset_path = os.path.join(MY_DATA_DIR, local_dir, "filtered_dataset.parquet")
    # 如果文件已存在且设置了恢复标志，则跳过处理
    if RESUME and os.path.exists(test_path):
        return
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True, split="train")

    golden_extraction_target=(ExprExtractionConfig(),)
    filtered_dataset_path = os.path.join(MY_DATA_DIR, local_dir, "filtered_dataset.parquet")
    # if os.path.exists(filtered_dataset_path):
    if False:
        print(f"Loading the filtered dataset from {filtered_dataset_path}...", flush=True)
        dataset = datasets.load_dataset("parquet", data_files=filtered_dataset_path)
    else:
        def filter_fn(example):
            if not example['reward_model']['ground_truth']:
                return False
            try:
                # Try to convert answer to a number
                float(example['reward_model']['ground_truth'])
                is_number = True
            except (ValueError, TypeError):
                is_number = False
                print("Answer is not a number:", example['reward_model']['ground_truth'])
            
            # golden_answer = '\\boxed{' + str(example['reward_model']['ground_truth']) + "}"
            golden_answer = example['reward_model']['ground_truth']
            extracted_golds = parse(golden_answer, golden_extraction_target, parsing_timeout=5)
            # 过滤掉code data
            return example['ability'] == "MATH" and len(extracted_golds) > 0
            return False
        dataset = dataset.filter(filter_fn)        # 保存过滤后的数据集到 parquet 文件
        if not os.path.exists(os.path.dirname(filtered_dataset_path)):
            makedirs(os.path.dirname(filtered_dataset_path))
        dataset.to_parquet(filtered_dataset_path)
    print("Size of DAPO-Math dataset after filtering:", len(dataset))

    # 分割数据集为训练集和测试集
    _ = dataset.train_test_split(test_size=1000, seed=42)
    train_dataset, test_dataset = _["train"], _["test"]
    test_dataset = test_dataset.shuffle(42).select(range(100))  # only select the first 100 samples for testing

    # 为每个数据项添加一个表示唯一ID的行
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")[-1]["content"]
            prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
            suffix = "\n\nRemember to put your answer on its own line after \"Answer:\"."
            question = question[len(prefix):-len(suffix)]  # reformat the original prompt by removing prefix and suffix
            # 这里需要根据实际数据结构调整键名，假设数据结构与MATH-500类似
            example['data_source'] = "dapomath"
            example['prompt'] = format_question_to_prompt(question)
            example['reward_model']['ground_truth'] = str(example['reward_model']['ground_truth'])
            example['extra_info'] = {"split": split, "index": idx}
            return example

        return process_fn
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    # 保存数据集到 parquet 文件
    test_dataset.to_parquet(test_path)
    train_dataset.to_parquet(train_path)
    print("Size of DAPO-Math train dataset:", len(train_dataset))
    print("Size of DAPO-Math test dataset:", len(test_dataset))


if __name__ == "__main__":
    argsort = argparse.ArgumentParser()
    argsort.add_argument("--resume", action="store_true")   
    RESUME = argsort.parse_args().resume
    MY_DATA_DIR = os.getenv("MY_DATA_DIR")

    # process_numinamath_dataset()
    # process_math500_dataset()
    # process_amc_dataset()
    # process_dapomath_dataset()
    # process_math_dataset()

    print("Done Preprocessing!")
