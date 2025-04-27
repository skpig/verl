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

import datasets
from numpy import argsort

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def format_question_to_prompt(question):
    system_prompt = """
When tackling complex reasoning tasks, you should first thinks about the reasoning process in the mind and then provides the answer. 

You should strictly follow the format below:

## Reasoning step 1:
Your reasoning process here

## Reasoning step 2:
Your reasoning process here

## Reasoning step 3:
Your reasoning process here
...
## Reasoning step n:
Your reasoning process here

## Answer:
Your answer here
"""
    user_prompt = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
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
    test_dataset = dataset["validation"].shuffle(42).select(range(100)) # only select the first 100 samples for testing

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


if __name__ == "__main__":
    argsort = argparse.ArgumentParser()
    argsort.add_argument("--resume", action="store_true")   
    RESUME = argsort.parse_args().resume
    MY_DATA_DIR = os.getenv("MY_DATA_DIR")
    

    process_numinamath_dataset()
    process_math500_dataset()
    process_amc_dataset()

    print("Done Preprocessing!")