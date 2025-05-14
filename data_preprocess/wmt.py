"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def gen_dataset(
    num_samples: int,
    num_operands: int = 6,
    max_target: int = 1000,
    min_number: int = 1,
    max_number: int = 100,
    operations: List[str] = ['+', '-', '*', '/'],
    seed_value: int = 42,
) -> List[Tuple]:
    """Generate dataset for countdown task.
    
    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        operations: List of allowed operations
        seed_value: Random seed for reproducibility
        
    Returns:
        List of tuples containing (target, numbers, solution)
    """
    seed(seed_value)
    samples = []
    
    for _ in tqdm(range(num_samples)):
        # Generate random target
        target = randint(1, max_target)
        
        # Generate random numbers
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]
        
        
        samples.append((target, numbers))
    
    return samples

def make_prefix(cn_text, template_type):
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        raise NotImplementedError("Base template is not implemented.")
        """This works for any base model"""
        return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

###User: Using the numbers {numbers}, create an expression that equals {dialogue}. You can use basic arithmetic operations (+, -, *, /). Each number can only be used once, while each operation could be used for arbitrary times. Show your reasoning process in <think> </think> tags. You may conduct multiple turns of reasoning. And return the final expression in only **ONE** pair of <answer> </answer> tag (e.g. <answer>2*3+1</answer>). In summary, your response should be formatted in "<think> Your thinking process 1 </think>\n<think> Your thinking process 2 </think>\n...\n<answer> Your final expression </answer>".

###Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'chat':
        """This works for Chat Models"""
        return [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': "Translate the following Chinese sentences into English.\n\n" + cn_text},
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/data/datasets/wmt')
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--test_size', type=int, default=200)

    args = parser.parse_args()
    # args.local_dir += f'_{args.template_type}'
    # if os.path.exists(args.local_dir):
    #     print(f"Directory {args.local_dir} already exists. Exiting.")
    #     exit(0)

    TEST_SIZE = args.test_size

    raw_dataset = load_dataset('wmt/wmt19', 'zh-en', split='validation')

    test_dataset = raw_dataset.select(range(TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example['translation']['zh'], template_type=args.template_type)
            solution = example['translation']['en']
            data = {
                "data_source": "wmt",
                "prompt": question,
                "template_type": args.template_type,
                "ability": "translation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    print("Test dataset saved to:", os.path.join(args.local_dir, 'test.parquet'))