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

import random
import time
import traceback
import os
import re
from tracemalloc import start
try:
    from math_verify import parse
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.utils import timeout
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from math_verify.grader import verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

import math

# 定义不同检查对应的位
ANSWER_MATCH_BIT = 1
REASONING_ORDER_BIT = 2
def verify_format(model_output: str, prompt_id: int):
    """
    Verify if the answer is in a valid format.
    返回值为位掩码，不同位代表不同检查结果。
    """
    result = 0
    
    # 检查是否有且仅有一个 "## Answer:" 
    if prompt_id in [0, 1, 2, 3]:
        answer_matches = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
    else:
        raise NotImplementedError(f"Prompt ID {prompt_id} is not supported for answer extraction in math verify.")
    if len(answer_matches) == 1:
        result |= ANSWER_MATCH_BIT
    
    # 1. 提取所有<think>...</think>标签
    think_blocks = [i for i in re.findall(r'<think>.*?</think>', model_output, re.DOTALL)]
    
    # 2. 将所有标签拼接起来
    concatenated = ''.join(think_blocks)
    
    # 3. 移除所有空白字符进行比较
    text_no_whitespace = re.sub(r'\s+', '', model_output)
    concatenated_no_whitespace = re.sub(r'\s+', '', concatenated)
    
    # 如果移除空白后完全相同，说明字符串只由<think>...</think>标签组成
    if text_no_whitespace == concatenated_no_whitespace:
        result |= REASONING_ORDER_BIT

    num_steps = len(think_blocks)
    return result, num_steps


# def verify_format(model_output: str):
#     """
#     Verify if the answer is in a valid format.
#     返回值为位掩码，不同位代表不同检查结果。
#     """
#     result = 0
    
#     # 检查是否有且仅有一个 "## Answer:" 
#     answer_matches = re.findall(r'## Answer:', model_output)
#     if len(answer_matches) == 1:
#         result |= ANSWER_MATCH_BIT
    
#     # 查找所有 "## Reasoning step [1-9]+:" 
#     # 修改正则表达式以支持多位数的编号
#     reasoning_pattern = r'## Reasoning step ([0-9]+):'
#     reasoning_matches = [(match.start(), match.group(1)) for match in re.finditer(reasoning_pattern, model_output)]

#     num_steps = -1
    
#     if reasoning_matches:
#         reasoning_steps = [int(step) for _, step in reasoning_matches]
#         expected_steps = list(range(2, len(reasoning_steps) + 2))
#         if reasoning_steps == expected_steps:
#             result |= REASONING_ORDER_BIT
#             num_steps = len(reasoning_steps)
    
#     return result, num_steps

def extract_answer(model_output: str, prompt_id: int) -> str:
    if prompt_id in [0, 1, 2, 3]:
        extraction = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
    else:
        raise NotImplementedError(f"Prompt ID {prompt_id} is not supported for answer extraction in math verify.")
    if len(extraction) == 0:
        return "None extraction"
    else:
        return extraction[-1].strip() # use the last extracted answer


total_time1 = 0
pred_extract_num1 = 0
gold_extract_num1 = 0
def verify1(solution_str, ground_truth):
    global total_time1, pred_extract_num1, gold_extract_num1
    start_time = time.time()

    gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig())
    # gold_extraction_target=(LatexExtractionConfig()),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig())

    print("====== Golden ======")
    print(ground_truth)
    print("====== Solution ======")
    print(solution_str)
    print("============")
    print("Verify1 extracted predictions")
    extracted_predictions = parse(solution_str, pred_extraction_target, parsing_timeout=3)
    print("Verify1 parser targets")
    extracted_golds = parse(ground_truth, gold_extraction_target, parsing_timeout=3)
    pred_extract_num1 += len(extracted_predictions)
    gold_extract_num1 += len(extracted_golds)

    if len(extracted_golds) == 0:
        print(f"Warning: No gold targets found for the ground truth: {ground_truth}")
        total_time1 += time.time() - start_time
        return
    if len(extracted_predictions) == 0:
        print(f"Warning: No predictions extracted from the solution string: {solution_str}")
        total_time1 += time.time() - start_time
        return
    
    print("Verify1 verify begin")
    rtn = verify(extracted_golds, extracted_predictions)
    print("Verify1 verify end")
    total_time1 += time.time() - start_time

total_time2 = 0
pred_extract_num2 = 0
gold_extract_num2 = 0
def verify2(solution_str, ground_truth):
    global total_time2, pred_extract_num2, gold_extract_num2
    start_time = time.time()

    gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig())
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig())

    print("====== Golden ======")
    print(ground_truth)
    print("====== Solution ======")
    print(solution_str)
    print("============")
    print("Verify2 extracted predictions")
    solution_str = extract_answer(solution_str)
    extracted_predictions = parse(solution_str, pred_extraction_target, parsing_timeout=3)
    print("Verify2 parser targets")
    extracted_golds = parse(ground_truth, gold_extraction_target, parsing_timeout=3)
    pred_extract_num2 += len(extracted_predictions)
    gold_extract_num2 += len(extracted_golds)

    if len(extracted_golds) == 0:
        print(f"Warning: No gold targets found for the ground truth: {ground_truth}")
        total_time2 += time.time() - start_time
        return
    if len(extracted_predictions) == 0:
        print(f"Warning: No predictions extracted from the solution string: {solution_str}")
        total_time2 += time.time() - start_time
        return
    
    print("Verify2 verify begin")
    rtn = verify(extracted_golds, extracted_predictions)
    print("Verify2 verify end")
    total_time2 += time.time() - start_time

total_time3 = 0
pred_extract_num3 = 0
gold_extract_num3 = 0
def verify3(solution_str, ground_truth):
    global total_time3, pred_extract_num3, gold_extract_num3
    start_time = time.time()

    gold_extraction_target=(LatexExtractionConfig(),)
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig())

    print("====== Golden ======")
    print(ground_truth)
    print("====== Solution ======")
    print(solution_str)
    print("============")
    print("Verify3 extracted predictions")
    extracted_predictions = parse(solution_str, pred_extraction_target, parsing_timeout=3)
    print("Verify3 parser targets")
    extracted_golds = parse(ground_truth, gold_extraction_target, parsing_timeout=3)
    pred_extract_num3 += len(extracted_predictions)
    gold_extract_num3 += len(extracted_golds)

    if len(extracted_golds) == 0:
        print(f"Warning: No gold targets found for the ground truth: {ground_truth}")
        total_time3 += time.time() - start_time
        return
    if len(extracted_predictions) == 0:
        print(f"Warning: No predictions extracted from the solution string: {solution_str}")
        total_time3 += time.time() - start_time
        return
    
    print("Verify3 verify begin")
    rtn = verify(extracted_golds, extracted_predictions)
    print("Verify3 verify end")
    total_time3 += time.time() - start_time



def compute_score_for_statistics(data_source, solution_str, ground_truth, extra_info=None) -> bool:

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        verify1(solution_str, ground_truth_boxed)
        verify2(solution_str, ground_truth_boxed)
        verify3(solution_str, ground_truth_boxed)

        # print the statistics
        print(f"Total time for verify1: {total_time1:.2f}s, pred_extract_num1: {pred_extract_num1}, gold_extract_num1: {gold_extract_num1}")
        print(f"Total time for verify2: {total_time2:.2f}s, pred_extract_num2: {pred_extract_num2}, gold_extract_num2: {gold_extract_num2}")
        print(f"Total time for verify3: {total_time3:.2f}s, pred_extract_num3: {pred_extract_num3}, gold_extract_num3: {gold_extract_num3}")
        ret_score = 0
    except Exception:
        ret_score = 0.
        traceback.print_exc()
        print("Error detected in math_verify, returning 0 score.")
        # os.makedirs('.cache/reward_error', exist_ok=True)
        # with open(f'.cache/reward_error/grader_error_{os.getpid()}.log', 'w') as f:
        #     f.write(f"Error detected\n====\n{model_output}\n====\n{ground_truth}\n====\n")
        #     traceback.print_exc(file=f)
        #     f.write('\n')
    # except TimeoutException:
    #     print("Timeout detected, returning 0 score from math_verify.")
    #     ret_score = timeout_score

    # format_correctness, num_steps = verify_format(model_output)

    # extracted_model_output = parse(model_output, (ExprExtractionConfig(), LatexExtractionConfig()))
    # if len(extracted_model_output) == 0:
    #     extracted_model_output = "None extraction"
    # elif len(extracted_model_output) == 1:
    #     extracted_model_output = f"{extracted_model_output[0]}"
    # else:
    #     extracted_model_output = extracted_model_output[1] if isinstance(extracted_model_output[1], str) else f"{extracted_model_output[0]}"


def compute_score(data_source, solution_str, ground_truth, extra_info=None, is_valid=False, prompt_id=None) -> bool:
    assert prompt_id is not None, "prompt_id must be provided for math_verify"

    try:
        verify_format_w_timeout = timeout(2)(verify_format)
        format_correctness, num_steps = verify_format_w_timeout(solution_str, prompt_id)
    except TimeoutException:
        print("Timeout detected in format verification, returning 0 score.")
        # os.makedirs('.cache/reward_error', exist_ok=True)
        # with open(f'.cache/reward_error/format_error_{os.getpid()}.log', 'w') as f:
        #     f.write(f"Timeout detected in format verification\n==Solution==\n{solution_str}\n==Ground==\n{ground_truth}\n====\n")
        #     traceback.print_exc(file=f)
        #     f.write('\n')
        format_correctness, num_steps = 0, 0
    except Exception:
        print("Error detected in format verification, returning 0 score.")
        # os.makedirs('.cache/reward_error', exist_ok=True)
        # with open(f'.cache/reward_error/format_error_{os.getpid()}.log', 'w') as f:
        #     f.write(f"Error detected in format verification\n==Solution==\n{solution_str}\n==Ground==\n{ground_truth}\n====\n")
        #     traceback.print_exc(file=f)
        #     f.write('\n')
        format_correctness, num_steps = 0, 0

    try:
        # verify1(solution_str, ground_truth_boxed)
        # verify2(solution_str, ground_truth_boxed)
        # verify3(solution_str, ground_truth_boxed)

        # # print the statistics
        # print(f"Total time for verify1: {total_time1:.2f}s, pred_extract_num1: {pred_extract_num1}, gold_extract_num1: {gold_extract_num1}")
        # print(f"Total time for verify2: {total_time2:.2f}s, pred_extract_num2: {pred_extract_num2}, gold_extract_num2: {gold_extract_num2}")
        # print(f"Total time for verify3: {total_time3:.2f}s, pred_extract_num3: {pred_extract_num3}, gold_extract_num3: {gold_extract_num3}")
        # ret_score = 0


        # during training
        if not is_valid and data_source == "dapomath":
            extracted_predictions = extract_answer(solution_str, prompt_id) # only verify the answer part wrapped in <answer>...</answer>
            gold_extraction_target=(ExprExtractionConfig(),) # reduce computation time for training, since DAPOmath only requires ExprExtractionConfig
        # during validation
        else:
            # Wrap the ground truth in \boxed{} format for verification
            ground_truth = "\\boxed{" + ground_truth + "}"
            extracted_predictions = solution_str
            gold_extraction_target = (LatexExtractionConfig(), ExprExtractionConfig()) 
        pred_extraction_target=(
            ExprExtractionConfig(), 
            LatexExtractionConfig(),
            )

        # reduce computation time for training
        with open(".cache/current_solution.log", 'w') as f:
            f.write("====== Solution ======\n")
            f.write(solution_str)
            f.write("\n\n\n")
            f.write("====== Ground Truth ======\n")
            f.write(ground_truth)


        # print("====== Parse Golden ======")
        extracted_predictions = parse(extracted_predictions, pred_extraction_target, parsing_timeout=3)
        # print("====== Parse Solution ======")
        extracted_golds = parse(ground_truth, gold_extraction_target, parsing_timeout=3)
        
        if random.random() < 0.01:
            print(f"====== [Random Sample] Verify {len(extracted_golds)} golds and {len(extracted_predictions)} predictions ======")
        ret_score = verify(extracted_golds, extracted_predictions, timeout_seconds=3)

        if len(extracted_predictions) == 0:
            extracted_predictions = "N/A extraction"
        elif len(extracted_predictions) == 1:
            extracted_predictions = f"{extracted_predictions[0]}"
        else:
            extracted_predictions = extracted_predictions[1] if isinstance(extracted_predictions[1], str) else f"{extracted_predictions[0]}"

    except TimeoutException:
        print("Timeout detected, returning 0 score from math_verify.")
        ret_score = 0.
        extracted_predictions = "Timeout extraction"
        os.makedirs('.cache/reward_error', exist_ok=True)
        with open(f'.cache/reward_error/grader_error_{os.getpid()}.log', 'a') as f:
            f.write(f"Timeout detected\n==Solution==\n{solution_str}\n==Ground==\n{ground_truth}\n====\n")
            traceback.print_exc(file=f)
            f.write('\n')
    except Exception:
        ret_score = 0.
        extracted_predictions = "Error extraction"
        traceback.print_exc()
        print("Error detected in math_verify, returning 0 score.")
        os.makedirs('.cache/reward_error', exist_ok=True)
        with open(f'.cache/reward_error/grader_error_{os.getpid()}.log', 'a') as f:
            f.write(f"Error detected\n==Solution==\n{solution_str}\n==Ground==\n{ground_truth}\n====\n")
            traceback.print_exc(file=f)
            f.write('\n')



    return {
        "score": ret_score,
        "acc": 1 if ret_score > 0 else 0,
        "format": format_correctness,
        "pred": extracted_predictions,
        "#steps": num_steps,
    }
