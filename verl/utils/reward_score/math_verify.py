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

import traceback
import os
import re
try:
    from math_verify import parse
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

import math

# 定义不同检查对应的位
ANSWER_MATCH_BIT = 1
REASONING_ORDER_BIT = 2

def verify_format(model_output: str):
    """
    Verify if the answer is in a valid format.
    返回值为位掩码，不同位代表不同检查结果。
    """
    result = 0
    
    # 检查是否有且仅有一个 "## Answer:" 
    answer_matches = re.findall(r'## Answer:', model_output)
    if len(answer_matches) == 1:
        result |= ANSWER_MATCH_BIT
    
    # 查找所有 "## Reasoning step [1-9]+:" 
    # 修改正则表达式以支持多位数的编号
    reasoning_pattern = r'## Reasoning step ([0-9]+):'
    reasoning_matches = [(match.start(), match.group(1)) for match in re.finditer(reasoning_pattern, model_output)]

    num_steps = -1
    
    if reasoning_matches:
        reasoning_steps = [int(step) for _, step in reasoning_matches]
        expected_steps = list(range(2, len(reasoning_steps) + 2))
        if reasoning_steps == expected_steps:
            result |= REASONING_ORDER_BIT
            num_steps = len(reasoning_steps)
    
    return result, num_steps



def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> bool:
    model_output = solution_str
    timeout_score = 0.0


    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(), ExprExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    prediction_boxed = "\\boxed{" + model_output + "}"
    try:
        ret_score, (extracted_gold, extracted_model_output) = verify_func([ground_truth_boxed], [prediction_boxed])
    except Exception:
        ret_score = 0.
        os.makedirs('.cache/reward_error', exist_ok=True)
        with open(f'.cache/reward_error/grader_error_{os.getpid()}.log', 'w') as f:
            f.write(f"Error detected\n====\n{model_output}\n====\n{ground_truth}\n====\n")
            traceback.print_exc(file=f)
            f.write('\n')
    except TimeoutException:
        ret_score = timeout_score

    format_correctness, num_steps = verify_format(model_output)

    # extracted_model_output = parse(model_output, (ExprExtractionConfig(), LatexExtractionConfig()))
    # if len(extracted_model_output) == 0:
    #     extracted_model_output = "None extraction"
    # elif len(extracted_model_output) == 1:
    #     extracted_model_output = f"{extracted_model_output[0]}"
    # else:
    #     extracted_model_output = extracted_model_output[1] if isinstance(extracted_model_output[1], str) else f"{extracted_model_output[0]}"

    return {
        "score": ret_score,
        "acc": 1 if ret_score > 0 else 0,
        "format": format_correctness,
        "pred": extracted_model_output[-1] if len(extracted_model_output) > 0 else "None extraction",
        "#steps": num_steps,
    }
