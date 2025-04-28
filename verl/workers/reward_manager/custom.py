# Copyright 2024 PRIME team and/or its affiliates
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
import asyncio
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Optional

import torch
from sympy import ground_roots
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference, task_extra_info),  # Ensure synchronous
                ),
                timeout=timeout,
            )
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"==== Error processing completion ====\n {completion[:10]}\n==== Error: {e} ====")
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0)
            for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print("shut down failed: " + str(kill_err))
            raise

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


# 定义同步版本的计算函数
import concurrent.futures


def parallel_compute_score_sync(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=40
):
    scores = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        futures = []
        for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info):
            future = executor.submit(evaluation_func, task, completion, reference, task_extra_info)
            futures.append(future)
        #     try:
        #         result = evaluation_func(task, completion, reference, task_extra_info)
        #     except Exception as e:
        #         print(f"==== Error processing completion ====\n {completion[:10]}\n==== Error: {e} ====")
        #         traceback.print_exc()
        #     futures.append(result)
        # return futures

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=100.0) # 设置超时时间为 300 秒
                scores.append(result)
            except concurrent.futures.TimeoutError:
                print(f"计算超时: {future}")
                scores.append({
                    "score": 0,
                    "acc": 0,
                    "format": 0,
                    "pred": "Error",
                })
            except Exception as e:
                traceback.print_exc()
                print(f"计算出错: {e}")
                scores.append({
                    "score": 0,
                    "acc": 0,
                    "format": 0,
                    "pred": "Error",
                })

    return scores


class CustomRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        max_resp_len: Optional[int] = None,
        overlong_buffer_cfg: Optional[dict] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )

    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # valid_response_lst is a list of length N
        valid_response_lst = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_lst.append(valid_response_ids)
        valid_response_lst = self.tokenizer.batch_decode(valid_response_lst, skip_special_tokens=True)

        # ground_truch_lst is a list of length N
        ground_truth_lst = [i["ground_truth"] for i in data.non_tensor_batch["reward_model"]]

        # data_source_lst is a list of length N
        data_source_lst = data.non_tensor_batch["data_source"]

        # extra_info is a list of length N 
        extra_info_lst = data.non_tensor_batch.get("extra_info", None)
        if extra_info_lst is None:
            extra_info_lst = [None] * len(data_source_lst)

        assert len(valid_response_lst) == len(ground_truth_lst) == len(data_source_lst) == len(extra_info_lst), f"Length mismatch: valid_response_lst: {len(valid_response_lst)}, ground_truth_lst: {len(ground_truth_lst)}, data_source_lst: {len(data_source_lst)}, extra_info_lst: {len(extra_info_lst)}"

        # breakpoint()
        try:
            rtn_lst = parallel_compute_score_sync(
                self.compute_score,
                valid_response_lst,
                ground_truth_lst,
                data_source_lst,
                extra_info=extra_info_lst,
                num_processes=40,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"批量奖励计算时出现意外错误。全部设为 0.: {e}")
            rtn_lst = [0.0] * len(valid_response_lst)

        return rtn_lst

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        rtn_lst = self.verify(data)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            # get return
            result = rtn_lst[i]

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            data_source = data.non_tensor_batch["data_source"][i]
            ground_truth = data.non_tensor_batch["reward_model"][i]["ground_truth"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("=== [prompt] ===\n", prompt_str)
                print("=== [response] ===\n", response_str)
                print("=== [ground_truth] ===\n", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"=== [{key}] ===", value)
                else:
                    print("=== [score] ===", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
