import os
import time
import json
import logging
import copy
import uuid
import asyncio
import torch

import numpy as np
import pandas as pd
import verl.utils.torch_functional as verl_F

from enum import Enum
from dataclasses import dataclass
from typing import Union, List, Dict
from functools import lru_cache
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from transformers.utils import PaddingStrategy
from omegaconf import DictConfig
from mono_rl import DataProto
from alpha_seed.utils.dataset.rl_dataset import collate_fn
from alpha_seed.workers.agents.handlers import register_handler, TaskContext
from alpha_seed.workers.agents.handlers.base import ThreadedAgent
from alpha_seed.workers.agents.handlers.agentbench.proxy import proxy_server
from alpha_seed.workers.agents.handlers.agentbench.trajectory_manager import build_training_samples
from alpha_seed.workers.streaming_service.streaming_utils import internal_call
from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto
from bytedance import metrics


@lru_cache(maxsize=1)
def get_metrics_client():
    return metrics.Client(prefix='seed.agentless')


@register_handler("agent/agentbench/agentless")
class Agentless(ThreadedAgent):

    def __init__(self, tokenizer, llm, **kwargs):
        super().__init__(tokenizer, llm)

    def _extract_prompt_meta(self, item: DataProto) -> Dict:
        framework, dataset, index = item.non_tensor_batch['raw_prompt'][0][0].get('meta').split(":")
        return {"framework": framework, "dataset": dataset, "index": index}

    def _extract_row_dict(self, item: DataProto) -> Dict:
        return {
            **{
                k: item.batch[k][0] for k in item.batch.keys()
            },
            **{
                k: (lambda x: (x.tolist()[0] if hasattr(x, "tolist") else x))(item.non_tensor_batch[k]) for k in item.non_tensor_batch.keys(
                )
            }
        }

    def _preprocess(self, messages: List[Dict], prompt_meta: Dict, row_dict: Dict, meta_info: Dict,
                    max_prompt_length: int, truncation: str, sub_index: int) -> DataProto:
        prompt_with_chat_template = self.tokenizer.apply_chat_template(messages,
                                                                       add_generation_prompt=True,
                                                                       tokenize=False)
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=truncation)
        prompt = {
            "prompt_names": [""],
            "raw_prompt": messages.tolist() if hasattr(messages, "tolist") else messages,
            "input_ids": input_ids[0].to(torch.int32),
            "attention_mask": attention_mask[0].to(torch.int8)
        }

        _row_dict = {
            **row_dict,
            **prompt,
            **({
                'uid': str(uuid.uuid4())
            } if 'uid' in row_dict else {}),
            **({
                'rollout_id': str(uuid.uuid4())
            } if 'rollout_id' in row_dict else {}),
            **{
                'reward_model': {
                    **row_dict.get('reward_model', {}),
                    **{
                        'ground_truth': row_dict.get('reward_model', {}).get('ground_truth', '')
                    }
                },
                'index': (lambda x, y: int(x if str(x).isdigit() else (x.split('-')[1] if (len(x.split('-')) == 2 and x.split('-')[1].isdigit(
                                                                                           )) else y)))(str(
                             row_dict.get('index')).strip(), prompt_meta.get('index', 0))
            },
        }
        _item = DataProto.from_single_dict(collate_fn([_row_dict]))
        _item.meta_info = copy.copy(meta_info)
        return _item

    def _postprocess(self, completion, item: DataProto, rollout_config: DictConfig) -> DataProto:
        data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
        out = pack_to_dataproto(item, self.tokenizer, data_pack, rollout_config)  # dataproto

        def extra_fill_datapack():
            out.non_tensor_batch['data_pack'] = np.array([data_pack], dtype=object)

        def extra_fill_raw_response():
            raw_output_ids = completion['choices'][0]['message']['raw_output_ids']
            if self.tokenizer.eos_token_id == raw_output_ids[-1]:
                raw_output_ids = raw_output_ids[:-1]
            raw_response = ''.join(self.tokenizer.decode(raw_output_ids, skip_special_tokens=False))
            #  get_metrics_client().emit_store(
            #  "process_single_batch.agentbench.task_raw_response_len", len(raw_response), tags=_tagkv)
            if rollout_config.get('remove_think'):
                raw_response = (lambda x: raw_response
                                if x not in raw_response else raw_response.split(x)[-1])('</think>')
            #  get_metrics_client().emit_store(
            #  "process_single_batch.agentbench.task_response_len", len(raw_response), tags=_tagkv)
            out.non_tensor_batch['raw_response'] = np.array([raw_response], dtype=object)

        extra_fill_datapack()
        extra_fill_raw_response()
        return out

    def _build_records(self, item: DataProto, agentbench_score, **kwargs):
        item.pop(non_tensor_batch_keys=['data_pack', 'raw_response'])
        item.non_tensor_batch['reward_model'] = np.array([{
            **(item.non_tensor_batch.get('reward_model', [{}])[0]),
            **{
                'agentbench_score': agentbench_score
            },
            **kwargs
        }],
                                                         dtype=object)
        item.non_tensor_batch['raw_prompt'] = np.array([[{
            **item.non_tensor_batch['raw_prompt'][0][-1],
            **{
                'meta': None,
                'name': None
            }
        }]],
                                                       dtype=object)
        item.non_tensor_batch['agent_num_turns'] = np.array([kwargs.get('num_turns', 0)], dtype=object)
        item.non_tensor_batch['agent_num_tool_calls'] = np.array([kwargs.get('num_tool_calls', 0)], dtype=object)
        return item

    def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        os.environ["no_proxy"] = ""
        tokenizer = self.tokenizer
        config = context.config
        rollout_config = context.config.actor_rollout_ref.rollout

        retry = int(config.rollout_server.get('retry', 3))
        rollout_retry = int(config.rollout_server.get('rollout_retry', 3))
        backoff_interval = float(config.rollout_server.get('backoff_interval', 0.5))
        wait_for_request_timeout = float(config.rollout_server.get('wait_for_request_timeout', 3600))

        prompt_meta = self._extract_prompt_meta(item)
        row_dict = self._extract_row_dict(item)
        meta_info = copy.copy(item.meta_info)

        tagkv_common = {
            'trial_id': os.getenv('ARNOLD_TRIAL_ID', 'unk'),
            'framework': prompt_meta.get('framework', 'unk'),
        }

        class Status(Enum):
            NON_EXIST = 1
            FINISHED = 2
            WAIT_FOR_REQUEST = 3
            WAIT_FOR_ROLLOUT = 4
            RUN_ROLLOUT = 5
            TURN_FINISHED = 6

        def check_status(task, turn_task):
            if task is None:
                return None, Status.NON_EXIST, 86400 * 365

            if task.finished():
                return None, Status.FINISHED, task.total_elapsed()

            turn_task = turn_task or task.get_pending_task()

            if turn_task:
                if turn_task.touch_elapsed() is None:
                    return turn_task, Status.WAIT_FOR_ROLLOUT, turn_task.request_elapsed()
                elif turn_task.total_elapsed() is None:
                    return turn_task, Status.RUN_ROLLOUT, turn_task.touch_elapsed()
                else:
                    return turn_task, Status.TURN_FINISHED, turn_task.total_elapsed()
            else:
                return None, Status.WAIT_FOR_REQUEST, task.touch_elapsed()

        trajectory = []
        score = None
        for trial in range(retry):
            if score is not None:
                break
            tagkv = {**tagkv_common, **{'trial': str(trial + 1)}}

            get_metrics_client().emit_counter("process_single_batch.agentbench.request", 1, tags=tagkv)

            task_id = proxy_server.add_task(**prompt_meta)

            task_details = f"{task_id=}, {prompt_meta=}"
            logging.info(f"agentbench_handler: add task[{task_details}]")

            task = proxy_server.get_task(task_id)
            turn_task = None

            rollout_trial = 0
            trajectory = []

            while True:
                _tagkv = {**tagkv, **{'turn': str(len(trajectory) + 1)}}
                turn_task, status, elapsed = check_status(task, turn_task)
                if status == Status.NON_EXIST:
                    get_metrics_client().emit_counter("process_single_batch.agentbench.task_non_exist", 1, tags=_tagkv)
                    logging.exception(f"agentbench_handler: task[{task_details}] doesn't exist, break this trial")
                    break
                elif status == Status.WAIT_FOR_REQUEST:
                    if elapsed > wait_for_request_timeout:
                        get_metrics_client().emit_counter("process_single_batch.agentbench.task_wait_timeout",
                                                          1,
                                                          tags=_tagkv)
                        logging.exception(
                            f"agentbench_hadnler: task[{task_details}] wait_for_request {elapsed} seconds, break this trial"
                        )
                        break
                    time.sleep(backoff_interval)
                elif status == Status.WAIT_FOR_ROLLOUT:
                    logging.info(
                        f"agentbench_handler: task[{task_details}] trigger rollout, turn_task_id[{turn_task.task_id}]")
                    get_metrics_client().emit_counter("process_single_batch.agentbench.rollout_run", 1, tags=_tagkv)
                    get_metrics_client().emit_store("process_single_batch.agentbench.task_wait_interval",
                                                    turn_task.request_elapsed(),
                                                    tags=_tagkv)
                    task.trigger_task(turn_task.task_id)

                    turn_item = self._preprocess(turn_task.request.messages, prompt_meta, row_dict, meta_info,
                                                 config.data.max_prompt_length, config.data.truncation, len(trajectory))

                    try:
                        rollout_start_ts = time.time()
                        completion = self.llm.complete(turn_item, rollout_config)
                        rollout_end_ts = time.time()
                        get_metrics_client().emit_store("process_single_batch.agentbench.task_inner_rollout_interval",
                                                        rollout_end_ts - rollout_start_ts,
                                                        tags=_tagkv)
                    except asyncio.CancelledError:
                        logging.exception(
                            f"agentbench_handler: task[{task_details}] rollout was cancelled, turn_task_id[{turn_task.task_id}]"
                        )
                        break
                    except Exception as e:
                        logging.exception(
                            f"agentbench_handler: task[{task_details}] rollout caught exception[{e}], turn_task_id[{turn_task.task_id}]"
                        )
                        break

                    out = self._postprocess(completion, turn_item, rollout_config)
                    proxy_server.respond_turn(turn_task.task_id, out)
                elif status == Status.RUN_ROLLOUT:
                    get_metrics_client().emit_counter("process_single_batch.agentbench.rollout_fail", 1, tags=_tagkv)
                    rollout_trial += 1
                    logging.exception(
                        f"agentbench_hadnler: task[{task_details}] rollout failed, {rollout_trial} {rollout_retry=}, turn_task_id[{turn_task.task_id}]"
                    )
                    if rollout_trial > rollout_retry:
                        break
                    task.abort_task(turn_task.task_id)
                elif status == Status.TURN_FINISHED:
                    logging.info(
                        f"agentbench_hadnler: task[{task_details}] turn rollout succeeded, turn_task_id[{turn_task.task_id}]"
                    )
                    get_metrics_client().emit_counter("process_single_batch.agentbench.task_turn_success",
                                                      1,
                                                      tags=_tagkv)
                    get_metrics_client().emit_store("process_single_batch.agentbench.task_rollout_interval",
                                                    turn_task.touch_elapsed(),
                                                    tags=_tagkv)
                    trajectory.append(turn_task)
                    turn_task = None
                elif status == Status.FINISHED:
                    logging.info(f"agentbench_hadnler: task[{task_details}] rollout succeeded")
                    get_metrics_client().emit_counter("process_single_batch.agentbench.task_success", 1, tags=tagkv)
                    get_metrics_client().emit_store("process_single_batch.agentbench.task_interval",
                                                    task.total_elapsed(),
                                                    tags=tagkv)
                    score = task.result.score
                    break
                else:
                    get_metrics_client().emit_counter("process_single_batch.agentbench.task_unkonwn_exception",
                                                      1,
                                                      tags=_tagkv)
                    logging.exception(f"agentbench_hadnler: task[{task_details}] hit unknown status, break this trial")
                    break
            proxy_server.pop_task(task_id)

        if score is not None:
            train_samples = build_training_samples(self._build_records, task, score, trajectory, rollout_config,
                                                   context)
        else:
            logging.info(f"agentbench_hadnler: task[{prompt_meta=}] score is None")
            train_samples = []
        return train_samples


if __name__ == '__main__':
    import sys
    import alpha_seed
    from mono_rl import DataProto
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from torch.utils.data import RandomSampler
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from alpha_seed.utils.dataset.rl_dataset import RLHFDataset
    from alpha_seed.workers.agents.executor import LocalExecutor
    from alpha_seed.workers.agents.handlers import select_handler_fn

    def fill_required_fields(batch, config):
        for key in ["rollout_log_probs", "probs_gt_threshold_num", "probs_lt_threshold_sum", "off_policy_steps"]:
            if key not in batch:
                batch.batch[key] = torch.zeros(
                    batch.batch["input_ids"].shape[0],
                    config.data.max_response_length,
                    dtype=torch.bfloat16,
                    device=batch.batch["input_ids"].device,
                ).fill_(-1)
        batch.meta_info["generation_kwargs"] = OmegaConf.to_container(
            config.actor_rollout_ref.rollout.train_generate_kwargs, resolve=True)
        return batch

    config = OmegaConf.load(
        f'{os.path.dirname(os.path.abspath(__file__))}/../../../../../tasks/config/ppo_trainer.yaml')
    config.data.truncation = 'left'

    tokenizer_path = copy_local_path_from_hdfs(
        "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/models/p6dense-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    data_path = copy_local_path_from_hdfs(
        "hdfs://haruna/home/byte_data_seed/hdd_wlcb/user/jiangyun.jy/datasets/rl_data/swe_gym_agentless_full_valid_filterPass_1024.parquet"
    )

    dataset = RLHFDataset(
        parquet_files=data_path,
        tokenizer=tokenizer,
        prompt_key='prompt',
        answer_key='answer',
        use_ref_answer=True,
        max_prompt_length=16384,
        multi_prompts="all",
        num_prompts_per_data=1,
        return_raw_chat=True,
    )
    sampler = RandomSampler(dataset)

    dataloader = DataLoader(dataset=dataset, batch_size=1, sampler=sampler, collate_fn=collate_fn)

    batch = next(iter(dataloader))

    gen_batch = fill_required_fields(DataProto.from_single_dict(batch), config)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout,
                        force=True)

    def chat_completions(content, meta_info, config):
        response_ids = tokenizer.encode("Hello World!") + [tokenizer.eos_token_id]
        response_log_probs = [0.] * len(response_ids)
        response_probs_gt_threshold_num = [0] * len(response_ids)
        response_probs_lt_threshold_sum = [0] * len(response_ids)
        response_model_output_mask = [1] * len(response_ids)
        return {
            'choices': [{
                'message': {
                    'raw_output_ids': response_ids,
                    'response_log_probs': response_log_probs,
                    'response_probs_gt_threshold_num': response_probs_gt_threshold_num,
                    'response_probs_lt_threshold_sum': response_probs_lt_threshold_sum,
                    'model_output_mask': response_model_output_mask,
                    'is_finished': True,
                    'extra_data': {},
                    'metrics': {}
                }
            }]
        }

    client_executor = LocalExecutor('train', config, tokenizer, None, None, "train_rollout")
    for worker in client_executor.workers:
        worker.sync_llm.chat_completions = chat_completions

    context = TaskContext(
        config=config,
        tokenizer=tokenizer,
        global_step=1,
        server_host=None,
        server_port=None,
        is_train=True,
    )
    handler = select_handler_fn("agent/agentbench/agentless")

    for item in gen_batch.chunk(len(gen_batch)):
        print(len(asyncio.run(client_executor.submit(handler, item, context))))
