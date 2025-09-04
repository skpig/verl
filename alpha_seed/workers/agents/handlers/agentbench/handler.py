import os
import time
import logging
import copy
import uuid
import asyncio
import torch

import numpy as np
import verl.utils.torch_functional as verl_F

from enum import Enum
from dataclasses import dataclass, fields
from typing import List, Dict
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig
from mono_rl import DataProto
from alpha_seed.utils.dataset.rl_dataset import collate_fn
from alpha_seed.workers.agents.handlers import register_handler, TaskContext
from alpha_seed.workers.agents.handlers.base import ThreadedAgent
from alpha_seed.workers.agents.handlers.agentbench.proxy import get_metrics_client, get_proxy_client
from alpha_seed.workers.agents.handlers.agentbench.trajectory_manager import build_training_samples
from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto


class AgentHandler(ThreadedAgent):

    def __init__(self, tokenizer, llm, **kwargs):
        super().__init__(tokenizer, llm)
        self._pad_token_id = self.tokenizer.pad_token_id
        self._eos_token_id = self.tokenizer.eos_token_id

    def _extract_prompt_meta(self, item: DataProto) -> Dict:
        framework, dataset, index = item.non_tensor_batch['raw_prompt'][0][0].get('meta').split(":")
        agent_config = item.non_tensor_batch['extra_info'][0].get('agent_config', {})
        if isinstance(agent_config, str):
            agent_config = json.loads(agent_config)
        return {"framework": framework, "dataset": dataset, "index": index, "agent_config": agent_config}

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

    def _preprocess(self, task_id, turn_task, prompt_meta: Dict, row_dict: Dict, meta_info: Dict,
                    rsp_left_truncation: bool, max_new_tokens_per_turn: int, max_length: int, pad_to_max_length: bool,
                    truncation: str, sub_index: int, rollout_config: DictConfig, tagkv: Dict) -> DataProto:

        # just a copy from verl with minor change
        def _tokenize_and_postprocess_data(prompt: str,
                                           tokenizer: PreTrainedTokenizer,
                                           max_length: int,
                                           pad_to_max_length: bool,
                                           pad_token_id: int,
                                           left_pad=True,
                                           truncation='error'):
            """
            input_data is the output from tokenizer.
            """
            assert truncation in ['left', 'right', 'error']

            input_data = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

            input_ids = input_data['input_ids']
            attention_mask = input_data['attention_mask']

            assert input_ids.ndim == 2

            sequence_length = input_ids.shape[-1]
            truncated = False
            if pad_to_max_length and sequence_length < max_length:
                input_ids = verl_F.pad_sequence_to_length(input_ids,
                                                          max_seq_len=max_length,
                                                          pad_token_id=pad_token_id,
                                                          left_pad=left_pad)
                attention_mask = verl_F.pad_sequence_to_length(attention_mask,
                                                               max_seq_len=max_length,
                                                               pad_token_id=0,
                                                               left_pad=left_pad)
            elif sequence_length > max_length:
                if truncation == 'left':
                    # actually, left truncation may not be reasonable
                    input_ids = input_ids[:, -max_length:]
                    attention_mask = attention_mask[:, -max_length:]
                elif truncation == 'right':
                    input_ids = input_ids[:, :max_length]
                    attention_mask = attention_mask[:, :max_length]
                elif truncation == 'error':
                    raise NotImplementedError(f'{sequence_length=} is larger than {max_length=}')
                else:
                    raise NotImplementedError(f'Unknown truncation method {truncation}')
                truncated = True

            return input_ids, attention_mask, truncated

        if rsp_left_truncation:
            prompt_messages = []
            first_assistant_idx = len(messages)
            for idx, message in enumerate(messages):
                if message['role'] == 'assistant':
                    first_assistant_idx = idx
                    break
            prompt_messages = messages[:first_assistant_idx]
            response_messages = messages[first_assistant_idx:]
            prompt_with_chat_template = self.tokenizer.apply_chat_template(prompt_messages,
                                                                           add_generation_prompt=True,
                                                                           tokenize=False)
            p_ids, p_attention_mask, p_truncated = _tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                                  tokenizer=self.tokenizer,
                                                                                  max_length=max_length,
                                                                                  pad_to_max_length=False,
                                                                                  pad_token_id=0,
                                                                                  truncation=truncation)
            if len(response_messages) > 0:
                response_with_chat_template = self.tokenizer.apply_chat_template(response_messages,
                                                                                 add_generation_prompt=False,
                                                                                 tokenize=False)
                r_ids, r_attention_mask, r_truncated = _tokenize_and_postprocess_data(
                    prompt=response_with_chat_template,
                    tokenizer=self.tokenizer,
                    max_length=max_length - p_ids.shape[-1] - max_new_tokens_per_turn,
                    pad_to_max_length=False,
                    pad_token_id=0,
                    truncation='left')
                input_ids = torch.cat([p_ids, r_ids], dim=-1)
                attention_mask = torch.cat([p_attention_mask, r_attention_mask], dim=-1)
                truncated = p_truncated or r_truncated
            else:
                input_ids = p_ids
                attention_mask = p_attention_mask
                truncated = p_truncated
        else:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(turn_task.request.messages,
                                                                           add_generation_prompt=True,
                                                                           tokenize=False)
            input_ids, attention_mask, truncated = _tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                                  tokenizer=self.tokenizer,
                                                                                  max_length=max_length,
                                                                                  pad_to_max_length=pad_to_max_length,
                                                                                  pad_token_id=self._pad_token_id,
                                                                                  left_pad=True,
                                                                                  truncation=truncation)
        prompt = {"input_ids": input_ids[0].to(torch.int32), "attention_mask": attention_mask[0].to(torch.int8)}

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
            **{
                '__AGENTBENCH_task_id':
                    task_id,
                '__AGENTBENCH_turn_task_id':
                    turn_task.task_id,
                '__AGENTBENCH_traj_id':
                    turn_task.request.extra_info.traj_id if turn_task.request.extra_info is not None else task_id,
            },
        }
        _item = DataProto.from_single_dict(collate_fn([_row_dict]))
        _item.meta_info = copy.copy(meta_info)
        _item.meta_info['generation_kwargs']['max_new_tokens'] = max_new_tokens_per_turn
        return {'item': _item, 'truncated': truncated}

    def _postprocess(self, completion, item: DataProto, origin_prompt, rollout_config: DictConfig,
                     tagkv: Dict) -> DataProto:
        data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
        out = pack_to_dataproto(item, self.tokenizer, data_pack, rollout_config)  # dataproto
        out.non_tensor_batch['prompt_names'] = copy.deepcopy(origin_prompt.get('prompt_names'))
        out.non_tensor_batch['raw_prompt'] = copy.deepcopy(origin_prompt.get('raw_prompt'))

        def extra_fill_input_ids():
            out.non_tensor_batch['__AGENTBENCH_input_ids'] = np.array([item.batch['input_ids'][0]], dtype=object)

        def extra_fill_datapack():
            out.non_tensor_batch['__AGENTBENCH_data_pack'] = np.array([data_pack], dtype=object)

        def extra_fill_raw_response():
            raw_output_ids = completion['choices'][0]['message']['raw_output_ids']
            if self._eos_token_id == raw_output_ids[-1]:
                raw_output_ids = raw_output_ids[:-1]
            raw_response = ''.join(self.tokenizer.decode(raw_output_ids, skip_special_tokens=False))
            get_metrics_client().emit_timer("agentbench.handler.task_raw_response_len", len(raw_response), tags=tagkv)
            if rollout_config.get('remove_think'):
                raw_response = (lambda x: raw_response
                                if x not in raw_response else raw_response.split(x)[-1])('</think>')
            get_metrics_client().emit_timer("agentbench.handler.task_response_len", len(raw_response), tags=tagkv)
            out.non_tensor_batch['raw_response'] = np.array([raw_response], dtype=object)

        extra_fill_input_ids()
        extra_fill_datapack()
        extra_fill_raw_response()
        return {'item': out}

    def _build_records(self, item: DataProto, agentbench_score, **kwargs):
        item.pop(non_tensor_batch_keys=[
            'raw_response', '__AGENTBENCH_task_id', '__AGENTBENCH_turn_task_id', '__AGENTBENCH_traj_id',
            '__AGENTBENCH_input_ids', '__AGENTBENCH_data_pack'
        ])
        item.non_tensor_batch['reward_model'] = np.array([{
            **(item.non_tensor_batch.get('reward_model', [{}])[0]),
            **{
                'agentbench_score': agentbench_score
            },
            **kwargs
        }],
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
        wait_for_task_timeout = float(config.rollout_server.get('wait_for_task_timeout', 1200))
        wait_for_request_timeout = float(config.rollout_server.get('wait_for_request_timeout', 3600))

        origin_prompt = {
            'prompt_names': copy.deepcopy(item.non_tensor_batch['prompt_names']),
            'raw_prompt': copy.deepcopy(item.non_tensor_batch['raw_prompt'])
        }
        prompt_meta = self._extract_prompt_meta(item)
        row_dict = self._extract_row_dict(item)
        meta_info = copy.copy(item.meta_info)

        tagkv_common = {
            'trial_id': os.getenv('ARNOLD_TRIAL_ID', 'unk'),
            'framework': prompt_meta.get('framework', 'unk'),
        }
        max_new_tokens_per_turn = rollout_config.agent.max_new_tokens_per_turn
        max_length = (config.data.max_prompt_length + config.data.max_response_length) if prompt_meta.get(
            'framework', 'unk') != 'agentless' else config.data.max_prompt_length
        pad_to_max_length = prompt_meta.get('framework', 'unk') == 'agentless'
        truncation = config.data.truncation
        rsp_left_truncation = rollout_config.get('response_left_truncation', False)

        if rsp_left_truncation:
            assert rollout_config.agent.max_new_tokens_per_turn < config.data.max_response_length, \
                f'{rollout_config.agent.max_new_tokens_per_turn=} should be smaller ' \
                f'than {config.data.max_response_length=} when response_left_truncation is True'

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
            turn_task_id = get_proxy_client().fetch_pending_turn_task(
                task.task_id) if not turn_task else turn_task.task_id
            turn_task = get_proxy_client().get_turn_task_meta(turn_task_id) if turn_task_id else None
            if turn_task:
                if turn_task.touch_elapsed() is None:
                    return turn_task, Status.WAIT_FOR_ROLLOUT, turn_task.request_elapsed()
                elif turn_task.total_elapsed() is None:
                    return turn_task, Status.RUN_ROLLOUT, turn_task.touch_elapsed()
                else:
                    return turn_task, Status.TURN_FINISHED, turn_task.total_elapsed()
            else:
                return None, Status.WAIT_FOR_REQUEST, task.touch_elapsed()

        task = None
        score = None
        trajectory = []
        success = False

        @dataclass
        class TS:
            trigger_ts: float = None
            trigger_trial_ts: float = None
            trigger_completion_ts: float = None
            finish_completion_ts: float = None
            trigger_rollout_ts: float = None
            finish_rollout_ts: float = None
            finish_trial_ts: float = None
            finish_ts: float = None

            def _mark(self, name: str) -> bool:
                if getattr(self, name) is not None:
                    return False
                now = time.time()
                setattr(self, name, now)
                names = [f.name for f in fields(self)]
                idx = names.index(name)
                for prev in names[:idx]:
                    if getattr(self, prev) is None:
                        setattr(self, prev, now)
                return True

            def _reset_from(self, name: str):
                names = [f.name for f in fields(self)]
                idx = names.index(name)
                for n in names[idx:]:
                    setattr(self, n, None)

            def _interval(self, start_name: str, end_name: str):
                now = time.time()
                start = getattr(self, start_name) or now
                end = getattr(self, end_name) or now
                return end - start

            def trigger(self):
                return self._mark('trigger_ts')

            def trigger_trial(self):
                self._reset_from('trigger_trial_ts')
                return self._mark('trigger_trial_ts')

            def trigger_completion(self):
                return self._mark('trigger_completion_ts')

            def finish_completion(self):
                return self._mark('finish_completion_ts')

            def trigger_rollout(self):
                return self._mark('trigger_rollout_ts')

            def finish_rollout(self):
                return self._mark('finish_rollout_ts')

            def finish_trial(self):
                return self._mark('finish_trial_ts')

            def finish(self):
                return self._mark('finish_ts')

            def elapsed(self):
                return self._interval('trigger_ts', 'finish_ts')

            def trial_elapsed(self):
                return self._interval('trigger_trial_ts', 'finish_trial_ts')

            def completion_elapsed(self):
                return self._interval('trigger_completion_ts', 'finish_completion_ts')

            def rollout_elapsed(self):
                return self._interval('trigger_rollout_ts', 'finish_rollout_ts')

            def re_rollout(self):
                self._reset_from('trigger_rollout_ts')

            def next_turn(self):
                self._reset_from('trigger_completion_ts')
                self.trigger_completion()

        ts = TS()

        attempt_messages = []

        ts.trigger()
        for trial in range(retry):
            if score is not None:
                break
            tagkv = {**tagkv_common, **{'trial': str(trial + 1)}}

            get_metrics_client().emit_counter("agentbench.handler.request", 1, tags=tagkv)

            ts.trigger_trial()
            task_id = get_proxy_client().produce(**prompt_meta)

            task_details = f"{task_id=}, {prompt_meta=}"
            logging.info(f"agentbench_handler: add task[{task_details}]")

            task_meta_info = None
            turn_task_meta_info = None

            rollout_trial = 0
            trajectory = []

            while True:
                task_meta_info = get_proxy_client().get_task_meta(task_id)
                _tagkv = {**tagkv, **{'turn': str(len(trajectory) + 1)}}
                turn_task_meta_info, status, elapsed = check_status(task_meta_info, turn_task_meta_info)
                if status == Status.NON_EXIST:
                    elapsed = ts.trial_elapsed()
                    if elapsed > wait_for_task_timeout:
                        get_metrics_client().emit_counter("agentbench.handler.trigger_timeout", 1, tags=_tagkv)
                        logging.exception(f"agentbench_handler: task[{task_details}] doesn't exist, break this trial")
                        break
                    time.sleep(backoff_interval)
                elif status == Status.WAIT_FOR_REQUEST:
                    if ts.trigger_completion():
                        get_metrics_client().emit_timer("agentbench.handler.consume_elapsed",
                                                        ts.trial_elapsed(),
                                                        tags=tagkv)
                    if elapsed > wait_for_request_timeout:
                        get_metrics_client().emit_counter("agentbench.handler.wait_timeout", 1, tags=_tagkv)
                        logging.exception(
                            f"agentbench_handler: task[{task_details}] wait_for_request {elapsed} seconds, break this trial"
                        )
                        break
                    time.sleep(backoff_interval)
                elif status == Status.WAIT_FOR_ROLLOUT:
                    ts.finish_completion()
                    get_metrics_client().emit_timer("agentbench.handler.completion_elapsed",
                                                    ts.completion_elapsed(),
                                                    tags=_tagkv)

                    ts.trigger_rollout()

                    logging.info(
                        f"agentbench_handler: task[{task_details}] trigger rollout, turn_task_id[{turn_task_meta_info.task_id}]"
                    )
                    get_metrics_client().emit_counter("agentbench.handler.rollout_run", 1, tags=_tagkv)
                    get_metrics_client().emit_timer("agentbench.handler.rollout_task_wait_elapsed",
                                                    turn_task_meta_info.request_elapsed(),
                                                    tags=_tagkv)
                    get_proxy_client().trigger_turn(turn_task_meta_info.task_id)

                    turn_task = get_proxy_client().get_turn(turn_task_meta_info.task_id)
                    preprocess_output = self._preprocess(task_id, turn_task, prompt_meta, row_dict, meta_info,
                                                         rsp_left_truncation, max_new_tokens_per_turn, max_length,
                                                         pad_to_max_length, truncation, len(trajectory), rollout_config,
                                                         _tagkv)
                    if preprocess_output.get('truncated') and prompt_meta.get('framework', 'unk') != 'agentless':
                        _item = DataProto.from_dict(non_tensors={
                            'raw_response':
                                np.array([f'The task {turn_task_meta_info.task_id} is truncated'], dtype=str)
                        })
                    else:
                        turn_item = preprocess_output.get('item')
                        try:
                            completion = self.llm.complete(turn_item, rollout_config)
                        except asyncio.CancelledError:
                            logging.exception(
                                f"agentbench_handler: task[{task_details}] rollout was cancelled, turn_task_id[{turn_task_meta_info.task_id}]"
                            )
                            break
                        except Exception as e:
                            logging.exception(
                                f"agentbench_handler: task[{task_details}] rollout caught exception[{e}], turn_task_id[{turn_task_meta_info.task_id}]"
                            )
                            break
                        postprocess_output = self._postprocess(completion, turn_item, origin_prompt, rollout_config,
                                                               _tagkv)
                        _item = postprocess_output.get('item')
                    get_proxy_client().respond_turn(turn_task_meta_info.task_id, _item)
                elif status == Status.RUN_ROLLOUT:
                    get_metrics_client().emit_counter("agentbench.handler.rollout_fail", 1, tags=_tagkv)
                    rollout_trial += 1
                    logging.exception(
                        f"agentbench_handler: task[{task_details}] rollout failed, {rollout_trial} {rollout_retry=}, turn_task_id[{turn_task_meta_info.task_id}]"
                    )
                    if rollout_trial > rollout_retry:
                        break
                    get_proxy_client().abort_turn(turn_task_meta_info.task_id)
                    ts.re_rollout()
                elif status == Status.TURN_FINISHED:
                    ts.finish_rollout()
                    get_metrics_client().emit_timer("agentbench.handler.rollout_elapsed",
                                                    ts.rollout_elapsed(),
                                                    tags=_tagkv)

                    turn_task = get_proxy_client().get_turn(turn_task_meta_info.task_id)
                    logging.info(
                        f"agentbench_handler: task[{task_details}] turn rollout succeeded, turn_task_id[{turn_task_meta_info.task_id}]"
                    )
                    get_metrics_client().emit_counter("agentbench.handler.turn_success", 1, tags=_tagkv)
                    get_metrics_client().emit_timer("agentbench.handler.rollout_task_rollout_elapsed",
                                                    turn_task_meta_info.touch_elapsed(),
                                                    tags=_tagkv)
                    trajectory.append(turn_task)
                    turn_task_meta_info = None
                    ts.next_turn()
                elif status == Status.FINISHED:
                    ts.finish_completion()
                    get_metrics_client().emit_timer("agentbench.handler.score_elapsed",
                                                    ts.completion_elapsed(),
                                                    tags=_tagkv)
                    ts.finish_trial()

                    task = get_proxy_client().get_task(task_meta_info.task_id)
                    logging.info(f"agentbench_handler: task[{task_details}] rollout succeeded")
                    get_metrics_client().emit_counter("agentbench.handler.success", 1, tags=tagkv)
                    get_metrics_client().emit_timer("agentbench.handler.task_elapsed", task.total_elapsed(), tags=tagkv)
                    score = task.result.score
                    success = True
                    break
                else:
                    get_metrics_client().emit_counter("agentbench.handler.unkonwn_exception", 1, tags=_tagkv)
                    logging.exception(f"agentbench_handler: task[{task_details}] hit unknown status, break this trial")
                    break

            attempt_messages.append({'task_id': task_id, 'status': status.name, 'turn': len(trajectory) + 1})

            get_proxy_client().pop_task(task_id)

            get_metrics_client().emit_timer("agentbench.handler.trial_elapsed",
                                            ts.trial_elapsed(),
                                            tags={
                                                **tagkv,
                                                **{
                                                    'status': 'success' if success else 'fail'
                                                }
                                            })
            get_metrics_client().emit_timer("agentbench.handler.trial_total_turn",
                                            len(trajectory),
                                            tags={
                                                **tagkv,
                                                **{
                                                    'status': 'success' if success else 'fail'
                                                }
                                            })

        ts.finish()
        get_metrics_client().emit_timer("agentbench.handler.elapsed",
                                        ts.elapsed(),
                                        tags={
                                            **tagkv_common,
                                            **{
                                                'trial': str(trial),
                                                'status': 'success' if success else 'fail'
                                            }
                                        })
        get_metrics_client().emit_timer("agentbench.handler.trial",
                                        trial,
                                        tags={
                                            **tagkv_common,
                                            **{
                                                'status': 'success' if success else 'fail'
                                            }
                                        })
        get_metrics_client().emit_timer("agentbench.handler.total_turn",
                                        len(trajectory),
                                        tags={
                                            **tagkv_common,
                                            **{
                                                'trial': str(trial),
                                                'status': 'success' if success else 'fail'
                                            }
                                        })
        if score is not None:
            train_samples = build_training_samples(self._build_records,
                                                   task,
                                                   score,
                                                   trajectory,
                                                   rollout_config,
                                                   tokenizer=tokenizer,
                                                   max_prompt_length=config.data.max_prompt_length,
                                                   max_response_length=config.data.max_response_length)
        else:
            logging.info(f"agentbench_handler: task[{prompt_meta=}] score is None")
            train_samples = []

        for _ in train_samples:

            def off_policy_steps(start_step):
                return self.global_state.get_global_step() - start_step

            get_metrics_client().emit_timer("agentbench.handler.off_policy_steps",
                                            off_policy_steps(context.global_step),
                                            tags={
                                                **tagkv_common,
                                                **{
                                                    'trial': str(trial),
                                                }
                                            })

            def response_status(score):
                if score == -1:
                    return 'infer_fail'
                elif score == -100000:
                    return 'eval_fail'
                else:
                    return 'normal'

            status = response_status(score)
            get_metrics_client().emit_counter("agentbench.handler.response",
                                              1,
                                              tags={
                                                  **tagkv_common,
                                                  **{
                                                      'trial': str(trial),
                                                      'status': status
                                                  }
                                              })
        if not success:
            logging.exception(
                f"agentbench_handler: task[{prompt_meta=}] is failed, attempt messages are {attempt_messages}")
        return train_samples


@register_handler("agent/agentbench")
class Agentbench(AgentHandler):
    pass


@register_handler("agent/agentbench/agentless")
class Agentless(AgentHandler):
    pass


if __name__ == '__main__':
    import sys
    import yaml
    import ray
    import alpha_seed
    import alpha_seed.workers.agents.executor
    import alpha_seed.workers.agents.handlers.agentbench
    import importlib
    from mono_rl import DataProto
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from torch.utils.data import RandomSampler
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from alpha_seed.utils.dataset.rl_dataset import RLHFDataset
    from alpha_seed.workers.agents.executor import LocalExecutor
    from alpha_seed.workers.agents.handlers import select_handler_fn

    os.environ["AGENTBENCH_ENABLE"] = "True"
    os.environ["AGENTBENCH_DEBUG_MODE"] = "True"
    os.environ["AGENTBENCH_STORAGE_MODE"] = "ray"
    os.environ["AGENTBENCH_STORAGE_MODE"] = "local"
    os.environ["AGENTBENCH_STORAGE_SHARD_NUM"] = "3"

    importlib.reload(alpha_seed.workers.agents.handlers.agentbench)
    if os.environ["AGENTBENCH_STORAGE_MODE"] == 'ray':
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../../../../tasks/runtime_env/runtime_env.yaml'
                 ) as fin:
            runtime_env = yaml.safe_load(fin)
            for k in [
                    "AGENTBENCH_ENABLE", "AGENTBENCH_DEBUG_MODE", "AGENTBENCH_STORAGE_MODE",
                    "AGENTBENCH_STORAGE_SHARD_NUM"
            ]:
                runtime_env[k] = os.environ.get(k)
            print(runtime_env)
            ray.init(namespace="alphaseed", runtime_env=runtime_env, address='auto')

    def fill_required_fields(batch, config):
        for key in ["rollout_behavior_log_probs", "off_policy_steps", "model_output_mask"]:
            if key not in batch:
                batch.batch[key] = torch.zeros(
                    batch.batch["input_ids"].shape[0],
                    config.data.max_response_length,
                    dtype=torch.bfloat16,
                    device=batch.batch["input_ids"].device,
                ).fill_(-1)
        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
        batch.non_tensor_batch['rollout_id'] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
        batch.meta_info["generation_kwargs"] = OmegaConf.to_container(
            config.actor_rollout_ref.rollout.train_generate_kwargs, resolve=True)
        return batch

    config = OmegaConf.load(
        f'{os.path.dirname(os.path.abspath(__file__))}/../../../../../tasks/config/ppo_trainer.yaml')
    config.data.truncation = 'left'
    config.data.max_prompt_length = 16384
    config.data.max_response_length = 49152
    config.rollout_server.response_left_truncation = True
    config.rollout_server.agent.enable_monitoring = False
    config.rollout_server.agent.direct_submit_query = False
    config.elastic.resource_pools.stable_pool_names = ''

    tokenizer_path = copy_local_path_from_hdfs(
        "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/models/p6dense-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    data_path = copy_local_path_from_hdfs(
        "hdfs://harunawl/home/byte_data_seed_wl/user/jiangyun.jy/datasets/rl_data/F0808_openhands_rebench4968.parquet")

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

    alpha_seed.workers.agents.executor.get_agent_metrics_collector = lambda: False

    def chat_completions(content, meta_info, config):
        response_ids = tokenizer.encode("Hello World!") + [tokenizer.eos_token_id]
        response_log_probs = [0.] * len(response_ids)
        response_model_output_mask = [1] * len(response_ids)
        return {
            'choices': [{
                'message': {
                    'raw_output_ids': response_ids,
                    'response_log_probs': response_log_probs,
                    'model_output_mask': response_model_output_mask,
                    'is_finished': True,
                    'extra_data': {},
                    'metrics': {}
                }
            }]
        }

    loop = asyncio.new_event_loop()
    client_executor = LocalExecutor('train', config, tokenizer, None, None, None, "train_rollout", loop)
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
    handler = select_handler_fn("agent/agentbench")

    for item in gen_batch.chunk(len(gen_batch)):
        print(len(loop.run_until_complete(client_executor.submit(handler, item, context))))
