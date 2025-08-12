"""
Implement custom functions for code agent task
"""
import inspect
import json
import torch
import re
import asyncio
import numpy as np
import copy
import random

from uuid import uuid4
from functools import reduce
from transformers import PreTrainedTokenizer
from mono_rl import DataProto
from typing import List, Dict
import torch.nn.functional as F

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import register_handler, TaskContext
from alpha_seed.workers.agents.handlers.base import AsyncAgent, AsyncLLMInterface
from alpha_seed.utils.reward_score.response_post_proc import last_codeblock_postprocess
from alpha_seed.utils.reward_score.oj_utils import compute_score_client, compute_score
from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto, rmpad
from alpha_seed.workers.agents.handlers.tool.parser import _extract_messages_from_dataproto


class CodeParser:
    """Code parser for reponse extraction"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_code(self, input_text, last_response_strict):
        code = last_codeblock_postprocess(input_text=input_text, last_response_strict=last_response_strict)
        return code


class SandboxFeedback:
    """Get sandbox feedback from code"""

    def __init__(self, tokenizer):
        # psm placeholder for futher use
        self.code_parser = CodeParser(tokenizer)

    async def __call__(self, input_text, ground_truth, data_uid, config):
        code = self.code_parser.extract_code(input_text, config.reward_model.last_response_strict)
        params = {
            "solution_str": code,
            "ground_truth": ground_truth,
            "code_sandbox_psm": config.trainer.code_sandbox_psm,
            "data_uid": data_uid,
            "config": config,
        }
        # use remote sandbox result
        if inspect.iscoroutinefunction(compute_score_client):
            score = await compute_score_client(**params)
        else:
            score = await asyncio.to_thread(compute_score_client, **params)
        return score

        # if inspect.iscoroutinefunction(compute_score):
        #     score = await compute_score(**params)
        # else:
        #     score = await asyncio.to_thread(compute_score, **params)
        # return score


@register_handler("agent/competitive_coding/multiturn")
class CodeAgent(AsyncAgent):

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        super().__init__(tokenizer, llm, **kwargs)
        self.sandbox_feedback = SandboxFeedback(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    async def _call_sandbox(self, code, ground_truth, data_uid, config):
        result = await self.sandbox_feedback(code, ground_truth, data_uid, config)
        if isinstance(result, dict):
            score, score_msg = result["score"], result.get("msg", "")
        else:
            score, score_msg = result, ""
        return score, score_msg

    def summarize(self, token_ids):
        # remove last eos token id
        if token_ids[-1] == self.eos_token_id:
            token_ids = token_ids[:-1]

        # 将 tokens 转成字符串，建议不要自动清理空白
        text = self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

        # 匹配任何空白字符中可能存在的 <think>...</think> 区块
        pattern = r'<think>\s*.*?\s*</think>'

        # 替换所有此类区块为空字符串（删除）
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return text, cleaned_text

    def generate_new_prompt(self, score, score_msg):
        message = {
            'role':
                'user',
            'content':
                f'''Given the following feedback of above answer.\n===== Feedback Start =====\nThe score is : {score}\nThe environment message is: {score_msg}\n===== Feedback End =====\nAccording to the wrong code and environment message. Please re-generate a new solution with thinking.'''
        }
        return message

    def get_output(self, item, completion, initial_input, config):
        item = rmpad(item)
        item.meta_info['generation_kwargs']['max_new_tokens'] = config.data.max_response_length
        mask0_len = item.batch['input_ids'].shape[1] - initial_input['input_length']
        data_pack: DataPack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])

        message = completion['choices'][0]['message']
        data_pack.response_outputs = [
            item.batch['input_ids'][0][initial_input['input_length']:].tolist() + data_pack.response_outputs[0]
        ]
        data_pack.response_model_output_mask = [[False] * mask0_len + data_pack.response_model_output_mask[0]]
        data_pack.this_turn_off_policy_steps = [[0] * mask0_len + data_pack.this_turn_off_policy_steps[0]]
        data_pack.response_log_probs = [[0.0] * mask0_len + data_pack.response_log_probs[0]]

        # use initial_input as prompt inputs_ids and attention_mask
        item.batch['input_ids'] = initial_input["input_ids"]
        item.batch['attention_mask'] = initial_input["attention_mask"]

        # pack to dataproto
        out = pack_to_dataproto(item, self.tokenizer, data_pack, config.actor_rollout_ref.rollout)

        # pad to max length
        left_pad_len = config.data.max_prompt_length - initial_input['input_length']
        real_len = out.batch['attention_mask'].sum(-1)
        total_len = config.data.max_prompt_length + config.data.max_response_length
        right_pad_len = total_len - left_pad_len - real_len

        out.batch['attention_mask'] = F.pad(out.batch['attention_mask'][:, :real_len], (left_pad_len, right_pad_len),
                                            value=0)
        out.batch['input_ids'] = F.pad(out.batch['input_ids'][:, :real_len], (left_pad_len, right_pad_len),
                                       value=self.pad_token_id)

        return out

    async def _generate_code(self, messages: List[Dict], item: DataProto, context, max_length, max_prompt_length,
                             max_response_length, max_new_tokens_per_turn, num_turns):
        """Generate code response"""
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Tokenize the prompt and set it in item.batch
        # does not require padding when doing inference
        if num_turns == 1:
            max_tokenize_length = max_prompt_length
        else:
            max_tokenize_length = max_length
        prompt_data = await self.tokenizer.batch_encode_plus_async([prompt],
                                                                   add_special_tokens=False,
                                                                   max_length=max_tokenize_length,
                                                                   truncation=True)

        # set input and attn mask and uid
        item.batch['input_ids'] = torch.tensor(prompt_data.input_ids, dtype=torch.int32)
        item.batch['attention_mask'] = torch.tensor(prompt_data.attention_mask, dtype=torch.int8)

        prompt_length_before_generate = len(item.batch['input_ids'][0])

        # Call LLM with the enhanced prompt
        rollout_config = context.config.actor_rollout_ref.rollout
        max_new_tokens_this_turn = min(max_new_tokens_per_turn, max_length - prompt_length_before_generate)

        # For num_turns > 1, remained response length too short, give up
        if num_turns > 1 and max_new_tokens_this_turn < 4096:
            return None, prompt

        item.meta_info['generation_kwargs']['max_new_tokens'] = max_new_tokens_this_turn

        # different uid for each turn for remote callback
        item.meta_info['uid'] = str(uuid4())
        completion = await self.llm.complete(item, rollout_config)

        prompt_length_after_generate = len(item.batch['input_ids'][0])

        assert prompt_length_before_generate == prompt_length_after_generate

        return completion, prompt

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        """Main agent loop with multi turn code capability"""
        max_prompt_length = context.config.data.max_prompt_length
        max_response_length = context.config.data.max_response_length
        max_length = max_prompt_length + max_response_length
        max_turns = context.config.actor_rollout_ref.rollout.agent.max_turns
        target_score = context.config.actor_rollout_ref.rollout.agent.target_score
        enable_prefix_sampling = context.config.actor_rollout_ref.rollout.agent.enable_prefix_sampling
        show_error_case_prob = context.config.actor_rollout_ref.rollout.agent.show_error_case_prob
        max_new_tokens_per_turn = context.config.actor_rollout_ref.rollout.agent.max_new_tokens_per_turn
        ground_truth = item.non_tensor_batch['reward_model'][0]['ground_truth']
        is_train = context.is_train

        # data uid for each turn
        uid_list = []

        # Extract initial messages from DataProto
        messages = await _extract_messages_from_dataproto(item, max_prompt_length, self.tokenizer, self.tool_schemas)
        completion = None
        num_turns = 1
        assert num_turns <= max_turns, "max_turns should be >= 1"

        out_pool = []
        last_completion = None
        prompt_lengths = []
        response_lengths = []
        response_texts = []
        summarized_response_lengths = []
        summarized_response_texts = []

        item = rmpad(item)
        item.meta_info = copy.deepcopy(item.meta_info)
        initial_input = {
            'input_ids': item.batch['input_ids'],
            'attention_mask': item.batch['attention_mask'],
            'input_length': len(item.batch['input_ids'][0]),
        }

        while num_turns <= max_turns:
            # Generate response using LLM
            latest_completion, prompt = await self._generate_code(messages, item, context, max_length,
                                                                  max_prompt_length, max_response_length,
                                                                  max_new_tokens_per_turn, num_turns)

            # Remained response length too short, give up
            if latest_completion is None:
                break
            completion = latest_completion

            # get uid for this turn
            uid_list.append(item.non_tensor_batch['uid'][0])

            if not completion or 'choices' not in completion:
                # could be something error in completion
                completion_str = json.dumps(completion, indent=2)
                raise ValueError(f"completion should contain at least one choice, got\n{completion_str}")

            response_message = completion['choices'][0]['message']
            prompt_lengths.append(len(item.batch['input_ids'][0]))
            response_lengths.append(len(response_message['raw_output_ids']))

            # summarize上一轮的模型输出
            original_text, response_text = self.summarize(response_message['raw_output_ids'])
            response_texts.append(original_text)
            summarized_response_texts.append(response_text)
            messages.append({"role": "assistant", "content": response_text})

            # Parse response and get sandbox feedback
            score, score_msg = await self._call_sandbox(response_text, ground_truth, uid_list[-1], context.config)

            # enable_prefix_sampling 会把每一轮的output当作rollout batch
            if enable_prefix_sampling and is_train:
                out = self.get_output(item, completion, initial_input, context.config)
                out.non_tensor_batch['extra_data'][0]['score'] = {"score": score, "msg": score_msg}
                out.non_tensor_batch['agent_num_turns'] = np.array([num_turns])
                out_pool.append(out)

            # add sandbox feedback as new prompt
            if random.random() < show_error_case_prob:
                score_msg = score_msg.split("\n", 1)[0]
            messages.append(self.generate_new_prompt(score, score_msg))

            # Has Next Round
            has_next_round = (num_turns < max_turns) and (score < target_score) and (score_msg != "") and is_train
            if not has_next_round:
                break

            num_turns += 1

        # recover data uid from first round
        item.non_tensor_batch['uid'] = [uid_list[0]]

        if enable_prefix_sampling and is_train:
            out = DataProto.concat(out_pool)
        else:
            out = self.get_output(item, completion, initial_input, context.config)
            out.non_tensor_batch['extra_data'][0]['score'] = {"score": score, "msg": score_msg}
            out.non_tensor_batch['agent_num_turns'] = np.array([num_turns])
            assert out.meta_info['generation_kwargs'][
                'max_new_tokens'] == max_response_length, f"max_new_tokens should be {max_response_length}, but got {out.meta_info['generation_kwargs']['max_new_tokens']}"

        return out
