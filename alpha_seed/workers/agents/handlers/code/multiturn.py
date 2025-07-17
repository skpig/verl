"""
Implement custom functions for code agent task
"""
import inspect
import json
import torch
import re
import asyncio
import numpy as np

from uuid import uuid4
from functools import reduce
from transformers import PreTrainedTokenizer
from mono_rl import DataProto
from typing import Any, Tuple, List, Dict

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import register_handler, TaskContext
from alpha_seed.workers.agents.handlers.base import AsyncAgent, AsyncLLMInterface
from alpha_seed.utils.reward_score.response_post_proc import last_codeblock_postprocess
from alpha_seed.utils.reward_score.oj_utils import compute_score


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

    async def __call__(self, input_text, ground_truth, conf):
        code = self.code_parser.extract_code(input_text, conf["last_response_strict"])
        params = {
            "solution_str": code,
            "ground_truth": ground_truth,
            "code_sandbox_psm": conf["code_sandbox_psm"],
        }
        if inspect.iscoroutinefunction(compute_score):
            score = await compute_score(**params)
        else:
            score = await asyncio.to_thread(compute_score, **params)
        return score


@register_handler("competitive_coding/multiturn")
class CodeAgent(AsyncAgent):

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        super().__init__(tokenizer, llm, **kwargs)
        self.sandbox_feedback = SandboxFeedback(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    async def _extract_messages_from_dataproto(self, item: DataProto) -> List[Dict]:
        """Extract messages from DataProto for chat template"""
        messages = [{"role": "user", "content": item.non_tensor_batch['raw_prompt'][0][0]['content']}]
        return messages

    async def _call_sandbox(self, code, ground_truth, conf):
        result = await self.sandbox_feedback(code, ground_truth, conf)
        if isinstance(result, dict):
            score, score_msg = result["score"], result.get("msg", "")
        else:
            score, score_msg = result, ""
        return score, score_msg

    def summarize(self, token_ids):
        # 将 tokens 转成字符串，建议不要自动清理空白
        text = self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

        # 匹配任何空白字符中可能存在的 <think>...</think> 区块
        pattern = r'<think>\s*.*?\s*</think>'

        # 替换所有此类区块为空字符串（删除）
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned_text

    def generate_new_prompt(self, score, score_msg):
        message = {
            'role':
                'user',
            'content':
                f'''Given the following feedback of above answer.\n===== Feedback Start =====\nThe score is : {score}\nThe environment message is: {score_msg}\n===== Feedback End =====\nAccording to the wrong code and environment message. Please re-generate a new solution with thinking.'''
        }
        return message

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

        # set input and attn mask
        item.batch['input_ids'] = torch.tensor(prompt_data.input_ids, dtype=torch.int32)
        item.batch['attention_mask'] = torch.tensor(prompt_data.attention_mask, dtype=torch.int8)

        prompt_length_before_generate = len(item.batch['input_ids'][0])

        # Call LLM with the enhanced prompt
        rollout_config = context.config.actor_rollout_ref.rollout
        max_new_tokens_this_turn = min(max_new_tokens_per_turn, max_length - prompt_length_before_generate)
        item.meta_info['generation_kwargs']['max_new_tokens'] = max_new_tokens_this_turn
        completion = await self.llm.complete(item, rollout_config)
        # pack_to_dataproto will use max_length to pad
        item.meta_info['generation_kwargs']['max_new_tokens'] = max_response_length

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
        max_new_tokens_per_turn = context.config.actor_rollout_ref.rollout.agent.max_new_tokens_per_turn
        ground_truth = item.non_tensor_batch['reward_model'][0]['ground_truth']
        is_train = context.is_train

        # Extract initial messages from DataProto
        messages = await self._extract_messages_from_dataproto(item)
        initial_input_ids = None
        initial_attn_mask = None
        model_out_mask_list = []  # 记录每次llm输出的token长度和input长度， (True or False, length)
        log_probs_list: List[List[float]] = []  # 每一轮的output log probs，input部分总是-1
        last_turn_prompt_model_output_length = 0

        completion = None
        num_turns = 1
        assert num_turns <= max_turns, "max_turns should be >= 1"

        input_lengths = []
        response_lengths = []
        raw_output_ids = []
        response_texts = []
        all_input_ids = []
        all_prompts = []

        while num_turns <= max_turns:
            # Generate response using LLM
            completion, prompt = await self._generate_code(messages, item, context, max_length, max_prompt_length,
                                                           max_response_length, max_new_tokens_per_turn, num_turns)
            # pack_to_dataproto will use max_length to pad
            item.meta_info['generation_kwargs']['max_new_tokens'] = max_response_length

            # extract the initial input with system prompts at first round
            if initial_input_ids is None:
                initial_input_ids = item.batch['input_ids'][0]
                initial_attn_mask = item.batch['attention_mask'][0]
                last_turn_prompt_model_output_length = len(initial_input_ids)

            if not completion or 'choices' not in completion:
                # could be something error in completion
                completion_str = json.dumps(completion, indent=2)
                raise ValueError(f"completion should contain at least one choice, got\n{completion_str}")

            response_message = completion['choices'][0]['message']
            input_length = len(item.batch['input_ids'][0])
            response_length = len(response_message['raw_output_ids'])
            response_text = self.tokenizer.decode(response_message['raw_output_ids'])

            all_input_ids.append(item.batch['input_ids'][0].tolist())
            all_prompts.append(prompt)
            input_lengths.append(input_length)
            response_lengths.append(response_length)
            raw_output_ids.append(response_message['raw_output_ids'])
            response_texts.append(response_text)

            # 算这一轮新增给llm的长度
            incremental_input_length = input_length - last_turn_prompt_model_output_length
            try:
                assert incremental_input_length >= 0, f"incremental_input_length should be > 0, {input_length=} {last_turn_prompt_model_output_length=}, {input_lengths=}, {response_lengths=}, {num_turns=}"
            # input_length=1495 last_turn_prompt_model_output_length=2304, tem
            # p=[268, 1495], num_turns=2
            except:
                save_info = {
                    "all_input_ids": all_input_ids,
                    "input_length": input_length,
                    "last_turn_prompt_model_output_length": last_turn_prompt_model_output_length,
                    "input_lengths": input_lengths,
                    "response_lengths": response_lengths,
                    "response_texts": response_texts,
                    "num_turns": num_turns,
                    "messages": messages,
                    "raw_output_ids": raw_output_ids
                }
                with open("incremental_input_length_error.json", "w") as f:
                    json.dump(save_info, f)
                from hdfs_io.hdfs_io import hcopy, hmkdir
                hcopy(f"incremental_input_length_error.json", context.config.trainer.default_hdfs_dir)
                raise

            model_out_mask_list.append((False, incremental_input_length))
            model_out_mask_list.append((True, response_length))
            log_probs_list.append([-1] * incremental_input_length)
            log_probs_list.append(response_message['response_log_probs'])
            last_turn_prompt_model_output_length = input_length + response_length

            # length的退出逻辑，除去initial_input_ids (prompt_length)，所有的model response + env，超出max_response_length就退出
            # 规定每轮的最大输出长度
            if last_turn_prompt_model_output_length - len(initial_input_ids) > max_response_length:
                break

            # 增加上一轮的模型输出
            response_text = self.summarize(response_message['raw_output_ids'])
            messages.append({"role": "assistant", "content": response_text})

            # Parse response and get sandbox feedback
            sandbox_config = {
                "last_response_strict": context.config.reward_model.last_response_strict,
                "code_sandbox_psm": context.config.trainer.code_sandbox_psm,
            }
            score, score_msg = await self._call_sandbox(response_text, ground_truth, sandbox_config)

            # Has Next Round
            has_next_round = (num_turns + 1 < max_turns) and (score < target_score) and (score_msg != "") and is_train
            if not has_next_round:
                break

            # add sandbox feedback as new prompt
            messages.append(self.generate_new_prompt(score, score_msg))

        # extract all outputs and logprobs
        latest_output_ids = completion['choices'][0]['message']['raw_output_ids']
        entire_seq_list = item.batch['input_ids'][0].tolist() + latest_output_ids
        total_output_ids = entire_seq_list[len(initial_input_ids):]
        log_probs = reduce(lambda x, y: x + y, log_probs_list)[len(initial_input_ids):]
        item.batch['raw_output_ids'] = torch.tensor([total_output_ids], dtype=torch.int32)
        item.batch['rollout_log_probs'] = torch.tensor([log_probs], dtype=torch.bfloat16)

        # left pad and adjust original input
        left_pad_size = context.config.data.max_prompt_length - len(initial_input_ids)
        input_ids = torch.concat(
            [torch.tensor([self.pad_token_id] * left_pad_size, dtype=torch.int32), initial_input_ids])
        attention_mask = torch.concat([torch.tensor([0] * left_pad_size, dtype=torch.int8), initial_attn_mask])
        item.batch['input_ids'] = input_ids[None, :]  # (1, max_prompt_length)
        item.batch['attention_mask'] = attention_mask[None, :]  # (1, max_prompt_length)

        # 重新组装completion
        model_output_mask = []
        for mask, length in model_out_mask_list:
            model_output_mask.extend([mask] * length)
        completion['choices'][0]['message'].update({
            'prompt': self.tokenizer.decode(total_output_ids),
            'model_output_mask': model_output_mask,
            'raw_output_ids': total_output_ids,
            'response_log_probs': log_probs,
        })

        # 将completion转换为DataProto格式，与其他agent保持一致
        from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto
        # 使用internal_call后，应该有完整的alpha-seed格式
        data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
        # FIXME(lixiang): off policy steps在多轮里还不准
        out = pack_to_dataproto(item, self.tokenizer, data_pack, context.config.actor_rollout_ref.rollout)
        out.non_tensor_batch['agent_num_turns'] = np.array([num_turns])

        return out