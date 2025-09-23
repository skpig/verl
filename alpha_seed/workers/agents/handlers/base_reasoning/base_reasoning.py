"""
Implement custom functions for math expression task
"""
import copy
from functools import reduce

from transformers import PreTrainedTokenizer

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import register_handler, TaskContext
from alpha_seed.workers.agents.handlers.base import AsyncAgent, AsyncLLMInterface
from alpha_seed.workers.agents.envs.textbrowser import create_from_env_str as create_textbrowser_env_from_env_str
from alpha_seed.workers.agents.envs.search import create_from_env_str as create_search_env_from_env_str
from mono_rl import DataProto
from typing import Any, List, Dict
import json
import torch
from uuid import uuid4
import numpy as np
from alpha_seed.workers.agents.handlers.tool.parser import _extract_messages_from_dataproto


@register_handler("agent/base_reasoning_handler")
class BaseReasoningAgent(AsyncAgent):

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        super().__init__(tokenizer, llm, **kwargs)
        self.tool_schemas = []
        assert hasattr(tokenizer, 'pad_token'), 'we need `pad_token` to substitute the rollout ids'

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        """Main agent loop with tool calling capability"""
        max_prompt_length = context.config.data.max_prompt_length
        max_response_length = context.config.data.max_response_length
        cur_step = context.global_step
        item.meta_info = copy.deepcopy(item.meta_info)

        # Extract initial messages from DataProto
        messages = await _extract_messages_from_dataproto(item, max_prompt_length, self.tokenizer, self.tool_schemas)
        model_out_mask_list = []  # 记录每次llm输出的token长度和input长度， (True or False, length)
        log_probs_list: List[List[float]] = []  # 每一轮的output log probs，input部分总是-1

        completion, _ = await self._generate_with_tools(messages, item, context, max_prompt_length, max_response_length)
        initial_input_ids = item.batch['input_ids'][0]
        initial_attn_mask = item.batch['attention_mask'][0]

        if not completion or 'choices' not in completion:
            # could be something error in completion
            completion_str = json.dumps(completion, indent=2)
            raise ValueError(f"completion should contain at least one choice, got\n{completion_str}")

        # 这几个量直接用，最好不要改
        # 比如response_length指的是rollout出来的ids的length，不能是decode response_text得到的length，这两个不一定相等
        response_message = completion['choices'][0]['message']
        response_length = len(response_message['raw_output_ids'])

        model_out_mask_list.append((True, response_length))
        log_probs_list.append(response_message['response_log_probs'])

        # extract all outputs and logprobs
        latest_output_ids = completion['choices'][0]['message']['raw_output_ids']
        entire_seq_list = item.batch['input_ids'][0].tolist() + latest_output_ids
        total_output_ids = entire_seq_list[len(initial_input_ids):][:max_response_length]
        log_probs = reduce(lambda x, y: x + y, log_probs_list)[:max_response_length]
        item.batch['raw_output_ids'] = torch.tensor([total_output_ids], dtype=torch.int32)

        # left pad and adjust original input
        left_pad_size = max_prompt_length - len(initial_input_ids)
        input_ids = torch.concat(
            [torch.tensor([self.tokenizer.pad_token_id] * left_pad_size, dtype=torch.int32), initial_input_ids])
        attention_mask = torch.concat([torch.tensor([0] * left_pad_size, dtype=torch.int8), initial_attn_mask])
        item.batch['input_ids'] = input_ids[None, :][:, -max_prompt_length:]  # (1, max_prompt_length)
        item.batch['attention_mask'] = attention_mask[None, :][:, -max_prompt_length:]  # (1, max_prompt_length)

        # 重新组装completion
        model_output_mask = []
        for mask, length in model_out_mask_list:
            model_output_mask.extend([mask] * length)
        model_output_mask = model_output_mask[:max_response_length]
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
        out.non_tensor_batch['agent_num_turns'] = np.array([0])
        out.non_tensor_batch['agent_num_tool_calls'] = np.array([0])
        out.meta_info["cur_step"] = cur_step

        return out

    async def _generate_with_tools(self, messages: List[Dict], item: DataProto, context, max_prompt_length,
                                   max_response_length):
        """Generate response with tool schemas included"""
        # Apply chat template with tools
        prompt_with_tools = self.tokenizer.apply_chat_template(messages,
                                                               tools=self.tool_schemas,
                                                               add_generation_prompt=True,
                                                               tokenize=False)
        soi = context.config.data.special_tokens.soi
        eoi = context.config.data.special_tokens.eoi
        prompts = prompt_with_tools.replace("<image>", f"{soi}<ImageHere>{eoi}").split("<ImageHere>")
        if 'num_image_tokens' in item.non_tensor_batch:
            num_image_tokens = item.non_tensor_batch['num_image_tokens'][0]
        else:
            num_image_tokens = None
        if num_image_tokens is None or len(num_image_tokens) == 0:
            num_image_tokens = []
        img_idx = 0
        input_ids = []
        attention_masks = []
        for prompt in prompts:
            if prompt == "":
                continue
            prompt_data = await self.tokenizer.batch_encode_plus_async([prompt], add_special_tokens=False)
            input_ids += prompt_data.input_ids[0]
            attention_masks += prompt_data.attention_mask[0]
            if img_idx < len(num_image_tokens):
                num_img_token = num_image_tokens[img_idx]
                input_ids += [-100] * num_img_token
                attention_masks += [1] * num_img_token
            img_idx += 1

        # set input and attn mask
        item.batch['input_ids'] = torch.tensor([input_ids], dtype=torch.int32)[:, -max_prompt_length:]
        item.batch['attention_mask'] = torch.tensor([attention_masks], dtype=torch.int8)[:, -max_prompt_length:]

        prompt_length_before_generate = len(item.batch['input_ids'][0])
        # Call LLM with the enhanced prompt
        rollout_config = context.config.actor_rollout_ref.rollout
        item.meta_info['generation_kwargs']['max_new_tokens'] = max_response_length
        completion = await self.llm.complete(item, rollout_config)
        # pack_to_dataproto will use max_length to pad
        item.meta_info['generation_kwargs']['max_new_tokens'] = max_response_length

        prompt_length_after_generate = len(item.batch['input_ids'][0])

        assert prompt_length_before_generate == prompt_length_after_generate

        return completion, prompt_with_tools
