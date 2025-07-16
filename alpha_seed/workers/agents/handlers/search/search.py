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
import regex as re
from uuid import uuid4
from omegaconf import DictConfig
import numpy as np
import ray


class FunctionCall:

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class HermesToolParser:
    """Tool parser for Hermes format, adapted from verl"""

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.tool_call_start_token = config.rollout_server.tool_call_start_token
        self.tool_call_end_token = config.rollout_server.tool_call_end_token
        self.tool_call_regex = re.compile(
            f"{self.tool_call_start_token}(.*?){self.tool_call_end_token}".replace("|", "\|"), re.DOTALL)

    async def extract_tool_calls(self, response_text: str) -> List[FunctionCall]:
        """Extract tool calls from response text"""
        if self.tool_call_start_token not in response_text or self.tool_call_end_token not in response_text:
            return []

        matches = self.tool_call_regex.findall(response_text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                if isinstance(function_call, list):
                    for f in function_call:
                        name, arguments = f["name"], f["arguments"] if "arguments" in f else f["parameters"]
                        function_calls.append(
                            FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
                else:
                    name, arguments = function_call["name"], function_call[
                        "arguments"] if "arguments" in function_call else function_call["parameters"]
                    function_calls.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
            except Exception as e:
                pass  # Skip invalid tool calls
        return function_calls


@register_handler("agent/tool/search_and_text_browser")
class ToolAgent(AsyncAgent):

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        super().__init__(tokenizer, llm)
        self.search = create_search_env_from_env_str("deep_research/search@{}", tokenizer=tokenizer)
        self.textbrowser = create_textbrowser_env_from_env_str("deep_research/textbrowser@{}", tokenizer=tokenizer)
        self.tool_parser = HermesToolParser(tokenizer, self.config)
        self.tools = {"Search": self.search, "TextBrowser": self.textbrowser}
        # Get tool schema for the calculator
        self.tool_schemas = [
            self.search.get_openai_tool_schema().model_dump(exclude_unset=True, exclude_none=True),
            self.textbrowser.get_openai_tool_schema().model_dump(exclude_unset=True, exclude_none=True)
        ]
        assert hasattr(tokenizer, 'pad_token'), 'we need `pad_token` to substitute the rollout ids'

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        """Main agent loop with tool calling capability"""
        max_prompt_length = context.config.data.max_prompt_length
        max_response_length = context.config.data.max_response_length
        max_length = max_prompt_length + max_response_length
        max_turns = context.config.actor_rollout_ref.rollout.agent.max_turns
        max_new_tokens_per_turn = context.config.actor_rollout_ref.rollout.agent.max_new_tokens_per_turn
        item.meta_info = copy.deepcopy(item.meta_info)

        # Extract initial messages from DataProto
        messages = await self._extract_messages_from_dataproto(item, max_prompt_length)
        initial_input_ids = None
        initial_attn_mask = None
        model_out_mask_list = []  # 记录每次llm输出的token长度和input长度， (True or False, length)
        log_probs_list: List[List[float]] = []  # 每一轮的output log probs，input部分总是-1
        last_turn_prompt_model_output_length = 0

        completion = None
        num_turns = 1
        num_tool_calls = 0
        assert num_turns <= max_turns, "max_turns should be >= 1"

        prompt_lengths = []
        response_lengths = []
        raw_output_ids = []
        response_texts = []
        all_input_ids = []
        all_prompts = []
        response_info = []

        while num_turns <= max_turns:
            # Generate response using LLM
            completion, prompt = await self._generate_with_tools(messages, item, context, max_length, max_prompt_length,
                                                                 max_response_length, max_new_tokens_per_turn,
                                                                 num_turns, response_info)
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

            # 这几个量直接用，最好不要改
            # 比如response_length指的是rollout出来的ids的length，不能是decode response_text得到的length，这两个不一定相等
            response_message = completion['choices'][0]['message']
            prompt_length = len(item.batch['input_ids'][0])
            response_length = len(response_message['raw_output_ids'])
            response_text = response_message['prompt']

            all_input_ids.append(item.batch['input_ids'][0].tolist())
            all_prompts.append(prompt)
            prompt_lengths.append(prompt_length)
            response_lengths.append(response_length)
            raw_output_ids.append(response_message['raw_output_ids'])
            response_texts.append(response_text)
            response_info.append({
                'prompt_length': prompt_length,
                'response_length': response_length,
                'raw_output_ids': response_message['raw_output_ids']
            })

            # 算这一轮新增给llm的长度（可能是上一轮的tool call的结果等）
            incremental_input_length = prompt_length - last_turn_prompt_model_output_length
            assert incremental_input_length >= 0, f"incremental_input_length should be > 0, {prompt_length=} {last_turn_prompt_model_output_length=}, {prompt_lengths=}, {response_lengths=}, {num_turns=}"

            model_out_mask_list.append((False, incremental_input_length))
            model_out_mask_list.append((True, response_length))
            log_probs_list.append([-1] * incremental_input_length)
            log_probs_list.append(response_message['response_log_probs'])
            last_turn_prompt_model_output_length = prompt_length + response_length

            # length的退出逻辑，除去initial_input_ids (prompt_length)，所有的model response + env，超出max_response_length就退出
            # 规定每轮的最大输出长度
            if last_turn_prompt_model_output_length - len(initial_input_ids) >= max_response_length:
                break

            # 添加assistant的对话, 不能使用response_message['prompt']，这个会截断，可能是rebalance导致的，还在查
            # response_text = self.tokenizer.decode(response_message['raw_output_ids'])
            messages.append({
                "role": "assistant",
                "content": self.tokenizer.pad_token * len(response_message['raw_output_ids'])
            })

            # Parse tool calls from response
            tool_calls = await self.tool_parser.extract_tool_calls(response_text)
            num_tool_calls += len(tool_calls)

            if not tool_calls:
                # No tool calls, conversation ends
                break

            # Execute tool calls
            tool_responses = []
            for tool_call in tool_calls:
                tool_response = await self._call_tool(tool_call)
                if isinstance(tool_response, Exception):
                    break
                tool_responses.append(tool_response)

            # Add tool responses to conversation
            for tool_response in tool_responses:
                messages.append(tool_response)

            num_turns += 1

            # break if length is exceed the max length limit
            prompt_with_tools = self.tokenizer.apply_chat_template(messages,
                                                                   tools=self.tool_schemas,
                                                                   add_generation_prompt=True,
                                                                   tokenize=False)

            prompt_data = await self.tokenizer.batch_encode_plus_async([prompt_with_tools], add_special_tokens=False)
            if len(prompt_data.input_ids[0]) >= max_length:
                break

        # extract all outputs and logprobs
        latest_output_ids = completion['choices'][0]['message']['raw_output_ids']
        entire_seq_list = item.batch['input_ids'][0].tolist() + latest_output_ids
        total_output_ids = entire_seq_list[len(initial_input_ids):][:max_response_length]
        log_probs = reduce(lambda x, y: x + y, log_probs_list)[len(initial_input_ids):][:max_response_length]
        item.batch['raw_output_ids'] = torch.tensor([total_output_ids], dtype=torch.int32)
        item.batch['rollout_log_probs'] = torch.tensor([log_probs], dtype=torch.bfloat16)

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
        if context.config.trainer.use_remote_search:
            input_ids = entire_seq_list[:max_length]
            req_id = item.non_tensor_batch['uid'][0]
            reward_model = item.non_tensor_batch['reward_model'][0]
            reward_style = reward_model['style']
            ground_truth = reward_model['ground_truth']

            # note that the uid of padding dataproto should be None
            if req_id is not None:
                # get the sandbox ray handler
                handler = ray.get_actor('remote_client')
                # this is non-blocking
                handler.add_requests.remote(req_id=req_id,
                                            input_ids=input_ids,
                                            ground_truth=ground_truth,
                                            reward_style=reward_style)

        # 将completion转换为DataProto格式，与其他agent保持一致
        from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto
        # 使用internal_call后，应该有完整的alpha-seed格式
        data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
        # FIXME(lixiang): off policy steps在多轮里还不准
        out = pack_to_dataproto(item, self.tokenizer, data_pack, context.config.actor_rollout_ref.rollout)
        out.non_tensor_batch['agent_num_turns'] = np.array([num_turns])
        out.non_tensor_batch['agent_num_tool_calls'] = np.array([num_tool_calls])

        return out

    async def _extract_messages_from_dataproto(self, item, max_prompt_length) -> List[Dict]:
        """Extract messages from DataProto for chat template"""
        # For simplicity, assume it's a user message
        # In practice, you might need more sophisticated parsing
        empty_prompt = self.tokenizer.apply_chat_template([{
            "role": "user",
            "content": ""
        }],
                                                          tools=self.tool_schemas,
                                                          add_generation_prompt=True,
                                                          tokenize=False)
        empty_prompt_data = await self.tokenizer.batch_encode_plus_async([empty_prompt], add_special_tokens=False)
        remain_length = max(0, max_prompt_length - len(empty_prompt_data.input_ids[0]))
        if remain_length == 0:
            prompt = ""
        else:
            initial_prompt = item.non_tensor_batch['raw_prompt'][0][0]['content']
            prompt_data = await self.tokenizer.batch_encode_plus_async([initial_prompt], add_special_tokens=False)
            prompt_data = prompt_data.input_ids[0][-remain_length:]
            prompt = self.tokenizer.decode(prompt_data)
        messages = [{"role": "user", "content": prompt}]

        return messages

    async def _generate_with_tools(self, messages: List[Dict], item: DataProto, context, max_length, max_prompt_length,
                                   max_response_length, max_new_tokens_per_turn, num_turns, response_info):
        """Generate response with tool schemas included"""
        # Apply chat template with tools
        prompt_with_tools = self.tokenizer.apply_chat_template(messages,
                                                               tools=self.tool_schemas,
                                                               add_generation_prompt=True,
                                                               tokenize=False)

        # Tokenize the prompt and set it in item.batch
        # does not require padding when doing inference
        if num_turns == 1:
            max_tokenize_length = max_prompt_length
        else:
            max_tokenize_length = max_length
        prompt_data = await self.tokenizer.batch_encode_plus_async([prompt_with_tools], add_special_tokens=False)

        # set input and attn mask
        item.batch['input_ids'] = torch.tensor(prompt_data.input_ids, dtype=torch.int32)[:, -max_tokenize_length:]
        item.batch['attention_mask'] = torch.tensor(prompt_data.attention_mask, dtype=torch.int8)[:,
                                                                                                  -max_tokenize_length:]

        # 用rollout ids填充padded tokens
        for _resp_info in response_info:
            item.batch['input_ids'][0, _resp_info['prompt_length']:_resp_info['prompt_length'] +
                                    _resp_info['response_length']] = torch.tensor(_resp_info['raw_output_ids'])

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

        return completion, prompt_with_tools

    async def _call_tool(self, tool_call: FunctionCall) -> Dict[str, str]:
        """Execute a tool call and return the response"""
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)

            if tool_name not in self.tools:
                return {"role": "tool", "content": f"Error: Unknown tool {tool_name}", "name": tool_name}

            tool = self.tools[tool_name]
            instance_id = str(uuid4())

            # Execute the tool
            tool_response = await tool.step(instance_id, tool_name, tool_args)

            return {"role": "tool", "content": tool_response, "name": tool_name}

        except Exception as e:
            return {"role": "tool", "content": f"Error: {str(e)}", "name": tool_name}
