"""
Implement custom functions for math expression task
"""
import asyncio
from functools import reduce
import copy
import uuid
import base64
import logging

from transformers import PreTrainedTokenizer

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import register_handler, TaskContext
from alpha_seed.workers.agents.handlers.base import AsyncAgent
from alpha_seed.workers.agents.llm import AsyncLLMInterface
from alpha_seed.workers.agents.handlers.tool.parser import FunctionCall
from alpha_seed.workers.agents.handlers.vlm.parser import VisualCotParser
from mono_rl import DataProto
from typing import Any, List, Dict
import json
import torch
from uuid import uuid4
import numpy as np
from alpha_seed.workers.agents.envs.visual_cot import create_from_env_str
from alpha_seed.workers.agents.handlers.vlm import post_process_eval_result
from alpha_seed.utils.functional import import_from_string
from mono_rl.utils.dataset.dist_data_util import get_dist_data_manager
from alpha_seed.utils.reward_score.utils import Verifier
import ray

logger = logging.getLogger(__name__)


def get_local_data(ref_list, dist_data_manager):
    images_ref = ray.get(dist_data_manager.get_refs.remote(ref_list))
    images_bytes = ray.get(images_ref)[0].tolist()
    return images_bytes


@register_handler("agent/tool/visual_cot")
class VisualCotAgent(AsyncAgent):

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        super().__init__(tokenizer, llm, **kwargs)
        processor = kwargs['processor']
        self.visual_cot = create_from_env_str('visual_cot@',
                                              tokenizer=tokenizer,
                                              image_processor=processor.image_processor,
                                              config=kwargs['config'])
        self.processor = processor
        self.tools = {"visual_cot": self.visual_cot}
        self.tool_parser = VisualCotParser(tokenizer)
        self.dist_data_manager = get_dist_data_manager()
        reward_manager_cls = import_from_string(kwargs['config'].tasks.reward_manager)
        self.val_reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                                config=kwargs['config'],
                                                logger=None,
                                                rm_name="val",
                                                single_batch=True)

    def _reward_fn(self, item: DataProto, out: DataProto):
        reward_model = out.non_tensor_batch['reward_model'][0]
        reward_style = reward_model['style']
        verifier = Verifier.get_verifier(reward_style, self.config, self.tokenizer)
        req_id = item.non_tensor_batch['uid'][0]
        input_ids = out.batch['input_ids'][0]
        ground_truth = reward_model['ground_truth']
        verifier.add_requests(req_id=req_id, input_ids=input_ids, ground_truth=ground_truth, reward_style=reward_style)

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        out = await self.__call_internal__(item, context, **kwargs)
        if self.config.rollout_server.evals.enable:
            out = await post_process_eval_result(item, out, self.executor, self.val_reward_fn)
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._reward_fn, item, out)
        return out

    async def __call_internal__(self, item: DataProto, context: TaskContext, **kwargs):
        """Main agent loop with tool calling capability"""
        max_prompt_length = context.config.data.max_prompt_length
        max_response_length = context.config.data.max_response_length
        max_length = max_prompt_length + max_response_length
        max_turns = context.config.actor_rollout_ref.rollout.agent.max_turns
        item.meta_info = copy.deepcopy(item.meta_info)
        reward_model = item.non_tensor_batch.pop('reward_model')

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

        raw_output_ids = []
        last_data = None

        if 'images_bytes_ref' in messages[0]:
            images_bytes = get_local_data([messages[0]['images_bytes_ref']], self.dist_data_manager)
            messages[0]['images_bytes'] = images_bytes
        last_completion = None
        first_round_prompt_length = 0
        finish_reason = None
        incremental_input_length = 0
        # max_response_length_this_turn = max_response_length
        timeout = context.config.actor_rollout_ref.rollout.plugin.timeout

        input_ids_list = []
        image_data_ref_list = []
        image_bytes_list = []
        raw_output_list = []
        img_token_num_list = []

        while num_turns <= max_turns:
            # llm input is overlong
            new_input_ids = []
            for msg in messages:
                input_ids = self.get_input_ids(msg)
                new_input_ids.extend(input_ids)
            assert first_round_prompt_length <= max_prompt_length
            if first_round_prompt_length == 0:
                first_round_prompt_length = len(new_input_ids)
            response_length = len(new_input_ids) + len(input_ids_list) - first_round_prompt_length
            if response_length >= max_response_length:
                # 超长退出
                finish_reason = 'llm input is overlong'
                break
            input_ids_list.extend(new_input_ids)
            max_response_length_this_turn = max_response_length - response_length

            processed_msg = self.process_messages(messages, input_ids_list, image_data_ref_list, image_bytes_list,
                                                  raw_output_list, img_token_num_list)
            messages = []
            # Generate response using LLM
            completion = await self._generate_with_tools(processed_msg, item, context, max_response_length,
                                                         max_response_length_this_turn)
            last_completion = completion

            # extract the initial input with system prompts at first round
            if initial_input_ids is None:
                initial_input_ids = item.batch['input_ids'][0]
                initial_attn_mask = item.batch['attention_mask'][0]
                last_turn_prompt_model_output_length = len(initial_input_ids)
                first_round_prompt_length = len(initial_input_ids)

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
            # 算这一轮新增给llm的长度（可能是上一轮的tool call的结果等）
            incremental_input_length = prompt_length - last_turn_prompt_model_output_length
            last_data = processed_msg
            raw_output_ids.append(response_message['raw_output_ids'])

            model_out_mask_list.append((False, incremental_input_length))
            model_out_mask_list.append((True, response_length))
            log_probs_list.append([1] * incremental_input_length)
            assert len(response_message['response_log_probs']) == response_length
            log_probs_list.append(response_message['response_log_probs'])
            last_turn_prompt_model_output_length = prompt_length + response_length

            # length的退出逻辑，除去initial_input_ids (prompt_length)，所有的model response + env，超出max_response_length就退出
            # 规定每轮的最大输出长度
            this_turn_response_length = last_turn_prompt_model_output_length - len(initial_input_ids)
            logger.info(
                f"uid: {item.non_tensor_batch['uid'][0]} round num_turns: {num_turns}, last_turn_prompt_model_output_length: {last_turn_prompt_model_output_length}, initial_input_ids: {len(initial_input_ids)}, this_turn_response_length: {this_turn_response_length}"
            )

            if this_turn_response_length >= max_response_length:
                finish_reason = 'after llm generation is overlong'
                break

            # 增加上一轮的模型输出
            messages.append({"role": "assistant", "content": {"input_ids": response_message['raw_output_ids']}})
            images_bytes = processed_msg.get('images_bytes')
            # Parse tool calls from response
            tool_calls = await self.tool_parser.extract_tool_calls(response_text, images_bytes)
            num_tool_calls += len(tool_calls)

            if not tool_calls:
                # No tool calls, conversation ends
                finish_reason = 'no tool calls'
                break

            # Execute tool calls
            for tool_call in tool_calls:
                logger.info(f"start call tool num_turns: {num_turns} uid: {item.non_tensor_batch['uid'][0]}")
                tool_response = await self._call_tool(tool_call, timeout)
                logger.info(f"end call tool num_turns: {num_turns} uid: {item.non_tensor_batch['uid'][0]}")
                messages.append(tool_response)

            num_turns += 1
        if finish_reason is None and num_turns >= max_turns:
            finish_reason = 'max turns'
        logger.info(f'finish_reason: {finish_reason}')

        # extract all outputs and logprobs
        completion = last_completion
        latest_output_ids = completion['choices'][0]['message']['raw_output_ids']
        entire_seq_list = item.batch['input_ids'][0].tolist() + latest_output_ids
        total_output_ids = entire_seq_list[len(initial_input_ids):]
        log_probs = reduce(lambda x, y: x + y, log_probs_list)
        assert len(log_probs) == len(total_output_ids)
        # log_probs = log_probs[len(initial_input_ids):]
        item.batch['raw_output_ids'] = torch.tensor([total_output_ids], dtype=torch.int32)
        assert len(total_output_ids) <= max_response_length

        # left pad and adjust original input
        left_pad_size = context.config.data.max_prompt_length - len(initial_input_ids)
        input_ids = torch.concat(
            [torch.tensor([self.tokenizer.pad_token_id] * left_pad_size, dtype=torch.int32), initial_input_ids])
        attention_mask = torch.concat([torch.tensor([0] * left_pad_size, dtype=torch.int8), initial_attn_mask])
        item.batch['input_ids'] = input_ids[None, :]  # (1, max_prompt_length)
        item.batch['attention_mask'] = attention_mask[None, :]  # (1, max_prompt_length)

        # 重新组装completion
        model_output_mask = []
        for mask, length in model_out_mask_list:
            model_output_mask.extend([mask] * length)
        assert len(model_output_mask) == len(total_output_ids)
        # avoid concat error in verl, we pad it to max_turns
        image_data_ref = [None] * max_turns
        real_image_data_ref = last_data['image_data_ref']
        image_data_ref[:len(real_image_data_ref)] = real_image_data_ref
        assert sum(img_token_num_list[1:]) // 4 == len([i for i in total_output_ids if i == -100
                                                       ]), 'img token num not match'

        completion['choices'][0]['message'].update({
            'model_output_mask': model_output_mask,
            'raw_output_ids': total_output_ids,
            'response_log_probs': log_probs,
            'image_data_ref': image_data_ref
        })
        if context.config.actor_rollout_ref.rollout.vlm.return_raw_output:
            raw_outputs = last_data['raw_output_list']
            assert isinstance(raw_outputs, list)
            for ro in raw_outputs:
                assert isinstance(ro, dict)
            raw_output_ref = None
            if raw_outputs is not None and len(raw_outputs) > 0:
                raw_output_ref = ray.put(raw_outputs)
                ray.get(self.dist_data_manager.add_refs.remote([raw_output_ref]))
                raw_output_ref = raw_output_ref.hex()

            completion['choices'][0]['message'].update({'raw_output_ref': raw_output_ref})

        # 将completion转换为DataProto格式，与其他agent保持一致
        from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto
        # 使用internal_call后，应该有完整的alpha-seed格式
        data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
        data_pack.extra_data[0]['metrics'] = self.visual_cot.metrics
        # FIXME(lixiang): off policy steps在多轮里还不准
        out = pack_to_dataproto(item, self.tokenizer, data_pack, context.config.actor_rollout_ref.rollout)
        out.non_tensor_batch['agent_num_turns'] = np.array([num_turns])
        out.non_tensor_batch['agent_num_tool_calls'] = np.array([num_tool_calls])
        out.non_tensor_batch['reward_model'] = reward_model
        return out

    async def _extract_messages_from_dataproto(self, item, max_prompt_length) -> List[Dict]:
        """Extract messages from DataProto for chat template"""
        item.batch = item.batch.reshape(-1)
        input_ids = item.batch['input_ids']
        attention_mask = item.batch['attention_mask']
        valid_input_len = torch.sum(attention_mask)
        prompt_ids = input_ids[0, -valid_input_len:].tolist()
        assert len(prompt_ids) <= max_prompt_length
        data = {"prompt": prompt_ids}
        if 'image_data_ref' in item.non_tensor_batch:
            image_data_ref = item.non_tensor_batch['image_data_ref'][0]
            if image_data_ref is not None:
                data['image_data_ref'] = image_data_ref
        if 'images_bytes_ref' in item.non_tensor_batch:
            images_bytes_ref = item.non_tensor_batch['images_bytes_ref'][0]
            if images_bytes_ref is not None:
                data['images_bytes_ref'] = images_bytes_ref
        return [data]

    def process_messages(self, messages, input_ids_list, image_data_ref_list, image_bytes_list, raw_output_list,
                         img_token_num_list):
        for msg in messages:
            image_data_ref, images_bytes, raw_output, img_token_num = self.process_message(msg)
            if image_data_ref is not None:
                image_data_ref_list.append(image_data_ref)
            if images_bytes is not None:
                assert isinstance(images_bytes, list)
                image_bytes_list.extend(images_bytes)
            if raw_output is not None:
                raw_output_list.append(raw_output)
            img_token_num_list.append(img_token_num)
        processed_item = {
            'input_ids': input_ids_list,
            'image_data_ref': image_data_ref_list,
            'images_bytes': image_bytes_list,
            'raw_output_list': raw_output_list,
            'img_token_nums': img_token_num_list
        }
        return processed_item

    def get_input_ids(self, msg):
        if 'content' in msg:
            if isinstance(msg['content'], str):
                # mean some error in excution, need to add more error information in first happened lines
                logger.info(f"error outputs: {msg['content']}")
                input_ids = self.tokenizer.encode(str(msg['content']))
            else:
                input_ids = msg['content']['input_ids']
        else:
            input_ids = msg['prompt']
        return input_ids

    def process_message(self, msg):
        image_data_ref = None
        images_bytes = None
        raw_output = None
        img_token_num = 0

        if 'content' in msg and not isinstance(msg['content'], str):
            input_ids = msg['content']['input_ids']
            if 'pixel_values' in msg['content']:
                image_data = {
                    'pixel_values': msg['content']['pixel_values'],
                    'image_grid_hw': msg['content']['image_grid_hw']
                }
                img_token_num = image_data['pixel_values'].shape[0]
                image_data_ref = ray.put(image_data)
                ray.get(self.dist_data_manager.add_refs.remote([image_data_ref]))
                image_data_ref = image_data_ref.hex()
                raw_output = msg['content']['raw_output']
                assert isinstance(raw_output, dict)
                if raw_output is not None:
                    images_bytes = [base64.b64decode(raw_output["image_base64"])]
        else:
            if 'image_data_ref' in msg:
                image_data_ref = msg['image_data_ref']
                images_bytes = msg['images_bytes']
                img_token_num = msg.get('img_token_num', 0)
        return image_data_ref, images_bytes, raw_output, img_token_num

    async def _generate_with_tools(self, processed_items: List[Dict], item: DataProto, context, max_response_length,
                                   max_response_length_this_turn):
        """Generate response with tool schemas included"""
        image_data_ref = processed_items['image_data_ref']
        input_ids = processed_items['input_ids']

        # # Call LLM with the enhanced prompt
        rollout_config = context.config.actor_rollout_ref.rollout
        item.non_tensor_batch['image_data_ref'] = image_data_ref
        item.batch['input_ids'] = torch.tensor(input_ids, dtype=torch.int32).unsqueeze(0)
        item.batch['attention_mask'] = torch.ones_like(item.batch['input_ids'], dtype=torch.int8)
        item.meta_info['generation_kwargs']['max_new_tokens'] = max_response_length_this_turn

        completion = await self.llm.complete(item, rollout_config)
        # pack_to_dataproto will use max_length to pad
        item.meta_info['generation_kwargs']['max_new_tokens'] = max_response_length
        return completion

    async def _call_tool(self, tool_call: FunctionCall, timeout: int) -> Dict[str, str]:
        """Execute a tool call and return the response"""
        tool_name = tool_call.name
        tool_code = tool_call.arguments
        if not tool_call.status:
            return {"role": "tool", "content": self.tool_parser._format_ret(tool_code), "tool_name": tool_name}
        try:
            tool = self.tools[tool_name]

            # Execute the tool
            tool_state = False
            try:
                async with asyncio.timeout(timeout):
                    result = await tool.step(tool_code)
                    if isinstance(result, tuple) and len(result) == 2:
                        tool_state, tool_response = result
                    else:
                        tool_response = result
            except (asyncio.TimeoutError, TimeoutError):
                logger.info(f"tool {tool_name} timeout")
                tool_response = "execution timeout error."
            if not tool_state:
                # tool call error
                return {"role": "tool", "content": self.tool_parser._format_ret(tool_response), "tool_name": tool_name}

            return {"role": "tool", "content": tool_response, "tool_name": tool_name}
        except Exception as e:
            return {"role": "tool", "content": self.tool_parser._format_ret(str(e)), "tool_name": tool_name}
