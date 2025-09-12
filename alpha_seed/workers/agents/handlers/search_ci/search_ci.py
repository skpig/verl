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
from alpha_seed.workers.agents.handlers.tool.parser import FunctionCall, ToolParser, _extract_messages_from_dataproto
from mono_rl import DataProto
from typing import List, Dict
import json
import torch
import regex as re
from uuid import uuid4
import numpy as np
import ray
from alpha_seed.workers.agents.handlers.ci.tool import JupyterCI, JupyterCI_stateful

CODING_SNIPET_REGEX = (r'<escapeShell\s+type=["\']code["\']\s*,?\s*id=["\'](?P<id>\d+)["\']\s*,?\s*'
                       r'(name=["\'](?P<name>[^"\']+)["\']\s*)?>'
                       r'(?P<code>[\s\S]*?)</escapeShell>')

RAW_CODE_REGEX = r"```(?P<language>.*?)\s*\n(?P<code>[\s\S]*?)```"


@register_handler("agent/search_ci")
class SearchCIAgent(AsyncAgent):

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        super().__init__(tokenizer, llm, **kwargs)
        self.search = create_search_env_from_env_str("deep_research/search@{}", tokenizer=tokenizer)
        self.textbrowser = create_textbrowser_env_from_env_str("deep_research/textbrowser@{}", tokenizer=tokenizer)
        self.ci = JupyterCI()
        self.ci_stateful = JupyterCI_stateful()
        self.tool_parser = ToolParser(tokenizer, self.config)
        self.tools = {
            "jupyter_ci": self.ci,
            "JupyterCI_new": self.ci,
            "JupyterCI_stateful": self.ci_stateful,
            "Search": self.search,
            "GlobalSearch": self.search,
            "TextBrowser": self.textbrowser,
            "linkreader": self.textbrowser,
            "TextBrowserView": self.textbrowser
        }
        # Get tool schema for the calculator
        self.tool_schemas = [
            self.search.get_openai_tool_schema().model_dump(exclude_unset=True, exclude_none=True),
            self.textbrowser.get_openai_tool_schema().model_dump(exclude_unset=True, exclude_none=True),
            self.ci.get_openai_tool_schema().model_dump(exclude_unset=True, exclude_none=True)
        ]
        self.tool_schemas = [tool['function'] for tool in self.tool_schemas]

        self.ci_mode = None

        assert hasattr(tokenizer, 'pad_token'), 'we need `pad_token` to substitute the rollout ids'

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        """Main agent loop with tool calling capability"""
        max_prompt_length = context.config.data.max_prompt_length
        max_response_length = context.config.data.max_response_length
        max_length = max_prompt_length + max_response_length
        max_turns = context.config.actor_rollout_ref.rollout.agent.max_turns
        max_new_tokens_per_turn = context.config.actor_rollout_ref.rollout.agent.max_new_tokens_per_turn
        ci_sandbox_psm = context.config.trainer.ci_sandbox_psm
        self.ci_mode = context.config.rollout_server.agent.ci_mode
        item.meta_info = copy.deepcopy(item.meta_info)
        global_step = context.global_step

        # Extract initial messages from DataProto
        messages = await _extract_messages_from_dataproto(item, max_prompt_length, self.tokenizer, self.tool_schemas)
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

        def _parse_code_blocks(model_output: str):
            all_added_code_files = []

            code_matches = list(re.finditer(CODING_SNIPET_REGEX, model_output))

            for code_match in code_matches:
                raw_insert_item = code_match.group('code')
                idx = int(code_match.group('id'))
                name = str(code_match.group('name'))
                raw_code_match = re.search(RAW_CODE_REGEX, raw_insert_item)

                if raw_code_match:
                    language = raw_code_match.group('language')

                    code = raw_code_match.group('code')
                    if not (code.strip() == ''):
                        if name != '' and name is not None:
                            name = name.replace('"', '').replace("'", '')
                        all_added_code_files.append({'idx': idx, 'file_name': name, 'language': language, 'code': code})
            return all_added_code_files

        self.file_properties = {}

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
            try:
                assert incremental_input_length >= 0, f"incremental_input_length should be > 0, {prompt_length=} {last_turn_prompt_model_output_length=}, {prompt_lengths=}, {response_lengths=}, {num_turns=}"
            # input_length=1495 last_turn_prompt_model_output_length=2304, temp=[268, 1495], num_turns=2
            except:
                save_info = {
                    "all_input_ids": all_input_ids,
                    "prompt_len": prompt_length,
                    "last_turn_prompt_model_output_length": last_turn_prompt_model_output_length,
                    "prompt_lengths": prompt_lengths,
                    "response_lengths": response_lengths,
                    "response_texts": response_texts,
                    "num_turns": num_turns,
                    "messages": messages,
                    "raw_output_ids": raw_output_ids
                }
                with open("special_calculator_error.json", "w") as f:
                    json.dump(save_info, f)
                from hdfs_io.hdfs_io import hcopy, hmkdir
                hcopy(
                    f"special_calculator_error.json",
                    "hdfs://haruna/home/byte_data_seed/lf_lq/user/qiying.01/projects/alphaseed/experiments/tool_use_demo2"
                )
                raise

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
            # response_text = """<escapeShell type="code" id="0">```python\nprint("hello world")\n```</escapeShell><|FunctionCallBegin|>[{"name": "DoubaoCodeInterpreter", "parameters": {"id": "0"}}]<|FunctionCallEnd|>"""
            # response_text = "<|FunctionCallBegin|>" + json.dumps([{"name": "JupyterCI", "parameters": {"code": "print('hello world')"}}], ensure_ascii=False) + "<|FunctionCallEnd|>"

            #Chen: Here is the implementation of exculding all function call within thinking cot
            #FIXME: need to fix the hard code od thinking token
            def remove_think_block(text):
                start_tag = "<think_never_used_51bce0c785ca2f68081bfa7d91973934>"
                end_tag = "</think_never_used_51bce0c785ca2f68081bfa7d91973934>"

                start = text.find(start_tag)
                end = text.rfind(end_tag)

                if start != -1 and end != -1 and end > start:
                    end += len(end_tag)
                    return text[:start] + text[end:]
                return text

            response_text_excluded_thinking_cot = remove_think_block(response_text)
            all_added_code_files = _parse_code_blocks(response_text_excluded_thinking_cot)

            for i in range(len(all_added_code_files)):
                code_block_id = all_added_code_files[i]['idx']
                file_name = all_added_code_files[i]['file_name']
                code = all_added_code_files[i]['code']

                self.file_properties['id_' + str(code_block_id)] = {
                    'content': code,
                    'language': all_added_code_files[i]['language'],
                    'called': False
                }

            messages.append({
                "role": "assistant",
                "content": self.tokenizer.pad_token * len(response_message['raw_output_ids'])
                # "content": response_text,
            })

            # Parse tool calls from response
            tool_calls = await self.tool_parser.extract_tool_calls(response_text_excluded_thinking_cot)
            num_tool_calls += len(tool_calls)

            if not tool_calls:
                # No tool calls, conversation ends
                break

            # Execute tool calls
            tool_responses = []
            for tool_call in tool_calls:
                tool_response = await self._call_tool(
                    tool_call,
                    global_step,
                    ci_sandbox_psm,
                    initial_files=item.non_tensor_batch['extra_data'][0]['agent_env_initial_files'])
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
        item.batch['rollout_behavior_log_probs'] = torch.tensor([log_probs], dtype=torch.bfloat16)

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
        reward_model = item.non_tensor_batch['reward_model'][0]
        reward_style = reward_model['style']
        if context.config.trainer.use_remote_search and reward_style in [
                'code-sandbox', 'aider', 'verifier_service', 'deep_research_verifier', 'gaokao_verifier_service',
                'swe_repair_verifier'
        ]:
            input_ids = entire_seq_list[:max_length]
            req_id = item.non_tensor_batch['uid'][0]
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

        prompt_data = await self.tokenizer.batch_encode_plus_async([prompt_with_tools],
                                                                   add_special_tokens=False,
                                                                   max_length=max_tokenize_length,
                                                                   truncation=True)

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

    async def _call_tool(self,
                         tool_call: FunctionCall,
                         global_step,
                         ci_sandbox_psm,
                         initial_files=None) -> Dict[str, str]:
        """Execute a tool call and return the response"""
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)

            if "id" in tool_args:
                idx = tool_args['id']
                if not ('id_' + str(idx) in self.file_properties):
                    return {
                        "role": "tool",
                        "content": "plugin_error (code block not found): Code block {str(idx)} not found"
                    }

                elif self.file_properties[f'id_{idx}']['called']:
                    return {
                        "role":
                            "tool",
                        "content":
                            f"plugin_error (repeat call): Code block {str(idx)} has been called before and no change is detected. Please do not repeat running the same code block"
                    }
                else:
                    code_block = self.file_properties[f'id_{idx}']['content']
                    self.file_properties[f'id_{idx}']['called'] = True
                    if self.ci_mode == 'stateful':
                        tool_name = "JupyterCI_stateful"
                    else:
                        tool_name = "JupyterCI_new"
                    tool_args = {"code": [code_block]}

            if tool_name not in self.tools:
                return {"role": "tool", "content": f"Error: Unknown tool {tool_name}", "name": tool_name}

            tool = self.tools[tool_name]
            instance_id = str(uuid4())

            # Execute the tool
            tool_result = await tool.execute(instance_id,
                                             tool_args,
                                             tool_name=tool_name,
                                             global_step=global_step,
                                             ci_sandbox_psm=ci_sandbox_psm,
                                             initial_files=initial_files)
            tool_response = tool_result.result

            return {"role": "tool", "content": tool_response, "name": tool_name}

        except Exception as e:
            return {"role": "tool", "content": f"Error: {str(e)}", "name": tool_name}
