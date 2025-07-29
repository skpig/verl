import os
import re

import torch
from openai import AsyncOpenAI

from verl import DataProto
from typing import Optional, Tuple, List, Dict, Union
import asyncio
import aiohttp
import copy

from alpha_seed.workers.agents.handlers import TaskContext
from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto, is_ipv6
from alpha_seed.workers.agents.handlers import register_handler
from alpha_seed.workers.agents.handlers.base import AsyncAgent
import random
from verl.utils import torch_functional

try:
    from groot.action import AsyncEnv
    from groot.action import AsyncJupyterEnv  # name conflict with env_jupyter?
    from groot.action.search.toutiao_search import AsyncToutiaoSearch
    from groot.action.search.web_search import AsyncWebSearch
    from groot.action.search.global_search import AsyncGlobalSearch
    from groot.action.search.browser_reader import AsyncBrowserReader
    from groot.action.search.page_reader import AsyncPageReader
    from groot.action.search.global_search_apihub import AsyncGlobalSearchApiHub
    from groot.action.search.browser_reader_apihub import AsyncBrowserReaderApiHub
    from groot.action.call_llm import AsyncCallLLM
    from groot.state import BaseState
    from groot.llm import GPTAPI, AsyncGPTAPI, AsyncByteLLM
except:
    BaseState = object
    AsyncByteLLM = object
from transformers import PreTrainedTokenizer
import time
from alpha_seed.workers.agents.handlers.tool_use.verifier.judger import get_last_ans


def get_llm(model_psm: str, api_llm_cls, api_model_type: str = "Qwen2.5-32B-Instruct"):
    if model_psm.startswith("http:"):
        return api_llm_cls(model=api_model_type, openai_api_base=model_psm, key='')
    return AsyncByteLLM(mode="xperf", psm_cfg=dict(psm=model_psm))


async def tokenize_and_postprocess_data(prompt: str,
                                        tokenizer: Union[PreTrainedTokenizer, AsyncTokenizer],
                                        max_length: int,
                                        pad_token_id: int,
                                        left_pad=True,
                                        truncation='error'):
    """
    input_data is the output from tokenizer.
    """
    assert truncation in ['left', 'right', 'error']

    if isinstance(tokenizer, AsyncTokenizer):
        input_data = await tokenizer.batch_encode_plus_async([prompt], return_tensors='pt', add_special_tokens=False)
    else:
        input_data = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

    input_ids = input_data['input_ids']
    attention_mask = input_data['attention_mask']

    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = torch_functional.pad_sequence_to_length(input_ids,
                                                            max_seq_len=max_length,
                                                            pad_token_id=pad_token_id,
                                                            left_pad=left_pad)
        attention_mask = torch_functional.pad_sequence_to_length(attention_mask,
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

    return input_ids, attention_mask


async def encode_conversation(conversation: list[dict[str, str]],
                              tokenizer,
                              add_assistant_prompt: bool,
                              original_response: any = None) -> tuple[list[int], dict[str, int]]:
    num_tokens = {
        "system": 0,
        "user": 0,
        "assistant": 0,
        "tool": 0,
        "misc": 0,  # 最后的 assistant prompt
    }

    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    input_ids = []
    assistant_index = 0
    if isinstance(tokenizer, AsyncTokenizer):
        assistant_prefix_ids = await tokenizer.batch_encode_plus_async(["assistant\n"], add_special_tokens=False)
        assistant_prefix_ids = assistant_prefix_ids["input_ids"][0]
    else:
        assistant_prefix_ids = tokenizer("assistant\n", add_special_tokens=False)['input_ids']
    for msg in conversation:
        if msg['role'] == 'assistant' and original_response is not None:
            encoded = assistant_prefix_ids[:] + original_response[assistant_index]["choices"][0]["message"][
                "raw_output_ids"]
            input_ids += [bos] + encoded
            assistant_index += 1
            num_tokens[msg['role']] += len(encoded) + 1
        else:
            text = f"{msg['role']}\n{msg['content']}"
            if isinstance(tokenizer, AsyncTokenizer):
                encoded = await tokenizer.batch_encode_plus_async([text], add_special_tokens=False)
                encoded = encoded['input_ids'][0]
            else:
                encoded = tokenizer(text, add_special_tokens=False)['input_ids']
            input_ids += [bos] + encoded + [eos]
            num_tokens[msg['role']] += len(encoded) + 2
    if add_assistant_prompt:
        encoded = assistant_prefix_ids[:]
        input_ids += [bos] + encoded
        num_tokens['misc'] += len(encoded) + 1
    num_tokens['all'] = sum(num_tokens.values())
    return input_ids, num_tokens


def decode_conversation(input_ids: list[int], tokenizer) -> tuple[list[dict[str, str]], str]:
    # 会忽略所有不完整的消息（即没有被 bos_token 和 eos_token 包裹的消息）
    decoded_str = tokenizer.decode(input_ids, skip_special_tokens=False)
    pattern = re.compile(
        re.escape(tokenizer.bos_token)  # 精确匹配 tokenizer.bos_token
        + r'(system|user|assistant|tool)\n'  # role + 换行
        + r'(.*?)'  # content
        + r'(?=' + re.escape(tokenizer.eos_token) + r')',  # 前瞻，匹配到 tokenizer.eos_token 之前
        re.DOTALL,
    )
    matches = pattern.findall(decoded_str)
    conversation = [{'role': role, 'content': content} for role, content in matches]
    return conversation, decoded_str


def print_conversation(conversation: list[dict[str, str]]):
    print_str = "\n".join([f"[{msg['role']}]:\n{msg['content']}" for msg in conversation])
    print(f"*****BEGIN OF CONVERSATION*****\n{print_str}\n*****END OF CONVERSATION*****")


def extract_conversation(item: DataProto, tokenizer) -> tuple[list[dict[str, str]], str]:
    input_ids = item.batch['input_ids']
    attention_mask = item.batch['attention_mask']
    valid_input_len = torch.sum(attention_mask)
    prompt_ids = input_ids[0, -valid_input_len:].tolist()
    return decode_conversation(prompt_ids, tokenizer)


class LLMInterface:

    def __init__(self, tokenizer, config, host, port, meta_info):
        self.tokenizer = tokenizer
        self.config = config
        self.chat_completions = []
        if is_ipv6(host):
            host = f'[{host}]'
        self.host = host
        self.port = port
        self.meta_info = meta_info

        self.call_cnt = 0
        self.call_success = 0
        self.call_fail = 0

    async def chat(self, history: list[dict[str, str]], **kwargs):
        if kwargs:
            print("[WARNING] LLMInterface.chat() got unexpected kwargs:", kwargs)
        self.call_cnt += 1

        # print_conversation(history)
        prompt_ids, num_tokens_dict = await encode_conversation(history,
                                                                tokenizer=self.tokenizer,
                                                                add_assistant_prompt=True,
                                                                original_response=self.chat_completions)
        # print_conversation(decode_conversation(prompt_ids, tokenizer=self.tokenizer))

        # TODO: 这边 assume 了所有 system 和 user 都是 prompt，所有 assistant 和 tool 都是我们这个 agent 的 response
        max_tokens = self.config.train_generate_kwargs['max_new_tokens'] - num_tokens_dict[
            'assistant'] - num_tokens_dict['tool']
        # TODO @Fangkai: I'm not sure if there is any other usage of `max_new_tokens` so I simply add the option here.
        # This is used to avoid repetitive generation.
        if getattr(self.config.plugin, 'turn_max_new_tokens', -1) > 0:
            max_tokens = min(max_tokens, self.config.plugin.turn_max_new_tokens)
        if max_tokens < 10:
            print(f"[DEBUG] max_tokens < 10, skip rollout")
            return None

        data = {"prompt": prompt_ids}

        completion = None
        try:
            timeout = aiohttp.ClientTimeout(total=9600)
            session = aiohttp.ClientSession(timeout=timeout)
            generation_kwargs = self.meta_info['generation_kwargs']
            async with session.post(url=f"http://{self.host}:{self.port}/chat/completions",
                                    headers={"Authorization": "Bearer token-abc123"},
                                    json={
                                        "model": "rollout",
                                        "messages": data,
                                        "top_p": generation_kwargs['top_p'],
                                        "top_k": generation_kwargs['top_k'],
                                        "temperature": generation_kwargs['temperature'],
                                        "max_tokens": max_tokens,
                                        "max_length": self.config.prompt_length + self.config.response_length,
                                        "meta_info": self.meta_info,
                                    },
                                    timeout=timeout) as resp:
                ret = await resp.json()
                assert resp.status == 200, f"chat_completions failed msg: {ret}"
                completion = ret
        except asyncio.TimeoutError:
            print(f"[ERROR] chat_completions timeout after 9600 seconds, please check the server status")
            self.call_fail += 1
            return None
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
        finally:
            await session.close()

        self.call_success += 1

        self.chat_completions.append(completion)

        response_output_ids = completion["choices"][0]["message"]["raw_output_ids"]
        response = self.tokenizer.decode(response_output_ids, skip_special_tokens=False)
        return response

    @staticmethod
    def _check_before_pack(state: BaseState, completion_messages: list) -> None:
        # assistant message数量和记录到的chat completion数量一致
        assistant_messages = [msg for msg in state.history if msg['role'] == 'assistant']
        all_messages = [msg['content'] for msg in state.history]
        num_assistant_messages = len([msg for msg in state.history if msg['role'] == 'assistant'])
        if num_assistant_messages != len(completion_messages):
            print("[DEBUG] num_assistant_messages!= len(completion_messages)")
            print_conversation(state.history)
            breakpoint()
        assert num_assistant_messages == len(
            completion_messages
        ), f"num_assistant_messages={num_assistant_messages}, len(data_packs)={len(completion_messages)}"

        # 一开始的system和user message作为prompt，后面的assistant和tool message作为response
        if state.history[0]['role'] == 'system':
            assert len(state.history) >= 3
            assert state.history[1]['role'] == 'user'
            assert state.history[2]['role'] == 'assistant'
            start = 2
        elif state.history[0]['role'] == 'user':
            assert len(state.history) >= 2
            assert state.history[1]['role'] == 'assistant'
            start = 1
        else:
            raise AssertionError(f"Invalid conversation: {state.history}")
        # assistant / tool 必须交替
        for i in range(start, len(state.history)):
            role = state.history[i]['role']
            assert role in {'assistant', 'tool'}, f"Unexpected role {role} at index {i}"
            if (i - start) % 2 == 0:
                assert role == 'assistant', f"Expected assistant at index {i}, got {role}"
            else:
                assert role == 'tool', f"Expected tool at index {i}, got {role}"
        # 最后一条必须是 assistant（不对，由于调用chat的时候可能会直接被超长拒绝，最后一条也有可能是tool）
        # assert state.history[-1]['role'] == 'assistant', "Last message must be assistant"

    async def create_summarized_data_pack(self,
                                          state: BaseState,
                                          max_tokens: Optional[int] = None,
                                          stats: Dict = None,
                                          score: Union[Tuple, float, bool] = None) -> Tuple[DataPack, List[bool]]:
        completion_messages = [completion["choices"][0]["message"] for completion in self.chat_completions]
        # data_packs = [DataPack.create_from_completion(message) for message in completion_messages]

        self._check_before_pack(state, completion_messages)

        response_outputs = []
        response_log_probs = []
        model_output_mask = []

        assistant_index = 0
        for msg in state.history:
            if msg['role'] in {'system', 'user'}:  # 这部分被包含在prompt中，此处不需处理
                continue
            elif msg['role'] == 'assistant':
                response_outputs.extend(completion_messages[assistant_index]["raw_output_ids"])
                response_log_probs.extend(completion_messages[assistant_index]["response_log_probs"])
                model_output_mask.extend([True] * len(completion_messages[assistant_index]["raw_output_ids"]))
                assistant_index += 1
            elif msg['role'] == 'tool':
                tool_ids, _ = await encode_conversation([msg], tokenizer=self.tokenizer, add_assistant_prompt=True)
                response_outputs.extend(tool_ids)
                response_log_probs.extend([-1.0] * len(tool_ids))
                model_output_mask.extend([False] * len(tool_ids))
            else:
                raise AssertionError(f"Should not reach here")
        assert assistant_index == len(
            completion_messages
        ), f"assistant_index={assistant_index}, len(completion_messages)={len(completion_messages)}"

        # 截断
        if max_tokens is not None and len(response_outputs) > max_tokens:
            print(f"[DEBUG] create_summarized_data_pack时发生截断: 从{len(response_outputs)}截断至{max_tokens}")
            response_outputs = response_outputs[:max_tokens]
            response_log_probs = response_log_probs[:max_tokens]
            model_output_mask = model_output_mask[:max_tokens]

        answer_reached = False
        if state.history[-1]["role"] == "assistant":
            last_ans = get_last_ans(state.history[-1]["content"])
            if last_ans is not None:
                answer_reached = True
        extra_data = {
            "answer_reached": answer_reached,
        }
        action_fail = 0
        if stats is not None:
            extra_data["stats"] = stats
            for k, v in stats.items():
                if k.startswith("failed_"):  # Only consider failed actions, instead of invalid function call.
                    if v > 0:
                        action_fail += 1
                        break
        if score is not None:
            extra_data["score"] = score
        extra_data["call_fail"] = 1 if self.call_fail > 0 else 0
        extra_data["action_fail"] = 1 if action_fail > 0 else 0

        data_pack = DataPack(
            response_log_probs=[response_log_probs, ],
            this_turn_off_policy_steps=[[-1.0] * len(response_log_probs)],
            response_outputs=[response_outputs, ],
            response_model_output_mask=[model_output_mask, ],
            is_finished=[completion_messages[-1]["is_finished"], ],  # TODO: 只使用最后一个completion的is_finished和metrics。这边可能要检查is_finished和metrics是干嘛的
            metrics=completion_messages[-1]["metrics"],
            extra_data=[extra_data, ],
        )
        return data_pack, model_output_mask

    async def pack_out_dataproto(self,
                                 prompts: DataProto,
                                 state: BaseState,
                                 stats: Dict = None,
                                 score: Union[Tuple, float, bool] = None) -> DataProto:
        max_new_tokens = prompts.meta_info.get('generation_kwargs').get(
            'max_new_tokens', self.config.response_length)  # same as in pack_to_dataproto

        data_pack, model_output_mask = await self.create_summarized_data_pack(state,
                                                                              max_tokens=max_new_tokens,
                                                                              stats=stats,
                                                                              score=score)

        # TODO: 我们在推理时，实际上使用decode整理后又encode的system message和user message（开启了force system prompt模式）
        # 此处拼接训练数据时，仍旧使用的是原本数据提供的prompt DataProto
        out: DataProto = pack_to_dataproto(prompts, self.tokenizer, data_pack, self.config)

        # 添加上model_output_mask
        if len(model_output_mask) > max_new_tokens:
            model_output_mask = model_output_mask[:max_new_tokens]
        else:
            model_output_mask = model_output_mask + [False] * (max_new_tokens - len(model_output_mask))
        model_output_mask = [
            model_output_mask,
        ]
        out.batch['model_output_mask'] = torch.tensor(model_output_mask, dtype=torch.int8)
        return out
