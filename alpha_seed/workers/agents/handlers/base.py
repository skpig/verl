import asyncio
import copy

import torch
import aiohttp
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import TaskContext
from mono_rl import DataProto
from mono_rl.utils.network import is_ipv6


class AsyncLLMInterface:

    async def chat_completions(self, content: dict, meta_info: dict, config: DictConfig):
        raise NotImplementedError()

    @property
    def host(self) -> str:
        raise NotImplementedError()

    @property
    def port(self) -> int:
        raise NotImplementedError()

    async def complete(self, item: DataProto, config: DictConfig, prompt: str = ''):
        completion = None
        try:
            if prompt == '':
                item.batch = item.batch.reshape(-1)
                input_ids = item.batch['input_ids']
                attention_mask = item.batch['attention_mask']
                valid_input_len = torch.sum(attention_mask)
                prompt_ids = input_ids[0, -valid_input_len:].tolist()
                data = {"prompt": prompt_ids}
            else:
                data = {"prompt": prompt}
            if 'pixel_values_ref' in item.non_tensor_batch:
                assert len(item.non_tensor_batch['pixel_values_ref']) == 1
                pixel_values_ref = item.non_tensor_batch['pixel_values_ref'][0]
                if pixel_values_ref is not None:
                    data['pixel_values_ref'] = pixel_values_ref
                    data['image_grid_hw'] = item.non_tensor_batch['image_grid_hw'][0].tolist()
            meta_info = copy.copy(item.meta_info)
            # required for eos callback
            meta_info['uid'] = item.non_tensor_batch['uid'][0]
            meta_info['reward_model'] = item.non_tensor_batch['reward_model'][0]
            meta_info['server_meta'] = {
                'host': self.host,
                'port': self.port,
            }
            # required for tool calling
            if (key := 'extra_data') in item.non_tensor_batch:
                meta_info[key] = item.non_tensor_batch[key][0]

            completion = await self.chat_completions(data, meta_info, config)
        except asyncio.CancelledError:
            # Handle task cancellation (e.g., cleanup)
            print("Request was cancelled!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raise  # Re-raise to propagate the cancellation
        except Exception as e:
            print(f"Error occurred!!!!!!!!!!!!!!!!!!", e)
            raise  # Re-raise the exception to propagate it further
        return completion


class OpenAIAsyncClient(AsyncLLMInterface):

    def __init__(self, host: str, port: int):
        if is_ipv6(host):
            self._host = f'[{host}]'
        else:
            self._host = host
        self._port = port
        self.timeout = aiohttp.ClientTimeout(total=9600)
        self.url = f"http://{self._host}:{self._port}/chat/completions"

    async def chat_completions(self, content: dict, meta_info: dict, config: DictConfig):
        try:
            session = aiohttp.ClientSession(timeout=self.timeout)
            generation_kwargs = meta_info['generation_kwargs']

            # 构建ChatCompletionRolloutMessageParam格式的messages
            messages = {"prompt": content["prompt"]}

            # 如果有图像数据，添加图像相关字段
            if 'pixel_values_ref' in content:
                messages['pixel_values_ref'] = content['pixel_values_ref']
                messages['image_grid_hw'] = content['image_grid_hw']

            request_data = {
                "model": "rollout",  # 使用server期望的模型名
                "messages": messages,  # ChatCompletionRolloutMessageParam格式
                "top_p": generation_kwargs['top_p'],
                "top_k": generation_kwargs['top_k'],
                "temperature": generation_kwargs['temperature'],
                "max_tokens": generation_kwargs['max_new_tokens'],
                "max_length": config.prompt_length + config.response_length,
                "meta_info": meta_info,  # alpha-seed server需要meta_info
            }

            async with session.post(url=self.url, headers={"Authorization": "Bearer token-abc123"},
                                    json=request_data) as resp:
                if resp.status == 200:
                    ret = await resp.json()
                    # 保持alpha-seed server的原始响应格式
                    return ret
                else:
                    # 处理错误响应
                    try:
                        error_msg = await resp.json()
                    except:
                        error_msg = await resp.text()
                    raise Exception(f"API request failed with status {resp.status}: {error_msg}")
        except Exception as e:
            raise e
        finally:
            await session.close()

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port


class AsyncAgent:

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface):
        self.tokenizer = tokenizer
        self.llm = llm

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        raise NotImplementedError


class ThreadedAgent:

    def __init__(self, tokenizer: PreTrainedTokenizer, llm: AsyncLLMInterface):
        self.tokenizer = tokenizer
        self.llm = llm

    def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        raise NotImplementedError


def functional_agent(func):

    class FunctionAsyncAgent(AsyncAgent):

        def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface):
            super().__init__(tokenizer, llm)
            self.func = func

        async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
            return await self.func(item, context, **kwargs)

    return FunctionAsyncAgent
