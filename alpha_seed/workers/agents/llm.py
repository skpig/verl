import asyncio
import copy
from abc import ABC, abstractmethod

import aiohttp
import httpx
import torch
from omegaconf import DictConfig

from mono_rl import DataProto
from mono_rl.utils.network import is_ipv6


def make_reqeust_data_and_metadata(item: DataProto, prompt: str, host, port):
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
        'host': host,
        'port': port,
    }
    # required for tool calling
    if (key := 'extra_data') in item.non_tensor_batch:
        meta_info[key] = item.non_tensor_batch[key][0]

    return data, meta_info


class AsyncLLMInterface(ABC):

    @abstractmethod
    async def chat_completions(self, content: dict, meta_info: dict, config: DictConfig):
        raise NotImplementedError()

    @property
    @abstractmethod
    def host(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def port(self) -> int:
        raise NotImplementedError()

    async def complete(self, item: DataProto, config: DictConfig, prompt: str = ''):
        completion = None
        try:
            data, meta_info = make_reqeust_data_and_metadata(item, prompt, self.host, self.port)
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

    def __init__(self, host: str, port: int, max_connection: int):
        if is_ipv6(host):
            self._host = f'[{host}]'
        else:
            self._host = host
        self._port = port
        self.timeout = aiohttp.ClientTimeout(total=9600)
        self.max_connection = max_connection
        self.url = f"http://{self._host}:{self._port}/chat/completions"
        self._session = None
        self._session_mtx = asyncio.Lock()

    async def get_session(self):
        async with self._session_mtx:
            if self._session is None:
                connector = aiohttp.TCPConnector(
                    limit=self.max_connection,  # 总连接池大小
                    keepalive_timeout=300,  # 空闲x sec后再释放
                    enable_cleanup_closed=True,  # 启用清理已关闭的连接
                )
                self._session = aiohttp.ClientSession(connector=connector, timeout=self.timeout)
        return self._session

    async def chat_completions(self, content: dict, meta_info: dict, config: DictConfig):
        try:
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

            sess = await self.get_session()
            async with sess.post(url=self.url, headers={"Authorization": "Bearer token-abc123"},
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

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port


class SyncLLMInterface(ABC):

    @abstractmethod
    def chat_completions(self, content: dict, meta_info: dict, config: DictConfig):
        raise NotImplementedError()

    @property
    @abstractmethod
    def host(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def port(self) -> int:
        raise NotImplementedError()

    def complete(self, item: DataProto, config: DictConfig, prompt: str = ''):
        completion = None
        try:
            data, meta_info = make_reqeust_data_and_metadata(item, prompt, self.host, self.port)
            completion = self.chat_completions(data, meta_info, config)
        except asyncio.CancelledError:
            # Handle task cancellation (e.g., cleanup)
            print("Request was cancelled!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raise  # Re-raise to propagate the cancellation
        except Exception as e:
            print(f"Error occurred!!!!!!!!!!!!!!!!!!", e)
            raise  # Re-raise the exception to propagate it further
        return completion


class OpenAIClient(SyncLLMInterface):

    def __init__(self, host: str, port: int):
        if is_ipv6(host):
            self._host = f'[{host}]'
        else:
            self._host = host
        self._port = port
        self.timeout = 9600  # httpx timeout in seconds
        self.url = f"http://{self._host}:{self._port}/chat/completions"

    def chat_completions(self, content: dict, meta_info: dict, config: DictConfig):
        try:
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

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url=self.url,
                                       headers={"Authorization": "Bearer token-abc123"},
                                       json=request_data)

                if response.status_code == 200:
                    ret = response.json()
                    # 保持alpha-seed server的原始响应格式
                    return ret
                else:
                    # 处理错误响应
                    try:
                        error_msg = response.json()
                    except:
                        error_msg = response.text
                    raise Exception(f"API request failed with status {response.status_code}: {error_msg}")
        except Exception as e:
            raise e

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port
