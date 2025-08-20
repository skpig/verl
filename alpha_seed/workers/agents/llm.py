import asyncio
import copy
import inspect
from abc import ABC, abstractmethod
from functools import wraps

import aiohttp
import httpx
import ray
import torch
from omegaconf import DictConfig

from alpha_seed.workers.agents.monitor_ctx import current_agent_tracker, get_current_agent_tracker
from alpha_seed.workers.agents.monitoring import AgentTaskTracker
from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManager, RequestManagerRegisterCenter
from alpha_seed.workers.xperf_rollout.component.query import Query
from alpha_seed.workers.streaming_service.protocol import ChatCompletionRollout, ChoiceRollout, ChatCompletionMessageRollout, CompletionUsage
import uuid
import time
import numpy as np
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
    if 'image_data_ref' in item.non_tensor_batch:
        image_data_ref = item.non_tensor_batch['image_data_ref']
        if image_data_ref is not None and len(image_data_ref) > 0:
            assert isinstance(image_data_ref, list) or isinstance(
                image_data_ref, np.ndarray), f"type of image_data_ref error: {type(image_data_ref)}"
            if isinstance(image_data_ref, np.ndarray):
                image_data_ref = image_data_ref.tolist()
            data['image_data_ref'] = image_data_ref
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


def trace_llm_call(func):
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(self, item: DataProto, config: DictConfig, prompt: str = ''):
            try:
                tracker: AgentTaskTracker = current_agent_tracker.get()
            except LookupError:
                return await func(self, item, config, prompt)

            with tracker.llm_call(item) as capturer:
                completion = await func(self, item, config, prompt)
                capturer.capture_output(completion)
                return completion
    else:

        @wraps(func)
        def wrapper(self, item: DataProto, config: DictConfig, prompt: str = ''):
            try:
                tracker: AgentTaskTracker = get_current_agent_tracker()
            except AttributeError:
                return func(self, item, config, prompt)

            with tracker.llm_call(item) as capturer:
                completion = func(self, item, config, prompt)
                capturer.capture_output(completion)
                return completion

    return wrapper


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

    @trace_llm_call
    async def complete(self, item: DataProto, config: DictConfig, prompt: str = '') -> dict:
        completion = None
        data, meta_info = make_reqeust_data_and_metadata(item, prompt, self.host, self.port)
        try:
            completion = await self.chat_completions(data, meta_info, config)
        except asyncio.CancelledError:
            # Handle task cancellation (e.g., cleanup)
            print("Request was cancelled!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raise  # Re-raise to propagate the cancellation
        except asyncio.TimeoutError as e:
            uid = meta_info['uid']
            print(f"request timeout {uid=}, {data=}")
            raise
        except Exception as e:
            print(f"Error occurred!!!!!!!!!!!!!!!!!!", e)
            raise  # Re-raise the exception to propagate it further
        return completion


class OpenAIAsyncClient(AsyncLLMInterface):

    def __init__(self, host: str, port: int, max_connection: int, timeout: float = 9600):
        if is_ipv6(host):
            self._host = f'[{host}]'
        else:
            self._host = host
        self._port = port
        self.timeout = aiohttp.ClientTimeout(total=timeout)
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
            messages = {"prompt": content["prompt"], 'image_data_ref': content.get('image_data_ref')}

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


class DirectAsyncClient(AsyncLLMInterface):

    def __init__(self, request_manager_name):
        self.request_manager: RequestManager = RequestManagerRegisterCenter.get(request_manager_name)

    async def chat_completions(self, content: dict, meta_info: dict, config: DictConfig):
        # 从content和meta_info构造Query对象
        prompt = content["prompt"]
        if isinstance(prompt, str):
            input_ids = []
            input_prompt = prompt
        else:
            input_ids = prompt
            input_prompt = ""

        request_id = meta_info.get('uid', uuid.uuid4().hex)
        generation_kwargs = meta_info['generation_kwargs']

        # 构造sampling参数
        sampling_kwargs = {
            "top_k": generation_kwargs['top_k'],
            "top_p": generation_kwargs['top_p'],
            "temperature": generation_kwargs['temperature'],
            "max_new_tokens": generation_kwargs['max_new_tokens'],
            "max_length": config.prompt_length + config.response_length,
        }

        # 图像支持
        image_kwargs = None
        if 'image_data_ref' in content:
            image_kwargs = {'image_data_ref': content['image_data_ref']}

        # 创建Query对象
        query = Query.from_request(input_ids, input_prompt, request_id, sampling_kwargs, meta_info, image_kwargs)

        # 提交query到request_manager
        query_id = await self.request_manager.put_new_query.remote(query)

        # 等待完成
        finished_query = await self.request_manager.wait_until_finished.remote(query_id)

        # 构造response dict（模仿server的返回格式）
        message = ChatCompletionMessageRollout(
            role="assistant",
            prompt=finished_query.input_prompt + finished_query.output_prompt[0],  # input+output
            raw_output_ids=finished_query.output_tokens,
            response_log_probs=finished_query.log_probs,
            is_finished=finished_query.is_finished,
            model_output_mask=finished_query.model_output_mask,
            extra_data=finished_query.extra_data,
            metrics=finished_query.metrics)

        choice_data = ChoiceRollout(finish_reason="stop" if finished_query.is_finished else "length",
                                    index=0,
                                    message=message)

        usage = CompletionUsage(completion_tokens=finished_query.new_token_len,
                                prompt_tokens=finished_query.original_input_len,
                                total_tokens=finished_query.original_input_len + finished_query.new_token_len)

        response = ChatCompletionRollout(id=finished_query.id,
                                         choices=[choice_data],
                                         created=int(time.time()),
                                         model="rollout",
                                         usage=usage)

        # 转换成dict返回
        return response.dict()

    @property
    def host(self) -> str:
        return ""

    @property
    def port(self) -> int:
        return 0


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

    @trace_llm_call
    def complete(self, item: DataProto, config: DictConfig, prompt: str = '') -> dict:
        completion = None
        try:
            data, meta_info = make_reqeust_data_and_metadata(item, prompt, self.host, self.port)
            completion = self.chat_completions(data, meta_info, config)
        except Exception as e:
            print(f"Error occurred!!!!!!!!!!!!!!!!!!", e)
            raise  # Re-raise the exception to propagate it further
        return completion


class OpenAIClient(SyncLLMInterface):

    def __init__(self, host: str, port: int, timeout: float = 9600):
        if is_ipv6(host):
            self._host = f'[{host}]'
        else:
            self._host = host
        self._port = port
        self.timeout = timeout  # httpx timeout in seconds
        self.url = f"http://{self._host}:{self._port}/chat/completions"

    def chat_completions(self, content: dict, meta_info: dict, config: DictConfig):
        try:
            generation_kwargs = meta_info['generation_kwargs']

            # 构建ChatCompletionRolloutMessageParam格式的messages
            messages = {"prompt": content["prompt"], 'image_data_ref': content.get('image_data_ref')}

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
                                       headers={
                                           "Authorization": "Bearer token-abc123",
                                           "Connection": "keep-alive",
                                       },
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


class DirectClient(SyncLLMInterface):

    def __init__(self, request_manager_name):
        self.request_manager: RequestManager = RequestManagerRegisterCenter.get(request_manager_name)

    def chat_completions(self, content: dict, meta_info: dict, config: DictConfig):
        # 从content和meta_info构造Query对象
        prompt = content["prompt"]
        if isinstance(prompt, str):
            input_ids = []
            input_prompt = prompt
        else:
            input_ids = prompt
            input_prompt = ""

        request_id = uuid.uuid4().hex
        generation_kwargs = meta_info['generation_kwargs']

        # 构造sampling参数
        sampling_kwargs = {
            "top_k": generation_kwargs['top_k'],
            "top_p": generation_kwargs['top_p'],
            "temperature": generation_kwargs['temperature'],
            "max_new_tokens": generation_kwargs['max_new_tokens'],
            "max_length": config.prompt_length + config.response_length,
        }

        # 图像支持
        image_kwargs = None
        if 'image_data_ref' in content:
            image_kwargs = {'image_data_ref': content['image_data_ref']}

        # 创建Query对象
        query = Query.from_request(input_ids, input_prompt, request_id, sampling_kwargs, meta_info, image_kwargs)

        # 提交query到request_manager
        query_id = ray.get(self.request_manager.put_new_query.remote(query))

        # 等待完成
        finished_query = ray.get(self.request_manager.wait_until_finished.remote(query_id))

        # 构造response dict（模仿server的返回格式）
        message = ChatCompletionMessageRollout(
            role="assistant",
            prompt=finished_query.input_prompt + finished_query.output_prompt[0],  # input+output
            raw_output_ids=finished_query.output_tokens,
            response_log_probs=finished_query.log_probs,
            is_finished=finished_query.is_finished,
            model_output_mask=finished_query.model_output_mask,
            extra_data=finished_query.extra_data,
            metrics=finished_query.metrics)

        choice_data = ChoiceRollout(finish_reason="stop" if finished_query.is_finished else "length",
                                    index=0,
                                    message=message)

        usage = CompletionUsage(completion_tokens=finished_query.new_token_len,
                                prompt_tokens=finished_query.original_input_len,
                                total_tokens=finished_query.original_input_len + finished_query.new_token_len)

        response = ChatCompletionRollout(id=finished_query.id,
                                         choices=[choice_data],
                                         created=int(time.time()),
                                         model="rollout",
                                         usage=usage)

        # 转换成dict返回
        return response.dict()

    @property
    def host(self) -> str:
        return ""

    @property
    def port(self) -> int:
        return 0
