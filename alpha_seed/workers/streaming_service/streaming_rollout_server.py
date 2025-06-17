# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a XPerfGPT Rollout
"""
import traceback
from typing import Tuple

import ray
import uuid
import time
import asyncio
import uvicorn
import logging
import dill
import base64
from abc import ABC, abstractmethod
from http import HTTPStatus
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from alpha_seed.workers.streaming_service.protocol import (ChatCompletionRequest, ChatCompletion,
                                                           ChatCompletionMessageRollout, Choice, CompletionUsage,
                                                           ErrorResponse)
from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManagerRegisterCenter
from alpha_seed.workers.streaming_service.streaming_utils import get_node_ip, get_free_port
from alpha_seed.workers.xperf_rollout.component.query import AsyncQuery, Query


class OpenAIProxy(ABC):

    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()

    @abstractmethod
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        pass

    def setup_routes(self):

        @self.app.post("/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
            response = await self.create_chat_completion(request, raw_request)
            if isinstance(response, ErrorResponse):
                return JSONResponse(content=response.dict(), status_code=response.code)
            return JSONResponse(content=response.dict())

    def create_query(self, request: ChatCompletionRequest) -> Query:
        prompt = request.messages['prompt']
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = prompt
        request_id = uuid.uuid4().hex
        return Query.from_request(input_ids, request_id, request.to_sampling_params(), request.meta_info)

    def create_response(self, query: Query) -> JSONResponse:
        message = ChatCompletionMessageRollout(
            role="assistant",
            raw_output_ids=query.output_tokens,
            response_log_probs=query.new_token_log_probs,
            is_finished=query.is_finished,
            model_output_mask=query.model_output_mask,
            extra_data=query.extra_data,
            metrics=query.metrics,
        )
        choices = []
        choice_data = Choice(
            index=0,
            message=message,
            finish_reason="stop",
        )
        choices.append(choice_data)

        usage = CompletionUsage(completion_tokens=query.new_token_len,
                                prompt_tokens=query.input_len,
                                total_tokens=query.input_len + query.new_token_len)

        response = ChatCompletion(id=query.id,
                                  choices=choices,
                                  created=int(time.time()),
                                  model="rollout",
                                  object="chat.completion",
                                  usage=usage)
        return response

    def create_error_response(self,
                              message: str,
                              err_type: str = "BadRequestError",
                              status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(message=message, type=err_type, code=status_code.value)


class AsyncXPerfGPTRolloutServer(OpenAIProxy):

    def __init__(self, config, tokenizer=None, request_manager_name='standalone_rollout'):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.server = None
        self.server_task = None
        self.host = None
        self.port = None
        self.request_manager = RequestManagerRegisterCenter.get(request_manager_name)

    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        """
        此方法要等到这个请求生成完才返回
        """
        try:
            query: Query = self.create_query(request)
        except Exception as e:
            logging.exception("Error in [create_query]")
            return self.create_error_response(str(e))

        # submit query to request pool
        query_id = ray.get(self.request_manager.put_new_query.remote(query))

        # await prompt generation finished
        finished_query = await self.request_manager.wait_until_finished.remote(query_id)
        try:
            response = self.create_response(finished_query)
        except Exception as e:
            logging.exception("Error in [create_response]")
            return self.create_error_response(str(e))
        return response

    async def start_server(self) -> Tuple[str, int]:
        last_e = None
        last_tb = None
        self.host = get_node_ip()
        assert self.host is not None, "cannot find non-loopback ip address in this environment, please check manually"

        # 各种连接创建的很多，可能在获取端口后的一瞬间就被占了，这里重试几次尽量让server启动成功
        max_retries = 5
        for i in range(max_retries):
            try:
                self.port = get_free_port()
                config = uvicorn.Config(self.app,
                                        host=self.host,
                                        port=self.port,
                                        loop="asyncio",
                                        timeout_keep_alive=300,
                                        backlog=16384)
                logging.getLogger("uvicorn.access").disabled = True
                logging.getLogger("uvicorn").propagate = False
                self.server = uvicorn.Server(config)
                self.server.should_exit = True  # 加了这个后，.serve()调用只负责初始化，不阻塞loop
                await self.server.serve()  # 如果端口被占用，这里会抛出，重新获取一个
                self.server_task = asyncio.create_task(self.server.main_loop())
                return self.host, self.port
            except (OSError, SystemExit) as e:
                import traceback
                last_e = e
                last_tb = traceback.format_exc()
                continue

        print(last_tb)
        raise last_e

    async def stop_server(self):
        """Gracefully shutdown the server"""
        await self.server.shutdown()
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
            self.server_task = None

    async def __aenter__(self):
        """Async context manager startup"""
        await self.start_server()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager shutdown"""
        await self.stop_server()
