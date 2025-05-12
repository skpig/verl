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

import ray
import uuid
import time
import asyncio
import uvicorn
import logging
from abc import ABC, abstractmethod
from http import HTTPStatus
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from alpha_seed.workers.streaming_service.streaming_rollout import RemoteAsyncXPerfGPTRollout
from alpha_seed.workers.streaming_service.protocol import (ChatCompletionRequest, ChatCompletion,
                                                           ChatCompletionMessageRollout, Choice, CompletionUsage,
                                                           ErrorResponse)
from alpha_seed.workers.xperf_rollout.component.query import AsyncQuery


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

    def create_query(self, request: ChatCompletionRequest) -> AsyncQuery:
        prompt = request.messages['prompt']
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = prompt
        request_id = str(uuid.uuid4())
        return AsyncQuery.from_request(input_ids, request_id, request.to_sampling_params())

    def create_response(self, query: AsyncQuery) -> JSONResponse:
        message = ChatCompletionMessageRollout(
            role="assistant",
            raw_output_ids=query.global_new_token_ids,
            response_log_probs=query.new_token_log_probs,
            is_finished=query.is_finished,
            response_probs_gt_threshold_num=query.probs_gt_threshold_num,
            response_probs_lt_threshold_sum=query.probs_lt_threshold_sum,
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

        response = ChatCompletion(id=str(query.id),
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

    def __init__(self, config, tokenizer=None, model_hf_config=None, actor_cls=RemoteAsyncXPerfGPTRollout):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model_hf_config = model_hf_config
        self.actor_cls = actor_cls
        if self.config.actor_rollout_ref.rollout.mode == "server":
            self.world_size = self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
        else:
            self.world_size = self.config.rollout_server.nnodes * self.config.rollout_server.n_gpus_per_node
        self.tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.dp_size = self.config.actor_rollout_ref.rollout.get("attention_data_parallel_size", 1)
        self.mp_size = self.tp_size * self.dp_size
        self.replica_num = self.world_size // self.mp_size

        self.server_task = None
        self.workers = []
        self.inflight_query_num = 0

    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        try:
            query = self.create_query(request)
        except Exception as e:
            logging.exception("Error in [create_query]")
            return self.create_error_response(str(e))

        import random
        replica_index = random.randint(0, self.replica_num - 1)
        # load balance by random choice
        start_rank = replica_index * self.mp_size

        # use sync method. note that ray actor async method is an out-of-order execution
        try:
            remote_list = []
            # dispatch to one instance
            for rank_offset in range(self.mp_size):
                rank = start_rank + rank_offset
                remote_call = getattr(self.workers[rank], self.fused_worker_execute_fn_name)
                remote_call = remote_call.remote(f"{self.sub_cls_name}_fwmn_add_inflight_query", query)
                remote_list.append(remote_call)
            query_idx = ray.get(remote_list)[0]
        except Exception as e:
            logging.exception("Error in [add_inflight_query]")
            return self.create_error_response(str(e))

        # async call, cannot be blocked
        try:
            remote_list = []
            # dispatch to one instance
            for rank_offset in range(self.mp_size):
                rank = start_rank + rank_offset
                remote_call = getattr(self.workers[rank], "_async" + self.fused_worker_execute_fn_name)
                remote_call = remote_call.remote(f"{self.sub_cls_name}_fwmn_get_inflight_query", query_idx)
                remote_list.append(remote_call)
            await asyncio.gather(*remote_list, return_exceptions=True)
            query = ray.get(remote_list[0])
        except Exception as e:
            logging.exception("Error in [get_inflight_query]")
            return self.create_error_response(str(e))

        try:
            response = self.create_response(query)
        except Exception as e:
            logging.exception("Error in [create_response]")
            return self.create_error_response(str(e))
        return response

    def setup_actors(self):
        WorkerActor = ray.remote(num_cpus=1, num_gpus=1)(self.actor_cls)
        master_actor = WorkerActor.remote(self.config, self.tokenizer, self.model_hf_config, False, 0, self.world_size,
                                          None, None)
        self.workers.append(master_actor)
        master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
        logging.info("[setup_actors] workerActor initiating {} with cls {}".format(WorkerActor, self.actor_cls))

        for rank in range(1, self.world_size):
            worker = WorkerActor.remote(self.config, self.tokenizer, self.model_hf_config, False, rank, self.world_size,
                                        master_addr, master_port)
            self.workers.append(worker)
        logging.info("[setup_actors] init workerActor {}".format(len(self.workers)))

        remote_list = []
        for worker in self.workers:
            remote_list.append(worker.setup_distributed.remote())
        for worker in self.workers:
            remote_list.append(worker.setup_rollout.remote())
        ray.get(remote_list)

    def reset_inflight_query_num(self):
        self.inflight_query_num = 0

    def get_inflight_query_num(self):
        return self.inflight_query_num

    def attach_actors(self, worker_group):
        if worker_group is None:
            return
        self.worker_group = worker_group
        self.workers = worker_group._workers
        self.sub_cls_name = worker_group.sub_cls_name
        self.fused_worker_execute_fn_name = worker_group.fused_worker_execute_fn_name
        logging.info("[attach_actors] attach workerActor {}".format(len(self.workers)))

    async def start_server(self, host="0.0.0.0", port=8000):
        # get a free port and addr
        # from single_controller.base.worker import WorkerHelper
        # worker_helper = WorkerHelper()
        # free_port_addr = list(worker_helper.get_availale_master_addr_port())
        config = uvicorn.Config(self.app, host=host, port=8001, loop="asyncio", timeout_keep_alive=300, backlog=16384)
        logging.getLogger("uvicorn.access").disabled = True
        logging.getLogger("uvicorn").propagate = False
        server = uvicorn.Server(config)
        self.server_task = asyncio.create_task(server.serve())

    async def stop_server(self):
        """Gracefully shutdown the server"""
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
        self.server_task.cancel()
        try:
            await self.server_task
        except asyncio.CancelledError:
            pass
        await self.stop_server()
