import os
import time
import copy
import json
import uuid
import threading
import queue
import logging
import socket
import zlib
import asyncio
import uvicorn
import dataclasses
import ray

from typing import List, Any, Optional, Union
from functools import lru_cache
from dataclasses import dataclass
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException

from bytedance import metrics


@lru_cache(maxsize=1)
def get_metrics_client():
    return metrics.Client(prefix='seed.agentrl')


@lru_cache(maxsize=1)
def get_proxy_server():
    return ProxyServer()


@lru_cache(maxsize=1)
def get_proxy_client():
    return ProxyClient()


class ChatCompletionRequest(BaseModel):
    model: str  # task_id
    messages: List[dict]

    class ExtraInfo(BaseModel):
        turn: str
        session_id: str
        traj_id: Optional[str] = None

    extra_info: Optional[ExtraInfo] = None


@dataclass
class ChatCompletionResponse:
    model: str  # turn_task_id
    aborted: bool
    response: str
    payload: Any


class ScoresRequest(BaseModel):
    request_id: str  # task_id
    score: float
    extra: Optional[Union[dict, list]] = None


class Task:
    ## TaskArgs Define Start
    @dataclass
    class TaskArgs:
        framework: str = None
        dataset: str = None
        index: str = None
        model_type: str = None
        model_name: str = None
        model_connection_type: str = None
        model_connection: str = None
        task_category: str = None
        proxy_id: str = None

        def __init__(self, **kwargs):
            for field in self.__dataclass_fields__:
                setattr(self, field, kwargs.get(field, None))

    ## TaskArgs Define End

    ## TaskPayload Define Start
    @dataclass
    class TaskPayload:
        task_id: str = None  # turn_task_id under task_id
        create_timestamp: float = None
        request: object = None
        touch_timestamp: float = None
        touch_num: int = None
        finish_timestamp: float = None
        response: object = None

        def __init__(self, task_id, request):
            self.task_id = task_id
            self.create_timestamp = time.time()
            self.request = request
            self.touch_num = 0

        def get_meta_info(self):
            meta = copy.copy(self)
            meta.request = None
            return meta

        def request_elapsed(self):
            return time.time() - self.create_timestamp

        def touch_elapsed(self):
            return None if self.touch_timestamp is None else (time.time() - self.touch_timestamp)

        def total_elapsed(self):
            if self.finish_timestamp is not None and self.create_timestamp is not None:
                return self.finish_timestamp - self.create_timestamp
            return None

        def touch(self):
            self.touch_timestamp = time.time()
            self.touch_num += 1

        def untouch(self):
            self.touch_timestamp = None

        def respond(self, response):
            self.finish_timestamp = time.time()
            self.response = response

        def finished(self):
            return self.finish_timestamp is not None

    ## TaskPayload Define End

    def __init__(self, task_id, **task_args):
        self.create_timestamp = time.time()
        self.touch_timestamp = time.time()
        self.task_id = task_id  # task_id
        self.task_args = self.TaskArgs(**task_args)
        self.task_payloads = {}
        self.finish_timestamp = None
        self.result = None

    def get_meta_info(self):
        meta = copy.copy(self)
        meta.task_args = {}
        meta.task_payloads = {}
        meta.result = None
        return meta

    def request_elapsed(self):
        return time.time() - self.create_timestamp

    def touch_elapsed(self):
        return time.time() - self.touch_timestamp

    def total_elapsed(self):
        if self.finish_timestamp is not None and self.create_timestamp is not None:
            return self.finish_timestamp - self.create_timestamp
        return None

    def get_task_args(self):
        return dict(filter(lambda kv: kv[1] is not None, dataclasses.asdict(self.task_args).items()))

    def update_task_args(self, **task_args):
        self.task_args = dataclasses.replace(
            self.task_args,
            **dict(
                map(lambda kv: kv, filter(lambda kv: kv[0] in self.task_args.__dataclass_fields__, task_args.items()))))

    def task_num(self):
        return len(self.task_payloads)

    def fetch_pending_task_id(self):  # turn_task
        for task_payload in self.task_payloads.values():
            if task_payload.touch_elapsed() is None:
                return task_payload.task_id
        return None

    def exist_task(self, task_id):  # turn_task_id
        return task_id in self.task_payloads

    def get_task(self, task_id):  # turn_task_id
        return self.task_payloads.get(task_id)

    def request_task(self, task_id, request):  # turn_task_id
        if self.exist_task(task_id):
            logging.info(f"agentbench_proxy request_task: task[{task_id}] exists")
        self.task_payloads[task_id] = self.TaskPayload(task_id, request)
        self.touch_timestamp = time.time()

    def respond_task(self, task_id, response):  # turn_task_id
        if not self.exist_task(task_id):
            logging.info(f"agentbench_proxy respond_task: task[{task_id}] not found")
        self.task_payloads[task_id].respond(response)
        self.touch_timestamp = time.time()

    def trigger_task(self, task_id):  # turn_task_id
        if not self.exist_task(task_id):
            logging.info(f"agentbench_proxy trigger_task: task[{task_id}] not found")
        self.task_payloads[task_id].touch()
        self.touch_timestamp = time.time()

    def abort_task(self, task_id):  # turn_task_id
        if not self.exist_task(task_id):
            logging.info(f"agentbench_proxy abort_task: task[{task_id}] not found")
        self.task_payloads[task_id].untouch()
        self.touch_timestamp = time.time()

    def finalize(self, result):
        self.finish_timestamp = time.time()
        self.result = result
        self.touch_timestamp = time.time()

    def finished(self):
        return self.finish_timestamp is not None


class ShardedStorageBase:

    def __init__(self, name, gen_turn_task_id_func, extract_task_id_func):
        self._name = name
        self._gen_turn_task_id_func = gen_turn_task_id_func
        self._extract_task_id_func = extract_task_id_func
        self._requests = queue.Queue()
        self._tasks = {}

    def gen_turn_task_id(self, task_id):
        return self._gen_turn_task_id_func(task_id)

    def extract_task_id(self, turn_task_id):
        return self._extract_task_id_func(turn_task_id)

    def add_requests(self, requests):
        for request in requests:
            self._requests.put(request)

    def get_requests(self, limit=32):
        requests = []
        for i in range(limit):
            try:
                request = self._requests.get(block=False)
                requests.append(request)
            except Exception as e:
                pass
        return requests

    def task_exist(self, task_id):
        return task_id in self._tasks

    def turn_task_exist(self, turn_task_id):
        task_id = self._extract_task_id_func(turn_task_id)
        return task_id in self._tasks and \
            self._tasks[task_id].exist_task(turn_task_id)

    def get_task_meta(self, task_id):
        task = self._tasks.get(task_id)
        return task and task.get_meta_info()

    def get_turn_task_meta(self, turn_task_id):
        turn_task = self.get_turn(turn_task_id)
        return turn_task and turn_task.get_meta_info()

    def add_task(self, task_id, task):
        self._tasks[task_id] = task

    def pop_task(self, task_id):
        return self._tasks.pop(task_id, None)

    def get_task(self, task_id):
        return self._tasks.get(task_id)

    def finalize_task(self, task_id, request):
        task = self._tasks.get(task_id)
        task and task.finalize(request)

    def fetch_pending_turn_task(self, task_id):
        task = self._tasks.get(task_id)
        return task and task.fetch_pending_task_id()

    def turn_finished(self, turn_task_id):
        task_id = self._extract_task_id_func(turn_task_id)
        return task_id in self._tasks and \
            self._tasks[task_id].exist_task(turn_task_id) and \
            self._tasks[task_id].get_task(turn_task_id).finished()

    def get_turn(self, turn_task_id):
        task_id = self._extract_task_id_func(turn_task_id)
        return None if (task_id not in self._tasks) else self._tasks[task_id].get_task(turn_task_id)

    def request_turn(self, task_id, request):
        if task_id in self._tasks:
            turn_task_id = self._gen_turn_task_id_func(task_id)
            self._tasks[task_id].request_task(turn_task_id, request)
            return turn_task_id
        logging.info(f"agentbench_proxy request_turn: task[{task_id}] not found")

    def trigger_turn(self, turn_task_id):
        task_id = self._extract_task_id_func(turn_task_id)
        if task_id in self._tasks:
            self._tasks[task_id].trigger_task(turn_task_id)
        else:
            logging.info(f"agentbench_proxy trigger_turn: task[{task_id}] not found")

    def abort_turn(self, turn_task_id):
        task_id = self._extract_task_id_func(turn_task_id)
        if task_id in self._tasks:
            self._tasks[task_id].abort_task(turn_task_id)
        else:
            logging.info(f"agentbench_proxy abort_turn: task[{task_id}] not found")

    def respond_turn(self, turn_task_id, response):
        task_id = self._extract_task_id_func(turn_task_id)
        if task_id in self._tasks:
            self._tasks[task_id].respond_task(turn_task_id, response)
        else:
            logging.info(f"agentbench_proxy respond_turn: task[{task_id}] not found")

    def get_turn_result(self, turn_task_id):
        return self.get_turn(turn_task_id) if self.turn_finished(turn_task_id) else None


class StorageBase:
    _local_sharded_storage_registry = {}

    def __init__(self, mode, shard_num, gen_turn_task_id_func, extract_task_id_func):
        assert mode == 'ray' or os.getenv(
            "AGENTBENCH_DEBUG_MODE"), f"agentbench with {mode=}(not ray) is not recommended"
        self.mode = mode
        self.shard_num = int(shard_num)
        self._storages = [
            self.spawn_storage(mode,
                               f"AgentbenchShardedStorage_{i}",
                               gen_turn_task_id_func=gen_turn_task_id_func,
                               extract_task_id_func=extract_task_id_func) for i in range(self.shard_num)
        ]

    def spawn_storage(self, mode, name, **kwargs):

        def spawn_ray_storage(name, **kwargs):
            while True:
                try:
                    return ray.get_actor(name=name)
                except Exception as e:
                    try:
                        return ray.remote(ShardedStorageBase).options(name=name,
                                                                      scheduling_strategy="SPREAD",
                                                                      max_restarts=-1,
                                                                      max_task_retries=-1).remote(name, **kwargs)
                    except Exception as e:
                        time.sleep(0.5)

        def spawn_local_storage(name, **kwargs):
            if name not in StorageBase._local_sharded_storage_registry:
                StorageBase._local_sharded_storage_registry[name] = ShardedStorageBase(name, **kwargs)
            return StorageBase._local_sharded_storage_registry[name]

        if mode == 'ray':
            return spawn_ray_storage(name, **kwargs)
        else:
            return spawn_local_storage(name, **kwargs)

    @staticmethod
    def get_or_create(mode, shard_num, gen_turn_task_id_func=lambda x: x, extract_task_id_func=lambda x: x):
        return StorageBase(mode, shard_num, gen_turn_task_id_func, extract_task_id_func)

    @staticmethod
    def dynamic_call(obj, key, method_name, *args, **kwargs):
        index = zlib.crc32(str(key).encode('utf-8')) % obj.shard_num
        if obj.mode == "ray":
            method_ref = getattr(obj._storages[index],
                                 method_name).options(enable_task_events=False).remote(*args, **kwargs)
            x = ray.get(method_ref)
            return x
        else:
            method = getattr(obj._storages[index], method_name)
            return method(*args, **kwargs)

    @staticmethod
    def dynamic_call_foreach(obj, method_name, *args, **kwargs):
        rsp = []
        for index in range(obj.shard_num):
            if obj.mode == "ray":
                method_ref = getattr(obj._storages[index],
                                     method_name).options(enable_task_events=False).remote(*args, **kwargs)
                rsp.append(ray.get(method_ref))
            else:
                method = getattr(obj._storages[index], method_name)
                rsp.append(method(*args, **kwargs))
        return rsp


class Storage:

    def __init__(self, **kwargs):
        self.gen_turn_task_id_func = lambda x: f'{x}-{str(uuid.uuid4().hex)}'
        self.extract_task_id_func = lambda x: x.rsplit('-', 1)[0]
        self.storage = StorageBase(mode=os.getenv('AGENTBENCH_STORAGE_MODE', 'ray'),
                                   shard_num=int(os.getenv('AGENTBENCH_STORAGE_SHARD_NUM', '97')),
                                   gen_turn_task_id_func=self.gen_turn_task_id_func,
                                   extract_task_id_func=self.extract_task_id_func)

    def task_exist(self, task_id):
        return StorageBase.dynamic_call(self.storage, task_id, "task_exist", task_id)

    def add_task(self, task_id, payload):
        StorageBase.dynamic_call(self.storage, task_id, "add_task", task_id, payload)

    def get_task_meta(self, task_id):
        return StorageBase.dynamic_call(self.storage, task_id, "get_task_meta", task_id)

    def get_turn_task_meta(self, turn_task_id):
        task_id = self.extract_task_id_func(turn_task_id)
        return StorageBase.dynamic_call(self.storage, task_id, "get_turn_task_meta", turn_task_id)

    def get_task(self, task_id):
        return StorageBase.dynamic_call(self.storage, task_id, "get_task", task_id)

    def pop_task(self, task_id):
        StorageBase.dynamic_call(self.storage, task_id, "pop_task", task_id)

    def finalize_task(self, task_id, payload):
        StorageBase.dynamic_call(self.storage, task_id, "finalize_task", task_id, payload)

    def fetch_pending_turn_task(self, task_id):
        return StorageBase.dynamic_call(self.storage, task_id, "fetch_pending_turn_task", task_id)

    def request_turn(self, task_id, payload):
        return StorageBase.dynamic_call(self.storage, task_id, "request_turn", task_id, payload)

    def turn_task_exist(self, turn_task_id):
        task_id = self.extract_task_id_func(turn_task_id)
        StorageBase.dynamic_call(self.storage, task_id, "turn_task_exist", turn_task_id)

    def trigger_turn(self, turn_task_id):
        task_id = self.extract_task_id_func(turn_task_id)
        StorageBase.dynamic_call(self.storage, task_id, "trigger_turn", turn_task_id)

    def abort_turn(self, turn_task_id):
        task_id = self.extract_task_id_func(turn_task_id)
        StorageBase.dynamic_call(self.storage, task_id, "abort_turn", turn_task_id)

    def get_turn(self, turn_task_id):
        task_id = self.extract_task_id_func(turn_task_id)
        return StorageBase.dynamic_call(self.storage, task_id, "get_turn", turn_task_id)

    def respond_turn(self, turn_task_id, payload):
        task_id = self.extract_task_id_func(turn_task_id)
        StorageBase.dynamic_call(self.storage, task_id, "respond_turn", turn_task_id, payload)

    def get_turn_result(self, turn_task_id):
        task_id = self.extract_task_id_func(turn_task_id)
        return StorageBase.dynamic_call(self.storage, task_id, "get_turn_result", turn_task_id)


class ProxyClient(Storage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def produce(self, **kwargs):
        task_id = str(uuid.uuid4())
        task_args = {
            **{
                'task_id': task_id,
                'framework': kwargs.pop('framework'),
                'dataset': kwargs.pop('dataset'),
                'index': kwargs.pop('index'),
                'model_type': 'rl',
                'model_connection_type': 'url',
                'model_name': task_id,
                'task_category': kwargs.pop('category', 'normal'),
            },
            **kwargs
        }
        StorageBase.dynamic_call(self.storage, task_id, "add_requests", [task_args])
        return task_id

    def respond_turn(self, turn_task_id, data_proto):
        assert len(data_proto) == 1, f"respond_turn with len(data_proto) = {len(data_proto)}"
        assert 'raw_response' in data_proto.non_tensor_batch, f"raw_response should be in data_proto"
        response = data_proto.non_tensor_batch['raw_response'][0]
        super().respond_turn(
            turn_task_id,
            ChatCompletionResponse(model=None, aborted=data_proto.batch is None, response=response, payload=data_proto))


class ProxyServer(Storage):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        if getattr(self, "_initialized", False):
            return

        def get_ip():

            def get_ipv4():
                return os.getenv("MY_HOST_IP", None)

            def get_ipv6():
                return (lambda x: ("[" + x.strip('[').rstrip(']').strip(".") + "]")
                        if x else None)(os.getenv("MY_HOST_IPV6", None))

            def get_ip_ray():
                if os.getenv("WG_BACKEND", None) == "ray":
                    import ray
                    return ray._private.services.get_node_ip_address()
                return None

            def get_ip_torchrpc():
                if os.getenv("WG_BACKEND", None) == "torch_rpc":
                    from single_controller.torchrpc.k8s_client import get_ip_addr
                    return get_ip_addr()
                return None

            return get_ipv4() or get_ipv6() or get_ip_ray() or get_ip_torchrpc() or '0.0.0.0'

        def get_port():

            def get_free_port():
                with socket.socket() as sock:
                    sock.bind(('', 0))
                    return sock.getsockname()[1]

            port_idx = os.getenv('AGENTBENCH_PORT_IDX', '7')
            return int(kwargs.get('port', None) or os.getenv(f"PORT{port_idx}", get_free_port()))

        self._initialized = True
        super().__init__(**kwargs)
        self.proxy_id = str(uuid.uuid4())
        self.trial_id = os.getenv('ARNOLD_TRIAL_ID', 'unk')
        self.ip = get_ip()
        self.port = get_port()
        self.service_discovery = f"{self.ip}:{self.port}"

        self.tasks_queue = queue.Queue()
        self.app = FastAPI()
        self._init_api()
        self._consume_loop_thread = self._start_consume_loop()
        self._server_thread = self._start_server()

    def consume(self):
        tasks = StorageBase.dynamic_call_foreach(self.storage, "get_requests")
        return [task for sub_tasks in tasks for task in sub_tasks]

    def _start_consume_loop(self):

        def _consume_loop():
            while True:
                backoff = True
                try:
                    tasks = self.consume()
                    for task_args in tasks:
                        task_id = task_args.pop('task_id', None)
                        if not task_id:
                            continue

                        task = Task(task_id=task_id,
                                    **{
                                        **task_args,
                                        **{
                                            'model_connection': f'http://{self.service_discovery}/v1chat/completions',
                                            'proxy_id': self.proxy_id
                                        }
                                    })
                        self.add_task(task_id, task)
                        self.tasks_queue.put({
                            'task': {
                                'task_id': task.task_id,
                                'taskpool_uid': self.proxy_id,
                                "parameters": task.get_task_args()
                            },
                            'taskpool_uid': self.proxy_id
                        })
                        get_metrics_client().emit_counter("agentbench.proxy.produce",
                                                          1,
                                                          tags={
                                                              'trial_id': self.trial_id,
                                                              'status': 'success'
                                                          })
                        task_args = json.dumps(task.get_task_args())
                        logging.info(f"agentbench_proxy add task: {task_id=}, {task_args=}")
                        backoff = False
                except Exception as e:
                    logging.info(f'agentbench: got exception {e} in consume_loop')
                backoff and time.sleep(1)

        consume_loop_thread = threading.Thread(target=_consume_loop, daemon=True, name=f'agentbench/consume_loop')
        consume_loop_thread.start()
        return consume_loop_thread

    def _init_api(self):

        @self.app.get("/tasks/test")
        def _test():
            return {"task": "test task"}

        @self.app.get("/tasks/agentbench")
        def _agentbench():
            try:
                result = self.tasks_queue.get(block=False)
                get_metrics_client().emit_counter("agentbench.proxy.task",
                                                  1,
                                                  tags={
                                                      'trial_id': self.trial_id,
                                                      'status': 'success'
                                                  })
                logging.info(
                    f"agentbench_proxy fetch_task: task_id={result.get('task', {}).get('task_id') if type(result) is dict else 'unk'}"
                )
                return result
            except Exception as e:
                get_metrics_client().emit_counter("agentbench.proxy.task",
                                                  1,
                                                  tags={
                                                      'trial_id': self.trial_id,
                                                      'status': 'fail'
                                                  })
                return {"error": "No tasks available", "task": {}}

        @self.app.post("/tasks/agentbench/scores")
        def _agentbench_scores(request: ScoresRequest):
            task_id = request.request_id
            task = self.get_task(task_id)
            if task:
                self.finalize_task(task_id, request)
                logging.info(
                    f"agentbench_proxy scores: {task_id=}, score={request.score}, interact turns={task.task_num()}")

            get_metrics_client().emit_counter("agentbench.proxy.finalize",
                                              1,
                                              tags={
                                                  'trial_id': self.trial_id,
                                                  'status': 'success' if task else 'fail'
                                              })
            return {}

        @self.app.post("/v1chat/completions")
        def _completions(request: ChatCompletionRequest):
            task_id = request.model
            if not self.task_exist(task_id):
                logging.info(f"agentbench_proxy completions: {task_id=} not found, raise 422 HTTPException")
                raise HTTPException(status_code=422, detail=f"{task_id=} not found")
            turn_task_id = self.request_turn(task_id, request)
            get_metrics_client().emit_counter("agentbench.proxy.completion_request",
                                              1,
                                              tags={
                                                  'trial_id': self.trial_id,
                                                  'status': 'success' if turn_task_id else 'fail'
                                              })
            logging.info(f"agentbench_proxy completion: {task_id=}, {turn_task_id=}")
            message = {
                'task_id': task_id,
                'turn_task_id': turn_task_id,
                'category': 'completions_request',
                'message': request.messages
            }
            #  logging.info(f"AGENTBENCH_PROXY DEBUGGING: {json.dumps(message)}")
            return {"request_id": turn_task_id}

        @self.app.get("/v1chat/completions/results")
        def _completions_results(request: Request):
            turn_task_id = request.headers.get("request_id", "") or request.query_params.get("request_id", "")
            task_id = self.extract_task_id_func(turn_task_id)
            if not self.task_exist(task_id):
                logging.info(
                    f"agentbench_proxy completions results: {task_id=}/{turn_task_id=} not found, raise 422 HTTPException"
                )
                raise HTTPException(status_code=422, detail=f"{task_id=}/{turn_task_id=} not found")
            result = self.get_turn_result(turn_task_id)
            get_metrics_client().emit_counter("agentbench.proxy.completion_respond",
                                              1,
                                              tags={
                                                  'trial_id': self.trial_id,
                                                  'status': 'success' if result is not None else 'fail'
                                              })
            if result is not None:
                logging.info(f"agentbench_proxy completions results: {turn_task_id=}")
                if result.response.aborted:
                    logging.warning(
                        f"agentbench_proxy completion results: {task_id=}/{turn_task_id=} is aborted, raise 432 HTTPException"
                    )
                    raise HTTPException(status_code=432, detail=f"{result.response.response}")
                message = {
                    'task_id': task_id,
                    'turn_task_id': turn_task_id,
                    'category': 'completions_response',
                    'message': result.response.response
                }
                #  logging.info(f"AGENTBENCH_PROXY DEBUGGING: {json.dumps(message)}")
                return {
                    "id": turn_task_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": turn_task_id,
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": result.response.response,
                            "resoning_content": "",
                        },
                        "finish_reason": "stop",
                        "logprobs": [],
                        "input_ids": [],
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                }
            else:
                return {}

    def _start_server(self):
        config = uvicorn.Config(
            self.app,
            host=None,
            port=self.port,
            limit_concurrency=2000,
            access_log=True,
            log_config=uvicorn.config.LOGGING_CONFIG,
            log_level="info",
        )
        logging.getLogger("uvicorn.access").disabled = True
        logging.getLogger("uvicorn").propagate = False
        server = uvicorn.Server(config)
        logging.info(f"agentbench proxy server listens on port[{self.port}]")
        server_thread = threading.Thread(target=lambda: asyncio.run(server.serve()),
                                         daemon=True,
                                         name='agentbench/proxy')
        server_thread.start()
        return server_thread


if __name__ == "__main__":
    import yaml
    import ray
    import requests
    import torch
    import numpy
    import importlib
    import alpha_seed.workers.agents.handlers.agentbench
    from verl import DataProto

    os.environ["AGENTBENCH_ENABLE"] = "True"
    os.environ["AGENTBENCH_DEBUG_MODE"] = "True"
    os.environ["AGENTBENCH_STORAGE_MODE"] = "ray"
    os.environ["AGENTBENCH_STORAGE_MODE"] = "local"
    os.environ["AGENTBENCH_STORAGE_SHARD_NUM"] = "3"
    os.environ["AGENTBENCH_PORT_IDX"] = "-1"

    importlib.reload(alpha_seed.workers.agents.handlers.agentbench)
    if os.environ["AGENTBENCH_STORAGE_MODE"] == 'ray':
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/../../../../../tasks/runtime_env/runtime_env.yaml'
                 ) as fin:
            runtime_env = yaml.safe_load(fin)
            for k in [
                    "AGENTBENCH_ENABLE", "AGENTBENCH_DEBUG_MODE", "AGENTBENCH_STORAGE_MODE",
                    "AGENTBENCH_STORAGE_SHARD_NUM"
            ]:
                runtime_env[k] = os.environ.get(k)
            print(runtime_env)
            ray.init(namespace="alphaseed", runtime_env=runtime_env, address='auto')

    proxy_server = get_proxy_server()

    time.sleep(1)
    test_rsp = requests.get(f"http://{proxy_server.service_discovery}/tasks/test")
    print(f"{test_rsp=}, {test_rsp.text=}")

    task_id = get_proxy_client().produce(**{'framework': 'agentless', 'dataset': 'swe_gym_test', 'index': '0'})
    time.sleep(7)
    agentbench_api_result = requests.get(f"http://{proxy_server.service_discovery}/tasks/agentbench")
    print(f"{task_id=}, {agentbench_api_result=}, {agentbench_api_result.text=}")

    completions_api_result = requests.post(f"http://{proxy_server.service_discovery}/v1chat/completions",
                                           json={
                                               "model":
                                                   task_id,
                                               "messages": [{
                                                   "role": "system",
                                                   "content": "I am system"
                                               }, {
                                                   "role": "user",
                                                   "content": "I am user"
                                               }]
                                           })
    print(f"{task_id=}, {completions_api_result=}, {completions_api_result.text=}")

    turn_task_id = json.loads(completions_api_result.text).get('request_id')

    completions_task = get_proxy_client().get_turn(turn_task_id)
    print(f"{task_id=}, {completions_task.task_id=}, {completions_task.request=}")

    get_proxy_client().respond_turn(
        turn_task_id,
        DataProto.from_dict(tensors={
            'input_ids': torch.ones(1, 3, 8, 8),
            'attention_mask': torch.ones(1, 3, 8, 8),
        },
                            non_tensors={
                                'request_id':
                                    numpy.array([completions_task.task_id], dtype=str),
                                'raw_response':
                                    numpy.array(["<think>I don't know what I am thinking about</think>I am response"],
                                                dtype=str),
                            }))
    completions_results_api_result = requests.get(f"http://{proxy_server.service_discovery}/v1chat/completions/results",
                                                  params={"request_id": completions_task.task_id})
    print(
        f"{task_id=}, {completions_task.task_id=}, {completions_results_api_result=}, {completions_results_api_result.text=}"
    )

    agentbench_scores_api_result = requests.post(f"http://{proxy_server.service_discovery}/tasks/agentbench/scores",
                                                 json={
                                                     "request_id": task_id,
                                                     "score": 1
                                                 })
    print(f'{get_proxy_client().get_task(task_id).total_elapsed()=}, {get_proxy_client().get_task(task_id).result=}')
    get_proxy_client().pop_task(task_id)
    print(f'{get_proxy_client().get_task(task_id)=}')
