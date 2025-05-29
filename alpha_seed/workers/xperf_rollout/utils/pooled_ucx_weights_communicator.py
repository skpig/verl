import os
import traceback
import random
from typing import Any, Dict, List
import threading
import asyncio
import socket
import logging
import aiomonitor
from collections import defaultdict
from queue import Queue
from contextlib import contextmanager

import ucxx
import cupy as cp
import numpy as np
import torch
import torch.distributed as dist
from ucxx.exceptions import UCXConnectionResetError, UCXCanceledError

from alpha_seed.workers.xperf_rollout.utils.base_weights_communicator import WeightsCommunicator
from verl.utils.debug import log_gpu_memory_usage


def get_free_port():
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        # 绑定到一个随机的可用端口
        s.bind(('::', 0))
        # 获取绑定的端口号
        port = s.getsockname()[1]
    return port


class ConnectionPool:

    def __init__(self):
        self.pool = defaultdict(list)  # address -> [ep]
        self.in_use = {}  # ep -> address
        self._mutex = asyncio.Lock()

    async def get_connection(self, address: str):
        async with self._mutex:
            if self.pool[address]:
                # Get a free connection from the pool
                ep = self.pool[address].pop()
                self.in_use[ep] = address
                return ep

        ip_port = address.rsplit(":", 1)
        assert len(ip_port) == 2, f"expecting ip_port to be a list of 2 strings, got {ip_port}"
        ip = ip_port[0]
        ip = ip.removeprefix("[")
        ip = ip.removesuffix("]")
        port = int(ip_port[1])
        print(f"Creating endpoint with ip = {ip}, port = {port}")
        ep = None
        for try_count in range(50):
            try:
                ep = await ucxx.create_endpoint(ip, port)
            except Exception as e:
                logging.warning(f"Failed to create endpoint, retry for the {try_count + 1} time: {e}, {ip=}, {port=}")
                await asyncio.sleep(2)
            else:
                logging.info(f"Successfully created endpoint on {ip}:{port}")
                break

        assert ep is not None, "Endpoint cannot be created"
        self.in_use[ep] = address
        return ep

    def put_connection(self, ep):
        if ep in self.in_use:
            address = self.in_use.pop(ep)
            self.pool[address].append(ep)


class UCXWeightsCommunicator(WeightsCommunicator):

    def __init__(self, inference_engine, standalone, device_mesh):
        self.inference_engine = inference_engine
        self.standalone = standalone
        self.device_mesh = device_mesh  # 注意不开tp时这个是None

        self.source_address = ""  # 作为client时，默认要连到server的地址 ip:port 格式
        self.server_up = False  # 是否作为server启动
        self.address = ""  # 作为server启动时，server的地址 ip:port 格式
        self._setup_completed = threading.Event()

        self.server_finish_event = threading.Event()  # server侧的handler通知其他线程参数传输完成
        self.enter_ready = asyncio.Event()  # server侧的handler等待model参数准备好
        self.send_buffer_sema = asyncio.Semaphore(2)  # 由于send weight需要先copy到cupy buffer，为避免OOM，限制同时send数量

        # 作为actor server时，nccl world里的rank，用来做序号标记debug用
        self.rank = 0
        if dist.is_initialized():
            self.rank = dist.get_rank()

        # ucx endpoint接池类
        self.connection_pool = ConnectionPool()

    def setup_as_client(self, role, source_address: str):
        assert source_address is not None
        print(f"receive source address = {source_address}")
        self.source_address = source_address

        async def register(client_role):
            ep = await self.connection_pool.get_connection(source_address)
            try:
                register_msg = f"register:{client_role}"
                await ep.send_obj(register_msg.encode("utf-8"))
                ok = await ep.recv_obj()
            finally:
                self.connection_pool.put_connection(ep)

        def register_in_thread(my_role):
            asyncio.run(register(my_role))
            return True

        register_thread = threading.Thread(target=register_in_thread, args=(role,))
        register_thread.start()
        register_thread.join()
        print("register finished")
        self._setup_completed.set()

    def wait_for_setup_completed(self):
        self._setup_completed.wait()

    async def _event_handler(self, ep):
        while True:
            try:
                raw_tensor_key = await ep.recv_obj()
                if raw_tensor_key is None:
                    continue
                tensor_key = raw_tensor_key.decode("utf-8")

                if tensor_key.startswith("register:"):
                    print(f"tensor_key = {tensor_key} received")
                    await ep.send_obj(f"r{self.rank}".encode("utf-8"))
                elif tensor_key == "group_end":
                    print("tensor_key = group_end received")
                    self.server_finish_event.set()
                    await ep.send_obj(f"r{self.rank}".encode("utf-8"))
                elif tensor_key == "ping":
                    await asyncio.sleep(5)
                    await ep.send_obj("pong".encode("utf-8"))
                elif tensor_key.startswith("layers_weight."):
                    await self.enter_ready.wait()  # 等待actor sharding manager把xperf engine fsdp/megatron的参数准备好
                    layers_weight = self.inference_engine.engine.module.layers_weight
                    tensor_key = tensor_key.removeprefix("layers_weight.")
                    indices = tensor_key.split("-")
                    i = int(indices[0])
                    j = int(indices[1])
                    weight = layers_weight[i][j]
                    if isinstance(weight, torch.Tensor):
                        origin_dtype = weight.dtype
                        if origin_dtype == torch.float8_e4m3fn or origin_dtype == torch.uint8:
                            weight = weight.view(torch.int8)
                        if origin_dtype == torch.bfloat16:
                            weight = weight.view(torch.int16)

                        async with self.send_buffer_sema:
                            weight = weight.cuda()
                            dlpack = torch.utils.dlpack.to_dlpack(weight)
                            send_buf = cp.from_dlpack(dlpack).view(cp.uint8).copy()
                            cp.cuda.get_current_stream().synchronize()
                            await ep.send(send_buf)
                else:
                    # 注意不要乱发不知道的key，会处理不了
                    await self.enter_ready.wait()
                    weight_tensor = getattr(self.inference_engine.engine.module, tensor_key)
                    if weight_tensor.dtype == torch.bfloat16:
                        weight_tensor = weight_tensor.view(torch.int16)

                    async with self.send_buffer_sema:
                        weight_tensor = weight_tensor.cuda()
                        dlpack = torch.utils.dlpack.to_dlpack(weight_tensor)
                        send_buf = cp.from_dlpack(dlpack).view(cp.uint8).copy()
                        cp.cuda.get_current_stream().synchronize()
                        await ep.send(send_buf)
            except (UCXConnectionResetError, UCXCanceledError) as e:
                print("connection reset by client side, server handler ignored", e)
                break
            except Exception as e:
                traceback.print_exc()
                print(f"error found on server, but continuing event handle loop: {e}")

    async def _server(self, port_queue: Queue):
        for try_count in range(50):
            try:
                await asyncio.sleep(random.random() * 0.2)  # reduce port conflict probabilities in the same host
                port = get_free_port()
                lf = ucxx.create_listener(self._event_handler, port)
            except Exception as e:
                logging.warning(f"failed to create listener, retry for the {try_count + 1} time: {e}")
                await asyncio.sleep(2)
            else:
                logging.info(f"started listener on port {lf.port} with ip = {lf.ip}")
                port_queue.put(port)
                self.server_up = True
                break
        assert self.server_up, "server cannot started"
        while True:
            await asyncio.sleep(0.1)

    def _start_server(self, port_queue: Queue):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # monitor asyncio tasks, use `telnet 127.0.0.1 21000` to connect to the monitor
        print(f"aiomonitor started on rank={self.rank}, "
              f"use `telnet 127.0.0.1 {21000 + self.rank}` to connect to the monitor")
        with aiomonitor.Monitor(loop, termui_port=21000 + self.rank, console_enabled=False):
            loop.run_until_complete(self._server(port_queue))
        loop.close()

    def setup_as_server(self, ifname=None) -> str:
        ip = ucxx.get_address(ifname=ifname)
        assert ip is not None and ip != "", f"expecting ucxx.get_address return non-empty, got {ip}"
        port_queue = Queue()

        threading.Thread(target=self._start_server, args=(port_queue,), daemon=True, name="ucx-server").start()

        port = port_queue.get(block=True, timeout=300)

        if ":" in ip:
            ip = "[" + ip + "]"
        self.address = f"{ip}:{port}"
        return self.address

    def update_standalone_worker(self, role):
        print(f"update_standalone_worker called with role = {role}")

        if self.server_up:  # called on server, skipping
            self.server_finish_event.clear()
            self.enter_ready.set()  # notify client to read weights
            print(f"server up, waiting for finish signal from any client")
            self.server_finish_event.wait()  # wait for all clients finish reading
            self.enter_ready.clear()  # clear for next turn
            self.inference_engine.current_steps = 0
            print("server finished")
            return

        async def transfer_single_weight(tensor_key, buffer):
            """
            receive weight from server to this `buffer` object
            """
            ep = await self.connection_pool.get_connection(self.source_address)
            try:
                origin_dtype = buffer.dtype
                origin_shape = buffer.shape
                if (not tensor_key.startswith("layers_weight.")) and isinstance(buffer, torch.Tensor):
                    origin_dtype = buffer.dtype
                    if origin_dtype == torch.float8_e4m3fn or origin_dtype == torch.uint8:
                        buffer = buffer.view(torch.int8)
                        if self.inference_engine.engine.module.quant_mode == "WFP8":
                            origin_dtype = torch.float8_e4m3fn
                    if origin_dtype == torch.bfloat16:
                        buffer = buffer.view(torch.int16)

                await ep.send_obj(tensor_key.encode("utf-8"))

                if buffer.dtype == torch.bfloat16:
                    origin_dtype = torch.bfloat16
                    buffer = buffer.view(torch.int16)

                recv_buf = cp.empty(buffer.nelement() * buffer.element_size(), dtype=cp.uint8)
                await ep.recv(recv_buf)

                weight = torch.as_tensor(recv_buf, device=buffer.device).view(origin_dtype).reshape(origin_shape)
                if tensor_key.startswith("layers_weight."):
                    tensor_key = tensor_key.removeprefix("layers_weight.")
                    indices = tensor_key.split("-")
                    i = int(indices[0])
                    j = int(indices[1])
                    self.inference_engine.engine.module.layers_weight[i][j].data = weight.data
                else:
                    setattr(self.inference_engine.engine.module, tensor_key, weight)

                del recv_buf
                return True
            finally:
                self.connection_pool.put_connection(ep)

        async def transfer_weights():
            layers_weight = self.inference_engine.engine.module.layers_weight
            # blocking for first 3 weights
            await transfer_single_weight("layernorm_weight", self.inference_engine.engine.module.layernorm_weight)
            await transfer_single_weight("lm_head_weight", self.inference_engine.engine.module.lm_head_weight)
            await transfer_single_weight("wte_weight", self.inference_engine.engine.module.wte_weight)

            for layer, layer_weight in enumerate(layers_weight):
                for i, weight in enumerate(layer_weight):
                    if isinstance(weight, torch.Tensor):
                        await transfer_single_weight(f"layers_weight.{layer}-{i}", weight)

            self.inference_engine.current_steps = 0
            # free all recv buffer space to avoid OOM
            cp.get_default_memory_pool().free_all_blocks()

        print("start transfer_weights")

        def client_run():
            return asyncio.run(transfer_weights())

        client_thread = threading.Thread(target=client_run)
        client_thread.start()
        client_thread.join()
        print("finished transfer_weights")

        log_gpu_memory_usage(f'After {role} update')
        print("client finished")

    def update_standalone_worker_end(self, addresses: List[str]):

        async def send_group_end_signal(addr: str):
            ep = await self.connection_pool.get_connection(addr)
            try:
                print(f"ep({addr}) created to send group_end msg")
                group_end_msg = "group_end"
                await ep.send_obj(group_end_msg.encode("utf-8"))
                # server需要配合回复一个消息，并在这里接收，不然server可能根本收不到上面发的数据，不知道为什么
                ok = await ep.recv_obj()
                print(f"ep({addr}) received group_end response from: {ok}")
            finally:
                self.connection_pool.put_connection(ep)

        async def broadcast_group_end_signal():
            tasks = [asyncio.create_task(send_group_end_signal(addr)) for addr in addresses]
            await asyncio.gather(*tasks)

        def broadcast_group_end_signal_thread():
            return asyncio.run(broadcast_group_end_signal())

        t = threading.Thread(target=broadcast_group_end_signal_thread, daemon=True)
        t.start()
        t.join()
        print("client finished")
