import os
import time
import traceback
import random
from typing import Any, Dict, List, Optional
import threading
import asyncio
import socket
import logging
from collections import defaultdict
from queue import Queue

import ucxx
import cupy as cp
import numpy as np
import torch
import torch.distributed as dist
from ucxx.exceptions import UCXConnectionResetError, UCXCanceledError

from alpha_seed.utils.debug.aiomonitor import get_aiomonitor_cls
from alpha_seed.workers.xperf_rollout.utils.base_weights_communicator import WeightsCommunicator
from verl.utils.debug import log_gpu_memory_usage

logger = logging.getLogger(__file__)


def get_free_port():
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        # 绑定到一个随机的可用端口
        s.bind(('::', 0))
        # 获取绑定的端口号
        port = s.getsockname()[1]
    return port


def get_free_port_v4():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 绑定到一个随机的可用端口
        s.bind(('0.0.0.0', 0))
        # 获取绑定的端口号
        port = s.getsockname()[1]
    return port


class ConnectionPool:

    def __init__(self):
        self.pool = defaultdict(list)  # address -> [ep]
        self.in_use = {}  # ep -> address
        self._mutex = asyncio.Lock()

    async def create(self, address):
        ip_port = address.rsplit(":", 1)
        assert len(ip_port) == 2, f"expecting ip_port to be a list of 2 strings, got {ip_port}"
        ip = ip_port[0]
        ip = ip.removeprefix("[")
        ip = ip.removesuffix("]")
        port = int(ip_port[1])
        print(f"Creating endpoint with remote ip = {ip}, remote port = {port}")
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
        return ep

    async def get_connection(self, address: str):
        async with self._mutex:
            if self.pool[address]:
                # Get a free connection from the pool
                ep = self.pool[address].pop()
                self.in_use[ep] = address
                return ep

        ep = await self.create(address)
        self.in_use[ep] = address
        return ep

    def put_connection(self, ep):
        if ep in self.in_use:
            address = self.in_use.pop(ep)
            self.pool[address].append(ep)


class WeightsUpdatingInterrupt(Exception):
    pass


class WeightUpdateRWLock:

    def __init__(self):
        self._readers = {}  # ep -> ts
        self._read_lock = asyncio.Lock()  # 修改readers用的锁
        self._resource_lock = asyncio.Lock()  # 修改被加锁资源本身用的锁
        self._update_complete = asyncio.Event()  # 写锁具有更高优先级，获取到写锁之前不让新的读锁增加
        self._resource_owner = ''  # ['read', 'write', ''], 标记当前资源被哪里锁了

        # 初始状态下可让读锁
        self._update_complete.set()

    @property
    def resource_owner(self):
        return self._resource_owner

    async def acquire_read(self, ep):
        # 如果正准备update，就不让新来的acquire，避免不断有新来的client acquire下去导致update无法获取写锁
        await self._update_complete.wait()
        async with self._read_lock:
            self._readers[ep] = time.time()
            if len(self._readers) == 1:
                t0 = time.time()
                await self._resource_lock.acquire()
                t1 = time.time()
                self._resource_owner = 'read'
                print(f'read _resource_lock.acquire(), cost={t1 - t0:.3f}s')

    async def release_read(self, ep):
        async with self._read_lock:
            self._readers.pop(ep, None)
            if len(self._readers) == 0:
                if self._resource_owner != 'read':
                    return
                self._resource_lock.release()
                self._resource_owner = ''
                print('read _resource_lock.release()')

    async def acquire_update(self):
        self._update_complete.clear()
        t0 = time.time()
        await self._resource_lock.acquire()
        t1 = time.time()
        self._resource_owner = 'write'
        print(f'update _resource_lock.acquire(), cost={t1 - t0:.3f}s')

    def release_update(self):
        self._resource_lock.release()
        self._resource_owner = ''
        self._update_complete.set()
        print('update _resource_lock.release()')


class UCXWeightsCommunicator(WeightsCommunicator):

    def __init__(self, inference_engine, standalone: bool, device_mesh, enable_aiomonitor: bool = False):
        self.inference_engine = inference_engine
        self.standalone = standalone
        self.device_mesh = device_mesh  # 注意不开tp时这个是None
        self.enable_aiomonitor = enable_aiomonitor

        self.source_address = ""  # 作为client时，默认要连到server的地址 ip:port 格式
        self.server_up = False  # 是否作为server启动
        self.address = ""  # 作为server启动时，server的地址 ip:port 格式
        self._setup_completed = threading.Event()  # 已完成endpoint setup
        self._relay_weights_loaded = threading.Event()  # 已经进行过一次weights同步，根据此判断是否可以接受elastic client过来拉
        self.client_thread = None  # client thread to receive weights
        self.loop = None  # ucx server event loop

        self.server_finish_event = threading.Event()  # server侧的handler通知其他线程参数传输完成
        self.enter_ready = asyncio.Event()  # server侧的handler等待model参数准备好
        # relay server 参数更新期间的互斥区，等正在读的client读完，新来的client等着
        self.relay_server_update_lock = WeightUpdateRWLock()
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

    @property
    def has_setup(self):
        return self._setup_completed.is_set()

    @property
    def is_relay(self):
        return self.server_up and self.standalone

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
                elif tensor_key == "rank_start":
                    # 在relay即将更新的期间，等待其更新完成再开始
                    ack = 'ok'
                    if self.standalone:
                        # 防止死锁：当relay server正在更新权重时（写锁状态），
                        # 新启动的elastic client可能触发以下死锁场景：
                        # 1. relay server获取写锁，开始更新权重
                        # 2. driver调用stop_server_before_update，等待所有worker停止
                        # 3. 新elastic client启动，调用update_standalone_worker到达此处
                        # 4. 该worker因等待读锁而无法响应stop_server_before_update
                        # 5. 形成循环等待：driver等worker停止，worker等锁释放
                        # 解决方案：检测到写锁时直接跳过此次更新，避免死锁
                        if self.relay_server_update_lock.resource_owner == 'write':
                            ack = 'skip'
                        elif not self._relay_weights_loaded.is_set():
                            # 如果relay尚未进行过第一次weights update，也要让elastic standalone skip掉这次update
                            # 不然就会读到空的weights tensor
                            ack = 'skip'
                        else:
                            await self.relay_server_update_lock.acquire_read(ep)
                    await ep.send_obj(ack.encode("utf-8"))
                elif tensor_key == "rank_end":
                    if self.standalone:
                        await self.relay_server_update_lock.release_read(ep)
                    await ep.send_obj(f"r{self.rank}".encode("utf-8"))
                elif tensor_key == "group_end":
                    # 防止时序竞争：确保在设置server_finish_event前等待enter_ready
                    # 问题场景：
                    # 1. hybrid rollout调用update_standalone_worker，但尚未执行到enter_ready
                    # 2. standalone完成传输并广播group_end到达此处
                    # 3. 如果此时直接设置server_finish_event，后续进入的update_standalone_worker
                    #    会清除server_finish_event，导致等待该事件的代码永久阻塞
                    # 解决方案：等待enter_ready确保所有worker都已准备就绪再设置完成事件
                    logger.debug("ucx server received 'group_end'")
                    await self.enter_ready.wait()
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
                            # deref: 被dlpack转移之后的tensor，需要这样手动assign None才不会造成内存泄漏
                            weight = None
                            del send_buf
                        cp.get_default_memory_pool().free_all_blocks()
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
                        # deref: 被dlpack转移之后的tensor，需要这样手动assign None才不会造成内存泄漏
                        weight_tensor = None  # deref
                        del send_buf

                    cp.get_default_memory_pool().free_all_blocks()
            except (UCXConnectionResetError, UCXCanceledError) as e:
                if self.standalone:
                    await self.relay_server_update_lock.release_read(ep)
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
                if self.standalone:
                    # 对于relay server来说，因为不需要同步调用update_standalone_worker，因此一开始就把server设于enter状态
                    self.enter_ready.set()
                break
        assert self.server_up, "server cannot started"
        while True:
            await asyncio.sleep(0.1)

    def _start_server(self, port_queue: Queue):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # monitor asyncio tasks, use `telnet 127.0.0.1 21000` to connect to the monitor
        base_port = 21000
        if self.standalone:
            base_port = 22000
        if self.enable_aiomonitor:
            print(f"aiomonitor started on rank={self.rank}, "
                  f"use `telnet 127.0.0.1 {base_port + self.rank}` to connect to the monitor")
        Monitor = get_aiomonitor_cls(self.enable_aiomonitor)
        with Monitor(self.loop, termui_port=base_port + self.rank, console_enabled=False):
            self.loop.run_until_complete(self._server(port_queue))
        self.loop.close()

    def setup_as_server(self, ifname=None) -> str:
        ifname = os.environ.get('UCXX_IFNAME', ifname)
        ip = ucxx.get_address(ifname=ifname)
        assert ip is not None and ip != "", f"expecting ucxx.get_address return non-empty, got {ip}"
        port_queue = Queue()

        threading.Thread(target=self._start_server, args=(port_queue,), daemon=True, name="ucx-server").start()

        port = port_queue.get(block=True, timeout=300)

        if ":" in ip:
            ip = "[" + ip + "]"
        self.address = f"{ip}:{port}"
        return self.address

    def on_will_start_update(self):
        # relay server 要更新前，提前获取写锁，避免elastic client这时候来获取到了上一个版本的旧参数
        if self.is_relay:
            # 注意这个锁不是加在transfer_weights()调用前后，因为relay server只作为server，不会调用transfer_weights
            # 实际的transfer_weights由另一个weights_communicator调用，他们共享的是同一份underlying weights
            # 因此这个写锁在hook方法里调用
            asyncio.run_coroutine_threadsafe(self.relay_server_update_lock.acquire_update(), self.loop).result()

    def on_updated(self):
        # relay server 参数更新完，在等的client可以开始传输了
        if self.is_relay:
            self._relay_weights_loaded.set()
            self.relay_server_update_lock.release_update()

    def update_standalone_worker(self, role):
        source_addr = self.source_address  # noqa: py-spy

        # hybrid rollout才会走到这里
        # standalone rollout relay server总是up状态，这里跳过
        if self.server_up and not self.is_relay:
            self.server_finish_event.clear()
            self.enter_ready.set()  # notify client to read weights
            self.server_finish_event.wait()  # wait for all clients finish reading
            self.enter_ready.clear()  # clear for next turn
            self.inference_engine.current_steps = 0
            return

        async def await_start():
            ep = await self.connection_pool.get_connection(self.source_address)
            await ep.send_obj('rank_start'.encode('utf-8'))
            ack = await ep.recv_obj()
            return ep, ack

        async def await_end(ep):
            await ep.send_obj('rank_end'.encode('utf-8'))
            ack = await ep.recv_obj()
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

        print(f"start transfer_weights {self.standalone=} {self.is_relay=}")

        def client_run(exc_queue):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # monitor asyncio tasks, use `telnet 127.0.0.1 <port>` to connect to the monitor
            port = get_free_port_v4()
            if self.enable_aiomonitor:
                print(f"aiomonitor of client_transfer_weights started on rank={self.rank}, "
                      f"use `telnet 127.0.0.1 {port}` to connect to the monitor")
            Monitor = get_aiomonitor_cls(self.enable_aiomonitor)
            with Monitor(loop, termui_port=port, console_enabled=False):
                try:
                    # step 1: 通知server，client即将拉取参数，server若还没准备好，可以在这个时候先处理好了再返回
                    #   如果server是hybrid rollout：则等待参数就绪
                    #   如果server时relay server： 则等relay自己拉完参数后再返回
                    ep, ack = loop.run_until_complete(await_start())
                    if ack == 'skip':
                        print('client skip this weight transfer, will be updated soon')
                        e = WeightsUpdatingInterrupt()
                        tb = traceback.format_exc()
                        exc_queue.put((e, tb))
                        return
                    else:
                        # 没有relay server没有通知需要skip的话则继续运行，拉取参数
                        exc_queue.put((None, None))

                    # step 2: 真正开始拉取参数
                    loop.run_until_complete(transfer_weights())
                    loop.run_until_complete(await_end(ep))
                except Exception as e:
                    tb = traceback.format_exc()
                    exc_queue.put((e, tb))

                # receive完weights后立即回调，通知此rank，不管整个worker group的状态
                self.on_updated()

            loop.close()

        exc_queue = Queue()
        self.client_thread = threading.Thread(target=client_run, args=(exc_queue,), daemon=True)
        self.client_thread.start()

        # 如果是elastic client则只需关注是否已经开始传输，过exc_queue提前知道传输状态，异常原路抛出
        # 如果是stable client(as relay)，会等自己的传输完成才退出
        # 非elastic的server mode下，standalone rollout也会被setup as relay，来保证这里不会提前退出(虽然relay没有被用到)
        if self.standalone and not self.is_relay:
            exc, tb = exc_queue.get()
            if exc is not None:
                print(tb)
                raise exc
        else:
            # wait for transfer finish before returning. ensure the integrity of model weights
            self.client_thread.join()
            self.client_thread = None

        # 把queue里接下来还有的exc也抛出了
        while not exc_queue.empty():
            exc, tb = exc_queue.get()
            if exc is not None:
                print(tb)
                raise exc

        log_gpu_memory_usage(f'After {role} update')
        print("client finished")

    def update_standalone_worker_wait(self):
        t = self.client_thread
        self.client_thread = None
        if t is not None:
            t.join()

    def update_standalone_worker_end(self, addresses: List[str]):

        async def send_group_end_signal(addr: str, ucx_connection_concurrency):
            async with ucx_connection_concurrency:
                ep = await self.connection_pool.create(addr)
                try:
                    group_end_msg = "group_end"
                    await ep.send_obj(group_end_msg.encode("utf-8"))
                    # server需要配合回复一个消息，并在这里接收，不然server可能根本收不到上面发的数据，不知道为什么
                    ok = await ep.recv_obj()
                finally:
                    ep.close()

        async def broadcast_group_end_signal():
            # 单个client或server创建太多连接会被直接rst，这里控一下数量
            ucx_connection_concurrency = asyncio.Semaphore(32)
            print('will broadcast group_end signal to all trainer actors')
            tasks = [asyncio.create_task(send_group_end_signal(addr, ucx_connection_concurrency)) for addr in addresses]
            await asyncio.gather(*tasks)

        def broadcast_group_end_signal_thread():
            return asyncio.run(broadcast_group_end_signal())

        t = threading.Thread(target=broadcast_group_end_signal_thread, daemon=True)
        t.start()
        t.join()
