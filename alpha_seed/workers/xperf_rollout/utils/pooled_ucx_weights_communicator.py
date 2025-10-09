import copy
import math
import os
import time
import traceback
import random
from typing import Any, Dict, List, Optional, Tuple
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
import torch.utils.dlpack
from aiohttp import ClientConnectorError
from ucxx.exceptions import UCXConnectionResetError, UCXCanceledError, UCXEndpointTimeoutError, UCXUnreachableError

from alpha_seed.utils.debug.aiomonitor import get_aiomonitor_cls
from alpha_seed.workers.xperf_rollout.utils.base_weights_communicator import WeightsCommunicator, WeightsRankInfo
from alpha_seed.workers.xperf_rollout.utils.weights_comm_oob_control import WeightCommOOBControl, WeightsCommOOBControlClient
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


class ConnectionCannotBeCreated(Exception):

    def __init__(self, server_addr: str, remote_info: WeightsRankInfo):
        self.server_addr = server_addr
        self.remote_info = remote_info
        super().__init__()

    def __str__(self):
        remote_info = self.remote_info
        return (f"Unable to connect to {remote_info.ucx_address}. "
                f"name={remote_info.worker_name}, host={remote_info.ip}, dp={remote_info.dp_rank} "
                f"tp={remote_info.tp_rank}, rank={remote_info.rank}")

    def __reduce__(self):
        return self.__class__, (self.server_addr, self.remote_info)


class ConnectionPool:

    def __init__(self):
        self.pool = defaultdict(list)  # address -> [ep]
        self.in_use = {}  # ep -> address
        self._server_info: Dict[str, WeightsRankInfo] = {}  # address -> WeightsRankInfo
        self._blacklist = {}  # address -> recover_at, 连不上的endpoint会进这里，每多banned一次banned的时间翻倍
        self._blacklist_count = defaultdict(int)  # address -> 被banned了多少次
        self._mutex = asyncio.Lock()
        self._blacklist_duration_base = 1800  # unit: second

    async def create(self, address):
        ip_port = address.rsplit(":", 1)
        assert len(ip_port) == 2, f"expecting ip_port to be a list of 2 strings, got {ip_port}"
        ip = ip_port[0]
        ip = ip.removeprefix("[")
        ip = ip.removesuffix("]")
        port = int(ip_port[1])
        ep = None
        connect_one_ep_max_attempts = 3  # 不用尝试太多次，连不上可以尽快fail-over到另一个地址
        for attempt in range(connect_one_ep_max_attempts):
            try:
                ep = await ucxx.create_endpoint(ip, port)
            except UCXUnreachableError as e:
                # 这个ip:port一定不可用，直接返回重新换一个
                return None
            except Exception as e:
                # 这个e是None，还不太好判断如何识别一定连不上的错误
                logger.warning(f"Failed to create endpoint, retry for the {attempt + 1} time. "
                               f"{type(e)=} {e=}, {ip=}, {port=}")
                await asyncio.sleep(1)
            else:
                logger.info(f"Successfully created endpoint on {ip}:{port} after {attempt} attempts")
                break

        return ep

    async def get_connection(self, server_info: List[WeightsRankInfo]):
        """
        从给定的address列表里随机找一个，如果已经有可用的连接，优先返回一个可用的
        """
        connect_max_retries = 5
        now = time.time()
        servers = {s.ucx_address: s for s in server_info}
        self._server_info.update(servers)
        addresses = set(servers.keys())

        ep = None
        most_recent_address = None
        tried_addresses = defaultdict(int)  # str(addr) ->

        for attempt in range(connect_max_retries):
            # 这里先判断pool里和给定的addresses列表里如果有可用的addr，则直接从可用的部分选一个，避免建立过多连接
            # 否则直接从传入的addresses里选一个
            already_connected_addresses = set(self.pool.keys())
            blacklist = set(addr for addr, recover_at in self._blacklist.items() if now < recover_at)
            available_addresses = (already_connected_addresses & addresses - blacklist) or (addresses - blacklist)
            if not available_addresses:
                raise RuntimeError(f"No available addresses after {attempt} attempts. "
                                   f"connected in pool: {already_connected_addresses}, "
                                   f"candidate addresses: {addresses}, blacklist: {blacklist}")
            address = random.choice(list(available_addresses))
            most_recent_address = address
            tried_addresses[address] += 1

            # 如果连接池里已经有这个地址的连接的话则直接用
            ep = None
            async with self._mutex:
                if self.pool[address]:
                    # Get a free connection from the pool
                    ep = self.pool[address].pop()

            # pool里没有可用连接则新建一个
            if ep is None:
                logger.warning(f"creating new connection to {address}. "
                               f"candidate addresses({addresses}), available({available_addresses})")
                ep = await self.create(address)
            if ep is None:
                self.ban(address)
                continue

            # 所有拿到的连接都要测试一下ping
            remote_info = self.get_remote_info_by_address(address)
            ping_error = await self.ping(ep)
            if ping_error is not None:
                logger.warning(f"Endpoint connected but failed to ping {address}: {ping_error}, "
                               f"remote info: {remote_info}")
                self.ban(address)
                continue

            # 成功连接且ping通，才会保存到in use
            async with self._mutex:
                self.in_use[ep] = address
            if attempt >= 1:
                logger.warning(f"successfully connected to {address} after {attempt} major attempts "
                               f"on these addresses: {dict(tried_addresses)}. "
                               f"remote info: {remote_info}. blacklist summary: {self._blacklist_count}")
            break

        # 换了多个地址都没建立成功的话，则抛出
        if ep is None:
            raise ConnectionCannotBeCreated(most_recent_address, servers[most_recent_address])

        return ep

    async def put_connection(self, ep):
        async with self._mutex:
            if ep in self.in_use:
                address = self.in_use.pop(ep)
                self.pool[address].append(ep)

    def get_remote_info_by_address(self, addr):
        info: WeightsRankInfo = self._server_info.get(addr)
        if info is None:
            return f"no remote info because cannot find WeightsRankInfo from {addr}"
        return f"name={info.worker_name} host={info.ip} dp={info.dp_rank} tp={info.tp_rank} rank={info.rank}"

    def get_remote_info(self, ep):
        if ep is None:
            return "no remote info because ep is None"
        addr = self.in_use.get(ep)
        if addr is None:
            return f"no remote info because cannot find corresponding address of this {ep=}"
        return self.get_remote_info_by_address(addr)

    async def dispose(self, ep):
        """
        传输过程中遇到错误则丢弃这个连接
        """
        async with self._mutex:
            self.in_use.pop(ep, None)
        try:
            await ep.close()
        except:
            pass

    async def ping(self, ep):
        try:
            await ep.send_obj("ping".encode())
            pong = await ep.recv_obj()
            return None
        except Exception as e:
            return e

    def ban(self, address: str):
        self._blacklist_count[address] += 1
        banned_count = self._blacklist_count[address]
        recover_at = time.time() + banned_count * self._blacklist_duration_base
        self._blacklist[address] = recover_at
        remote_info = self.get_remote_info_by_address(address)
        logger.warning(f"Banned {address} the {banned_count}th times. will be recovered at {recover_at:.1f}. "
                       f"remote info: {remote_info}")

    def get_blacklist_summary(self):
        return dict(self._blacklist_count)

    def get_possible_dead_servers(self) -> List[WeightsRankInfo]:
        ret = []
        blacklist_addresses = list(self._blacklist_count.keys())
        for addr in blacklist_addresses:
            count = self._blacklist_count[addr]
            if count > 0:
                info = self._server_info.get(addr)
                if info is not None:
                    ret.append(info)
                    self._blacklist[addr] = time.time() + 8640000  # 100天后
        return ret


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
                logger.debug(f'read _resource_lock.acquire(), cost={t1 - t0:.3f}s')

    async def release_read(self, ep):
        async with self._read_lock:
            self._readers.pop(ep, None)
            if len(self._readers) == 0:
                if self._resource_owner != 'read':
                    return
                self._resource_lock.release()
                self._resource_owner = ''
                logger.debug('read _resource_lock.release()')

    async def acquire_update(self):
        self._update_complete.clear()
        t0 = time.time()
        await self._resource_lock.acquire()
        t1 = time.time()
        self._resource_owner = 'write'
        logger.debug(f'update _resource_lock.acquire(), cost={t1 - t0:.3f}s')

    def release_update(self):
        self._resource_lock.release()
        self._resource_owner = ''
        self._update_complete.set()
        logger.debug('update _resource_lock.release()')


async def send_tensor(ep, src: torch.Tensor, send_buf: cp.ndarray):
    dlpack = torch.utils.dlpack.to_dlpack(src.cuda())
    src = cp.from_dlpack(dlpack).reshape(-1).view(cp.uint8)
    total_bytes = src.nbytes
    chunk_size = send_buf.nbytes
    num_chunks = math.ceil(total_bytes / chunk_size)
    await ep.send_obj(f"{total_bytes}".encode())

    for i in range(num_chunks):
        chunk_len = min(chunk_size, total_bytes - i * chunk_size)
        send_buf[:chunk_len] = src[i * chunk_size:i * chunk_size + chunk_len]
        cp.cuda.get_current_stream().synchronize()  # 确保 copy 完成
        await ep.send(send_buf[:chunk_len])

    # deref: 被dlpack转移之后的tensor，需要这样手动assign None才不会造成内存泄漏
    dlpack = None


async def recv_tensor(ep, dst: torch.Tensor, recv_buf: cp.ndarray) -> torch.Tensor:
    total_bytes = await ep.recv_obj()  # 先收长度
    total_bytes = int(total_bytes.decode())
    assert dst.numel() * dst.element_size() >= total_bytes, \
        f"dst tensor too small: {dst.numel()} * {dst.element_size()} < {total_bytes}"

    chunk_size = recv_buf.nbytes
    num_chunks = math.ceil(total_bytes / chunk_size)
    offset = 0

    for _ in range(num_chunks):
        await ep.recv(recv_buf)

        chunk_len = min(chunk_size, total_bytes - offset)
        # 从 recv_buf（cupy array）切片，转成 dlpack 给 torch tensor 对应区域
        cupy_chunk = recv_buf[:chunk_len]
        # 构造 torch tensor 对应的子 tensor（1D byte view）
        torch_chunk = dst.reshape(-1).view(torch.uint8).narrow(0, offset, chunk_len)
        # 利用 DLPack 进行显存间拷贝
        dlpack = cupy_chunk.toDlpack()
        tmp = torch.utils.dlpack.from_dlpack(dlpack)
        torch_chunk.copy_(tmp)
        tmp = None
        offset += chunk_len

    return dst


async def send_tensor_multi(ep, src_list: List[torch.Tensor], send_buf: cp.ndarray):
    total_bytes = sum(t.numel() * t.element_size() for t in src_list)
    await ep.send_obj(str(total_bytes).encode())

    chunk_size = send_buf.nbytes
    offset_global = 0  # 全局发送字节计数
    send_pos = 0  # send_buf 写入偏移

    # 遍历所有tensor
    for t in src_list:
        t = t.cuda()
        t_bytes = t.numel() * t.element_size()
        t_view = t.view(torch.uint8).reshape(-1)  # 转为字节流，1D tensor

        offset_in_tensor = 0  # 当前tensor已发送字节

        while offset_in_tensor < t_bytes:
            space_in_buf = chunk_size - send_pos  # send_buf剩余空间
            send_len = min(space_in_buf, t_bytes - offset_in_tensor)

            # 拷贝torch tensor切片到send_buf对应位置
            dlpack = torch.utils.dlpack.to_dlpack(t_view[offset_in_tensor:offset_in_tensor + send_len])
            cupy_chunk = cp.from_dlpack(dlpack).view(cp.uint8)

            send_buf[send_pos:send_pos + send_len] = cupy_chunk

            send_pos += send_len
            offset_in_tensor += send_len
            offset_global += send_len

            # send_buf满了，发送出去，send_pos重置
            if send_pos == chunk_size:
                cp.cuda.get_current_stream().synchronize()
                await ep.send(send_buf)
                send_pos = 0

    # 发送最后一块（如果没满也要发）
    if send_pos > 0:
        cp.cuda.get_current_stream().synchronize()
        await ep.send(send_buf[:send_pos])


async def recv_tensor_multi(ep, dst_list: List[torch.Tensor], recv_buf: cp.ndarray) -> List[torch.Tensor]:
    total_bytes = await ep.recv_obj()
    total_bytes = int(total_bytes.decode())

    chunk_size = recv_buf.nbytes
    num_chunks = math.ceil(total_bytes / chunk_size)

    # 计算每个 tensor 字节范围的区间，用于定位写入
    tensor_offsets = []
    offset_accum = 0
    for t in dst_list:
        nbytes = t.numel() * t.element_size()
        tensor_offsets.append((offset_accum, offset_accum + nbytes, t))
        offset_accum += nbytes

    offset_global = 0  # 接收数据的全局偏移

    for _ in range(num_chunks):
        # 接收前sync一下，保证recv_buf中的数据已经被读取完，这个操作暂时假设很快，因为有阻塞asyncio的风险
        torch.cuda.synchronize()
        await ep.recv(recv_buf)
        chunk_len = min(chunk_size, total_bytes - offset_global)
        cupy_chunk = recv_buf[:chunk_len]

        # 当前chunk在整体数据中的范围
        chunk_start = offset_global
        chunk_end = offset_global + chunk_len

        for (t_start, t_end, t) in tensor_offsets:
            # chunk与tensor的重叠区间 [overlap_start, overlap_end)
            overlap_start = max(chunk_start, t_start)
            overlap_end = min(chunk_end, t_end)

            if overlap_start < overlap_end:
                # 计算本次重叠区间长度
                length = overlap_end - overlap_start

                # chunk内部对应偏移
                chunk_offset = overlap_start - chunk_start

                # tensor内部对应偏移（字节）
                tensor_offset = overlap_start - t_start

                # cupy_chunk[chunk_offset:chunk_offset+length] -> t[tensor_offset:tensor_offset+length]
                cupy_sub = cupy_chunk[chunk_offset:chunk_offset + length]
                torch_sub = t.reshape(-1).view(torch.uint8).narrow(0, tensor_offset, length)

                dlpack = cupy_sub.toDlpack()
                tmp = torch.utils.dlpack.from_dlpack(dlpack)
                torch_sub.copy_(tmp, non_blocking=True)
                tmp = None

        offset_global += chunk_len

    torch.cuda.synchronize()
    return dst_list


class UCXWeightsCommunicator(WeightsCommunicator):

    def __init__(self, inference_engine, standalone: bool, device_mesh, enable_aiomonitor: bool = False):
        self.inference_engine = inference_engine
        self.standalone = standalone
        self.device_mesh = device_mesh  # 注意不开tp时这个是None
        self.enable_aiomonitor = enable_aiomonitor

        self.source_info: List[WeightsRankInfo] = []  # 作为client时，默认要连到server的连接信息，使用时可随机选一个
        self.server_up = False  # 是否作为server启动
        self.server_task = None  # ucx server run起来的task
        self.ucxx_ip = ""  # ucx server的ip地址，setup_as_server后就不会变了
        self.ucx_address = ""  # ucx server当前的address ip:port格式
        self.ucx_address_lock = threading.Lock()
        self.port_queue = Queue()  # 每次启动(重启)server，新的port启动好后会塞进这里，从这里get出来一次
        self._setup_completed = threading.Event()  # 已完成endpoint setup
        self._relay_weights_loaded = threading.Event()  # 已经进行过一次weights同步，根据此判断是否可以接受elastic client过来拉
        self._client_weights_loaded = threading.Event()  # 已经进行过一次weights同步，client根据此判断是否允许在某一次weights update过程中中断
        self.client_thread = None  # client thread to receive weights
        self.loop = None  # ucx server event loop

        self.oob_control = WeightCommOOBControl(self._restart_server)
        self.oob_client = WeightsCommOOBControlClient()
        self._update_server_info_thread = None  # oob用于控制remote server重启的thread
        # relay server 参数更新期间的互斥区，等正在读的client读完，新来的client等着
        self.relay_server_update_lock = WeightUpdateRWLock()

        # send/recv buffer
        self.send_buffer_sema = asyncio.Semaphore(1)  # 由于现在用静态buffer，先用单并发测试
        self.buffer_chunk_size = 256 * 2**20  # ?MiB
        self.send_buffer = None
        self.recv_buffer = None

        # 作为actor server时，nccl world里的rank，用来做序号标记debug用
        self.rank = 0
        if dist.is_initialized():
            self.rank = dist.get_rank()

        # ucx endpoint接池类
        self.connection_pool = ConnectionPool()

    def setup_as_client(self, role, source_endpoint_info: List[WeightsRankInfo], init_recv_buffer: bool = True):
        assert source_endpoint_info is not None and len(source_endpoint_info) > 0 and isinstance(
            source_endpoint_info[0], WeightsRankInfo), f"{source_endpoint_info=}"
        self.source_info = source_endpoint_info

        if init_recv_buffer:
            self.recv_buffer = cp.empty(self.buffer_chunk_size, dtype=cp.uint8)

        # 这个本身没什么用，就是测一下是否通而已
        async def register(client_role):
            ep = await self.connection_pool.get_connection(self.source_info)
            try:
                register_msg = f"register:{client_role}"
                await ep.send_obj(register_msg.encode("utf-8"))
                ok = await ep.recv_obj()
            finally:
                await self.connection_pool.put_connection(ep)

        def register_in_thread(my_role):
            asyncio.run(register(my_role))
            return True

        register_thread = threading.Thread(target=register_in_thread, args=(role,))
        register_thread.start()
        register_thread.join()
        logger.info("register finished")
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
        self.oob_control.incr_ep()
        while True:
            try:
                raw_tensor_key = await ep.recv_obj()
                if raw_tensor_key is None:
                    continue
                tensor_key = raw_tensor_key.decode("utf-8")

                if tensor_key.startswith("register:"):
                    logger.info(f"tensor_key = {tensor_key} received")
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
                    self.oob_control.incr_update()
                    await ep.send_obj(ack.encode("utf-8"))
                elif tensor_key == "rank_end":
                    if self.standalone:
                        await self.relay_server_update_lock.release_read(ep)
                    await ep.send_obj(f"r{self.rank}".encode("utf-8"))
                elif tensor_key == "ping":
                    await ep.send_obj("pong".encode("utf-8"))
                elif tensor_key == "pull_weights":
                    # 等待actor sharding manager把xperf engine fsdp/megatron的参数准备好
                    await self.oob_control.enter_ready.wait()
                    weight_list = [
                        self.inference_engine.engine.module.layernorm_weight,
                        self.inference_engine.engine.module.lm_head_weight,
                        self.inference_engine.engine.module.wte_weight,
                    ]
                    layers_weight = self.inference_engine.engine.module.layers_weight
                    for layer_weight in layers_weight:
                        weights = [w for w in layer_weight if isinstance(w, torch.Tensor)]
                        weight_list.extend(weights)

                    async with self.send_buffer_sema:
                        await send_tensor_multi(ep, weight_list, self.send_buffer)

                elif tensor_key.startswith("layers_weight."):
                    # unused code block, keep for a moment
                    # 等待actor sharding manager把xperf engine fsdp/megatron的参数准备好
                    await self.oob_control.enter_ready.wait()
                    layers_weight = self.inference_engine.engine.module.layers_weight
                    tensor_key = tensor_key.removeprefix("layers_weight.")
                    indices = tensor_key.split("-")
                    i = int(indices[0])
                    j = int(indices[1])
                    weight = layers_weight[i][j]

                    async with self.send_buffer_sema:
                        await send_tensor(ep, weight, self.send_buffer)
                else:
                    # unused code block, keep for a moment
                    # 注意不要乱发不知道的key，会处理不了
                    await self.oob_control.enter_ready.wait()
                    weight_tensor = getattr(self.inference_engine.engine.module, tensor_key)

                    async with self.send_buffer_sema:
                        await send_tensor(ep, weight_tensor, self.send_buffer)
            except (UCXConnectionResetError, UCXCanceledError, UCXEndpointTimeoutError) as e:
                # ignore client side error and break the dead loop
                #   UCXEndpointTimeoutError: any potential hand in client side
                #   (others): client crashed
                if self.standalone:
                    await self.relay_server_update_lock.release_read(ep)
                break
            except Exception as e:
                traceback.print_exc()
                logger.warning(f"error found on server, but continuing event handle loop: {e}")

    async def _server(self, port_queue: Queue):
        for try_count in range(50):
            try:
                await asyncio.sleep(random.random() * 0.2)  # reduce port conflict probabilities in the same host
                port = get_free_port()
                logger.warning(f"try to listen on port {port}")
                lf = ucxx.create_listener(self._event_handler, port)
            except Exception as e:
                logger.warning(f"failed to create listener, retry for the {try_count + 1} time: {e=}")
                await asyncio.sleep(2)
            else:
                logger.warning(f"started listener on port {lf.port} with ip = {lf.ip}")
                port_queue.put(port)
                self.server_up = True
                if self.standalone:
                    # 对于relay server来说，因为不需要同步调用update_standalone_worker，因此一开始就把server设于enter状态
                    self.oob_control.enter_ready.set()
                break
        assert self.server_up, "server cannot start"
        # 注意这里不能用asyncio.Future()会导致server工作不正常，原因不明，要用while true asyncio.sleep
        while True:
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                lf.close()
                break

    def _start_server(self, port_queue: Queue):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        server_start_count = 0
        base_port = 21000
        if self.standalone:
            base_port = 22000
        if self.enable_aiomonitor:
            print(f"aiomonitor started on rank={self.rank}, "
                  f"use `telnet 127.0.0.1 {base_port + self.rank}` to connect to the monitor")
        Monitor = get_aiomonitor_cls(self.enable_aiomonitor)
        with Monitor(self.loop, termui_port=base_port + self.rank, console_enabled=False):
            while True:
                self.server_task = self.loop.create_task(self._server(port_queue))
                try:
                    self.loop.run_until_complete(self.server_task)
                except asyncio.CancelledError:
                    logger.warning("ucx server task cancelled, try to restart a new one")
                    time.sleep(0.5)
                    server_start_count += 1

    def _restart_server(self, current_ucx_address: str) -> Optional[str]:
        """
        重启并返回新的地址
        """
        with self.ucx_address_lock:
            if self.ucx_address != current_ucx_address:
                return None
            assert self.server_task is not None, "server has not been started"
            self.server_task.cancel()  # 触发重启
            summary = self.oob_control.get_server_stats_summary()
            logger.warning(f"restarting ucx server of {current_ucx_address}, waiting for it to be ready. "
                           f"previous {summary}")
            port = self.port_queue.get()  # 等待ucx server启动
            self.ucx_address = f"{self.ucxx_ip}:{port}"
            logger.warning(f"ucx server started on rank={self.rank} with new addr={self.ucx_address}")
            return self.ucx_address

    def _start_oob_server(self):
        # oob的loop和thread都跟ucx server独立开，排除干扰
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        base_port = 23000
        if self.standalone:
            base_port = 24000
        if self.enable_aiomonitor:
            print(f"aiomonitor started on rank={self.rank}, "
                  f"use `telnet 127.0.0.1 {base_port + self.rank}` to connect to the oob server monitor")
        Monitor = get_aiomonitor_cls(self.enable_aiomonitor)
        with Monitor(loop, termui_port=base_port + self.rank, console_enabled=False):
            loop.run_until_complete(self.oob_control.run_server_forever())
        loop.close()

    def setup_as_server(self, ifname=None) -> Tuple[str, str]:
        ifname = os.environ.get('UCXX_IFNAME', ifname)
        ip = ucxx.get_address(ifname=ifname)
        assert ip is not None and ip != "", f"expecting ucxx.get_address return non-empty, got {ip}"

        # init send buffer
        self.send_buffer = cp.empty(self.buffer_chunk_size, dtype=cp.uint8)

        threading.Thread(target=self._start_server, args=(self.port_queue,), daemon=True, name="ucx-server").start()
        threading.Thread(target=self._start_oob_server, daemon=True, name="ucx-oob-server").start()

        port = self.port_queue.get()  # 等待ucx server启动

        if ":" in ip:
            ip = "[" + ip + "]"
        self.ucxx_ip = ip
        self.ucx_address = f"{ip}:{port}"

        # wait for control server ready
        self.oob_control.ctrl_server_started.wait()
        return self.ucx_address, self.oob_control.get_address()

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
        else:
            self._client_weights_loaded.set()

    def update_standalone_worker(self, role):
        # hybrid rollout才会走到这里
        # standalone rollout relay server总是up状态，这里跳过
        if self.server_up and not self.is_relay:
            self.oob_control.serve_until_finish()
            self.inference_engine.current_steps = 0
            return

        async def await_start():
            ep = await self.connection_pool.get_connection(self.source_info)
            await ep.send_obj('rank_start'.encode('utf-8'))
            ack = await ep.recv_obj()
            return ep, ack

        async def await_end(ep):
            await ep.send_obj('rank_end'.encode('utf-8'))
            ack = await ep.recv_obj()
            return

        async def transfer_weights(ep):
            weight_list = [
                self.inference_engine.engine.module.layernorm_weight,
                self.inference_engine.engine.module.lm_head_weight,
                self.inference_engine.engine.module.wte_weight,
            ]
            layers_weight = self.inference_engine.engine.module.layers_weight
            for layer_weight in layers_weight:
                weights = [w for w in layer_weight if isinstance(w, torch.Tensor)]
                weight_list.extend(weights)
            await ep.send_obj(f"pull_weights".encode("utf-8"))
            await recv_tensor_multi(ep, weight_list, self.recv_buffer)

            self.inference_engine.current_steps = 0

        logger.info(f"start transfer_weights {self.standalone=} {self.is_relay=}")

        def client_run(exc_queue: Queue):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # monitor asyncio tasks, use `telnet 127.0.0.1 <port>` to connect to the monitor
            port = get_free_port_v4()

            transfer_max_retries = 5
            num_try = 0
            last_e, last_tb = None, None
            error_count = defaultdict(int)  # error_type ->
            success = False

            if self.enable_aiomonitor:
                print(f"aiomonitor of client_transfer_weights started on rank={self.rank}, "
                      f"use `telnet 127.0.0.1 {port}` to connect to the monitor")
            Monitor = get_aiomonitor_cls(self.enable_aiomonitor)
            with Monitor(loop, termui_port=port, console_enabled=False):
                # 在这一层做ep重试
                ep = None
                remote_info = None
                while True:
                    num_try += 1
                    if num_try > transfer_max_retries:
                        break
                    # 每一轮重试清空，以最后一轮的exc为准
                    with exc_queue.mutex:
                        exc_queue.queue.clear()
                    try:
                        # step 1: 通知server，client即将拉取参数，server若还没准备好，可以在这个时候先处理好了再返回
                        #   如果server是hybrid rollout：则等待参数就绪
                        #   如果server时relay server： 则等relay自己拉完参数后再返回
                        ep, ack = loop.run_until_complete(await_start())
                        remote_info = self.connection_pool.get_remote_info(ep)
                        if ack == 'skip':
                            # 如果这个client之前就已经完整load过参数，说明这次访问到了一个可能还在准备中的relay
                            # 放弃这次尝试，重新找另一个relay(不占用重试次数)
                            if self._client_weights_loaded.is_set():
                                num_try -= 1
                                loop.run_until_complete(self.connection_pool.dispose(ep))
                                continue

                            # 全新的client，则先跳过这次参数同步，等下一次全局同步
                            logger.warning('client skip this weight transfer, will be updated soon')
                            e = WeightsUpdatingInterrupt()
                            exc_queue.put((e, traceback.format_exc()))
                            return
                        else:
                            # 没有relay server没有通知需要skip的话则继续运行，拉取参数
                            exc_queue.put((None, None))

                        # step 2: 真正开始拉取参数
                        loop.run_until_complete(transfer_weights(ep))
                        loop.run_until_complete(await_end(ep))
                        success = True
                        break
                    except ConnectionCannotBeCreated as e:
                        # 如果连接没法建立，则直接重试另一个endpoint
                        last_e, last_tb = e, traceback.format_exc()
                        continue
                    except (UCXCanceledError, UCXConnectionResetError) as e:
                        # 传输过程中中断，则丢弃此连接重试
                        retries_remain = transfer_max_retries - num_try
                        logger.warning(
                            f"errors occurred during transfer: {e=}. will retry another {retries_remain} times."
                            f" remote info: {remote_info}")
                        last_e, last_tb = e, traceback.format_exc()
                        loop.run_until_complete(self.connection_pool.dispose(ep))
                        ep = None
                        continue
                    except Exception as e:
                        # 其他错误要传到外层抛出，不再重试
                        last_e, last_tb = e, traceback.format_exc()
                        break
                    finally:
                        if last_e is not None:
                            error_count[type(last_e).__name__] += 1
                        if ep is not None:
                            loop.run_until_complete(self.connection_pool.put_connection(ep))

                # receive完weights后立即回调，通知此rank，不管整个worker group的状态
                if success:
                    self.on_updated()
                    if num_try > 1:
                        logger.warning(f"finally complete transfer after {num_try} attempts, "
                                       f"from this remote: {remote_info}, recent errors: {dict(error_count)} "
                                       f"blacklist summary: {dict(self.connection_pool.get_blacklist_summary())} "
                                       f"{last_e=}\n{last_tb}")
                else:
                    if last_e is not None:
                        exc_queue.put((last_e, last_tb))
                    logger.warning(f"failed to complete transfer after {num_try} attempts, "
                                   f"recent errors: {dict(error_count)} "
                                   f"blacklist summary: {dict(self.connection_pool.get_blacklist_summary())}")

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
                logger.warning(tb)
                raise exc
        else:
            # wait for transfer finish before returning. ensure the integrity of model weights
            self.client_thread.join()
            self.client_thread = None

        # 把queue里接下来还有的exc也抛出了
        while not exc_queue.empty():
            exc, tb = exc_queue.get()
            if exc is not None:
                logger.warning(tb)
                raise exc

        log_gpu_memory_usage(f'After {role} update', logger)
        logger.info("client finished")

    def update_standalone_worker_wait(self):
        t = self.client_thread
        self.client_thread = None
        if t is not None:
            t.join()

    def update_standalone_worker_end(self, servers: List[WeightsRankInfo]):
        addresses = [s.oob_address for s in servers]

        async def broadcast_group_end_signal():
            logger.info('will broadcast group_end signal to all trainer actors')
            tasks = [asyncio.create_task(self.oob_client.group_end(addr)) for addr in addresses]
            await asyncio.gather(*tasks)

        def broadcast_group_end_signal_thread():
            return asyncio.run(broadcast_group_end_signal())

        t = threading.Thread(target=broadcast_group_end_signal_thread, daemon=True)
        t.start()
        t.join()

    # unused, but keep it for diagnosis purpose
    def restart_stuck_ucx_server(self):
        servers = self.connection_pool.get_possible_dead_servers()
        if not servers:
            return

        async def update_server_info(s: WeightsRankInfo):
            new_server_info = copy.copy(s)
            try:
                new_ucx_address = await self.oob_client.restart_ucx_server(s.oob_address, s.ucx_address)
                if new_ucx_address:
                    new_server_info.ucx_address = new_ucx_address
                    self.source_info.append(new_server_info)
                    logger.warning(f"ucx server restarted with new address: {new_ucx_address} in "
                                   f"remote {new_server_info.worker_name} dp={new_server_info.dp_rank} "
                                   f"tp={new_server_info.tp_rank}")
            except ClientConnectorError as e:
                # ignore client connection error if the client already dead
                pass
            except Exception as e:
                logger.warning(f"some errors {e=} during restarting")
                traceback.print_exc()
                # ignore errors, because remote server may already be dead
                pass

        async def parallel_run_update_server_info(servers):
            tasks = [asyncio.create_task(update_server_info(s)) for s in servers]
            return await asyncio.gather(*tasks)

        def parallel_run_update_server_info_thread(servers):
            return asyncio.run(parallel_run_update_server_info(servers))

        if self._update_server_info_thread is not None:
            self._update_server_info_thread.join()
        self._update_server_info_thread = threading.Thread(target=parallel_run_update_server_info_thread,
                                                           args=(servers,),
                                                           daemon=True)
        self._update_server_info_thread.start()
