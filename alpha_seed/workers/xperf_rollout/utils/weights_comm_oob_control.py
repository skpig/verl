import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from alpha_seed.workers.streaming_service.streaming_utils import get_node_ip, get_free_port

logger = logging.getLogger(__file__)


@dataclass
class DiagnosisCounter:
    connection_accepted: int = 0  # 接受一个新的连接endpoint
    weights_update_count: int = 0  # 总共传输的weights次数(await start即算1次)


class WeightCommOOBControl:

    def __init__(self, restart_server_fn: Callable[[str], str]):
        self.restart_server_fn = restart_server_fn
        self.app = FastAPI()
        self.setup_routes()
        self.enter_ready = asyncio.Event()  # server侧的handler通知其他线程参数传输完成
        self.server_finish_event = threading.Event()  # server侧的handler等待model参数准备好
        self.diagnosis_counter = DiagnosisCounter()

        # uvicorn server
        self.ctrl_server_started = threading.Event()
        self.server = None
        self.server_task = None
        self.host = None
        self.port = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="oob-sync-worker")

    def setup_routes(self):

        @self.app.get("/group_end")
        async def group_end():
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
            return JSONResponse({})

        @self.app.post("/restart_ucx_server")
        async def restart_ucx_server(req: Request):
            req_body = await req.json()
            current_ucx_address = req_body.get("ucx_address", "")
            # 触发ucx server重启，注意可能要花数秒时间，之后再改成async，现在同步也够用
            loop = asyncio.get_event_loop()
            addr = await loop.run_in_executor(self.executor, self.restart_server_fn, current_ucx_address)
            if addr is None:
                # 已经重启过无需操作，或者重启失败
                return JSONResponse({"status": "skip"})
            else:
                return JSONResponse({"status": "ok", "address": addr})

    def serve_until_finish(self):
        self.server_finish_event.clear()
        self.enter_ready.set()  # notify client to read weights
        self.server_finish_event.wait()  # wait for all clients finish reading
        self.enter_ready.clear()  # clear for next turn

    async def run_server_forever(self):
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
                self.ctrl_server_started.set()
                await asyncio.Future()
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

    def get_address(self):
        ip = self.host
        if ':' in ip:
            ip = f"[{ip}]"
        return f"{ip}:{self.port}"

    def incr_ep(self):
        self.diagnosis_counter.connection_accepted += 1

    def incr_update(self):
        self.diagnosis_counter.weights_update_count += 1

    def get_server_stats_summary(self) -> str:
        return (f"server has accepted {self.diagnosis_counter.connection_accepted} connections "
                f"and finished {self.diagnosis_counter.weights_update_count}x updates")


class WeightsCommOOBControlClient:

    async def group_end(self, addr: str):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'http://{addr}/group_end') as resp:
                await resp.json()

    async def restart_ucx_server(self, addr: str, ucx_address: str):
        body = {"ucx_address": ucx_address}
        timeout = aiohttp.ClientTimeout(total=600)  # should be enough to restart the server
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f'http://{addr}/restart_ucx_server', json=body) as resp:
                j_resp = await resp.json()
                return j_resp.get('address')