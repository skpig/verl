import subprocess
import threading
import time
from collections import deque
from typing import List, Tuple, Deque
import ray

from alpha_seed.utils.profile.timeline import Tracer, export_chrome_trace
from alpha_seed.workers.streaming_service.rollout_request_manager import get_all_request_manager_actors


class RolloutQueryTimeline:

    def __init__(self, config):
        self.config = config
        self.save_path = None
        # 记录最近每个step的时间 (step, ts)
        self.step_timestamp: Deque[Tuple[int, float]] = deque(maxlen=3 * self.config.dump_every_n_steps)
        self.step_timestamp.append((0, 0.))  # 初始从最开始

    def step(self, global_step):
        self.step_timestamp.append((global_step, time.time()))
        if global_step % self.config.dump_every_n_steps == 0:
            # trigger dump
            start_step, start_time = self.step_timestamp[0]
            threading.Thread(target=self._dump,
                             args=(start_time, start_step, global_step),
                             name="query-trace-dump",
                             daemon=True).start()

    def _dump(self, after_ts: float, start_step: int, step: int):
        spans: List[dict] = Tracer.merge_all(after_ts=after_ts)

        for req_mgr in get_all_request_manager_actors():
            request_spans = ray.get(req_mgr.dump_request_trace.remote(after_ts))
            spans.extend(request_spans)

        # step是即将开始的next step，所以之类-1表示结束到上一个step为止
        self.save_path = export_chrome_trace(f'query_trace_step{start_step}-{step - 1}.json.gz', spans)
        # delay upload to make sure file be flushed properly and visible to subprocess call
        time.sleep(1)
        ret = subprocess.call([f"/opt/tiger/mlx_deploy/bin/mlx asset upload {self.save_path}"], shell=True)
        if ret != 0:
            print("upload profiler trace fail. please see the log around")
