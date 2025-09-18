import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict

from alpha_seed.workers.streaming_service.rollout_request import Request
from alpha_seed.workers.xperf_rollout.component.query import ProcessEventType


@dataclass
class FinishedEventStats:
    """
    统计在完成缓冲区里还没有被取走的events
    """
    waiting_count: int  # finished wait event数
    done_count: int  # finished done event数
    finished_size: int  # 某step完成数
    finished_staging_size: int  # 目前暂存区里完成的未被取走的数量
    finished_accumulated_size: int  # 历史记录区里的数量
    done_request_ids: List[str]


@dataclass
class RequestDigest:
    query_id: str
    pool_name: str  # 属于哪个request manager
    assigned: bool  # 是否被挑选出来staging阶段但还没指定具体某个engine
    assigned_engine_id: str
    assigned_engine_name: str
    global_step: int  # 从第几个global step提交的
    assigned_at: float
    updated_at: float
    input_length: int
    output_length: int
    aborted_count: int
    stale_count: int


@dataclass
class ProgressStat:
    pool_name: str
    step: int
    total: int
    finished: int
    token_throughput: float
    prefill_throughput: float
    running_queries: int
    pending_queries: int
    active_engines: int
    oldest_updated_time: float  # 最老的更新时间
    oldest_query_time: float  # 目前最老的query开始跑的时间戳
    latest_query_time: float  # 目前最新的query开始的时间戳


@dataclass
class RequestPoolInternalDiagnosis:
    out_of_order_count: int = 0  # update接收乱序计数

    def to_dict(self):
        return {
            'rollout/query/update_out_of_order_count': self.out_of_order_count,
        }


class FiniteDict:
    """
    有限大小的dict，有个FIFO队列记录key，所以key重复了可能会被提前pop掉
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self._map = {}
        self._queue = deque()  # 只存key，不要设定最大长度，手动判断pop
        self._mutex = threading.Lock()

    def add(self, key, data):
        with self._mutex:
            self._map[key] = data
            self._queue.append(key)
            if len(self._queue) > self.max_size:
                pop_key = self._queue.popleft()
                # add进来的key可能重复，所以这里pop要判None
                self._map.pop(pop_key, None)

    def get(self, key, default=None):
        return self._map.get(key, default)

    def __getitem__(self, key):
        return self._map[key]

    def __len__(self):
        return len(self._map)


class ReservoirSamples:

    def __init__(self, sample_size=512):
        self.sample_size = sample_size
        self.samples: List[float] = []
        self._sorted = False

        # 统计用的变量
        self.count = 0
        self.sum = 0.0
        self._min = float('inf')
        self._max = float('-inf')

    def add(self, values: list[float]):
        for v in values:
            # 更新基本统计
            self.count += 1
            self.sum += v
            self._min = min(v, self._min)
            self._max = max(v, self._max)

            # Reservoir Sampling采样逻辑
            if len(self.samples) < self.sample_size:
                self.samples.append(v)
            else:
                # 第count个数据，有sample_size/count的概率替换samples中随机位置的元素
                r = random.randint(0, self.count - 1)
                if r < self.sample_size:
                    self.samples[r] = v
        self._sorted = False

    def percentile(self, p: float):
        # p: 0.0 ~ 1.0
        if not self.samples:
            return 0
        if not self._sorted:
            self.samples = sorted(self.samples)
            self._sorted = True
        k = int(len(self.samples) * p)
        k = min(k, len(self.samples) - 1)
        return self.samples[k]

    def min(self):
        return self._min if self.count > 0 else 0

    def max(self):
        return self._max if self.count > 0 else 0

    def mean(self):
        return self.sum / self.count if self.count > 0 else 0


@dataclass
class StepStat:
    """
    统计每个step的
    """
    step: int  # 第几个global step
    pool_name: str  # 那个request manager

    # 记录所有query创建到末次从proxy dispatch的时间差(用来衡量proxy分发能力)
    shed_delay: ReservoirSamples = field(default_factory=lambda: ReservoirSamples())
    # 记录query进入engine后在waiting队列的等待时间(用来衡量engine内部调度和负载状况)
    wait_delay: ReservoirSamples = field(default_factory=lambda: ReservoirSamples())
    # 记录query创建来到开始prefill的时间差(用来表示query实际调度状况)
    run_delay: ReservoirSamples = field(default_factory=lambda: ReservoirSamples())


class RequestStatCollector:

    def __init__(self):
        self.steps: Dict[int, StepStat] = {}  # step ->
        self.mutex = threading.Lock()

    def finish(self, pool_name: str, req: Request):
        # shed delay
        shed_delays = []
        en_pool = req.last_pending_reschedule_ts
        last_recv = req.query.received_time
        shed_delays.append((last_recv - en_pool) / 1e3)
        for his in req.stale_histories:
            delay = (his.received_time - his.last_pending_reschedule_ts) / 1e3
            shed_delays.append(delay)

        # wait delay
        wait_delays = []
        for his in req.stale_histories:
            wait_delays.extend(self._accumulate_wait_delay_from_events_list(his.process_events))
        wait_delays.extend(self._accumulate_wait_delay_from_events_list(req.query.process_events))

        # run delay(unit: s)
        run_delays = []
        for his in req.stale_histories:
            # 按request pool到开始prefill的时差算
            delay = (his.first_scheduled_time - his.last_pending_reschedule_ts) / 1e3
            if delay > 2e-6:
                # delay 太短认为是没有实际开始prefill，忽略
                run_delays.append(delay)
        delay = (req.query.first_scheduled_time - req.last_pending_reschedule_ts) / 1e3
        run_delays.append(delay)

        with self.mutex:
            if req.global_step not in self.steps:
                self.steps[req.global_step] = StepStat(req.global_step, pool_name)
            self.steps[req.global_step].shed_delay.add(shed_delays)
            self.steps[req.global_step].wait_delay.add(wait_delays)
            self.steps[req.global_step].run_delay.add(run_delays)

    def _accumulate_wait_delay_from_events_list(self, events) -> list:
        wait_delays = []
        for i in range(1, len(events)):
            prev_event = events[i - 1]
            this_event = events[i]
            dur = (this_event.ts_ms - prev_event.ts_ms) / 1e3
            if this_event.event == ProcessEventType.PREFILL_START:
                # prefill start 之前在waiting queue里等了多久
                wait_delays.append(dur)
        return wait_delays

    def get_step_metrics(self, step: int) -> Dict[str, float]:
        if step not in self.steps:
            return {}
        stat = self.steps[step]
        return {
            "rollout/query/shed_delay_min": stat.shed_delay.min(),
            "rollout/query/shed_delay_mean": stat.shed_delay.mean(),
            "rollout/query/shed_delay_p95": stat.shed_delay.percentile(0.95),
            "rollout/query/shed_delay_p99": stat.shed_delay.percentile(0.99),
            "rollout/query/shed_delay_max": stat.shed_delay.max(),
            "rollout/query/wait_delay_min": stat.wait_delay.min(),
            "rollout/query/wait_delay_mean": stat.wait_delay.mean(),
            "rollout/query/wait_delay_p95": stat.wait_delay.percentile(0.95),
            "rollout/query/wait_delay_p99": stat.wait_delay.percentile(0.99),
            "rollout/query/wait_delay_max": stat.wait_delay.max(),
            "rollout/query/run_delay_min": stat.run_delay.min(),
            "rollout/query/run_delay_mean": stat.run_delay.mean(),
            "rollout/query/run_delay_p95": stat.run_delay.percentile(0.95),
            "rollout/query/run_delay_p99": stat.run_delay.percentile(0.99),
            "rollout/query/run_delay_max": stat.run_delay.max(),
        }
