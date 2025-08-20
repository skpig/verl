import random
import threading
from dataclasses import dataclass, field
from typing import List, Dict

from alpha_seed.workers.streaming_service.rollout_request import Request


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


class FiniteDict:
    """
    有限大小的dict，有个FIFO队列记录key，所以key重复了可能会被提前pop掉
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self._map = {}
        self._queue = []
        self._mutex = threading.Lock()

    def add(self, key, data):
        with self._mutex:
            self._map[key] = data
            self._queue.append(key)
            if len(self._queue) > self.max_size:
                pop_key = self._queue.pop(0)
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

    # 记录所有query创建到末次从proxy dispatch的时间差
    shed_delay: ReservoirSamples = field(default_factory=lambda: ReservoirSamples())


class RequestStatCollector:

    def __init__(self):
        self.steps: Dict[int, StepStat] = {}  # step ->
        self.mutex = threading.Lock()

    def finish(self, pool_name: str, req: Request):
        shed_delays = []
        en_pool = req.last_pending_reschedule_ts
        last_recv = req.query.received_time
        shed_delays.append((last_recv - en_pool) / 1e3)
        for his in req.stale_histories:
            delay = (his.received_time - his.last_pending_reschedule_ts) / 1e3
            shed_delays.append(delay)
        with self.mutex:
            if req.global_step not in self.steps:
                self.steps[req.global_step] = StepStat(req.global_step, pool_name)
            self.steps[req.global_step].shed_delay.add(shed_delays)

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
        }
