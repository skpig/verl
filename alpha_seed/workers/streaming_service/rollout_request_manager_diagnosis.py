import threading
from dataclasses import dataclass
from typing import List


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
    assigned_at: float
    updated_at: float
    input_length: int
    output_length: int
    aborted_count: int
    stale_count: int


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
