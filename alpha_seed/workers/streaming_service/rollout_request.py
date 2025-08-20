import copy
import time
from dataclasses import dataclass, field
from typing import List, Optional

from alpha_seed.workers.xperf_rollout.component.query import QueryProcessEvent, Query


@dataclass
class StaleHistory:
    assigned_engine_id: str
    assigned_engine_name: str
    start_step: int  # 最早开始的step
    # 下面时间戳均为stale之前在engine侧的时间戳 (默认unit: ms)
    last_pending_reschedule_ts: float  # 此history span进入request pool的时间
    dispatch_time: float  # 从request pool分出去的时间 (unit: s)
    received_time: float  # 进入engine waiting队列的时间
    first_scheduled_time: float  # 开始prefill时间
    first_token_time: float  # 最早开始decode时间(prefill完成时间)
    end_ts: float  # 发现stale的时刻
    stale_action: str  # 是什么直接导致的stale，如release/engine-died/...
    stale_reason: str  # 基于什么原因要做这个stale的操作，如内存不够/rebalance/...
    length_generated: int  # 在这一次stale之前，decode了多少(不含prefill部分)
    update_count: int  # stale之前update过多少次
    release_count: int  # 这次stale区间内kv cache 被release的次数
    process_events: List[QueryProcessEvent]  # query对象在这次的engine上的event记录

    @classmethod
    def from_request(cls, req: 'Request', stale_action, stale_reason) -> 'StaleHistory':
        delta_generated = req.query.new_token_len - (len(req.query.input_ids) - req.query.original_input_len)
        # 时间单位都是ms
        received_time = req.query.received_time or req.last_pending_reschedule_ts
        first_scheduled_time = req.query.first_scheduled_time or (received_time + 1e-3)
        first_token_time = req.query.first_token_time or (first_scheduled_time + 1e-3)
        history = StaleHistory(
            assigned_engine_id=req.assigned_engine_id,  # noqa
            assigned_engine_name=req.assigned_engine_name,
            start_step=req.global_step,
            last_pending_reschedule_ts=req.last_pending_reschedule_ts,
            dispatch_time=req.last_assigned_at,
            received_time=received_time,
            first_scheduled_time=first_scheduled_time,
            first_token_time=first_token_time,
            end_ts=time.time() * 1e3,
            stale_action=stale_action,
            stale_reason=stale_reason,
            length_generated=delta_generated,
            update_count=req.update_count,
            release_count=req.query.release_count,
            process_events=copy.deepcopy(req.query.process_events),
        )
        return history


@dataclass
class AbortHistory:
    engine_id: str  # 从哪个engine abort
    timestamp: float  # 何时abort


@dataclass
class Request:
    request_id: str  # 同Query.id
    query: Query
    finished: bool = False  # 是否已经结束生成
    assigned: bool = False  # 是否已经分出去了（分出去了，但不确定具体分配的engine_id，取决于passive/active模式
    assigned_engine_id: Optional[str] = None  # 被分配到的engine
    assigned_engine_name: Optional[str] = None  # 比engine_id更可读的name，作为key优先使用engine_id
    updated_at: float = None  # 标记最后更新时间，并发更新时可以判断数据是否过期 (unit: s)

    # fields need to assign back
    global_step: int = 0  # 当前这个query来自哪个global step的sample
    last_assigned_at: float = 0  # 最近一次调度到此engine的时间 (unit: s)
    # 最近一次各种原因被重新放回池里等待调度的时间戳 (unit: ms), 这个字段可以用来计算request pool的schedule delay
    last_pending_reschedule_ts: float = 0
    update_count: int = 0  # 统计更新了多少次
    stale_histories: List[StaleHistory] = field(default_factory=list)  # 记录所有更换过的engine
    # 只记录从proxy主动abort的记录（新的在前），在重新分发时，会跳过从最近abort的engine，避免抖动
    abort_histories: List[AbortHistory] = field(default_factory=list)

    def is_recent_aborted_from(self, engine_id: str, cool_down_seconds) -> bool:
        now = time.time()
        for his in reversed(self.abort_histories):
            elapsed = now - his.timestamp
            if elapsed > cool_down_seconds:
                # 由于abort_histories是新的在后面，那前面的超过CD的都不管了
                break
            if his.engine_id == engine_id:
                # 遇到任意一个engine则true
                return True
        return False
