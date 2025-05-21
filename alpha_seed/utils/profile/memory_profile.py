from collections import defaultdict

import torch
import time
import socket

from alpha_seed.utils.profile.timeline import TracingEvent, Tracer
from dataclasses import dataclass
from typing import Union, Optional, List
from queue import Queue


@dataclass
class SnapshotWithTime(object):
    snapshot: Optional[List]
    ts: float
    only_before: Optional[List]
    last_ts: float
    only_after: Optional[List]
    name: str
    memory_size: Optional[List]

    def compare(self, before: 'SnapshotWithTime') -> 'SnapshotWithTime':  # 拿到和前一时刻的diff snapshot
        if before is None:
            return SnapshotWithTime(snapshot=None,
                                    ts=self.ts,
                                    only_before=None,
                                    last_ts=self.last_ts,
                                    only_after=self.snapshot,
                                    name=self.name,
                                    memory_size=self.memory_size)

        def _seg_key(seg):
            return (seg['address'], seg['total_size'])

        before_segs = set(_seg_key(seg) for seg in before.snapshot)
        after_segs = set(_seg_key(seg) for seg in self.snapshot)

        only_before_segs = []
        for seg in before.snapshot:
            if _seg_key(seg) not in after_segs:
                only_before_segs.append(seg)
        only_after_segs = []
        for seg in self.snapshot:
            if _seg_key(seg) not in before_segs:
                only_after_segs.append(seg)
        return SnapshotWithTime(snapshot=None,
                                ts=self.ts,
                                only_before=only_before_segs,
                                last_ts=self.last_ts,
                                only_after=only_after_segs,
                                name=self.name,
                                memory_size=self.memory_size)


@dataclass
class BlockList(object):
    blocks: List
    state: str
    total_size: int

    def append_block(self, block):
        self.blocks.append(block)
        self.total_size += block["size"]


def block_with_state(seg):
    blocks_inactive = BlockList([], "inactive", 0)
    blocks_active_allocated = BlockList([], "active_allocated", 0)
    blocks_active_awaiting_free = BlockList([], "active_awaiting_free", 0)

    for block in seg["blocks"]:
        if block["state"] == "active_allocated":
            blocks_active_allocated.append_block(block)
        elif block["state"] == "inactive":
            blocks_inactive.append_block(block)
        elif block["state"] == "active_awaiting_free":
            blocks_active_awaiting_free.append_block(block)
        else:
            print(f'unknown block state:{block["state"]}')

    return (blocks_inactive, blocks_active_allocated, blocks_active_awaiting_free)


class StackNode(object):

    def __init__(self, name, size) -> None:
        self.stack_name = name
        self.size = size
        self.children = {}
        self.begin_ts = -1
        self.parent_node = None

    def has_children(self, name: str):
        return name in self.children.keys()

    def insert_children(self, node):
        self.children[node.stack_name] = node
        node.parent_node = self


def get_frame_name(frame):
    return frame["filename"].split('/')[-1] + ":" + str(frame["line"]) + ":" + frame["name"]


def construct_stack_tree(block_list: BlockList):
    root_node = StackNode(None, 0)
    block_size_sum = 0
    history_size_sum = 0

    for block in block_list.blocks:
        block_size = block["size"]
        block_size_sum += block_size

        if "history" in block:
            for history in block["history"]:
                real_size = history["real_size"]
                history_size_sum += real_size

                cur_node = root_node
                for i, frame in enumerate(reversed(history["frames"])):
                    if i == 0:
                        cur_node.size += real_size
                    frame_name = get_frame_name(frame)
                    if cur_node.has_children(frame_name):
                        cur_node.children[frame_name].size += real_size
                    else:
                        cur_node.insert_children(StackNode(frame_name, real_size))
                    cur_node = cur_node.children[frame_name]

    if block_size_sum > history_size_sum:
        root_node.insert_children(StackNode("<gaps>", block_size_sum - history_size_sum))

    return root_node


def debug_stack_tree(node: StackNode):
    debug_stack_node(node)
    if len(node.children) == 0:
        return
    for child in node.children.values():
        debug_stack_tree(child)


def debug_stack_node(node: StackNode):
    print(f"name: {node.stack_name}, size: {node.size}")


# trace
@dataclass
class MemoryEvent(TracingEvent):
    name: str
    cat: str
    pid: Union[str, int]
    tid: Union[str, int]

    # 起始和持续时间长度（单位都是us）
    ts: float
    dur: float

    args: Optional[dict] = None

    def to_objects(self) -> List[dict]:
        return [{
            'name': self.name,
            'cat': self.cat,
            'pid': self.pid,
            'tid': self.tid,
            'args': self.args or {},
            'ts': self.ts,
            'dur': self.dur,
            'ph': 'X',
        }]


class DummyMemorySampler(object):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MemorySnapshotBucket(object):

    def __init__(self, tracer: 'MemoryFlamegraphTracer', bucket_name: str):
        self.tracer = tracer
        self.bucket_name = bucket_name
        self.start_snapshot = None
        self.end_snapshot = None

    def __enter__(self):
        if not self.tracer.started:
            return DummyMemorySampler()

        t0 = time.time_ns() / 1e3
        snapshot = torch.cuda.memory._snapshot()
        self.start_snapshot = SnapshotWithTime(snapshot=snapshot, ts=t0, only_before=None, last_ts=0, only_after=None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        snapshot = torch.cuda.memory._snapshot()
        t0 = time.time_ns() / 1e3
        last_ts = self.start_snapshot.ts
        self.end_snapshot = SnapshotWithTime(snapshot=snapshot,
                                             ts=t0,
                                             only_before=None,
                                             last_ts=last_ts,
                                             only_after=None)

        # compare snapshot to get the diff
        diff_snapshot = self.end_snapshot.compare(self.start_snapshot)

        # save diff
        saved_bucket = self.tracer.diff_snapshot_buckets[self.bucket_name]
        if not saved_bucket:
            saved_bucket.append(diff_snapshot)
        else:
            # TODO merge
            saved_bucket[0] + diff_snapshot


class MemorySnapshotSampling(object):

    def __init__(self, tracer: 'MemoryFlamegraphTracer'):
        self.tracer = tracer

    def __enter__(self):
        self.tracer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracer.stop()


class MemoryFlamegraphTracer(object):

    def __init__(self):
        self.last_snapshot = None
        self.diff_snapshot = None  # 当前正在处理的diff_snapshot
        self.last_ts = time.time_ns() / 1e3  # 存储的是last_snapshot的获取时间
        self.local_tracer = Tracer.get_instance()
        self.pid = socket.gethostbyname(socket.gethostname())
        self.diff_snapshot_list: List[SnapshotWithTime] = []
        self.diff_snapshot_buckets = defaultdict(list)  # 把每个snapshot按bucket分组存起来 bucket_name -> SnapshotWithTime
        self._started = False

    def trace_node_stack(self, root_node: StackNode, blocks_pre_ts: float, total_size: int, memory_size: List,
                         blocks_dur: float):  # 例如：blocklist.total_size 是inanctive的size
        if not root_node.children:
            return
        root_node.begin_ts = blocks_pre_ts
        q = Queue()
        for child in root_node.children.values():
            q.put(child)
        while not q.empty():
            layer_size = q.qsize()
            last_parent_node = None
            cur_ts = blocks_pre_ts
            while layer_size > 0:
                cur_node = q.get()
                if cur_node.parent_node != last_parent_node:
                    cur_node.begin_ts = cur_node.parent_node.begin_ts
                else:
                    cur_node.begin_ts = cur_ts
                cur_dur = (cur_node.size / total_size) * blocks_dur
                self.local_tracer.trace(
                    MemoryEvent(
                        name=cur_node.stack_name +
                        '[{:.2%}, {:.2%}]'.format(cur_node.size / memory_size[0], cur_node.size / memory_size[1]),
                        cat="memory_snapshot",
                        pid=self.pid,
                        tid="memory",
                        ts=cur_node.begin_ts,
                        dur=cur_dur,
                        args={
                            "size": cur_node.size,
                        }))
                cur_ts = cur_node.begin_ts + cur_dur
                last_parent_node = cur_node.parent_node
                layer_size -= 1

                for child in cur_node.children.values():
                    q.put(child)

    def trace_segs(self, only_state: str, diff_snapshot: SnapshotWithTime, segments_size: List[int],
                   snapshot_state_size: List):
        snapshot_dur = diff_snapshot.ts - diff_snapshot.last_ts
        cur_snapshot_state_size = snapshot_state_size[0]
        only_before_dur = snapshot_dur * (cur_snapshot_state_size / (snapshot_state_size[0] + snapshot_state_size[1]))
        segments = None
        if only_state == "only_before":
            all_dur = only_before_dur
            seg_pre_ts = diff_snapshot.last_ts

            segments = diff_snapshot.only_before

        else:
            cur_snapshot_state_size = snapshot_state_size[1]
            all_dur = snapshot_dur * (cur_snapshot_state_size / (snapshot_state_size[0] + snapshot_state_size[1]))
            seg_pre_ts = diff_snapshot.last_ts + only_before_dur

            segments = diff_snapshot.only_after

        snapshot_dur = only_before_dur + snapshot_dur * (cur_snapshot_state_size /
                                                         (snapshot_state_size[0] + snapshot_state_size[1]))
        self.local_tracer.trace(
            MemoryEvent(name=self.diff_snapshot.name,
                        cat="memory_snapshot",
                        pid=self.pid,
                        tid="memory",
                        ts=self.diff_snapshot.last_ts,
                        dur=snapshot_dur,
                        args={}))

        self.local_tracer.trace(
            MemoryEvent(name=only_state +
                        '[{:.2%}, {:.2%}]'.format(cur_snapshot_state_size / diff_snapshot.memory_size[0],
                                                  cur_snapshot_state_size / diff_snapshot.memory_size[1]),
                        cat="memory_snapshot",
                        pid=self.pid,
                        tid="memory",
                        ts=seg_pre_ts,
                        dur=all_dur,
                        args={
                            "size": cur_snapshot_state_size,
                            "name": self.diff_snapshot.name
                        }))
        end_time = seg_pre_ts + all_dur

        for i, seg in enumerate(segments):
            args = {"size": segments_size[i], "stream": seg["stream"]}
            seg_dur = (segments_size[i] / cur_snapshot_state_size) * all_dur
            if seg_pre_ts + seg_dur > end_time:
                seg_dur = end_time - seg_pre_ts
            self.local_tracer.trace(
                MemoryEvent(name=f'seg_{seg["address"]}' + '[{:.2%}, {:.2%}]'.format(
                    segments_size[i] / diff_snapshot.memory_size[0], segments_size[i] / diff_snapshot.memory_size[1]),
                            cat="memory_snapshot",
                            pid=self.pid,
                            tid="memory",
                            ts=seg_pre_ts,
                            dur=seg_dur,
                            args=args))

            block_lists = block_with_state(seg)
            blocks_pre_ts = seg_pre_ts
            for block_list in block_lists:
                if len(block_list.blocks) != 0:
                    blocks_dur = (block_list.total_size / segments_size[i]) * seg_dur
                    self.local_tracer.trace(
                        MemoryEvent(name=block_list.state +
                                    '[{:.2%}, {:.2%}]'.format(block_list.total_size / diff_snapshot.memory_size[0],
                                                              block_list.total_size / diff_snapshot.memory_size[1]),
                                    cat="memory_snapshot",
                                    pid=self.pid,
                                    tid="memory",
                                    ts=blocks_pre_ts,
                                    dur=blocks_dur,
                                    args={
                                        "size": block_list.total_size,
                                    }))

                    root_node = construct_stack_tree(block_list)
                    # print(f"state debug {block_list.state}")
                    # debug_stack_tree(root_node)
                    self.trace_node_stack(root_node, blocks_pre_ts, block_list.total_size, diff_snapshot.memory_size,
                                          blocks_dur)
                    blocks_pre_ts += blocks_dur
            seg_pre_ts += seg_dur

    def trace_snapshot(self, diff_snapshot: SnapshotWithTime):
        before_segments_size = []
        after_segments_size = []
        snapshot_state_size = [0, 0]
        if diff_snapshot.only_before is not None:
            for seg in diff_snapshot.only_before:
                before_segments_size.append(seg["total_size"])
                snapshot_state_size[0] += seg["total_size"]

        for seg in diff_snapshot.only_after:
            after_segments_size.append(seg["total_size"])
            snapshot_state_size[1] += seg["total_size"]

        if diff_snapshot.only_before is not None and self.diff_snapshot.only_before:
            self.trace_segs("only_before", diff_snapshot, before_segments_size, snapshot_state_size)
        if diff_snapshot.only_after is not None and self.diff_snapshot.only_after:
            self.trace_segs("only_after", diff_snapshot, after_segments_size, snapshot_state_size)

    def memory_diff_checkpoint(self, diff_name: str):
        snapshot = torch.cuda.memory._snapshot()
        t0 = time.time_ns() / 1e3
        cur_memory_reserved = torch.cuda.memory_reserved()
        total_memory = 0
        for device_id in range(torch.cuda.device_count()):
            total_memory += torch.cuda.get_device_properties(device_id).total_memory
        memory_size = [cur_memory_reserved, total_memory]
        cur_snapshot = SnapshotWithTime(snapshot=snapshot,
                                        ts=t0,
                                        only_before=None,
                                        last_ts=self.last_ts,
                                        only_after=None,
                                        name=diff_name,
                                        memory_size=memory_size)
        self.diff_snapshot = cur_snapshot.compare(self.last_snapshot)
        self.diff_snapshot_list.append(self.diff_snapshot)
        self.last_snapshot = cur_snapshot
        self.last_ts = t0

    def sample_bucket(self, bucket_name: str):
        raise RuntimeError("do not call, it is Working In Progress")
        return MemorySnapshotBucket(self, bucket_name)

    def sampling(self, enabled=True):
        if not enabled:
            return DummyMemorySampler()
        else:
            return MemorySnapshotSampling(self)

    def start(self):
        torch.cuda.memory._record_memory_history(True)
        self._started = True

    @property
    def started(self):
        return self._started

    def stop(self):
        self._started = False
        torch.cuda.memory._record_memory_history(False)

    def trace_memory_flamegraph(self):
        for snapshot in self.diff_snapshot_list:
            self.trace_snapshot(snapshot)
