import itertools
import warnings
from contextlib import contextmanager
import functools
import gc
import gzip
import inspect
try:
    import ujson as json
except ImportError:
    import json
import os
import random
import socket
import sys
import threading
import time
import csv

from collections import defaultdict
from dataclasses import dataclass
from types import FrameType
from typing import Union, Optional, List, Any, Dict, Tuple


def filter_by(tl_file: str, pred: callable):
    with open(tl_file, 'r') as f:
        full_tl_json = json.load(f)

    full_tl_json = list(filter(pred, full_tl_json))

    with open(tl_file, 'w') as f:
        json.dump(full_tl_json, f)


def merge_into(tl_file: str, user_trace_span: List[dict]):
    with open(tl_file, 'r') as f:
        full_tl_json = json.load(f)
        f.flush()

    full_tl_json.extend(user_trace_span)

    with open(tl_file, 'w') as f:
        json.dump(full_tl_json, f)
        f.flush()


def tl_time_between(start_ts: float, end_ts: float) -> callable:
    start_ts_ns = start_ts * 1e6
    end_ts_ns = end_ts * 1e6

    def pred(span: dict):
        if start_ts_ns <= span['ts'] <= end_ts_ns:
            return True
        return False

    return pred


def export_chrome_trace(tl_file: str, spans: List[dict]) -> str:
    save_path = os.path.abspath(tl_file)
    if tl_file.endswith('.gz'):
        with gzip.open(save_path, 'wb') as gf:
            data = json.dumps(spans)
            gf.write(data.encode('utf-8'))
            gf.flush()
    else:
        with open(save_path, 'w') as f:
            json.dump(spans, f, indent=2)
            f.flush()
    print(f'timeline file generated at {save_path}')
    return save_path


def trace_json_to_csv(tl_file: str, out_csv_file: str):
    """
    将trace出来的json转成csv，纯粹为了方便查看和分析数据用
    """
    with open(tl_file, 'r') as f:
        full_tl_json: List[dict] = json.load(f)

    columns = set()
    for event in full_tl_json:
        columns = columns.union(event.keys())
    columns = list(columns)

    with open(out_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)
        for event in full_tl_json:
            row = [event.get(c, '') for c in columns]
            writer.writerow(row)


class TracingEvent(object):
    """
    chrome trace event format see doc:
      https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#
    """

    def to_objects(self) -> List[dict]:
        raise NotImplementedError


@dataclass
class CompleteEvent(TracingEvent):
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

    @property
    def ts_to_sort(self):
        return self.ts


class CoherentCompleteEvent(TracingEvent):
    """
    用于表示一组连贯的CompleteEvent，他们拥有同样的pid和tid，并拼在一起
    """

    def __init__(self, events: List[CompleteEvent], sorted_by_index: int = 0):
        # events should be sorted in order of ts
        # sorted_by_index: 根据第一个event的ts来决定sort ts
        self.events = events
        self.sorted_by_index = sorted_by_index
        assert len(events) > 0, "need at least 1 event"

    @property
    def pid(self):
        return self.events[0].pid

    @pid.setter
    def pid(self, v):
        for e in self.events:
            e.pid = v

    @property
    def tid(self):
        return self.events[0].tid

    @tid.setter
    def tid(self, v):
        for e in self.events:
            e.tid = v

    @property
    def ts(self):
        return min(e.ts for e in self.events)

    @property
    def ts_to_sort(self):
        return self.events[self.sorted_by_index].ts_to_sort

    @property
    def dur(self):
        # 必须这样算，忽略中间的空隙，已最后一个边界为准
        end_ts = max(e.ts + e.dur for e in self.events)
        return end_ts - self.ts

    def to_objects(self) -> List[dict]:
        obj = []
        for e in self.events:
            obj.extend(e.to_objects())
        return obj


@dataclass
class BeginEvent(TracingEvent):
    name: str
    cat: str
    pid: Union[str, int]
    tid: Union[str, int]

    # 起始和持续时间长度（单位都是us）
    ts: float
    stack: Optional[List[int]] = None

    args: Optional[dict] = None

    def to_objects(self) -> List[dict]:
        return [{
            'name': self.name,
            'cat': self.cat,
            'pid': self.pid,
            'tid': self.tid,
            'args': self.args or {},
            'ts': self.ts,
            'ph': 'B',
        }]


@dataclass
class EndEvent(TracingEvent):
    name: str
    cat: str
    pid: Union[str, int]
    tid: Union[str, int]

    # 起始和持续时间长度（单位都是us）
    ts: float
    stack: Optional[List[int]] = None

    args: Optional[dict] = None

    def to_objects(self) -> List[dict]:
        return [{
            'name': self.name,
            'cat': self.cat,
            'pid': self.pid,
            'tid': self.tid,
            'args': self.args or {},
            'ts': self.ts,
            'ph': 'E',
        }]


@dataclass
class FlowEvent(TracingEvent):
    # {"ph": "f", "id": 246, "pid": "172.20.133.93", "tid": 13, "ts": 1669171992173028, "cat": "async_gpu", "name": "cudaLaunchKernel", "bp": "e"}
    name: str
    cat: str

    # list of (pid, tid, ts)
    flows: List[Tuple[str, str, float]]

    def to_objects(self) -> List[dict]:
        gen_id = random.randint(1000, 9999999)
        ret = []
        for f in self.flows:
            pid, tid, ts = f
            ret.append({
                'name': self.name,
                'cat': self.cat,
                'pid': pid,
                'tid': tid,
                'ts': ts,
                'ph': 't',
                'bp': 'e',
                'id': gen_id,
            })
        ret[0]['ph'] = 's'
        ret[-1]['ph'] = 'f'
        return ret


@dataclass
class CounterEvent(TracingEvent):
    name: str
    pid: Union[str, int]

    # 起始和持续时间长度（单位都是us）
    ts: float

    # 计数的数据序列
    data: Dict[str, Union[int, float]]

    def to_objects(self) -> List[dict]:
        return [{
            'name': self.name,
            'pid': self.pid,
            'args': self.data,
            'ts': self.ts,
            'ph': 'C',
        }]


class CombinedEvents(TracingEvent):
    """
    将多个tracing event合并一起，表示成1个event，最后按顺序展开每个object
    """

    def __init__(self, events: List[TracingEvent]):
        self.events = events

    def to_objects(self) -> List[dict]:
        obj = []
        for e in self.events:
            obj.extend(e.to_objects())
        return obj


@dataclass
class ProcessMetadataEvent(TracingEvent):
    pid: Union[str, int]
    sort_index: int
    process_name: Optional[str] = None
    process_labels: List[str] = None

    def to_objects(self) -> List[dict]:
        ret = [{
            'name': 'process_sort_index',
            'pid': self.pid,
            'ph': 'M',
            'args': {
                'sort_index': self.sort_index,
            },
        }]
        if self.process_labels is not None:
            ret.append({
                'name': 'process_labels',
                'pid': self.pid,
                'ph': 'M',
                'args': {
                    'labels': ','.join(self.process_labels),
                },
            })
        if self.process_name is not None:
            ret.append({
                'name': 'process_name',
                'pid': self.pid,
                'ph': 'M',
                'args': {
                    'name': self.process_name,
                },
            })
        return ret


@dataclass
class ThreadMetadataEvent(TracingEvent):
    pid: Union[str, int]
    tid: Union[str, int]
    sort_index: int
    thread_name: Optional[str] = None

    def to_objects(self) -> List[dict]:
        ret = [{
            'name': 'thread_sort_index',
            'pid': self.pid,
            'tid': self.tid,
            'ph': 'M',
            'args': {
                'sort_index': self.sort_index,
            },
        }]
        if self.thread_name is not None:
            ret.append({
                'name': 'thread_name',
                'pid': self.pid,
                'tid': self.tid,
                'ph': 'M',
                'args': {
                    'name': self.thread_name,
                },
            })
        return ret


class DummyEvent(TracingEvent):

    def to_objects(self) -> List[dict]:
        return [{
            'name': 'dummy',
            'cat': 'dummy',
            'pid': random.randint(1, 100),
            'tid': random.randint(1, 100),
            'args': {
                'content': '*' * random.randint(100, 1000),
            },
            'ts': random.randint(1, 9999),
            'dur': random.randint(1, 100),
            'ph': 'i',
        }]


class Tracer(object):

    @classmethod
    def get_instance(cls):
        tracer = getattr(_local_tracers, 'tracer', None)
        if tracer is None:
            tid = threading.current_thread().ident
            _local_tracers.tracer = Tracer()
            with _tracer_map_mtx:
                _local_tracer_map[tid] = _local_tracers.tracer
            return _local_tracers.tracer
        else:
            return tracer

    def __init__(self):
        # local data store (access from current thread only)
        self._buffer_size = 256
        self.current_buf: List[Optional[TracingEvent]] = [None] * self._buffer_size
        self.current_pos: int = 0
        self.merged_buffers: List[List[TracingEvent]] = []  # [[buf0], [buf1], ...]
        self._gc_trace_disabled = False

    def trace(self, evt: TracingEvent):
        self.current_buf[self.current_pos] = evt
        self.current_pos += 1
        if self.current_pos == self._buffer_size:
            self._rotate()

    @contextmanager
    def complete_event(self, pid, tid, category, name):
        t0 = time.time() * 1e6
        yield
        t1 = time.time() * 1e6
        self.trace(CompleteEvent(
            pid=pid,
            tid=tid,
            name=name,
            cat=category,
            ts=t0,
            dur=t1 - t0,
        ))

    def _rotate(self):
        self.merged_buffers.append(self.current_buf)  # noqa
        self.current_buf = [None] * self._buffer_size
        self.current_pos = 0

    @staticmethod
    def merge_all() -> List[dict]:
        with _tracer_map_mtx:
            print(f'got {len(_local_tracer_map)} tracers in all threads')
            ret = []
            for tid, tracer in _local_tracer_map.items():
                total_events_count = len(tracer.merged_buffers) * tracer._buffer_size + tracer.current_pos
                print(f'thread({tid}) generated {total_events_count} events')
                for buf in tracer.merged_buffers:
                    buf_obj = []
                    for e in buf:
                        buf_obj.extend(e.to_objects())
                        ret.extend(e.to_objects())
                    # ret.extend(buf_obj)
                for e in tracer.current_buf:
                    if e is None:
                        break
                    ret.extend(e.to_objects())
            return ret


@dataclass
class ThreadSlot:
    tid: int
    latest_end_ts: float


class WaterfallSlotTracer:

    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self._thread_slots = defaultdict(list)
        self._sort_alignment_ts = time.time()

    def trace(self, evt: TracingEvent):
        if isinstance(evt, (CompleteEvent, CoherentCompleteEvent)):
            self.allocate_thread_slot(evt)
        self.tracer.trace(evt)

    def allocate_thread_slot(self, evt: Union[CompleteEvent, CoherentCompleteEvent]):
        pid = evt.pid

        # get available thread slot
        available_slot = None
        for t in self._thread_slots[pid]:
            if t.latest_end_ts <= evt.ts:
                available_slot = t
                break

        # update latest empty ts
        if available_slot is None:
            available_slot = ThreadSlot(
                tid=len(self._thread_slots[pid]),
                latest_end_ts=evt.ts + evt.dur,
            )
            self._thread_slots[pid].append(available_slot)
        else:
            available_slot.latest_end_ts = evt.ts + evt.dur

        # assign tid to event
        evt.tid = available_slot.tid


class OrderedTracer:

    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self._buffered_events: List[Union[CompleteEvent, CoherentCompleteEvent]] = []

    def trace(self, evt: Union[CompleteEvent, CoherentCompleteEvent]):
        self._buffered_events.append(evt)

    # 每个step结束后调用一次，全局flush
    def reorder_flush(self):
        self._buffered_events.sort(key=lambda e: (e.pid, e.ts_to_sort))
        for pid, events in itertools.groupby(self._buffered_events, key=lambda e: e.pid):
            for i, e in enumerate(events):
                e.tid = i
        self.tracer.trace(CombinedEvents(self._buffered_events))
        self._buffered_events = []


def make_pid() -> str:
    pid = ''
    try:
        pid = socket.gethostname()
        pid = socket.gethostbyname(pid)
    except socket.gaierror:
        # 如果ipv4失败则使用 getaddrinfo 获取地址信息
        try:
            addr_info = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6)
            # 提取第一个 IPv6 地址
            ipv6_address = addr_info[0][4][0]
            return ipv6_address
        except socket.gaierror:
            pass
    except Exception:
        pass
    if not pid:
        pid = '<err>'
    return pid


class GCEventTracer(object):

    def __init__(self):
        self._gc_trace_enabled = True
        self.pid: str = ''
        with _tracer_map_mtx:
            self.tracer = Tracer()
            _local_tracer_map['gc-tracer'] = self.tracer

    def _gc_trace_callback(self, phase: str, info: dict):
        if not self._gc_trace_enabled:
            return
        t0 = time.time_ns() / 1e3
        info = dict(info)
        info['from-thread'] = threading.current_thread().name
        if phase == 'start':
            self.tracer.trace(BeginEvent('gc', cat='gc', pid=self.pid, tid='gc', ts=t0, args=info))
        elif phase == 'stop':
            self.tracer.trace(EndEvent('gc', cat='gc', pid=self.pid, tid='gc', ts=t0, args=info))

    def start(self):
        self.pid = make_pid()
        self._gc_trace_enabled = True
        if self._gc_trace_callback not in gc.callbacks:
            gc.callbacks.append(self._gc_trace_callback)

    def stop(self):
        self._gc_trace_enabled = False


_local_tracers = threading.local()
_local_tracer_map: Dict[str, Tracer] = {}  # tid -> Tracer
_tracer_map_mtx = threading.Lock()

# use this to hack the ray, let ray know this object(_tracer_map_mtx) is not going to serialize
_merge_test_code = compile('_tracer_map_mtx.locked()', '<_merge>', 'eval')


def _merging():
    return eval(_merge_test_code)


def trace_me(func):
    """
    trace 某个指定的函数调用前后
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapped(*args, **kw):
        local_tracer = Tracer.get_instance()
        pid = socket.gethostbyname(socket.gethostname())
        tid = threading.current_thread().name
        t0 = time.time_ns() / 1e3

        r = func(*args, **kw)

        t1 = time.time_ns() / 1e3
        dt = t1 - t0
        local_tracer.trace(
            CompleteEvent(
                name=func.__name__,
                cat='hot_func',
                pid=pid,
                tid=tid,
                ts=t0,  # us
                dur=dt,  # us
                args={},
            )
        )

        return r

    return wrapped


longest_sys_paths = list(sorted([os.path.abspath(p) for p in sys.path], key=lambda p: -len(p)))


def _strip_leading_sys_path(p: str):
    for sys_p in longest_sys_paths:
        if p.startswith(sys_p):
            return p[len(sys_p):].lstrip('/')
    return None


class _old_tracer_sentinel(object):  # noqa
    pass


class CallStackTracer(object):

    def __init__(self, max_depth=15, stopped=False, ignore_functions: List[str] = None):
        self._enabled = True  # dynamic enable/disable control, can partially disable during tracing, DO NOT change it outside the class
        self._global_enabled = not stopped  # global enable/disable control, if disabled, will not register systrace callback
        self.max_depth = max_depth
        self._ignore_functions = set(ignore_functions or [])
        self._pid = make_pid()
        self._trace_event_cache = {}  # id(frame) -> Event
        self._thread_local = threading.local()

    def enable(self):
        self._global_enabled = True

    def disable(self):
        self._global_enabled = False

    def start(self):
        assert self._global_enabled, "cannot manually start without global enabled. please call .enable() first"
        # 不使用上下文管理时，不允许trace嵌套
        self._thread_local.current_depth = 0
        self._thread_local.old_tracer = sys.settrace(self._trace_call)  # noqa
        if self._thread_local.old_tracer is not None:
            warnings.warn("does not support nested tracing. will be undetermined behavior")

    def stop(self):
        old_tracer = getattr(self._thread_local, 'old_tracer', None)
        sys.settrace(old_tracer)

    def ignore_function(self, func_name):
        return func_name in self._ignore_functions

    @contextmanager
    def tracing(self):
        """
        usage:
            tracer = CallStackTracer()

            def some_function():
                print("Inside some_function")

            with tracer.tracing():
                some_function()
        """
        self._thread_local.current_depth = 0

        # 不能用None判断，因为当前没有设置sys.settrace时，返回的是None，结束后需要将这个None设置回去
        old_tracer = _old_tracer_sentinel
        if self._global_enabled:
            # 允许trace嵌套，每次嵌套后depth按最深的记
            old_tracer = sys.settrace(self._trace_call)  # noqa
        try:
            yield
        finally:
            # avoid being disabled during tracing
            if old_tracer is not _old_tracer_sentinel:
                sys.settrace(old_tracer)

    def trace_into(self, func):
        """
        trace从某个函数开始的整个调用栈
        :param func: 要被trace的那个函数
        :return:
        """

        @functools.wraps(func)
        def wrapper(*args, **kw):
            with self.tracing():
                return func(*args, **kw)

        return wrapper

    def _trace_return(self, frame: FrameType, event: str, arg: Any):
        """
        same as _trace_call
        """

        if not self._enabled:
            return
        if _merging():
            # no trace when merge_all called, otherwise cause recursively tracing
            # event collection
            return

        if event == 'return':
            self._thread_local.current_depth -= 1
            local_tracer = Tracer.get_instance()
            t1 = time.time_ns() / 1e3
            e = self._trace_event_cache.pop(id(frame), None)
            if e is not None:
                e.dur = t1 - e.ts
                local_tracer.trace(e)

    def _skip_until_return(self, frame: FrameType, event: str, arg: Any):
        if event == 'return':
            self._enabled = True

    def _trace_call(self, frame: FrameType, event: str, arg):
        """
        Trace functions should have three arguments: frame, event, and arg.
        :param frame: the current stack frame
        :param event: a string: 'call', 'line', 'return', 'exception' or 'opcode
        :param arg: depends on the event type
        :return: to cascade tracing into deeper stacks, return this trace func or another
        """

        if self._thread_local.current_depth > self.max_depth:
            return
        if not self._enabled:
            return
        if _merging():
            # no trace when merge_all called, otherwise cause recursively tracing
            # event collection
            return

        caller = frame.f_back
        caller_loc = '<root>'
        func_name = frame.f_code.co_name
        filename = os.path.basename(frame.f_code.co_filename)
        if caller:
            caller_loc = f'{os.path.basename(caller.f_code.co_filename)}:{caller.f_lineno}'
        filename_from_module = _strip_leading_sys_path(frame.f_code.co_filename) or filename

        # 如果是成员函数的话，函数名要加上类名，提示更友好
        var_names = frame.f_code.co_varnames
        if len(var_names) > 0 and var_names[0] in ('self', 'this', 'cls'):
            this_obj = frame.f_locals.get(var_names[0])
            if inspect.isclass(this_obj):
                # class method
                func_name = f'{this_obj.__name__}.{func_name}'
            else:
                # member func
                this_cls = getattr(this_obj, '__class__', None)
                if this_cls is not None and inspect.isclass(this_cls):
                    func_name = f'{this_cls.__name__}.{func_name}'

        if self.ignore_function(func_name):
            # 通过_enable控制，在这个{func_name} return之前，都不采集，
            # 在下面的的_skip_until_return里的return时重新打开采集
            self._enabled = False
            return self._skip_until_return

        e_name = f'{filename_from_module}:{frame.f_lineno}#{func_name} (at {caller_loc})'
        root_module = filename_from_module.split(os.path.sep, 1)[0]
        category = f'call:{root_module}'
        if event == 'call':
            self._thread_local.current_depth += 1
            t0 = time.time_ns() / 1e3
            tid = threading.current_thread().name
            e = CompleteEvent(name=e_name, cat=category, pid=self._pid, tid=tid, ts=t0, dur=0)
            # be careful, never store frame object,
            # otherwise will case local scope memory leak
            self._trace_event_cache[id(frame)] = e
            # 返回这个_trace_return，让{func_name}在结束时自动自动记录event结束时间
            return self._trace_return


def trace_into(max_depth=15):
    stack_tracer = CallStackTracer(max_depth=max_depth)
    return stack_tracer.trace_into


class _TraceTestClass(object):

    @staticmethod
    def _trace_test_static_func():
        pass

    @classmethod
    def _trace_test_cls_func(cls):
        pass

    def _trace_test_member_func(self):
        if 1 + 1 == 2:
            return 1
        else:
            return 0  # noqa


def _trace_test_inner(arg: int):
    print('inner', arg)
    return arg + 1


@trace_into()
def _trace_test_nested():
    return 0


@trace_into()
def _trace_test():
    tc = _TraceTestClass()
    a = tc._trace_test_member_func()
    b = _trace_test_inner(a)
    print('outer', a, b)
    _TraceTestClass._trace_test_static_func()
    _TraceTestClass._trace_test_cls_func()
    _trace_test_nested()


@trace_into()
def _merge_when_tracing():
    tracer = Tracer.get_instance()
    spans = tracer.merge_all()
    print(spans)


if __name__ == '__main__':
    _merge_when_tracing()
