import queue
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from itertools import groupby
from typing import Dict, Any, List, Tuple, Optional, Literal, Iterable

import ray

from alpha_seed.utils.profile.timeline import Tracer, WaterfallSlotTracer, CoherentCompleteEvent, CompleteEvent
from alpha_seed.utils.server_client import is_local_ray_instance
from alpha_seed.workers.agents.tool import ToolResult


@dataclass
class Segment:
    trajectory_id: int
    sid: str  # 用来关联start-end的id

    @property
    def action(self) -> str:
        raise NotImplementedError()

    @property
    def role(self) -> str:
        raise NotImplementedError()

    @property
    def ts(self) -> float:
        raise NotImplementedError()

    @property
    def num_tokens(self) -> int:
        """
        返回此segment的总token数，
        多轮对话里，后一轮只展示增量的内容，但发送按照完整的上下文发送，此方法返回发送的总token数，
        数字可能不等于消耗的量，因为有prefix cache之类的
        """
        return 0

    def decode(self, tokenizer):
        pass

    def to_digest(self) -> str:
        """
        返回segment摘要信息，最好一行搞定
        """
        pass

    def to_content(self) -> str:
        """
        返回segment完整内容
        """
        pass

    def to_dict(self):
        ret = asdict(self)
        ret['type'] = self.__class__.__name__
        ret['role'] = self.role
        ret['action'] = self.action
        return ret


@dataclass
class StartSegment(Segment):
    start_ts: float

    @property
    def ts(self) -> float:
        return self.start_ts


@dataclass
class EndSegment(Segment):
    end_ts: float

    @property
    def ts(self) -> float:
        return self.end_ts


@dataclass
class LLMStartSegment(StartSegment):
    input_ids: List[int] = field(default_factory=list)
    input_prompt: str = ""
    input_len: int = 0  # 这个存了之后就不变了，因为input_ids会被truncate掉上一轮之前的内容

    def __init__(self, sid: str, start_ts: float, input_ids: List[int]):
        super().__init__(0, sid, start_ts)
        self.input_ids = input_ids
        self.input_len = len(input_ids)

    @property
    def action(self) -> str:
        return "llm"

    @property
    def role(self) -> str:
        return "user"

    @property
    def num_tokens(self) -> int:
        return self.input_len

    def decode(self, tokenizer):
        if not self.input_prompt:
            token_slice = [list(g) for _, g in groupby(self.input_ids, key=lambda x: x >= 0)]
            prompt_frags = []
            for token_ids in token_slice:
                if token_ids[0] >= 0:
                    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    prompt_frags.append(text)
                else:
                    image_placeholder = f"[image placeholder {len(token_ids)} tokens]"
                    prompt_frags.append(image_placeholder)
            self.input_prompt = " ".join(prompt_frags)

    def to_digest(self) -> str:
        return f"{self.input_prompt[:80]}..."

    def to_content(self) -> str:
        return self.input_prompt

    def to_dict(self):
        ret = super().to_dict()
        ret.pop('input_ids')
        return ret


@dataclass
class LLMEndSegment(EndSegment):
    output_ids: List[int] = field(default_factory=list)
    output_prompt: str = ""

    def __init__(self, sid: str, end_ts: float, output_ids: List[int]):
        super().__init__(0, sid, end_ts)
        self.output_ids = output_ids

    @property
    def action(self):
        return "llm"

    @property
    def role(self):
        return "assistant"

    @property
    def num_tokens(self) -> int:
        return len(self.output_ids)

    def decode(self, tokenizer):
        if not self.output_prompt:
            token_slice = [list(g) for _, g in groupby(self.output_ids, key=lambda x: x >= 0)]
            prompt_frags = []
            for token_ids in token_slice:
                if token_ids[0] >= 0:
                    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    prompt_frags.append(text)
                else:
                    image_placeholder = f"[image placeholder {len(token_ids)} tokens]"
                    prompt_frags.append(image_placeholder)
            self.output_prompt = " ".join(prompt_frags)

    def to_digest(self) -> str:
        return f"{self.output_prompt[:30]}...{self.output_prompt[-30:]}"

    def to_content(self) -> str:
        return self.output_prompt

    def to_dict(self):
        ret = super().to_dict()
        ret["output_len"] = len(self.output_ids)
        ret.pop('output_ids')
        return ret


@dataclass
class ToolStartSegment(StartSegment):
    instance_id: str
    tool_name: str
    parameters: dict = field(default_factory=dict)

    def __init__(self, sid: str, start_ts: float, instance_id: str, tool_name: str, parameters: dict):
        super().__init__(0, sid, start_ts)
        self.instance_id = instance_id
        self.tool_name = tool_name
        try:
            self.parameters = {k: str(v) for k, v in parameters.items()}
        except Exception as e:
            traceback.print_exc()
            self.parameters = {'exception_in_capturing': str(e)}

    @property
    def action(self):
        return "tool"

    @property
    def role(self):
        return "call"

    def to_digest(self) -> str:
        return f"{self.tool_name}({self.parameters})"

    def to_content(self) -> str:
        return self.to_digest()


@dataclass
class ToolEndSegment(EndSegment):
    result: Optional[ToolResult] = field(default_factory=ToolResult)

    def __init__(self, sid: str, end_ts: float, result: Optional[ToolResult]):
        super().__init__(0, sid, end_ts)
        self.result = result

    def set_result(self, result: ToolResult):
        self.result = result.to_serializable()

    @property
    def action(self):
        return "tool"

    @property
    def role(self):
        return "result"

    def to_digest(self) -> str:
        if self.result and self.result.error_traceback:
            # 特殊字符串识别
            return (f"[System/Framework exception during tool call]\n"
                    f"{self.result.error_traceback}")
        if self.result and self.result.result is not None:
            result = self.result.result.lstrip()
            if len(result) > 40:
                return f"{result[:40]}..."
            return result
        return "(No result)"

    def to_content(self) -> str:
        if self.result and self.result.error_traceback:
            return (f"[System/Framework exception during tool call]\n"
                    f"{self.result.error_traceback}")
        return self.result.result


def make_llm_seg_pair(start_ts, input_ids) -> Tuple[LLMStartSegment, LLMEndSegment]:
    sid = uuid.uuid4().hex[:12]
    start = LLMStartSegment(sid, start_ts, input_ids)
    end = LLMEndSegment(sid, 0, [])
    return start, end


def make_tool_seg_pair(start_ts, instance_id, tool_name, parameter) -> Tuple[ToolStartSegment, ToolEndSegment]:
    sid = uuid.uuid4().hex[:12]
    start = ToolStartSegment(sid, start_ts, instance_id, tool_name, parameter)
    end = ToolEndSegment(sid, 0, None)
    return start, end


TrajectoryPhase = Literal["init", "processing", "llm", "tool", "env", "finished"]


@dataclass
class AgentIdentity:
    uid: str  # agent loop uid
    agent_name: str  # agent class name
    global_step: int  # 从第几个global_step过来的的task，无论task持续多少个step


class Trajectory:
    """
    主要是trace用的trajectory对象，里面包含各个不同的action的内容，
    用于trace，不是用于训练。
    """

    def __init__(self, agent_ident, traj_id, segments=None):
        self.agent_ident: AgentIdentity = agent_ident
        self.trajectory_id: int = traj_id
        # 只不断append，不删除
        self.segments: List[Segment] = segments or []
        self.finished: bool = False
        # 标记已经上传了的序号，即前N个segments
        self.collected_seq: int = 0
        # 是否在tool call中遇到训练程序错误(不是因输入导致的错误)
        self.has_tool_exception: bool = self._has_tool_exception_segment()

    def _has_tool_exception_segment(self):
        for seg in self.segments:
            if isinstance(seg, ToolEndSegment):
                if seg.result is not None and seg.result.error_traceback:
                    return True
        return False

    def append(self, segment: Segment):
        segment.trajectory_id = self.trajectory_id
        # 如果时llm segment，要把前一轮的input_ids和output_ids给去掉，去掉重复内容
        if isinstance(segment, LLMStartSegment):
            for prev_turn in reversed(self.segments):
                if isinstance(prev_turn, LLMStartSegment):
                    segment.input_ids = segment.input_ids[prev_turn.input_len:]
                    break
            for prev_turn in reversed(self.segments):
                if isinstance(prev_turn, LLMEndSegment):
                    segment.input_ids = segment.input_ids[len(prev_turn.output_ids):]
                    break
        if isinstance(segment, ToolEndSegment):
            if segment.result.error_traceback:
                self.has_tool_exception = True
        self.segments.append(segment)

    def decode(self, tokenizer):
        for seg in self.segments:
            seg.decode(tokenizer)

    def get_paired_segments(self) -> List[Tuple[StartSegment, Optional[EndSegment]]]:
        seg_map = {}  # sid -> (start, end)
        for seg in self.segments:
            if seg.sid not in seg_map:
                seg_map[seg.sid] = [None, None]
            if isinstance(seg, StartSegment):
                seg_map[seg.sid][0] = seg
            elif isinstance(seg, EndSegment):
                seg_map[seg.sid][1] = seg
        return sorted(seg_map.values(), key=lambda x: x[0].ts if x[0] is not None else x[1].ts)

    def get_staging_segments(self) -> Iterable[Segment]:
        next_seq = len(self.segments)
        start_seq = self.collected_seq
        self.collected_seq = next_seq
        for i in range(start_seq, next_seq):
            yield self.segments[i]

    def to_dict(self):
        return {
            'agent_ident': asdict(self.agent_ident),
            'trajectory_id': self.trajectory_id,
            'finished': self.finished,
            'segments': [seg.to_dict() for seg in self.segments],
        }


class TrajectoryFactory:
    """
    管理某个agent task实例下的多个trajectory，根据实际情况再添加fork方法之类
    """

    def __init__(self, uid: str, agent_name: str, global_step: int):
        self.agent_ident = AgentIdentity(uid, agent_name, global_step)
        self.trajectories: List[Trajectory] = []
        self._mutex = threading.Lock()

    def get(self) -> Trajectory:
        """
        拿当前default的那个trajectory
        未分裂之前，就只拿唯一的那一个，如有分裂则拿第一个
        """
        with self._mutex:
            if not self.trajectories:
                traj_id = len(self.trajectories)
                traj = Trajectory(self.agent_ident, traj_id)
                self.trajectories.append(traj)
            return self.trajectories[0]

    def get_staging_segments(self, tokenizer) -> List[Segment]:
        ret = []
        for traj in self.trajectories:
            if traj.finished:
                continue
            for seg in traj.get_staging_segments():
                seg.decode(tokenizer)
                ret.append(seg)
        return ret

    def finish(self, tokenizer):
        for traj in self.trajectories:
            traj.finished = True
            traj.decode(tokenizer)


class TrajectoryTracer:

    def __init__(self):
        self._max_events = 10_000  # 注意CombinedEvent算1个，不要存太多，dump可能太大了也看不了
        self.tracer = Tracer.get_instance("agent", max_events=self._max_events)
        self.waterfall_tracer = WaterfallSlotTracer(self.tracer)

    def trace(self, trajectory: Trajectory):
        events = self._make_trace_event(trajectory)
        for event in events:
            self.waterfall_tracer.trace(event)

    def _make_trace_event(self, trajectory: Trajectory) -> List[CoherentCompleteEvent]:
        """
        [LLM][Tool]...循环
        """
        coherent_events = []
        for start_seg, end_seg in trajectory.get_paired_segments():
            if start_seg is None:
                continue

            event_cat = start_seg.action
            args: Dict[str, Any] = {
                "uid": trajectory.agent_ident.uid,
            }
            if isinstance(start_seg, LLMStartSegment):
                event_name = start_seg.action
                args["input_prompt"] = self._normalize_string(start_seg.input_prompt)
                args["input_len"] = start_seg.input_len
                if isinstance(end_seg, LLMEndSegment):
                    args["output_prompt"] = self._normalize_string(end_seg.output_prompt)
                    args["output_len"] = len(end_seg.output_ids)
            elif isinstance(start_seg, ToolStartSegment):
                event_name = start_seg.tool_name
                args["instance_id"] = start_seg.instance_id
                args["parameters"] = start_seg.parameters
                if isinstance(end_seg, ToolEndSegment):
                    args["result"] = end_seg.result.result
                    args["retries"] = end_seg.result.retries
                    args["max_attempts"] = end_seg.result.max_attempts
                    args["success"] = end_seg.result.success
            else:
                event_name = start_seg.__class__.__name__

            start_ts = start_seg.start_ts * 1e6
            end_ts = end_seg.end_ts * 1e6 if end_seg is not None else time.time() * 1e6
            dur = end_ts - start_ts
            ce = CompleteEvent(
                name=event_name,
                cat=event_cat,
                pid=trajectory.agent_ident.agent_name,
                tid=0,
                ts=start_ts + 1,
                dur=dur - 1,
                args=args,
            )
            coherent_events.append(ce)
        return [CoherentCompleteEvent(coherent_events)]

    def _normalize_string(self, val: str) -> str:
        # 有些字符会导致perfetto UI显示不了，这里暂时转义掉
        return (val.replace("\"", "[(dquote)]").replace("\'",
                                                        "[(squote)]").replace("$",
                                                                              "[(dollar)]").replace("{",
                                                                                                    "｛")  # 注意右边的是全角字符
                .replace("}", "｝")  # 注意右边的是全角字符
               )

    def dump_trajectory_trace(self) -> List[dict]:
        tracer_spans = Tracer.merge_all("agent")
        buffered_spans = self.waterfall_tracer.dump()
        return tracer_spans + buffered_spans


@ray.remote
class TrajectoryCollector:

    def __init__(self, config):
        self.config = config
        # 已完成的task traj
        self.trajectories: Dict[str, List[Trajectory]] = defaultdict(list)  # uid ->
        # 存正在运行中还没结束的task
        self.segments: Dict[str, List[Segment]] = defaultdict(list)  # uid ->
        self.agent_ident_map: Dict[str, AgentIdentity] = {}  # uid -> agent_ident

        # 提交的traj和segments都先进queue，保证单线程执行，避免segments和traj并发导致segments不一致
        self._input_queue = queue.Queue()  # [input type: str, data: Any]

        self._max_save_traj = 60_000  # 够多了，避免OOM
        self._fifo = deque(maxlen=self._max_save_traj + 10)  # 记录traj顺序的 uid

        self.tracer = TrajectoryTracer()
        self._tracer_flush_interval = 30 * 60  # 30 min

        threading.Thread(target=self._input_queue_consumer_loop, daemon=True, name="input-consumer-loop").start()

    def collect(self, trajectories: List[Trajectory]):
        self._input_queue.put(("collect", trajectories))

    def _collect(self, trajectories: List[Trajectory]):
        for traj in trajectories:
            self.trajectories[traj.agent_ident.uid].append(traj)
            self.segments.pop(traj.agent_ident.uid, None)
            self.agent_ident_map.pop(traj.agent_ident.uid, None)

            self._fifo.append(traj.agent_ident.uid)
            if len(self._fifo) > self._max_save_traj:
                pop_uid = self._fifo.popleft()
                self.trajectories.pop(pop_uid, None)

            self.tracer.trace(traj)

    def collect_segments(self, segments: Dict[str, List[Segment]], agent_idents: Dict[str, AgentIdentity]):
        self._input_queue.put(("collect_segments", (segments, agent_idents)))

    def _collect_segments(self, segments: Dict[str, List[Segment]], agent_idents: Dict[str, AgentIdentity]):
        for uid, segs in segments.items():
            # 忽略已经完成的
            if uid in self.trajectories:
                continue
            self.segments[uid].extend(segs)
        self.agent_ident_map.update(agent_idents)

    def _input_queue_consumer_loop(self):
        last_tracer_flush_time = time.time()
        while True:
            # 通过这个队列，使collect和collect_segments串行执行，可以保证不会在已完成的traj里面混入未完成的segments
            type_, data = self._input_queue.get()
            if type_ == "collect":
                self._collect(data)
            elif type_ == "collect_segments":
                segments, agent_idents = data
                self._collect_segments(segments, agent_idents)

            # waterfall tracer flush暂时放这个循环里
            if time.time() - last_tracer_flush_time > self._tracer_flush_interval:
                self.tracer.waterfall_tracer.flush()
                last_tracer_flush_time = time.time()

    def get_all(self,
                max_num_segs: int,
                task_type: Optional[str] = None,
                with_exception_only: bool = False) -> List[Trajectory]:
        ret = self._build_traj_from_segments(self.segments, task_type, max_num_segs, with_exception_only)
        current_seg_count = sum([len(t.segments) for t in ret])
        if current_seg_count >= max_num_segs:
            return ret
        for traj in list(self.trajectories.values()):
            if task_type is not None:
                traj = [t for t in traj if t.agent_ident.agent_name == task_type]
            if with_exception_only:
                traj = [t for t in traj if t.has_tool_exception]
            ret.extend(traj)
            current_seg_count += sum([len(t.segments) for t in traj])
            if current_seg_count >= max_num_segs:
                break
        return ret

    def get_running(self, task_type: Optional[str] = None) -> List[Trajectory]:
        return self._build_traj_from_segments(self.segments, task_type)

    def get(self, uid: str, traj_id: Optional[int] = None) -> List[Trajectory]:
        ret = []
        if uid in self.trajectories:
            ret = self.trajectories[uid]
        elif uid in self.segments:
            ret = self._build_traj_from_segments({uid: self.segments[uid]})
        if traj_id is not None:
            ret = [traj for traj in ret if traj.trajectory_id == traj_id]
        return ret

    def get_task_complete_stats(self) -> Dict[int, dict]:
        """
        返回每个global step的agent task的完成数统计，用于observability
        """
        running_stats = defaultdict(int)  # global_step ->
        complete_stats = defaultdict(int)  # global_step ->
        for agent_ident in list(self.agent_ident_map.values()):
            running_stats[agent_ident.global_step] += 1
        for traj in list(self.trajectories.values()):
            if not traj:
                continue
            complete_stats[traj[0].agent_ident.global_step] += 1

        ret = {}
        for global_step in set(running_stats.keys()) & set(complete_stats.keys()):
            ret[global_step] = dict(
                global_step=global_step,  # task from global step
                running=running_stats[global_step],  # number of running tasks
                completed=complete_stats[global_step],  # number of completed tasks
            )
        return ret

    def _build_traj_from_segments(self,
                                  segments: Dict[str, List[Segment]],
                                  task_type: Optional[str] = None,
                                  max_num_segs: Optional[int] = None,
                                  with_exception_only: bool = False) -> List[Trajectory]:
        trajs: Dict[Tuple[str, int], List[Segment]] = defaultdict(list)  # (uid, traj_id) -> List[Segments]
        max_num_segs = max_num_segs or 999999999  # 很大不会超过的数就行

        uids = list(segments.keys())
        for uid in uids:
            # 跳过已经完成的
            if uid in self.trajectories:
                continue
            segs = segments.get(uid)
            # 跳过空seg的traj，这里先这样，之后有必要再修改成维护暂时的traj而不是segments
            if not segs:
                continue
            for seg in segs:
                trajs[uid, seg.trajectory_id].append(seg)

        ret = []
        accumulated_num_segs = 0
        for (uid, traj_id), segs in trajs.items():
            ident = self.agent_ident_map.get(uid)
            if not ident:
                continue
            if task_type is not None and ident.agent_name != task_type:
                continue
            traj = Trajectory(ident, traj_id, segs)
            if with_exception_only and not traj.has_tool_exception:
                continue
            ret.append(traj)
            accumulated_num_segs += len(segs)
            if accumulated_num_segs >= max_num_segs:
                break
        return ret

    def dump_trajectory_trace(self):
        return self.tracer.dump_trajectory_trace()


def init_agent_trajectory_collector(config, stable_pool_name):
    resources = {}
    if stable_pool_name and not is_local_ray_instance():
        resources = {stable_pool_name: 1}
    max_concurrency = config.rollout_server.agent.max_workers * 2 + 5  # 额外5个监控用
    return TrajectoryCollector.options(name="TrajectoryCollector",
                                       get_if_exists=True,
                                       max_concurrency=max_concurrency,
                                       resources=resources).remote(config)  # noqa


def get_agent_trajectory_collector():
    return ray.get_actor("TrajectoryCollector")
