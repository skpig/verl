import copy
import time
from dataclasses import asdict
from typing import List, Optional, Callable

from omegaconf import DictConfig

from alpha_seed.utils.profile.timeline import Tracer, WaterfallSlotTracer, CompleteEvent, CoherentCompleteEvent, \
    CombinedEvents, FlowEvent, TracingEvent
from alpha_seed.workers.streaming_service.rollout_request import StaleHistory, Request
from alpha_seed.workers.xperf_rollout.component.query import QueryProcessEvent, ProcessEventType


def colorize_step(step: int) -> str:
    # 把step (0-25) 转成a-z的字母循环，用于渲染D slice
    # 渲染成这样的效果  D.1.a  D.2.b
    step = step % 26
    return f"{step}.{chr(step + ord(b'a'))}"


def get_worker_group_name_normalizer(query_trace_config: DictConfig) -> Callable[[str], str]:
    if query_trace_config.renderer.merge_elastic_replicas:

        def normalize(wg_name: str) -> str:
            # 为了简单不把对象和配置传来传去，
            # 先hardcode渲染规则在这里，注意跟着ElasticRolloutManager的命名一起修改
            # wg name格式一般是 ElasticAsyncXPerfGPTRollout_train_rollout_el_644_
            #   _el_ 表示elastic
            #   _st_ 表示stable
            if '_el_' in wg_name:
                return wg_name.split('_el_')[0] + '_elastic'
            return wg_name

        return normalize
    else:
        return lambda x: x


class QueryTracer:

    def __init__(self, query_trace_config: DictConfig, rm_name: str):
        self.query_trace_config = query_trace_config
        self._rm_name = rm_name
        self._wg_name_normalizer = get_worker_group_name_normalizer(query_trace_config)

        self.tracer = Tracer.get_instance(retention_hours=query_trace_config.retention_hours)
        self.waterfall_tracer = WaterfallSlotTracer(self.tracer)
        self._pending_events_to_flows: List[List[CompleteEvent | CoherentCompleteEvent]] = []  # 每个List[CE]要串在一起

    def set_global_step(self, global_step: int):
        self.waterfall_tracer.flush()
        evts_to_flows = self._pending_events_to_flows
        self._pending_events_to_flows = []
        for events in evts_to_flows:
            flows = self._make_trace_flow_event(events)
            self.tracer.trace(CombinedEvents(flows))

    def dump_request_trace(self,
                           with_extra_events: List[List[CompleteEvent | CoherentCompleteEvent]] = None,
                           after_ts: float = 0.) -> List[dict]:
        with_extra_events = with_extra_events or []
        tracer_spans = Tracer.merge_all(after_ts=after_ts)

        # 分配thread slot
        buffered_spans = self.waterfall_tracer.dump(with_extra_events)

        # 同一个query用flow event连起来
        pending_events_to_flows = copy.copy(self._pending_events_to_flows) + with_extra_events
        flow_spans = []
        for events in pending_events_to_flows:
            flows = self._make_trace_flow_event(events)
            flow_spans.extend(CombinedEvents(flows).to_objects())
        return tracer_spans + buffered_spans + flow_spans

    def trace(self, req: Request, persist=True) -> List[CoherentCompleteEvent]:
        # persist: 是否要把event放到waterfall tracer里自动排序
        events = self._make_trace_event2(req)
        # 如果req是一个很新，还没开始gen的，不会有任何events
        if persist and events:
            for evt in events:
                self.waterfall_tracer.trace(evt)
            self._pending_events_to_flows.append(events)
        return events

    def trace_event(self, event: TracingEvent):
        self.tracer.trace(event)

    def _make_trace_event2(self, req: Request) -> List[CoherentCompleteEvent]:
        cce = []
        if req.query.process_events:
            cce = self._make_wpd_coherent_seq2(req, req.query.process_events)
        histories: List[CoherentCompleteEvent] = []
        for idx, his in enumerate(req.stale_histories):
            if len(his.process_events) > 0:
                his_cce = self._make_history_wpd_coherent_seq(req, idx, his)
                histories.extend(his_cce)
        return histories + cce

    def _make_wpd_coherent_seq2(self, req: Request, events: List[QueryProcessEvent]) -> List[CoherentCompleteEvent]:
        """
        根据query的event还原最后一次调度到engine的W/P/D trace，
        比纯query时间戳的要更精确，能反应decoding过程中因kv满了被evict的情况
        """
        meta_info = copy.deepcopy(req.query.meta_info)
        self._remove_garbage_from_metainfo(meta_info)
        uid = meta_info.get("uid", req.query.id)
        events_obj = [asdict(e) for e in events]

        ret: List[CoherentCompleteEvent] = []
        coherent_list: List[CompleteEvent] = []
        coherent_sort_idx = -1

        wg_name = self._wg_name_normalizer(req.assigned_engine_name)
        wait_delay = 0
        for i in range(1, len(events)):
            prev_event = events[i - 1]
            this_event = events[i]
            start_ts = prev_event.ts_ms * 1e3
            dur = (this_event.ts_ms - prev_event.ts_ms) * 1e3
            extra_args = {}
            if this_event.event == ProcessEventType.PREFILL_START:
                wait_delay = dur / 1e6  # unit: s
                if self.query_trace_config.renderer.no_waiting_spans:
                    continue
                event_name, event_cat = "W", "rollout-wait"
            elif this_event.event == ProcessEventType.PREFILL_DONE:
                event_name, event_cat = "P", "rollout-prefill"
                extra_args = {
                    'wait_delay': wait_delay,
                    **this_event.info,
                }
                if wg_name != req.assigned_engine_name:
                    extra_args["wg_name"] = req.assigned_engine_name  # noqa
                # 按第一个prefill event来对齐
                if coherent_sort_idx == -1:
                    coherent_sort_idx = len(coherent_list)
            elif this_event.event == ProcessEventType.FINISHED:
                event_name, event_cat = f"D.{colorize_step(req.global_step)}", "rollout-decode"
                extra_args = {
                    "update_count": req.update_count,
                    "meta_info": meta_info,
                    "sample_kwargs": {
                        "top_k": req.query.top_k,
                        "top_p": req.query.top_p,
                        "temperature": req.query.temperature,
                    },
                    **this_event.info,
                }
            elif this_event.event == ProcessEventType.EVICTED:
                event_name, event_cat = this_event.info.get("reason", "Evict"), "rollout-evicted"
                if prev_event.event == ProcessEventType.PREFILL_DONE:
                    event_name = f"D.{colorize_step(req.global_step)}"
                extra_args = {
                    **this_event.info,
                }
            else:
                # 忽略其他不相关的event类型
                continue

            ce = CompleteEvent(
                name=event_name,
                cat=event_cat,
                pid=f'{self._rm_name} {wg_name}',
                tid=0,
                ts=start_ts + 1,
                dur=dur - 1,
                args={
                    'query_id': req.query.id,
                    'uid': uid,  # 用这个来跟踪整个trajectory
                    'original_input_len': req.query.original_input_len,
                    'step': req.global_step,
                    'stale_count': len(req.stale_histories),
                    'abort_count': len(req.abort_histories),
                    'age': (start_ts / 1e3 - req.query.created_time) / 1e3,  # 相对于query生命周期的延迟，此event-创建时间
                    'shed_delay':
                        (req.query.received_time - req.last_pending_reschedule_ts) / 1e3,  # 相对于上次进入request pool的延迟
                    'enqueue_delay': (req.query.enqueue_time - req.query.created_time) / 1e3,
                    **extra_args,
                },
            )
            coherent_list.append(ce)

            # 每evicted就commit一次coherent events，每个WPD独立一个块，不需要一整个粘一起
            # [W][P][D evicted] --> [W][P][D] ...
            if event_cat in ("rollout-decode", "rollout-evicted"):
                if coherent_sort_idx == -1:
                    coherent_sort_idx = 0
                coherent_list[-1].args['processing_events'] = events_obj
                coherent_list[-1].args['abort_histories'] = [asdict(his) for his in req.abort_histories]
                ret.append(CoherentCompleteEvent(coherent_list, coherent_sort_idx))
                coherent_sort_idx = -1
                coherent_list = []

        last_event = events[-1]

        # 如果还在跑着一半的query也添加到最后面
        if not req.finished:
            event_name, event_cat = "", ""
            if last_event.event == ProcessEventType.PREFILL_START:
                event_name, event_cat = "P(running)", "rollout-prefill"
            elif last_event.event == ProcessEventType.PREFILL_DONE:
                event_name, event_cat = f"D.{colorize_step(req.global_step)}(running)", "rollout-decode"

            if event_name:
                start_ts = last_event.ts_ms * 1e3
                dur = (req.updated_at * 1e3 - last_event.ts_ms) * 1e3  # 按持续到最近一次update
                args = {
                    "query_id":
                        req.query.id,
                    "uid":
                        uid,  # 用这个来跟踪整个trajectory
                    'original_input_len':
                        req.query.original_input_len,
                    # 如果engine更新过参数，这个值也会显示为更新参数前已经decode的长度
                    'previous_generated_len':
                        len(req.query.input_ids) - req.query.original_input_len,
                    # 本次decode的token数
                    'length_generated':
                        req.query.new_token_len - (len(req.query.input_ids) - req.query.original_input_len),
                    'step':
                        req.global_step,
                    'stale_count':
                        len(req.stale_histories),
                    'abort_count':
                        len(req.abort_histories),
                    'abort_histories': [asdict(his) for his in req.abort_histories],
                    # 相对于query生命周期的延迟，此event-创建时间
                    'age': (start_ts / 1e3 - req.query.created_time) / 1e3,
                    # 相对于上次进入request pool的延迟
                    'shed_delay': (req.query.received_time - req.last_pending_reschedule_ts) / 1e3,
                    'enqueue_delay': (req.query.enqueue_time - req.query.created_time) / 1e3,
                    'processing_events': [asdict(e) for e in events],
                }
                if wg_name != req.assigned_engine_name:
                    args["wg_name"] = req.assigned_engine_name
                ce = CompleteEvent(
                    name=event_name,
                    cat=event_cat,
                    pid=f'{self._rm_name} {wg_name}',
                    tid=0,
                    ts=start_ts + 1,
                    dur=dur - 1,
                    args=args,
                )
                coherent_list.append(ce)

        if coherent_sort_idx == -1:
            coherent_sort_idx = 0
        if coherent_list:
            ret.append(CoherentCompleteEvent(coherent_list, coherent_sort_idx))

        return ret

    def _make_history_wpd_coherent_seq(self, req: Request, stale_idx: int,
                                       history: StaleHistory) -> List[CoherentCompleteEvent]:
        uid = req.query.meta_info.get("uid", req.query.id)
        coherent_sort_idx = -1
        wait_delay = 0
        events = history.process_events
        events_obj = [asdict(e) for e in events]
        ret: List[CoherentCompleteEvent] = []
        coherent_list: List[CompleteEvent] = []
        his_wg_name = self._wg_name_normalizer(history.assigned_engine_name)
        for i in range(1, len(events)):
            prev_event = events[i - 1]
            this_event = events[i]
            start_ts = prev_event.ts_ms * 1e3
            dur = (this_event.ts_ms - prev_event.ts_ms) * 1e3
            extra_args = {}
            if this_event.event == ProcessEventType.PREFILL_START:
                # prefill start 之前在waiting queue里等了多久
                wait_delay = dur / 1e6
                if self.query_trace_config.renderer.no_waiting_spans:
                    continue
                event_name, event_cat = "W", "rollout-wait"
            elif this_event.event == ProcessEventType.PREFILL_DONE:
                event_name, event_cat = "P", "rollout-prefill"
                extra_args = {
                    'wait_delay': wait_delay,
                    **this_event.info,
                }
                if his_wg_name != history.assigned_engine_name:
                    extra_args["wg_name"] = history.assigned_engine_name  # noqa
                if coherent_sort_idx == -1:
                    coherent_sort_idx = len(coherent_list)  # 选P对应的位置，不能用i
            elif this_event.event == ProcessEventType.EVICTED:
                event_name, event_cat = this_event.info.get("reason", "Evict"), "rollout-evicted"
                if prev_event.event == ProcessEventType.PREFILL_DONE:
                    event_name = f"D.{colorize_step(history.start_step)}"
                extra_args = {
                    'stale_action': history.stale_action,
                    'stale_reason': history.stale_reason,
                    **this_event.info,
                }
            else:
                # 忽略其他不相关的event类型
                continue

            ce = CompleteEvent(
                name=event_name,
                cat=event_cat,
                pid=f'{self._rm_name} {his_wg_name}',
                tid=0,
                ts=start_ts + 1,
                dur=dur - 1,
                args={
                    'stale_count': stale_idx,
                    'query_id': req.query.id,
                    'uid': uid,
                    'step': history.start_step,
                    'update_count': history.update_count,
                    'release_count': history.release_count,
                    'original_input_len': req.query.original_input_len,
                    'age': (start_ts / 1e3 - req.query.created_time) / 1e3,  # 相对于query生命周期的延迟，此event-创建时间
                    'shed_delay': (history.received_time - history.last_pending_reschedule_ts) / 1e3,
                    **extra_args,
                },
            )
            coherent_list.append(ce)

            # 每evicted就commit一次coherent events，每个WPD独立一个块，不需要一整个粘一起
            # [W][P][D evicted] --> [W][P][D] ...
            if event_cat == "rollout-evicted":
                if coherent_sort_idx == -1:
                    coherent_sort_idx = 0
                coherent_list[-1].args['processing_events'] = events_obj
                ret.append(CoherentCompleteEvent(coherent_list, coherent_sort_idx))
                coherent_sort_idx = -1
                coherent_list = []

        # append 最后一个 stale event
        last_event = events[-1]
        name = None
        if last_event.event == ProcessEventType.PREFILL_DONE:
            name = f"D.{colorize_step(history.start_step)}"
        elif last_event.event in [ProcessEventType.RECEIVED, ProcessEventType.EVICTED]:
            # 最后一个事件如果是evicted的话，说明最后的状态是waiting
            if not self.query_trace_config.renderer.no_waiting_spans:
                name = f"W -> {history.stale_reason}"
        elif last_event.event == ProcessEventType.PREFILL_START:
            name = f"P -> {history.stale_reason}"
        else:
            name = f"unknown -> {history.stale_reason}"

        if name is not None:
            last_event_end_ts = last_event.ts_ms * 1e3
            stale_decode = CompleteEvent(
                name=name,
                cat='rollout-stale',
                pid=f'{self._rm_name} {his_wg_name}',
                tid=0,
                ts=last_event_end_ts + 1,
                dur=history.end_ts * 1e3 - last_event_end_ts - 1,
                args={
                    'stale_count': stale_idx,
                    'query_id': req.query.id,
                    'uid': uid,
                    'step': history.start_step,
                    'stale_action': history.stale_action,
                    'stale_reason': history.stale_reason,
                    'update_count': history.update_count,
                    'release_count': history.release_count,
                    'length_generated': history.length_generated,
                    'processing_events': events_obj,
                    'wg_name': history.assigned_engine_name if his_wg_name != history.assigned_engine_name else '',
                },
            )
            coherent_list.append(stale_decode)

        if coherent_sort_idx == -1:
            coherent_sort_idx = 0
        if coherent_list:
            ret.append(CoherentCompleteEvent(coherent_list, coherent_sort_idx))

        return ret

    def _make_trace_flow_event(self, events: List[CompleteEvent | CoherentCompleteEvent]) -> List[FlowEvent]:
        # 将一个query对象的生命周期内的event串起来
        # 这个函数跟上面那个_make_trace_event配合用，先通过waterfall tracer分配了tid之后，再调用这个构造flow
        flows = []
        for i in range(len(events) - 1):
            ev0 = events[i]
            ev1 = events[i + 1]
            from_ = (ev0.pid, ev0.tid, ev0.ts + ev0.dur - 1)
            to = (ev1.pid, ev1.tid, ev1.ts + 1)
            flow = FlowEvent(name=f'f-{ev0.args["query_id"][:10]}-{i}', cat='f', flows=[from_, to])
            flows.append(flow)

        # close the gaps inside coherent events to connect the entire flow thread
        for event in events:
            if isinstance(event, CoherentCompleteEvent):
                for i in range(len(event.events) - 1):
                    ce0 = event.events[i]
                    ce1 = event.events[i + 1]
                    from_ = (ce0.pid, ce0.tid, ce0.ts + ce0.dur - 1)
                    to_ = (ce1.pid, ce1.tid, ce1.ts + 1)
                    flow = FlowEvent(name=f"cce-{ce0.args['query_id'][:10]}-{i}", cat="stf", flows=[from_, to_])
                    flows.append(flow)

        return flows

    def _remove_garbage_from_metainfo(self, meta_info: dict):
        # pop掉没用又冗余的字段
        meta_info.get("extra_data", {}).pop("config", None)
        meta_info.get("extra_data", {}).pop("agent_env_initial_files", None)
        meta_info.get("generation_kwargs", {}).pop("plugin_config", None)
        meta_info.get("reward_model", {}).pop("ground_truth", None)
