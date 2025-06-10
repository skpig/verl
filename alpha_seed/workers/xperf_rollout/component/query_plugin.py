from typing import *
import asyncio
from dataclasses import dataclass, field
from transformers import AutoTokenizer
import torch.distributed as dist
from alpha_seed.workers.agents.envs import BaseEnv, create_agent_envs_from_str

# from alpha_seed.workers.agents.plugins.tp_plugin_manager import get_tp_plugin_manager, TP_PluginManager, WrappedFuture
from alpha_seed.workers.agents.plugins.plugin_manager import (
    PluginManager,
    get_plugin_manager,
    PluginCallReq,
    PluginCallResp,
)
from enum import Enum
import pickle
import dill
import base64
import concurrent.futures


class TokenRole(Enum):
    Assistant = 0
    Tool = 1


@dataclass
class TokenRange:

    start: int
    end: int
    role: TokenRole

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def __post_init__(self):
        self.sanity_check()

    def sanity_check(self):
        assert isinstance(self.start, int) and isinstance(self.end, int)
        assert isinstance(self.role, TokenRole)
        assert 0 <= self.start <= self.end


def _add_range_to_range_list(range_list: List[TokenRange], new_range: TokenRange):
    new_range.sanity_check()
    if len(range_list) == 0:
        range_list.append(new_range)
    else:
        last_range = range_list[-1]
        if last_range.role != new_range.role:
            range_list.append(new_range)
        else:
            assert new_range.start > last_range.end
            if last_range.end + 1 == new_range.start:
                # extend old range
                last_range.end = new_range.end
                last_range.sanity_check()
            else:
                range_list.append(new_range)


def _range_intersect(x: TokenRange, y: TokenRange) -> Union[None, TokenRange]:
    if x.role != y.role:
        return None
    start = max(x.start, y.start)
    end = min(x.end, y.end)
    if start <= end:
        return TokenRange(start=start, end=end, role=x.role)
    return None


def _range_list_intersect(ranges_a: List[TokenRange], ranges_b: List[TokenRange]):
    ranges_a = sorted(ranges_a)
    ranges_b = sorted(ranges_b)
    i, j = 0, 0

    result = []
    while i < len(ranges_a) and j < len(ranges_b):
        range_a, range_b = ranges_a[i], ranges_b[j]
        intersected = _range_intersect(range_a, range_b)
        if intersected is not None:
            _add_range_to_range_list(result, intersected)

        if range_a.end <= range_b.end:
            i += 1
        else:
            j += 1

    return result


@dataclass
class EnvStates:
    finished: bool = False
    reward: float = -1.0
    metrics: Dict = field(default_factory=dict)


class WrappedFuture:
    """Requires batch_sync_tp_plugin_queries to manually set result"""

    def __init__(self, inner_future: Union[concurrent.futures.Future, None]):
        self.inner_future = inner_future
        self._done = False
        self._result = None

    def done(self) -> bool:
        return self._done

    def result(self) -> Any:
        assert self._done, "get result while pending"
        return self._result

    def set_result(self, result):
        if self.inner_future is not None:
            assert self.inner_future.done(), "set_result while inner_future is still running"
            self.inner_future = None
        assert not self._done, "set_result when already done"
        self._done = True
        self._result = result


class DetachedEnv:

    def __init__(self, state_dict: Dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict


@dataclass
class QueryPlugin:

    def __init__(self, config: Dict):
        self.config: Dict = config
        assert self.config["enable"]
        self.call_round: int = 0
        self.plugin_metrics: Dict[str, Union[List, int]] = dict()
        self.output_ranges: List[TokenRange] = []
        self.plugin_match_state = None
        self.envs: List[BaseEnv] = []
        self.env_states: List[EnvStates] = []  # only set when is_finished
        self.all_env_finished: bool = False  # need to sync tp
        self.pending_call_reqs: List[PluginCallReq] = []
        self.futures: List[WrappedFuture] = []

    def attach_session(self, session, query):
        from alpha_seed.workers.xperf_rollout.component.query import Query
        extra_data = query.meta_info.get('extra_data', {})
        env_strs = extra_data.get('agent_env', [])

        self._query: Query = query
        self.tokenizer = session.tokenizer
        self.tp_group = session.tp_group
        self.plugin_manager: PluginManager = get_plugin_manager(self.config, tokenizer=self.tokenizer)
        self.plugin_match_state = self.plugin_manager.get_match_state()
        if self.true_call:
            self.envs = create_agent_envs_from_str(env_strs, tokenizer=self.tokenizer)

    def detach(self):
        if self._query is None:
            # already detached
            return
        is_finished = self._query.is_finished
        self._query = None
        self.tokenizer = None
        self.tp_group = None
        self.plugin_manager = None
        self.futures.clear()
        if is_finished:
            self.env_states = [
                EnvStates(finished=env.finished, reward=env.reward, metrics=env.metrics) for env in self.envs
            ]
            self.envs = []
        else:
            self.envs = [DetachedEnv(state_dict=env.state_dict()) for env in self.envs]

    @property
    def pause_condition(self) -> str:
        return self.config["pause_condition"]

    @property
    def max_round(self) -> int:
        return self.config["max_round"]

    @property
    def edit_history(self) -> str:
        return self.config["edit_history"]

    @property
    def result_apply_chat_template(self) -> bool:
        return self.config["pause_condition"] == 'on_eos'

    @property
    def timeout(self) -> float:
        return float(self.config["timeout"])

    @property
    def is_exec_sequential(self) -> bool:
        return self.config['execution_mode'] == 'sequential'

    @property
    def true_call(self) -> bool:
        """For current rank, actually trigger plugin call, or wait for rank 0 results"""
        return (self.tp_group is None) or (self.tp_group.rank() == 0)

    def record_model_token(self, token_id: int):
        query = self._query
        # maintain token range
        cur_idx = (len(query.input_ids) + len(query.new_token_ids) - query.original_input_len - 1)
        _add_range_to_range_list(
            self.output_ranges,
            TokenRange(start=cur_idx, end=cur_idx, role=TokenRole.Assistant),
        )

        # try trigger plugin call
        if (self.max_round is not None) and self.call_round >= self.max_round:
            # skip if reach max round
            return
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        call_reqs = self.plugin_manager.add_token_match(token, state=self.plugin_match_state)
        if len(call_reqs) > 0:
            self.pending_call_reqs.extend(call_reqs)

    def trigger_plugin_call(self):
        for call_req in self.pending_call_reqs:
            if self.true_call:
                dep_fut = None if (len(self.futures) == 0 or
                                   not self.is_exec_sequential) else self.futures[-1].inner_future

                inner_fut = self.plugin_manager.async_call(call_req=call_req,
                                                           envs=self.envs,
                                                           timeout=self.timeout,
                                                           deps=None if dep_fut is None else [dep_fut])
            else:
                inner_fut = None
            self.futures.append(WrappedFuture(inner_future=inner_fut))
        self.pending_call_reqs.clear()

    def meet_pause_condition(self) -> bool:
        """Check if query need to be paused or is already paused"""
        if len(self.pending_call_reqs) == 0:
            # check if already in paused condition (i.e. have pending tool calls)
            return len(self.futures) > 0
        # has pending call_str_dict but haven't triggered
        if self.pause_condition == "on_trigger":
            return True
        elif self.pause_condition == "on_eos":
            if len(self._query.new_token_ids) > 0:
                last_token = self._query.new_token_ids[-1]
            else:
                last_token = self._query.input_ids[-1]
            return last_token == self.tokenizer.eos_token_id
        else:
            raise ValueError(f"unsupported pause_condition: {self.pause_condition}")

    def all_plugin_call_done(self) -> bool:
        return all([fut.done() for fut in self.futures])

    def do_edit_history(self):
        # TODO: implement
        assert self.edit_history is None, f"{self.edit_history} not supported yet"

    def try_resume_from_paused(self):
        """If all plugin call finished, append plugin token ids to response.
        Variables need to be maintained:
            - input_ids
            - new_token_log_probs, probs_gt_threshold_num, probs_lt_threshold_num
            - output_ranges
        """
        if not self.all_plugin_call_done():
            return
        resps: List[PluginCallResp] = [fut.result() for fut in self.futures]
        self.futures.clear()
        self.call_round += 1
        results_str_list = []

        # format results to str
        for resp in resps:
            if resp is None:
                continue
            plugin_resp = resp.plugin_resp
            results_str_list.append(plugin_resp.output)
            self._update_plugin_metrics(resp.metrics)

        results_str = "\n".join(results_str_list)
        if self.result_apply_chat_template:
            # chat_template = "{% for message in messages %}{% set role = message['role'] %}{{  '\n' + role + '\n' + message['content'] | trim + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant\n'}}{% endif %}"
            chat = [{"role": "user", "content": results_str}]
            has_new_round = not self.all_env_finished
            results_str = self.tokenizer.apply_chat_template(chat, add_generation_prompt=has_new_round, tokenize=False)

        if len(results_str) == 0:
            # skip if empty result
            return
        plugin_tokens_ids = self.tokenizer(results_str, padding=False, return_tensors="pt",
                                           add_special_tokens=False).input_ids.tolist()[0]
        token_len = len(plugin_tokens_ids)
        query = self._query
        query.to_context_phase()
        cur_idx = len(query.input_ids) - query.original_input_len
        query.input_ids.extend(plugin_tokens_ids)
        query.accepted_len.extend([-1] * token_len)
        query.new_token_log_probs.extend([0.0] * token_len)
        query.probs_gt_threshold_num.extend([0] * token_len)
        query.probs_lt_threshold_sum.extend([0.0] * token_len)
        new_range = TokenRange(start=cur_idx, end=cur_idx + token_len - 1, role=TokenRole.Tool)
        _add_range_to_range_list(self.output_ranges, new_range)
        self.do_edit_history()

    @property
    def env_state_b64(self) -> str:
        return base64.b64encode(dill.dumps(self.env_states)).decode('utf-8')

    @property
    def model_output_mask(self) -> List[bool]:
        prev_end = -1
        ret = []
        for out_range in self.output_ranges:
            assert (out_range.start == prev_end + 1), f"{out_range.start} != {prev_end} + 1"
            prev_end = out_range.end
            ret.extend([True if out_range.role == TokenRole.Assistant else False] * out_range.length)
        return ret

    def _update_plugin_metrics(self, metrics: Dict):
        for key, val in metrics.items():
            if key not in self.plugin_metrics:
                self.plugin_metrics[key] = val
            else:
                self.plugin_metrics[key] += val

    @property
    def metrics(self) -> Dict:
        ret = dict()
        ret.update(self.plugin_metrics)
        for env_state in self.env_states:
            metrics = env_state.metrics
            ret.update(metrics)
        return ret

    def get_resume_state(self) -> Dict:
        """states that should be consistent between on-policy and off-policy steps"""
        assert not self.meet_pause_condition(), "serialize while query is paused"
        state = dict()
        state["call_round"] = self.call_round
        state["plugin_match_state"] = self.plugin_match_state
        state["env_state_dicts"] = [env.state_dict() for env in self.envs]
        state['pending_call_reqs'] = self.pending_call_reqs
        return state

    def set_resume_state(self, state: Dict):
        self.call_round = state["call_round"]
        self.plugin_match_state = state["plugin_match_state"]
        if self.true_call:
            assert len(self.envs) == (env_state_len :=
                                      len(state['env_state_dicts'])), f"{len(self.envs)} != {env_state_len}"
            for env, state_dict in zip(self.envs, state['env_state_dicts']):
                env.load_state_dict(state_dict)
        self.pending_call_reqs = state["pending_call_reqs"]


def batch_sync_tp_plugin_queries(queries: List[QueryPlugin], tp_group: dist.ProcessGroup):
    """Batch synchronizing plugin call results and env states of plugin queries"""
    if len(queries) == 0:
        return
    assert all([tp_group == query.tp_group for query in queries])

    results = []
    true_call = (tp_group is None) or (tp_group.rank() == 0)
    for i, query in enumerate(queries):
        if not query.meet_pause_condition():
            continue
        assert query.true_call == true_call
        if true_call:
            if not all([fut.inner_future is None or fut.inner_future.done() for fut in query.futures]):
                continue

            def get_result(fut: WrappedFuture):
                return fut.inner_future.result() if fut.inner_future is not None else fut.result()

            all_env_finished = all([env.finished for env in query.envs])
            plugin_results = [get_result(fut) for fut in query.futures]
            results.append((i, plugin_results, all_env_finished))

    if tp_group is not None and tp_group.size() > 1:
        tp_src_rank = dist.get_global_rank(tp_group, group_rank=0)
        to_broadcast = [results]
        dist.broadcast_object_list(to_broadcast, src=tp_src_rank, group=tp_group)
        results = to_broadcast[0]
    # update synchronized results to query
    for i, plugin_results, all_env_finished in results:
        query = queries[i]
        query.all_env_finished = all_env_finished
        for res, fut in zip(plugin_results, query.futures):
            if not fut.done():
                fut.set_result(res)
