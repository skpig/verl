import time
import traceback
from typing import List, Tuple, Optional

import ray
import yaml
from rich.console import Console
from rich.table import Table, box

from scripts.query_tool.utils import compact_list_fields, FlowStyleList, represent_flow_list, represent_multiline_str

yaml.add_representer(FlowStyleList, represent_flow_list)
yaml.add_representer(str, represent_multiline_str)


def make_summary_table(agent_stats):
    # 创建实时更新的表格
    table = Table(title="Agent Overview", show_header=True, box=box.SIMPLE_HEAD)
    table.add_column("Metric")
    table.add_column("Value")

    if not agent_stats:
        return table
    table.add_row("Executor Class", f"{agent_stats['executor_class']}")
    table.add_row("Workers x Concurrency", f"{agent_stats['max_workers']} x {agent_stats['worker_max_concurrency']}")
    table.add_row("Tasks (Active/Pending)", f"{agent_stats['active_tasks']}/{agent_stats['pending_tasks']}")
    table.add_row("Completed Tasks", f"{agent_stats['completed_tasks']}")
    table.add_row("Avg Throughput", f"{agent_stats['avg_throughput']:.3f} rps")
    return table


def make_agent_task_table(agent_stats):
    agent_details_table = Table(title="Agent Task Stats", show_header=True, box=box.SIMPLE_HEAD)
    agent_details_table.add_column("Agent Type")
    agent_details_table.add_column("Active")
    agent_details_table.add_column("Pending")
    agent_details_table.add_column("Complete")
    agent_details_table.add_column("LLM Calls")
    agent_details_table.add_column("Tool Calls (succ/retry/MAE/err)")
    agent_details_table.add_column("Exec(s) (avg)")
    agent_details_table.add_column("Tool(s) (min/avg/max)")
    agent_details_table.add_column("Wait(s)")
    agent_details_table.add_column("Create(ms)")

    if not agent_stats:
        return agent_details_table

    # 添加agent类型详细统计
    all_agent_types = agent_stats["agent_type_pending"].keys()  # data maybe not ready
    for agent_type in all_agent_types:
        active = agent_stats["agent_type_active"].get(agent_type, 0)
        pending = agent_stats["agent_type_pending"].get(agent_type, 0)
        complete = agent_stats["agent_type_complete"].get(agent_type, 0)
        perf = agent_stats["agent_type_perf"].get(agent_type, {})
        tool_success = perf.get("tool_success", 0)
        tool_retries = perf.get("tool_retries", 0)
        tool_error = perf.get("tool_error", 0)
        tool_MAE = perf.get("tool_max_attempts_exceeds", 0)
        llm_success = perf.get("llm_success", 0)
        agent_details_table.add_row(
            agent_type,
            f"{active}",
            f"{pending}",
            f"{complete}",
            f"{llm_success}",
            f"{tool_success}/{tool_retries}/{tool_MAE}/{tool_error}",
            f"{perf.get('avg_execution_time', 0):.1f}",  # 执行时间
            f"{perf.get('min_tool_call_time', 0):.1f}/{perf.get('avg_tool_call_time', 0):.1f}/{perf.get('max_tool_call_time', 0):.1f}",  # tool call时间
            f"{perf.get('avg_waiting_time', 0):.1f}",  # 执行时间
            f"{perf.get('avg_creation_time', 0) * 1e3:.1f}"  # 创建到开始执行的时间
        )
    return agent_details_table


def make_tool_use_table(agent_stats):
    table = Table(title="Tool Calls Stats", show_header=True, box=box.SIMPLE_HEAD)
    table.add_column("Tool")
    table.add_column("Succ")
    table.add_column("Retries")
    table.add_column("Error")
    table.add_column("MAE")
    table.add_column("Time(s)")

    if not agent_stats:
        return table

    tool_perf = agent_stats["tool_use_perf"]
    all_tools = tool_perf.keys()
    for tool_name in all_tools:
        perf = tool_perf[tool_name]
        table.add_row(
            tool_name,
            f"{perf.get('success_count', 0)}",
            f"{perf.get('retried_count', 0)}",
            f"{perf.get('error_count', 0)}",
            f"{perf.get('max_attempts_exceeds', 0)}",
            f"{perf.get('avg_time', 0):.3f}",
        )

    return table


def render_agent_watch_data(agent_stats: dict,
                            collector_info: dict,
                            query_cost: float,
                            show_debug_info: bool = False) -> str:
    """渲染agent watch数据为表格"""
    try:

        if "error" in agent_stats:
            return (f"Error: {agent_stats['error']}\n"
                    f"Traceback: {agent_stats.get('traceback')}")

        agent_summary_table = make_summary_table(agent_stats)
        agent_details_table = make_agent_task_table(agent_stats)
        agent_tool_use_table = make_tool_use_table(agent_stats)

        agent_stats_notes = """
    MAE: 某次 tool call中max attempts exceeds 的次数
    Time(s): 主要表示asyncio中的调度延迟，可能比实际调用的耗时略长一点
        """

        # 捕获所有表格输出
        console = Console(color_system=None, force_terminal=False)
        with console.capture() as capture:
            console.print()
            console.print(agent_summary_table)
            console.print()
            console.print(agent_details_table)
            console.print()
            console.print(agent_tool_use_table)
            console.print()
            console.print(agent_stats_notes)

            console.print()
            console.print(f"query delay: {query_cost*1e3:.1f} ms")

            if show_debug_info:
                worker_submission_delay = collector_info["worker_submission_delay"]
                console.print("[DEBUG INFO]")
                console.print("worker metrics submission delay")
                console.print(worker_submission_delay)
                console.print("[DEBUG INFO END]")

        return capture.get()

    except Exception as e:
        tb = traceback.format_exc()
        return f"Error rendering data: \n{tb}\n{str(e)}"


def _format_timestamp(ts: float) -> str:
    """Convert timestamp to human-readable format"""
    # Extract millis from the float part
    ms = int((ts - int(ts)) * 1_000)
    return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{ms:03d}"


def _format_duration(start_ts: float, end_ts: float) -> str:
    """Format duration in human-readable format"""
    if end_ts == 0:  # Still running
        duration = time.time() - start_ts
        return f"{duration:.1f}s [green][R][/]"
    else:
        duration = end_ts - start_ts
        return f"{duration:.1f}s"


ACTION_COLORS = {
    "llm": "green",
    "tool": "dodger_blue2",
    "missing": "red",
}

ROLE_COLORS = {
    "user": "white",
    "call": "bright_yellow",
    "assistant": "green",
    "result": "dodger_blue2",
}


def _colorize(v):
    color = ACTION_COLORS.get(v) or ROLE_COLORS.get(v)
    if not color:
        return v
    return f"[{color}]{v}[/]"


def _colorize_with(role, v):
    color = ACTION_COLORS.get(role) or ROLE_COLORS.get(role)
    if not color:
        return v
    return f"[{color}]{v}[/]"


def list_agent_tasks_str(width,
                         list_all: bool,
                         task_type: Optional[str],
                         limit: int = 2000,
                         no_color: bool = False) -> str:
    from alpha_seed.workers.agents.trajectory import TrajectoryCollector, Trajectory
    try:
        traj_collector: TrajectoryCollector = ray.get_actor("TrajectoryCollector")  # noqa
        if list_all:
            trajectories: List[Trajectory] = ray.get(traj_collector.get_all.remote(limit, task_type))
        else:
            trajectories: List[Trajectory] = ray.get(traj_collector.get_running.remote(task_type))

        # Create table
        table = Table(title="Agent Tasks", show_header=True, box=box.SIMPLE_HEAD)
        table.add_column("Agent", style="white")
        table.add_column("UID", style="cyan", overflow="fold")
        table.add_column("Traj ID", style="cyan")
        table.add_column("Role", style="white")
        table.add_column("Start", style="magenta")
        table.add_column("Dur", style="yellow")
        table.add_column("Segments", style="white")
        table.add_column("Tokens", style="cyan")

        if not trajectories:
            table.add_row("", "", "", "", "", "", "No running tasks", "")
        else:
            for traj in trajectories:
                # Format segments info
                paired_segments = traj.get_paired_segments()
                uid, agent_name, traj_id = traj.agent_ident.uid, traj.agent_ident.agent_name, traj.trajectory_id
                for i, (start_seg, end_seg) in enumerate(paired_segments):
                    if start_seg is not None:
                        start_time = _format_timestamp(start_seg.start_ts)
                        end_ts = end_seg.end_ts if end_seg is not None else 0
                        duration = _format_duration(start_seg.start_ts, end_ts)

                        if i > 0:
                            uid, agent_name, traj_id = "", "", ""

                        # seg start
                        table.add_row(
                            agent_name,
                            uid,
                            f"{traj_id}",
                            _colorize(start_seg.role),
                            start_time,
                            duration,
                            _colorize_with(start_seg.role, start_seg.to_digest()),
                            f"{start_seg.num_tokens}",
                        )
                    else:
                        table.add_row(
                            agent_name,
                            uid,
                            f"{traj_id}",
                            "",
                            "",
                            "",
                            _colorize_with("missing", "(start segment is missing)"),
                            "",
                        )

                    # seg end
                    if end_seg is not None:
                        table.add_row("", "", "", _colorize(end_seg.role), "", "",
                                      _colorize_with(end_seg.role, end_seg.to_digest()), f"{end_seg.num_tokens}")

                if not paired_segments:
                    # maybe no segments in this traj
                    table.add_row(traj.agent_ident.agent_name, traj.agent_ident.uid, f"{traj.trajectory_id}", "", "",
                                  "", "(no segments)", "")

                # delimiter
                table.add_row("", "", "", "", "", "", "-" * 80, "")

        # Capture table output
        console = Console(force_terminal=not no_color, width=width, no_color=no_color)
        with console.capture() as capture:
            console.print()
            console.print(table)
            console.print()
            if table.row_count >= limit:
                console.print(f"(data may be truncated to max {limit} segments(rows))")

        return capture.get()

    except Exception as e:
        tb = traceback.format_exc()
        return f"Error getting agent tasks: \n{tb}\n{str(e)}"


def get_agent_task_str(width: int, uid: str, traj_id: Optional[int], no_color: bool = False) -> Tuple[str, str]:
    from alpha_seed.workers.agents.trajectory import TrajectoryCollector, Trajectory
    try:
        traj_collector: TrajectoryCollector = ray.get_actor("TrajectoryCollector")  # noqa
        trajectories: List[Trajectory] = ray.get(traj_collector.get.remote(uid, traj_id))

        if not trajectories:
            return "", f"Trajectory not found: uid={uid}, traj_id={traj_id}"

        # Create segments table using paired segments
        segments_table = Table(title=f"Agent: {trajectories[0].agent_ident.agent_name} | Task: {uid}",
                               show_header=True,
                               box=box.SIMPLE_HEAD)
        segments_table.add_column("Traj ID", style="cyan", width=4)
        segments_table.add_column("Turn No.", style="cyan", width=4)
        segments_table.add_column("Start", style="magenta", width=12)
        segments_table.add_column("Duration", style="yellow", width=10)
        segments_table.add_column("Role", style="green", width=10)
        segments_table.add_column("Content", style="white")
        segments_table.add_column("Tokens", style="cyan", width=10)

        for traj in trajectories:
            for i, (start_seg, end_seg) in enumerate(traj.get_paired_segments()):
                if start_seg:
                    start_time = _format_timestamp(start_seg.start_ts)
                    end_ts = end_seg.end_ts if end_seg else 0
                    duration = _format_duration(start_seg.start_ts, end_ts)
                    # Start segment row
                    segments_table.add_row(
                        str(traj.trajectory_id) if i == 0 else "",
                        str(i),
                        start_time,
                        duration,
                        _colorize(start_seg.role),
                        _colorize_with(start_seg.role, start_seg.to_content()),
                        f"{start_seg.num_tokens}",
                    )
                else:
                    # Start segment row (missing)
                    segments_table.add_row(
                        str(traj.trajectory_id) if i == 0 else "",
                        str(i),
                        "",
                        "",
                        "",
                        _colorize_with("missing", "(start segment is missing)"),
                        "",
                    )

                # End segment row (if exists)
                if end_seg:
                    segments_table.add_row(
                        "",
                        "",
                        "",
                        "",
                        _colorize(end_seg.role),
                        _colorize_with(end_seg.role, end_seg.to_content()),
                        f"{end_seg.num_tokens}",
                    )
                else:
                    # Show waiting state for incomplete segments
                    segments_table.add_row(
                        "",
                        "",
                        "",
                        "",
                        _colorize("result" if start_seg.action == "tool" else "assistant"),
                        "(waiting)",
                        "",
                    )

        # Capture output
        console = Console(force_terminal=not no_color, width=width, no_color=no_color)
        with console.capture() as capture:
            console.print()
            console.print(segments_table)

        return capture.get(), ""

    except Exception as e:
        tb = traceback.format_exc()
        return "", f"Error getting agent task detail: \n{tb}\n{str(e)}"


def get_agent_task_detail_yaml(uid: str) -> Tuple[str, str]:
    from alpha_seed.workers.agents.trajectory import TrajectoryCollector, Trajectory
    try:
        traj_collector: TrajectoryCollector = ray.get_actor("TrajectoryCollector")  # noqa
        trajectories: List[Trajectory] = ray.get(traj_collector.get.remote(uid))
    except Exception as e:
        tb = traceback.format_exc()
        return "", tb

    objs = [t.to_dict() for t in trajectories]
    objs = compact_list_fields(objs)
    yaml_str = yaml.dump(objs, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return yaml_str, ""
