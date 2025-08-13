import json
import time
import traceback
from collections import defaultdict

import ray
from rich.console import Console
from rich.table import Table, box


def make_summary_table(agent_stats):
    # 创建实时更新的表格
    table = Table(title="Agent Overview", show_header=True, box=box.SIMPLE_HEAD)
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Workers", agent_stats["workers"])
    table.add_row("Tasks (Active/Pending)", agent_stats["tasks"])
    table.add_row("Completed Tasks", str(agent_stats["completed_tasks"]))
    table.add_row("Avg Throughput", f"{agent_stats['avg_throughput']:.2f} rps")
    return table


def make_agent_task_table(agent_stats):
    agent_details_table = Table(title="Agent Task Stats", show_header=True, box=box.SIMPLE_HEAD)
    agent_details_table.add_column("Agent Type")
    agent_details_table.add_column("Active")
    agent_details_table.add_column("Pending")
    agent_details_table.add_column("Complete")
    agent_details_table.add_column("LLM Calls")
    agent_details_table.add_column("Tool Calls (succ/err)")
    agent_details_table.add_column("Exec(s)")
    agent_details_table.add_column("Tool(s)")
    agent_details_table.add_column("Wait(s)")
    agent_details_table.add_column("Create(ms)")

    # 添加agent类型详细统计
    all_agent_types = agent_stats["agent_type_pending"].keys()
    for agent_type in all_agent_types:
        active = agent_stats["agent_type_active"].get(agent_type, 0)
        pending = agent_stats["agent_type_pending"].get(agent_type, 0)
        complete = agent_stats["agent_type_complete"].get(agent_type, 0)
        perf = agent_stats["agent_type_perf"].get(agent_type, {})
        tool_success = perf.get("tool_success", 0)
        tool_error = perf.get("tool_error", 0)
        llm_success = perf.get("llm_success", 0)
        agent_details_table.add_row(
            agent_type,
            f"{active}",
            f"{pending}",
            f"{complete}",
            f"{llm_success}",
            f"{tool_success}/{tool_error}",
            f"{perf.get('avg_execution_time', 0):.1f}",  # 执行时间
            f"{perf.get('avg_tool_call_time', 0):.1f}",  # tool call时间
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
    table.add_column("Time(s)")

    tool_perf = agent_stats["tool_use_perf"]
    all_tools = agent_stats["tool_use_perf"].keys()
    for tool_name in all_tools:
        perf = tool_perf[tool_name]
        table.add_row(
            tool_name,
            f"{perf.get('success_count', 0)}",
            f"{perf.get('retried_count', 0)}",
            f"{perf.get('error_count', 0)}",
            f"{perf.get('avg_time', 0):.3f}",
        )

    return table


def render_agent_watch_data(agent_stats: dict) -> str:
    """渲染agent watch数据为表格"""
    try:

        if "error" in agent_stats:
            return (f"Error: {agent_stats['error']}\n"
                    f"Traceback: {agent_stats.get('traceback')}")

        # 使用无颜色的Console来避免乱码
        console = Console(color_system=None, force_terminal=False)

        agent_summary_table = make_summary_table(agent_stats)
        agent_details_table = make_agent_task_table(agent_stats)
        agent_tool_use_table = make_tool_use_table(agent_stats)

        # 捕获所有表格输出
        with console.capture() as capture:
            console.print()
            console.print(agent_summary_table)
            console.print()
            console.print(agent_details_table)
            console.print()
            console.print(agent_tool_use_table)

        return capture.get()

    except Exception as e:
        tb = traceback.format_exc()
        return f"Error rendering data: \n{tb}\n{str(e)}"
