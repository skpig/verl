import time
from typing import Any, List, Dict

import yaml


class FlowStyleList(list):
    pass


def represent_flow_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


def represent_multiline_str(dumper, data):
    """Custom representer for multiline strings to use literal block style (|)"""
    if '\n' in data and len(data) > 50:  # Only for strings with newlines and longer than 50 chars
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


yaml.add_representer(FlowStyleList, represent_flow_list)
yaml.add_representer(str, represent_multiline_str)


def compact_list_fields(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: compact_list_fields(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        if any(isinstance(x, int) for x in obj) or any(isinstance(x, float) for x in obj):
            return FlowStyleList(obj)
        else:
            return [compact_list_fields(v) for v in obj]
    else:
        return obj


def _print_list(l) -> List[str]:
    output = ['-' * 80]
    output += [f' {v}' for v in l]
    output += ['-' * 80]
    return output


def _render_progress_bar(done: int, total: int, width=40):
    if total == 0:
        return "[N/A]"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(width * ratio)
    empty = width - filled
    bar = '█' * filled + '-' * empty
    percent = f"{ratio * 100:.1f}%"
    return f"[{bar}] {percent}"


def render_rollout_progress(stats: List[dict], task_complete_stats: Dict[str, dict]) -> str:
    lines = []
    now = time.time()
    lines.append("")
    for stat in stats:
        # 处理时间戳为0的情况（表示没有相关查询）
        if stat["oldest_query_time"] == 0:
            oldest = "N/A"
        else:
            oldest = int(now - stat['oldest_query_time'])

        if stat["latest_query_time"] == 0:
            latest = "N/A"
        else:
            latest = int(now - stat['latest_query_time'])

        if stat["oldest_updated_time"] == 0:
            least_recent_update = "N/A"
        else:
            least_recent_update = int(now - stat['oldest_updated_time'])

        bar = _render_progress_bar(stat['finished'], stat['total'], width=40)
        step_completion = task_complete_stats.get(str(stat['step']), {"running": 0, "completed": 0})
        step_total_tasks = step_completion["running"] + step_completion["completed"]
        line = (f"{stat['pool_name']:13s} | "
                f"Step {stat['step']}: {bar} | "
                f"{step_completion['completed']} / {step_total_tasks} | "
                f"P/D {stat['prefill_throughput']:.0f}/{stat['token_throughput']:.1f} TPS | "
                f"assigned {stat['running_queries']} pending {stat['pending_queries']} done {stat['finished']} | "
                f"old {oldest} LRU {least_recent_update} new {latest} (sec ago) | "
                f"Engine: active {stat['active_engines']}")
        lines.append(line)
    lines.append("")
    lines.append("Notes:")
    lines.append("  assigned: LLM queries assigned to engine")
    lines.append("  pending: LLM queries pending in request pool")
    lines.append("  old: the earliest query in the running queue")
    lines.append("  LRU: least recent updated: the most staled query in the running queue")
    lines.append("  new: the latest query in the running queue")
    lines.append("")
    return "\n".join(lines)
