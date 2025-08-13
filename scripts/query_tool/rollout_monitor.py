import time
from dataclasses import asdict
from functools import reduce
from typing import List, Optional, Tuple

import ray
import yaml
from rich.console import Console
from rich.table import Table, box

from scripts.query_tool.utils import _print_list, compact_list_fields, FlowStyleList, represent_flow_list

# 这一行很重要，一定要在这里调用才能在这个文件里生效
yaml.add_representer(FlowStyleList, represent_flow_list)


def list_running_queries_str(request_managers: list):
    from alpha_seed.workers.streaming_service.rollout_request_manager_diagnosis import RequestDigest
    all_running_queries: List[List[RequestDigest]] = ray.get(
        [rm.get_inflight_query_digest.remote() for _, rm in request_managers])
    running_query_digest_flatten = reduce(lambda a, b: a + b, all_running_queries)

    if not running_query_digest_flatten:
        return "No inflight queries found."

    # Create rich table (3-line table style: no vertical lines)
    table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, width=200)
    table.add_column("Pool", style="cyan", no_wrap=True, width=15)
    table.add_column("Query ID", style="cyan", no_wrap=True, width=26)
    table.add_column("Engine Name", style="green", no_wrap=True, width=35)
    table.add_column("Input", style="blue", justify="right", no_wrap=True)
    table.add_column("Output", style="blue", justify="right", no_wrap=True)
    table.add_column("Aborted", style="red", justify="right", no_wrap=True)
    table.add_column("Stale", style="red", justify="right", no_wrap=True)
    table.add_column("Assigned", style="yellow", no_wrap=True)
    table.add_column("Updated", style="yellow", no_wrap=True)

    # Helper function to format relative time using humanize library if available
    def format_relative_time(timestamp):
        if not timestamp:
            return 'N/A'

        import humanize
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        return humanize.naturaltime(dt)

    # 排序
    # 默认按照assigned (越旧越前)
    running_query_digest_flatten.sort(key=lambda row: row.assigned_at)

    # 开始渲染
    for digest in running_query_digest_flatten:
        assigned_time = format_relative_time(digest.assigned_at)
        updated_time = format_relative_time(digest.updated_at)

        # Use original content without truncation - let rich handle overflow
        table.add_row(digest.pool_name, digest.query_id, digest.assigned_engine_name or '-', str(digest.input_length),
                      str(digest.output_length), str(digest.aborted_count), str(digest.stale_count), assigned_time,
                      updated_time)

    # Render table to string with wide console
    console = Console(width=200)
    with console.capture() as capture:
        console.print(table)
        console.print(f"\nTotal inflight queries: {len(running_query_digest_flatten)}")

    return capture.get()


def list_finished_queries_str(request_managers: list):
    finished_query_ids = ray.get([rm.get_finished_query_ids.remote() for _, rm in request_managers])
    finished_query_ids = reduce(lambda a, b: a + b, finished_query_ids)  # flatten
    output = _print_list(finished_query_ids)
    output += [f"total finished queries: {len(finished_query_ids)}"]
    return '\n'.join(output)


def get_query_details_str(request_managers: list, query_id: str) -> Tuple[str, str]:
    from alpha_seed.workers.streaming_service.rollout_request_manager import Request
    for name, rm in request_managers:
        req: Optional[Request] = ray.get(rm.get_by_id.remote(query_id))
        if req:
            obj = asdict(req)
            obj = compact_list_fields(obj)
            yaml_str = yaml.dump(obj, sort_keys=False, allow_unicode=True, default_flow_style=False)
            meta_info = f"Query from {name}, "
            if req.finished:
                finished_ago = time.time() - req.query.finished_time / 1e3
                meta_info += f"finished {finished_ago:.0f}s ago"
            else:
                run_time = req.query.created_time / 1e3
                meta_info += f"still running for {run_time:.0f}s"
            return f"---\n{yaml_str}\n", meta_info
    return "", f"Query with ID {query_id} not found."


def evict_query(request_managers: list, query_id: str):
    from alpha_seed.workers.streaming_service.rollout_request_manager import Request
    for name, rm in request_managers:
        req: Optional[Request] = ray.get(rm.get_by_id.remote(query_id))
        if not req:
            continue

        engine_id = req.assigned_engine_id
        if engine_id is None:
            return f"Query({query_id}) in pool({name}) has not been assigned."

        released_query_ids = ray.get(rm.release_by_ids.remote([query_id], engine_id, "manually trigger by query_tool"))
        if released_query_ids:
            return f"Queries({released_query_ids}) in pool({name}) has been evicted from engine({engine_id})."

    all_names = [n for n, _ in request_managers]
    return f"Query with ID {query_id} not found in all pools({all_names})."


def list_all_pools_str():
    from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManagerRegisterCenter
    rmrc: RequestManagerRegisterCenter = ray.get_actor('RequestManagerRegisterCenter')  # noqa
    pools = ray.get(rmrc.get_all_names.remote())
    output = ['Available pools:']
    output += _print_list(pools)
    output += [f'total pools: {len(pools)}']
    return '\n'.join(output)


def get_statistics_str(request_manager) -> str:
    from alpha_seed.workers.streaming_service.rollout_request_manager_diagnosis import FinishedEventStats
    throughput = ray.get(request_manager.get_estimated_throughput.remote()).values() or [0]
    concurrency = ray.get(request_manager.get_concurrency.remote())
    concurrency_values = concurrency.values() or [0]
    finished_stats: FinishedEventStats = ray.get(request_manager.get_finished_stats.remote())

    # Create rich table for statistics display
    table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE_HEAD, width=80)
    table.add_column("Metric", style="bold white", no_wrap=True, width=25)
    table.add_column("Value", style="green", justify="right", width=20)
    table.add_column("Unit", style="dim", width=15)

    # Add throughput and concurrency rows
    table.add_row("Active engines", f"{len(concurrency)}", "")
    table.add_row("Rollout throughput(min)", f"{min(throughput)}", "TPS")
    table.add_row("Rollout throughput(max)", f"{max(throughput)}", "TPS")
    table.add_row("Rollout throughput(total)", f"{sum(throughput)}", "TPS")
    table.add_row("Rollout concurrency(min)", f"{min(concurrency_values)}", "requests")
    table.add_row("Rollout concurrency(max)", f"{max(concurrency_values)}", "requests")
    table.add_row("Rollout concurrency(total)", f"{sum(concurrency_values)}", "requests")
    table.add_row("Rollout finished events(done)", f"{finished_stats.done_count}", "requests")
    table.add_row("Rollout finished events(wait)", f"{finished_stats.waiting_count}", "requests")
    table.add_row("Rollout finished staging", f"{finished_stats.finished_staging_size}", "requests")
    table.add_row("Rollout finished total", f"{finished_stats.finished_accumulated_size}", "requests")

    # Render table to string
    console = Console(width=80)
    with console.capture() as capture:
        console.print("\n[bold yellow]📊 Request Manager Statistics[/bold yellow]")
        console.print(table)
        console.print("")

    return capture.get()
