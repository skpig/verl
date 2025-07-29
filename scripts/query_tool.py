import json
import os
import signal
import socket
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from dataclasses import asdict
from typing import Optional, Any, List
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import box

import yaml
import ray
import argparse

SOCKET_PATH = os.environ.get('SOCKET_PATH', "/tmp/query_tool.sock")


# flow-style list wrapper
class FlowStyleList(list):
    pass


def represent_flow_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(FlowStyleList, represent_flow_list)


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


def _render_progress_lines(stats: List[dict]) -> str:
    lines = []
    now = time.time()
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
        line = (f"Step {stat['step']}: {bar} | "
                f"done {stat['finished']} / {stat['total']} | "
                f"{stat['token_throughput']:.1f} TPS | "
                f"running {stat['running_queries']} pending {stat['pending_queries']} | "
                f"old {oldest} LRU {least_recent_update} new {latest} (sec ago) | "
                f"Engine: active {stat['active_engines']}")
        lines.append(line)
    lines.append("")
    lines.append("Notes:")
    lines.append("  old: the earliest query in the running queue")
    lines.append("  LRU: least recent updated: the most staled query in the running queue")
    lines.append("  new: the latest query in the running queue")
    lines.append("")
    return "\n".join(lines)


def list_running_queries_str(request_manager):
    from alpha_seed.workers.streaming_service.rollout_request_manager_diagnosis import RequestDigest
    running_query_digest: List[RequestDigest] = ray.get(request_manager.get_inflight_query_digest.remote())

    if not running_query_digest:
        return "No inflight queries found."

    # Create rich table (3-line table style: no vertical lines)
    table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, width=200)
    table.add_column("Query ID", style="cyan", no_wrap=True, width=26)
    table.add_column("Engine Name", style="green", no_wrap=True, width=26)
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
    running_query_digest.sort(key=lambda row: row.assigned_at)

    # 开始渲染
    for digest in running_query_digest:
        assigned_time = format_relative_time(digest.assigned_at)
        updated_time = format_relative_time(digest.updated_at)

        # Use original content without truncation - let rich handle overflow
        table.add_row(digest.query_id, digest.assigned_engine_name or '-', str(digest.input_length),
                      str(digest.output_length), str(digest.aborted_count), str(digest.stale_count), assigned_time,
                      updated_time)

    # Render table to string with wide console
    console = Console(width=200)
    with console.capture() as capture:
        console.print(table)
        console.print(f"\nTotal inflight queries: {len(running_query_digest)}")

    return capture.get()


def list_finished_queries_str(request_manager):
    finished_query_ids = ray.get(request_manager.get_finished_query_ids.remote())
    output = _print_list(finished_query_ids)
    output += [f"total finished queries: {len(finished_query_ids)}"]
    return '\n'.join(output)


def get_query_details_str(request_manager, query_id):
    from alpha_seed.workers.streaming_service.rollout_request_manager import Request
    req: Optional[Request] = ray.get(request_manager.get_by_id.remote(query_id))
    if req:
        obj = asdict(req)
        obj = compact_list_fields(obj)
        yaml_str = yaml.dump(obj, sort_keys=False, allow_unicode=True, default_flow_style=False)
        return f"---\n{yaml_str}\n"
    else:
        return f"Query with ID {query_id} not found."


def evict_query(request_manager, query_id):
    from alpha_seed.workers.streaming_service.rollout_request_manager import Request
    req: Optional[Request] = ray.get(request_manager.get_by_id.remote(query_id))
    if not req:
        return f"Query with ID {query_id} not found."

    engine_id = req.assigned_engine_id
    if engine_id is None:
        return f"Query({query_id}) has not been assigned."

    released_query_ids = ray.get(
        request_manager.release_by_ids.remote([query_id], engine_id, "manually trigger by query_tool"))
    if released_query_ids:
        return f"Queries({released_query_ids}) has been evicted from engine({engine_id})."


def list_all_pools_str():
    from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManagerRegisterCenter
    rmrc: RequestManagerRegisterCenter = ray.get_actor('RequestManagerRegisterCenter')  # noqa
    pools = ray.get(rmrc.get_all_names.remote())
    output = ['Available pools:']
    output += _print_list(pools)
    output += [f'total pools: {len(pools)}']
    return '\n'.join(output)


def get_statistics_str(request_manager):
    from alpha_seed.workers.streaming_service.rollout_request_manager_diagnosis import FinishedEventStats
    throughput = ray.get(request_manager.get_estimated_throughput.remote()).values() or [0]
    concurrency = ray.get(request_manager.get_concurrency.remote())
    concurrency_values = concurrency.values() or [0]
    finished_stats: FinishedEventStats = ray.get(request_manager.get_finished_stats.remote())

    # Create rich table for statistics display
    table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED, width=80)
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


def handle_client(conn, server):
    from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManagerRegisterCenter, RequestManager
    try:
        data = conn.recv(4096).decode()
        args = yaml.safe_load(data)

        pool = args.get("pool")
        cmd = args["command"]
        response = ""

        rm: RequestManager = RequestManagerRegisterCenter.get(pool) if pool else None

        if cmd == "list":
            response = list_running_queries_str(rm)
        elif cmd == "list-finished":
            response = list_finished_queries_str(rm)
        elif cmd == "get":
            response = get_query_details_str(rm, args["query_id"])
        elif cmd == "evict":
            response = evict_query(rm, args["query_id"])
        elif cmd == "show-stats":
            response = get_statistics_str(rm)
        elif cmd == "list-pools":
            response = list_all_pools_str()
        elif cmd == "watch-all":
            while True:
                try:
                    stat = ray.get(rm.get_progress.remote())
                    payload = json.dumps([asdict(s) for s in stat]) + '\n'
                    conn.sendall(payload.encode())
                    time.sleep(2)
                except Exception:
                    print("\n[client disconnected]\n")
                    break
        elif cmd == "stop-daemon":
            print("Stopping daemon ...")
            conn.sendall("daemon stopped".encode())
            server.close()
            os.kill(os.getpid(), signal.SIGTERM)
        else:
            response = f"Unknown command: {cmd}"

        if response:
            conn.sendall(response.encode())

    except Exception as e:
        conn.sendall(f"Error: {str(e)}".encode())
    finally:
        conn.close()


def check_server_alive() -> bool:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        try:
            client.connect(SOCKET_PATH)
        except (FileNotFoundError, ConnectionRefusedError):
            return False
        except Exception as e:
            print("another error occurred:", e, "will restart the daemon")
            return False
    return True


@ray.remote
def start_server():
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(SOCKET_PATH)
    except OSError:
        # Socket file already exists
        print("Daemon already running?", file=sys.stderr)
        sys.exit(1)

    server.listen()

    print("Daemon started and listening at", SOCKET_PATH)

    try:
        while True:
            conn, _ = server.accept()
            threading.Thread(target=handle_client, args=(conn, server), daemon=True).start()
    finally:
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)


def send_to_daemon(cmd, args_dict):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        try:
            client.connect(SOCKET_PATH)
        except FileNotFoundError:
            print("❌ Daemon is not running. Please check the daemon job by `ray list jobs`")
            sys.exit(1)

        # request
        client.sendall(yaml.dump(args_dict).encode())

        # response
        try:
            if cmd in ['watch-all']:
                # watch 类的用console实时刷新画面
                console = Console()
                with Live(auto_refresh=True, console=console) as live:
                    buffer = ""
                    while True:
                        chunk = client.recv(65536)
                        if not chunk:
                            break

                        buffer += chunk.decode()

                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if not line.strip():
                                continue
                            try:
                                stats = json.loads(line)
                                if stats:
                                    live.update(_render_progress_lines(stats))  # 你已有的渲染函数
                                else:
                                    live.update("(no running steps, please hold on ...)")
                            except Exception as e:
                                console.print(f"[red]Parse error:[/red] {e}")
            else:
                # 非watch类的直接打印
                response_chunks = []
                while True:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    response_chunks.append(chunk)
                response = b''.join(response_chunks).decode()
                print(response)
        except KeyboardInterrupt:
            print("\nStopped by user.")


def main():
    parser = argparse.ArgumentParser(description='Query Tool for RequestManager')
    parser.add_argument('--pool', help='Request pool name')

    subparsers = parser.add_subparsers(dest='command', required=True)

    daemon_cmd = subparsers.add_parser('daemon', help='start the daemon')
    daemon_cmd.add_argument('--namespace', default='alphaseed', help='ray cluster namespace')
    subparsers.add_parser('stop-daemon', help='start the daemon')
    subparsers.add_parser('check-daemon', help='check the daemon liveness')

    # list-pools (does not require pool)
    subparsers.add_parser('list-pools', help='List all available pools')

    # list running queries (requires --pool)
    subparsers.add_parser('list', help='List running queries')

    # list finished queries
    subparsers.add_parser('list-finished', help='List finished queries')

    # watch queries
    subparsers.add_parser('watch-all', help='watch query processing progress')

    # get query details
    get_parser = subparsers.add_parser('get', help='Get details of a query')
    get_parser.add_argument('query_id', help='ID of the query')

    # evict query
    evict = subparsers.add_parser('evict', help='Evict a query from engine and put it back to the request pool')
    evict.add_argument('query_id', help='ID of the query')

    # show stats
    subparsers.add_parser('show-stats', help='Show statistics')

    args = parser.parse_args()
    if args.command == 'daemon':
        from alpha_seed.utils.server_client import is_local_ray_instance
        if ray.is_initialized() is False:
            ray.init(namespace=args.namespace)
        if is_local_ray_instance():
            ray.get(start_server.remote())
        else:
            ray.get(start_server.options(num_cpus=0, resources={'head': 1}).remote())
    elif args.command == 'check-daemon':
        if check_server_alive():
            exit(0)
        else:
            exit(1)
    else:
        send_to_daemon(args.command, vars(args))


if __name__ == "__main__":
    main()
