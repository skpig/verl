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
    for stat in stats:
        bar = _render_progress_bar(stat['finished'], stat['total'], width=40)
        line = (f"Step {stat['step']}: {bar} | "
                f"done {stat['finished']} / {stat['total']} | "
                f"{stat['token_throughput']:.1f} TPS")
        lines.append(line)
    return "\n".join(lines)


def list_running_queries_str(request_manager):
    running_query_ids = ray.get(request_manager.get_inflight_query_ids.remote())
    output = _print_list(running_query_ids)
    output += [f"total inflight queries: {len(running_query_ids)}"]
    return '\n'.join(output)


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


def list_all_pools_str():
    from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManagerRegisterCenter
    rmrc: RequestManagerRegisterCenter = ray.get_actor('RequestManagerRegisterCenter')  # noqa
    pools = ray.get(rmrc.get_all_names.remote())
    output = ['Available pools:']
    output += _print_list(pools)
    output += [f'total pools: {len(pools)}']
    return '\n'.join(output)


def get_statistics_str(request_manager):
    throughput = ray.get(request_manager.get_estimated_throughput.remote())
    concurrency = ray.get(request_manager.get_concurrency.remote())
    return f"\nEstimated Throughput: {throughput}\nConcurrency: {concurrency}\n"


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


def start_server():
    if ray.is_initialized() is False:
        ray.init(namespace="alphaseed")

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

    subparsers.add_parser('daemon', help='start the daemon')
    subparsers.add_parser('stop-daemon', help='start the daemon')

    # list-pools (does not require pool)
    subparsers.add_parser('list-pools', help='List all available pools')

    # list running queries (requires --pool)
    list_parser = subparsers.add_parser('list', help='List running queries')

    # list finished queries
    finished_parser = subparsers.add_parser('list-finished', help='List finished queries')

    # watch queries
    finished_parser = subparsers.add_parser('watch-all', help='watch query processing progress')

    # get query details
    get_parser = subparsers.add_parser('get', help='Get details of a query')
    get_parser.add_argument('query_id', help='ID of the query')

    # show stats
    stats_parser = subparsers.add_parser('show-stats', help='Show statistics')

    args = parser.parse_args()
    if args.command == 'daemon':
        start_server()
    else:
        send_to_daemon(args.command, vars(args))


if __name__ == "__main__":
    main()
