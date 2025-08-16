import json
import os
import signal
import socket
import sys
import threading
import time
import warnings
from functools import reduce

from alpha_seed.workers.streaming_service.rollout_request_manager import get_all_request_manager_actors_with_names
from scripts.query_tool.agent_monitor import render_agent_watch_data
from scripts.query_tool.rollout_monitor import list_running_queries_str, list_finished_queries_str, \
    get_query_details_str, evict_query, list_all_pools_str, get_statistics_str
from scripts.query_tool.utils import FlowStyleList, represent_flow_list, render_rollout_progress

warnings.filterwarnings("ignore", category=UserWarning)

from dataclasses import asdict, dataclass
from rich.console import Console
from rich.live import Live

import yaml
import ray
import argparse

SOCKET_PATH = os.environ.get('SOCKET_PATH', "/tmp/query_tool.sock")

# flow-style list wrapper

yaml.add_representer(FlowStyleList, represent_flow_list)


def handle_client(conn, server):
    from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManagerRegisterCenter, RequestManager, get_all_request_manager_actors
    from alpha_seed.workers.streaming_service.rollout_request_manager_diagnosis import ProgressStat
    from alpha_seed.workers.agents.metrics_collector import get_agent_metrics_collector

    @dataclass
    class RealTimeStats:
        rollout: ProgressStat
        agent: dict  # 定义见 collector.get_basic_stats

    try:
        data = conn.recv(4096).decode()
        print(f"[daemon] cmd received: {data}")
        args = json.loads(data)

        pool = args.get("pool")
        cmd = args["command"]
        response = ""

        # 用不到request pool信息的命令放这里
        if cmd == "stop-daemon":
            print("Stopping daemon ...")
            conn.sendall("daemon stopped".encode())
            server.close()
            os.kill(os.getpid(), signal.SIGTERM)
        elif cmd == "list-pools":
            response = list_all_pools_str()

        # 其他所有需要request manager的命令
        else:
            rms = get_all_request_manager_actors_with_names()

            if cmd == "list":
                response = list_running_queries_str(rms)
            elif cmd == "list-finished":
                response = list_finished_queries_str(rms)
            elif cmd == "get":
                stdout, stderr = get_query_details_str(rms, args["query_id"])
                response = json.dumps({"stdout": stdout, "stderr": stderr})
            elif cmd == "evict":
                response = evict_query(rms, args["query_id"])
            elif cmd == "show-stats":
                # 集成agent统计信息到show-stats命令
                response = ""
                for _, rm in rms:
                    response += get_statistics_str(rm) + "\n\n"
            elif cmd == "watch-all":
                collector = get_agent_metrics_collector()
                while True:
                    try:
                        rollout_stats = ray.get([rm.get_progress.remote() for _, rm in rms])
                        rollout_stats = reduce(lambda x, y: x + y,
                                               rollout_stats)  # [[rollout], [val]] => [rollout, val]
                        agent_stats = ray.get(collector.get_basic_stats.remote())
                        rt_stats = RealTimeStats(rollout_stats, agent_stats)

                        # flush frame buffer
                        payload = json.dumps(asdict(rt_stats)) + '\n'
                        conn.sendall(payload.encode())
                        time.sleep(2)
                    except Exception:
                        print("\n[client disconnected]\n")
                        break
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
def query_tool_daemon():
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
        client.sendall(json.dumps(args_dict).encode())

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
                                rt_stats = json.loads(line)

                                # rollout
                                if rt_stats['rollout']:
                                    rollout_frame_buf = render_rollout_progress(rt_stats['rollout'])
                                else:
                                    rollout_frame_buf = "\n(no running steps, please hold on ...)\n"

                                # agent
                                agent_frame_buf = render_agent_watch_data(rt_stats['agent'])

                                # flush frame
                                live.update(f'{rollout_frame_buf}\n{agent_frame_buf}')
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
                # check json format first
                if response.startswith("{"):
                    try:
                        structured_resp = json.loads(response)
                        stdout = structured_resp["stdout"]
                        stderr = structured_resp["stderr"]
                        print(stdout, file=sys.stdout, flush=True)
                        print(stderr, file=sys.stderr, flush=True)
                    except Exception as e:
                        print(response)
                else:
                    print(response)
        except KeyboardInterrupt:
            print("\nStopped by user.")


def main():
    parser = argparse.ArgumentParser(description='Query Tool for RequestManager')
    parser.add_argument('--pool', help='[DEPRECATED] Request pool name. inputting takes no effect')

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
            ray.get(query_tool_daemon.remote())
        else:
            ray.get(query_tool_daemon.options(num_cpus=0, resources={'head': 1}).remote())
    elif args.command == 'check-daemon':
        if check_server_alive():
            exit(0)
        else:
            exit(1)
    else:
        send_to_daemon(args.command, vars(args))


if __name__ == "__main__":
    main()
