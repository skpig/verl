import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import traceback
import warnings
from functools import reduce

from scripts.query_tool.agent_monitor import render_agent_watch_data, list_agent_tasks_str, get_agent_task_detail_yaml, \
    get_agent_task_str
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
    from alpha_seed.workers.streaming_service.rollout_request_manager import get_all_request_manager_actors_with_names
    from alpha_seed.workers.streaming_service.rollout_request_manager_diagnosis import ProgressStat
    from alpha_seed.workers.agents.metrics_collector import get_agent_metrics_collector
    from alpha_seed.workers.agents.trajectory import get_agent_trajectory_collector

    @dataclass
    class RealTimeStats:
        rollout: ProgressStat
        agent: dict  # 定义见 collector.get_basic_stats
        agent_collector_info: dict  # 见 collector.agent_collector_info
        task_complete_stats: dict  # 见 get_task_complete_stats
        query_cost: float  # 读取一轮数据花多久

    try:
        while True:
            data = conn.recv(4096).decode().strip()
            if not data:
                time.sleep(0.1)
                continue
            print(f"[daemon] cmd received: {data}")
            args = json.loads(data)
            break

        cmd = args["command"]
        console_width = args['console_width']
        response = ""

        # 用不到request pool信息的命令放这里
        if cmd == "stop-daemon":
            print("Stopping daemon ...")
            conn.sendall("daemon stopped".encode())
            server.close()
            try:
                os.remove(SOCKET_PATH)
            except FileNotFoundError:
                pass
            os.kill(os.getpid(), signal.SIGTERM)
        elif cmd == "list-pools":
            response = list_all_pools_str()

        # 其他所有需要request manager的命令
        else:
            rms = get_all_request_manager_actors_with_names()
            assert rms, "no request manager found, please check the actors of your job"

            if cmd == "list":
                step = args.get("step")
                response = list_running_queries_str(console_width, rms, step)
            elif cmd == "list-finished":
                response = list_finished_queries_str(rms)
            elif cmd == "list-tasks":
                list_all = args.get('all', False)
                task_type = args['type']  # filter by agent class name
                limit = args['limit']
                no_color = args['no_color']
                response = list_agent_tasks_str(console_width, list_all, False, task_type, limit, no_color)
            elif cmd == "list-exception-tasks":
                task_type = args['type']  # filter by agent class name
                limit = args['limit']
                no_color = args['no_color']
                response = list_agent_tasks_str(console_width, True, True, task_type, limit, no_color)
            elif cmd == "get":
                stdout, stderr = get_query_details_str(rms, args["query_id"])
                response = json.dumps({"stdout": stdout, "stderr": stderr})
            elif cmd == "get-task":
                uid = args['uid']
                format = args['format']
                traj_id = args['traj']  # filter by trajectory id
                no_color = args['no_color']
                if format == "yaml":
                    stdout, stderr = get_agent_task_detail_yaml(uid)
                elif format == "table":
                    stdout, stderr = get_agent_task_str(console_width, uid, traj_id, no_color)
                else:
                    stdout = ""
                    stderr = f"unsupported format({format}), check `scripts/query_tool.sh --help`"
                response = json.dumps({"stdout": stdout, "stderr": stderr})
            elif cmd == "evict":
                response = evict_query(rms, args["query_id"])
            elif cmd == "show-stats":
                # 集成agent统计信息到show-stats命令
                response = ""
                for _, rm in rms:
                    response += get_statistics_str(rm) + "\n\n"
            elif cmd == "dump-trace":
                # dump
                spans = []

                # task runner
                task_runner = ray.get_actor("task_runner")
                task_runner_spans = ray.get(task_runner.dump_trace_spans.remote())
                spans.extend(task_runner_spans)

                # request manager
                for _, rm in rms:
                    request_spans = ray.get(rm.dump_request_trace.remote())
                    spans.extend(request_spans)

                # rollout manager
                rollout_mgr = ray.get_actor("RolloutManager")
                rollout_spans = ray.get(rollout_mgr.dump_trace_spans.remote())
                spans.extend(rollout_spans)

                # export
                from alpha_seed.utils.profile.timeline import export_chrome_trace
                save_path = export_chrome_trace('query_trace.json.gz', spans)
                response = f"trace.json.gz saved to {save_path} with {len(spans)} spans"

                # upload
                will_not_upload = args.get("no_upload")
                if not will_not_upload:
                    time.sleep(1)  # 等flush完成，被subprocess可见
                    result = subprocess.run(
                        f"/opt/tiger/mlx_deploy/bin/mlx asset upload {save_path}",
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                    stdout = result.stdout
                    stderr = result.stderr
                    response += f"\n{stdout}\n{stderr}"
                    if result.returncode != 0:
                        response += "\n[ERROR] upload profiler trace fail. please see the log around"
            elif cmd == "dump-task-trace":
                # dump
                from alpha_seed.workers.agents.trajectory import TrajectoryCollector
                traj_collector: TrajectoryCollector = ray.get_actor("TrajectoryCollector")  # noqa
                spans = ray.get(traj_collector.dump_trajectory_trace.remote())

                # export
                from alpha_seed.utils.profile.timeline import export_chrome_trace
                save_path = export_chrome_trace('agent_task_trace.json.gz', spans)
                response = f"agent_task_trace.json.gz saved to {save_path} with {len(spans)} spans"

                # upload
                will_not_upload = args.get("no_upload")
                if not will_not_upload:
                    time.sleep(1)  # 等flush完成，被subprocess可见
                    result = subprocess.run(
                        f"/opt/tiger/mlx_deploy/bin/mlx asset upload {save_path}",
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                    stdout = result.stdout
                    stderr = result.stderr
                    response += f"\n{stdout}\n{stderr}"
                    if result.returncode != 0:
                        response += "\n[ERROR] upload profiler trace fail. please see the log around"
            elif cmd == "watch-all":
                collector = get_agent_metrics_collector()
                traj_collector = get_agent_trajectory_collector()
                while True:
                    try:
                        t0 = time.time()
                        rollout_stats = ray.get([rm.get_progress.remote() for _, rm in rms])
                        rollout_stats = reduce(lambda x, y: x + y,
                                               rollout_stats)  # [[rollout], [val]] => [rollout, val]
                        agent_stats = ray.get(collector.get_basic_stats.remote())
                        agent_collector_info = ray.get(collector.get_collector_info.remote())
                        task_complete_stats = ray.get(traj_collector.get_task_complete_stats.remote())
                        t1 = time.time()
                        rt_stats = RealTimeStats(rollout_stats, agent_stats, agent_collector_info, task_complete_stats,
                                                 t1 - t0)

                        # flush frame buffer
                        payload = json.dumps(asdict(rt_stats)) + '\n'
                        conn.sendall(payload.encode())
                        time.sleep(2)
                    except BrokenPipeError as e:
                        print(f"\n[client disconnected]\n")
                        break
                    except Exception as e:
                        print(f"\n[client disconnected] with error {e}\n")
                        traceback.print_exc()
                        break
            else:
                response = f"Unknown command: {cmd}\n"

        if response:
            conn.sendall(response.encode())

    except Exception as e:
        conn.sendall(f"Error: {str(e)}\n".encode())
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

        # attach console info
        console = Console()
        console_width = console.width
        args_dict['console_width'] = console_width
        mode = os.fstat(sys.stdout.fileno()).st_mode
        if stat.S_ISREG(mode):
            # 仅当重定向到文件时才no color
            args_dict['no_color'] = True

        # request
        client.sendall(json.dumps(args_dict).encode())

        # response
        try:
            if cmd in ['watch-all']:
                show_debug_info = args_dict.get("debug", False)

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
                                    rollout_frame_buf = render_rollout_progress(
                                        rt_stats['rollout'],
                                        rt_stats['task_complete_stats'],
                                    )
                                else:
                                    rollout_frame_buf = "\n(no running steps, please hold on ...)\n"

                                # agent
                                agent_frame_buf = render_agent_watch_data(rt_stats['agent'],
                                                                          rt_stats['agent_collector_info'],
                                                                          rt_stats['query_cost'], show_debug_info)

                                # flush frame
                                live.update(f'{rollout_frame_buf}\n{agent_frame_buf}')
                            except ValueError as e:
                                console.print(f"[red]Parse error:[/red] {e}, {line}")
                            except Exception as e:
                                console.print(f"[red]error:[/red] {e}, {line}")
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
    list_parser = subparsers.add_parser('list', help='List running queries')
    list_parser.add_argument("--step", type=int, default=None, help='filter by global step')

    # list finished queries
    subparsers.add_parser('list-finished', help='List finished queries')

    # watch queries
    watch_all_parser = subparsers.add_parser('watch-all', help='watch query processing progress')
    watch_all_parser.add_argument('--debug', action='store_true', help='show query tool internal diagnosis info')

    # get query details
    get_parser = subparsers.add_parser('get', help='Get details of a query (yaml)')
    get_parser.add_argument('query_id', help='ID of the query')

    # evict query
    evict = subparsers.add_parser('evict', help='Evict a query from engine and put it back to the request pool')
    evict.add_argument('query_id', help='ID of the query')

    # show stats
    subparsers.add_parser('show-stats', help='Show statistics')

    # dump query trace
    dump_trace_parser = subparsers.add_parser('dump-trace', help='dump query trace intermediately')
    dump_trace_parser.add_argument('--no-upload', action='store_true', help='not to upload to merlin automatically')

    # tasks related commands
    list_tasks_parser = subparsers.add_parser('list-tasks', help='List agent tasks')
    list_tasks_parser.add_argument('--all', action='store_true', help='including finished tasks(trajectories)')
    list_tasks_parser.add_argument('--type',
                                   default=None,
                                   help='agent task type, same aka agent class name '
                                   '`class XXAgent(AsyncAgent):` -> XXAgent, default to get all in the same UID')
    list_tasks_parser.add_argument('--limit', default=500, type=int, help='number of segments to list at one time')
    list_tasks_parser.add_argument('--no-color', action='store_true', help='output without coloring')

    list_exc_tasks_parser = subparsers.add_parser('list-exception-tasks',
                                                  help='List agent tasks, which has any errors during the agent loop')
    list_exc_tasks_parser.add_argument('--type',
                                       default=None,
                                       help='agent task type, same aka agent class name '
                                       '`class XXAgent(AsyncAgent):` -> XXAgent, default to get all in the same UID')
    list_exc_tasks_parser.add_argument('--limit', default=500, type=int, help='number of segments to list at one time')
    list_exc_tasks_parser.add_argument('--no-color', action='store_true', help='output without coloring')

    get_task_parser = subparsers.add_parser('get-task', help='Get details of an agent task (--format)')
    get_task_parser.add_argument('uid', help='UID of the agent task')
    get_task_parser.add_argument('--format', default='table', help='dump format, table/yaml')
    get_task_parser.add_argument('--no-color', action='store_true', help='output without coloring')
    get_task_parser.add_argument('--traj',
                                 type=int,
                                 default=None,
                                 help='trajectory_id, default to get all in the same UID')

    # dump traj trace
    dump_task_trace_parser = subparsers.add_parser('dump-task-trace', help='dump agent trace intermediately')
    dump_task_trace_parser.add_argument('--no-upload',
                                        action='store_true',
                                        help='not to upload to merlin automatically')

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
