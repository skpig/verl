import os
import requests
import inspect
from typing import Tuple, Union, List, Literal
from packaging.version import Version

_USE_CUDA_TIMER = False


def version_checker():
    NDTIMELINE_BASE_VERSION = "2.2.5"
    NDTIMELINE_HIGH_VERSION = "3.0.0"
    try:
        from bytedance.ndtimeline import __version__
        if Version(__version__) < Version(NDTIMELINE_BASE_VERSION) or Version(__version__) >= Version(
                NDTIMELINE_HIGH_VERSION):
            raise RuntimeError(
                f"bytedance.ndtimeline's version should be >={NDTIMELINE_BASE_VERSION} <{NDTIMELINE_HIGH_VERSION},"
                f"but {__version__} found, set use_cuda_timer=False in config file to disable it or install bytedance.ndtimeline properly"
            )
    except ImportError:
        raise RuntimeError(
            f"bytedance.ndtimeline's version should be >={NDTIMELINE_BASE_VERSION} <{NDTIMELINE_HIGH_VERSION}")


def get_cuda_timer_hires_persist_predicate():
    CUDA_TIMER_HIRES_INTERVAL_ARGS = os.getenv("CUDA_TIMER_HIRES_RECORD_ARGS", "1/5")
    last_for, record_every = CUDA_TIMER_HIRES_INTERVAL_ARGS.strip().split("/")
    record_every = int(record_every)
    last_for = int(last_for)
    assert (last_for
            <= record_every), "CUDA_TIMER_HIRES_RECORD_ARGS last_for should not be greater equal than record_every"

    def pred(iteration):
        # iteration starts from 1
        iteration -= 1
        # 对齐到最近的record_every整倍数
        start_iter = iteration // record_every * record_every
        return start_iter <= iteration < start_iter + last_for

    return pred


enable_by_global_step = get_cuda_timer_hires_persist_predicate()

# CUDA_TIMER_HIRES_RECORD_ARGS == 1/10
# 0 True
# [1, 10) False
# 10 True
# [10, 20) False
# ...


def use_cuda_timer():
    global _USE_CUDA_TIMER
    return _USE_CUDA_TIMER


def set_cuda_timer_option(turn_on):
    global _USE_CUDA_TIMER
    if turn_on:
        version_checker()
        # use warning level log in alpha seed
        if "NDTIMELINE_LOG_LEVEL" not in os.environ:
            os.environ["NDTIMELINE_LOG_LEVEL"] = "WARNING"
    _USE_CUDA_TIMER = turn_on


def get_all_actor_functions(instance: "verl.single_controller.base.worker.WorkerHelper"):
    class_name = instance._get_ray_actor_cls_name()
    method_prefix = instance._get_ray_method_prefix()
    method_names = [name for name, _ in inspect.getmembers(instance, predicate=inspect.ismethod)]
    method_names = [name for name in method_names if name != "do_ndtimeline_action"]  # ignore this function
    method_names = [f'{class_name}.{method_prefix}{name}' for name in method_names if not name.startswith("__")]
    all_names = method_names.copy()
    for name in method_names:
        all_names.extend([f'{name}::deserialize_arguments', f'{name}::execute', f'{name}::store_outputs'])
    return all_names


def set_global_step(step):
    if not use_cuda_timer():
        return
    import bytedance.ndtimeline as nd
    nd.set_global_step(step)


def inc_step(step: int = 1):
    if not use_cuda_timer():
        return
    import bytedance.ndtimeline as nd
    nd.inc_step(step)


def require_flush(global_step):
    if not use_cuda_timer():
        return False
    this_step_enable = enable_by_global_step(global_step)
    next_step_enable = enable_by_global_step(global_step + 1)
    # 1. if this_step_enable is False and next_step_enable is True:
    #    flush is required to enable timers
    # 2. if this_step_enable is True and next_step_enable is False:
    #    flush is required to disable timers and flush metrics of this step
    # 3. if this_step_enable is True and next_step_enable is True:
    #    flush is required to flush metrics of this step
    return this_step_enable or next_step_enable


def flush():
    if not use_cuda_timer():
        return
    import bytedance.ndtimeline as nd
    if nd.NDTimerManagerSingleton.is_initialized():
        global_step = nd.NDTimerManagerSingleton().global_step
        next_step_enabled = enable_by_global_step(global_step + 1)
        nd.flush(next_iter_enabled=next_step_enabled)


def init_ndtimers(mesh_shape: Union[Tuple[int, int], Tuple[int]], ray_class_instance: object):
    if not use_cuda_timer():
        return
    version_checker()
    import bytedance.ndtimeline as nd
    # wangchenyuan.99: deliberately not compatable with ray in lower verison
    import ray
    actor_name = ray.get_runtime_context().get_actor_name()
    ray_timer_names = get_all_actor_functions(ray_class_instance)
    if not nd.NDTimerManagerSingleton.is_initialized():
        nd.init_ndtimers(mode="fsdp",
                         mesh_shape=mesh_shape,
                         enable_streamer=True,
                         report_to_merlin=False,
                         actor_name=actor_name,
                         ray_timer_names=ray_timer_names)
        print("ndtimeline initialized")
    else:
        extend_timers(ray_timer_names)


def extend_timers(names: List[str]):
    if not use_cuda_timer():
        return
    import bytedance.ndtimeline as nd
    assert nd.NDTimerManagerSingleton.is_initialized()
    nd.extend_timers(names)


def do_ndtimeline_action(action: Literal["flush", "inc_step", "set_global_step", "flush_and_inc"], *args, **kwargs):
    if not use_cuda_timer():
        return
    import bytedance.ndtimeline as nd
    if action == "flush":
        flush()
    elif action == "inc_step":
        nd.inc_step()
    elif action == "set_global_step":
        nd.set_global_step(kwargs["global_step"])
    elif action == "flush_and_inc":
        flush()
        nd.inc_step()
    else:
        raise ValueError(f"Unknown action {action}")


def report_topo(all_meta):
    import os
    if os.getenv("ARNOLD_REGION") is None:
        print("ARNOLD_REGION not set, skipped report topo meta to megavision")
        return
    if os.getenv("ARNOLD_REGION") == "CN":
        base_url = "https://arnold-x-keeper.byted.org"
    else:
        base_url = "https://arnold-x-keeper-sg.byted.org"
    url = base_url + "/xray-perf/api/v1/topology/topology"
    import ray
    ray_job_id = str(ray.get_runtime_context().get_job_id())
    run_id = int(os.getenv("ROBUST_RUN_ID", 0))
    trial_id = int(os.getenv("ARNOLD_TRIAL_ID", 0))
    merlin_id = os.getenv("MERLIN_JOB_ID", "unknown_merlin_job")
    headers = {"Content-Type": "application/json"}
    body = {
        "trial_id": trial_id,
        "run_id": run_id,
        "role_id": 0,  # ignore
        "merlin_id": merlin_id,
        "ray_job_id": ray_job_id,
        "framework": "alphaseed",
        "alphaseed_meta": {
            "num_roles": len(all_meta),
            "data": all_meta,
        }
    }
    params = {
        "run_id": run_id,
        "trial_id": trial_id,
        "menlin_id": merlin_id,
        "ray_job_id": ray_job_id,
        "parse": 1,
    }
    for _ in range(2):
        try:
            resp = requests.post(url, headers=headers, json=body, params=params)
            resp.raise_for_status()
            return
        except Exception as e:
            print(f"fail to upload meta info to megavision with {body}: {e}")
