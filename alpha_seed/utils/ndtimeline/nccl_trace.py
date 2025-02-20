import os
import logging
from enum import Enum
from packaging.version import Version

_USE_NCCL_TRACE = True


class DumpType(Enum):
    initial = "initial"


def use_nccl_trace():
    global _USE_NCCL_TRACE
    return _USE_NCCL_TRACE


# (wangchenyuan.99): no need for set, will be refactored soon
def set_nccl_trace_option():
    global _USE_NCCL_TRACE

    if os.getenv("TRAINER_DISABLE_NCCL_TRACE", "false").lower() != "true" and check_version_for_nccl_trace():
        _USE_NCCL_TRACE = True
        logging.info("nccl trace is enabled")
    else:
        _USE_NCCL_TRACE = False


def check_version_for_nccl_trace():
    NDTIMELINE_BASE_VERSION = "2.2.5"
    try:
        from bytedance.ndtimeline import EmergencyServer, FlightRecorderDumper
        from bytedance.ndtimeline import __version__
        if Version(__version__) < Version(NDTIMELINE_BASE_VERSION):
            logging.warning(
                f"bytedance.ndtimeline's version should be >={NDTIMELINE_BASE_VERSION} to allow nccl trace dumping,"
                f"but {__version__} found, set use_cuda_timer=False in config file to disable it or install bytedance.ndtimeline properly"
            )
            return False
        return True
    except (ImportError, ModuleNotFoundError):
        logging.warning("ndtimeline EmergencyServer,FlightRecorderDumper is not found")
        return False


def init_emergency_server(local_rank, actor_name):
    if use_nccl_trace():
        from bytedance.ndtimeline import EmergencyServer, FlightRecorderDumper

        fr_dumper = FlightRecorderDumper(actor_name=actor_name)
        EmergencyServer.init(local_rank=local_rank, fr_dumper=fr_dumper)


def upload_process_group(trigger_timestamp, dump_type):
    if use_nccl_trace():
        from bytedance.ndtimeline import EmergencyServer, FlightRecorderDumper

        dumper = EmergencyServer.fr_dumper
        dumper.dump(trigger_timestamp=trigger_timestamp, dump_type=dump_type)
        logging.info(f'dump {dump_type} upload process group')
    else:
        logging.warning(f'flight recorder dumper not available, please use the latest ndtimeline version')


def safe_invoke(obj, method_name, *args, **kwargs):
    method = getattr(obj, method_name)
    if not callable(method):
        return False

    return method(*args, **kwargs)
