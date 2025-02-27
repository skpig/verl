import logging
from enum import Enum
from .check import use_cuda_timer


class DumpType(Enum):
    initial = "initial"


def use_nccl_trace():
    return use_cuda_timer()


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
