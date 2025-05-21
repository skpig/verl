from dataclasses import dataclass

from omegaconf import DictConfig

from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto


@dataclass
class TaskContext:
    config: DictConfig
    tokenizer: object
    global_step: int
    server_host: str
    server_port: int


def select_handler_fn(handler_type: str):
    if handler_type == "math/aiohttp":
        from .math.aiohttp_handler import process_single_batch

        return process_single_batch
    else:
        raise NotImplementedError(f"unsupported handler type: {handler_type}")
