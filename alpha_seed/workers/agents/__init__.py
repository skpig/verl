from dataclasses import dataclass
from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto


@dataclass
class TaskContext:
    config: dict
    tokenizer: object
    reward_fn: callable
    global_step: int
