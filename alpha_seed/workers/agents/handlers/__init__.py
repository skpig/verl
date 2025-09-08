from dataclasses import dataclass
from typing import Callable, Dict, Optional
import importlib
import pkgutil
from omegaconf import DictConfig
import os

from transformers import PreTrainedTokenizer

from alpha_seed.workers.agents import load_external_module


class GlobalState:

    def __init__(self):
        self.global_step = 0

    def set_global_step(self, global_step: int):
        self.global_step = global_step

    def get_global_step(self) -> int:
        """
        这个是当前trainer正在train的step
        """
        return self.global_step


@dataclass
class TaskContext:
    config: DictConfig
    global_step: int
    server_host: str
    server_port: int
    is_train: bool
    # 尽量不要用这个字段，只是用来兼容旧的代码的，Executor给handler_fn传参时会一并给这个对象赋值
    #  see: alpha_seed.workers.agents.executor.RayActorExecutor
    tokenizer: Optional[PreTrainedTokenizer] = None


def auto_import_submodules(package_name: str):
    package = importlib.import_module(package_name)
    for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(name)


_HANDLER_REGISTRY: Dict[str, Callable] = {}


def register_handler(name: str):

    def decorator(fn: Callable):
        if name in _HANDLER_REGISTRY:
            print(f"[WARN] handler {name} is overrided")
        _HANDLER_REGISTRY[name] = fn
        return fn

    return decorator


def select_handler_fn(handler_type: str, external_lib: str = None) -> Callable:
    _ = load_external_module(package_name=handler_type,
                             external_lib=external_lib,
                             external_path=os.environ.get('EXTERNAL_HANDLER_PATH', None))

    if handler_type not in _HANDLER_REGISTRY:
        raise NotImplementedError(f"unsupported handler type: {handler_type}")
    return _HANDLER_REGISTRY[handler_type]


auto_import_submodules("alpha_seed.workers.agents.handlers")
