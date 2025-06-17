from dataclasses import dataclass
from typing import Callable, Dict
import importlib
import pkgutil
from omegaconf import DictConfig
import os


@dataclass
class TaskContext:
    config: DictConfig
    tokenizer: object
    global_step: int
    server_host: str
    server_port: int


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


def select_handler_fn(handler_type: str) -> Callable:
    # Load an external handler dynamically
    # Example, EXTERNAL_HANDLER_PATH=/opt/tiger/agents_extension/handlers
    # `handler_type` should be the same as the registered name, as well as the module's filepath under EXTERNAL_HANDLER_PATH.
    #   - If `handler_type` == 'xx', the handler should be registered in /opt/tiger/agents_extension/handlers/xx.py
    #   - If `handler_type` == 'xx/yy', the handler should be registered in /opt/tiger/agents_extension/handlers/xx/yy.py
    import importlib
    external_path = os.environ.get('EXTERNAL_HANDLER_PATH', None)
    if (external_path is not None) and os.path.isfile((mod_path := os.path.join(external_path, f"{handler_type}.py"))):
        spec = importlib.util.spec_from_file_location(handler_type, mod_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    if handler_type not in _HANDLER_REGISTRY:
        raise NotImplementedError(f"unsupported handler type: {handler_type}")
    return _HANDLER_REGISTRY[handler_type]


auto_import_submodules("alpha_seed.workers.agents.handlers")
