import importlib
from typing import Any


def import_from_string(import_str: str) -> Any:
    if '.' in import_str:
        module_name, obj_name = import_str.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, obj_name)
    else:
        return importlib.import_module(import_str)


if __name__ == "__main__":
    print(import_from_string("tasks.vlm.reward_manager.VLMRewardManager"))
