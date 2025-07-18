import os
import sys
import re
import importlib
import importlib.machinery
from functools import cache
from pathlib import Path

MODULE_CACHE = dict()


@cache
def load_external_module(package_name: str, external_lib: str = None, external_path: str = None):
    """Load an external module from either a submodule of `external_lib` or a python file under `external_path`
    Args:
        package_name: xxx, xxx/yyy, or xxx/yyy/...
        external_lib: if not None, import the package as a submodule
        external_path: if not None and `external_lib` is None, import the package from the single python file.
    Returns:
        module: None if not found
    """
    if external_lib is not None:
        importlib.import_module(external_lib)
        try:
            module = importlib.import_module('.' + package_name.replace('/', '.'), package=external_lib)
            return module
        except ModuleNotFoundError:
            return None
        except Exception as e:
            raise (e)
    if external_path is not None:
        if os.path.isfile((mod_path := os.path.join(external_path, f"{package_name}.py"))):
            global MODULE_CACHE
            if mod_path in MODULE_CACHE:
                return MODULE_CACHE[mod_path]
            spec = importlib.util.spec_from_file_location(package_name, mod_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            MODULE_CACHE[mod_path] = module
            return module
        else:
            return None
