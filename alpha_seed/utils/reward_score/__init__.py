"""
This sub-package should contain all the rule-based reward score
"""

import pkgutil
import importlib

NON_AGENT_PLACE_HOLDER_SCORE = -99.0


def auto_import_submodules(package_name: str):
    package = importlib.import_module(package_name)
    for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(name)
        except ImportError as e:
            pass


auto_import_submodules("alpha_seed.utils.reward_score")


def select_remote_rm_fn(config, **kwargs):
    if config.trainer.remote_rm_type == "qrm":
        from alpha_seed.utils.reward_score.qrm_service import init_qrm_server
        return init_qrm_server
    elif config.trainer.remote_rm_type == "grm":
        from alpha_seed.utils.reward_score.grm_service import init_grm_server
        return init_grm_server
    elif config.trainer.remote_rm_type == "orm":
        pass
    else:
        raise NotImplementedError(f"{config.trainer.remote_rm_type=} not implemented")
