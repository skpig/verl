from typing import *
import os
from abc import ABC, abstractmethod
from alpha_seed.workers.agents import load_external_module


class BaseEnv(ABC):

    @abstractmethod
    async def step(self, action: str) -> str:
        raise NotImplemented

    @abstractmethod
    def action_supported(self, action: str) -> bool:
        raise NotImplemented

    @property
    def finished(self) -> bool:
        return False

    @property
    def reward(self) -> float:
        return 0.0

    @property
    def metrics(self) -> Dict:
        return {}

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state_dict: Dict):
        pass


def create_agent_envs_from_str(env_strs: Union[None, List[str], str],
                               external_lib: str = None,
                               **kwargs) -> List[BaseEnv]:
    if env_strs is None:
        return []
    if isinstance(env_strs, str):
        env_strs = [env_strs]

    external_path = os.environ.get('EXTERNAL_ENV_PATH', None)
    envs = []
    for env_str in env_strs:
        env_name = env_str.split('@')[0]

        module = load_external_module(package_name=env_name, external_lib=external_lib, external_path=external_path)
        if module is None:
            import importlib
            module = importlib.import_module(f".{env_name.replace('/', '.')}", package="alpha_seed.workers.agents.envs")

        env = module.create_from_env_str(env_str, **kwargs)
        envs.append(env)
    return envs
