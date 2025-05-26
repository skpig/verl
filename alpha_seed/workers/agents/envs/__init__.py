from typing import *
import os
from abc import ABC, abstractmethod


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


def create_agent_envs_from_str(env_strs: Union[None, List[str], str], **kwargs) -> List[BaseEnv]:
    if env_strs is None:
        return []
    if isinstance(env_strs, str):
        env_strs = [env_strs]

    envs = []
    for env_str in env_strs:
        env_name = env_str.split('@')[0]

        import importlib
        external_path = os.environ.get('EXTERNAL_ENV_PATH', None)
        if (external_path is not None) and os.path.isfile((mod_path := os.path.join(external_path, f"{env_name}.py"))):
            spec = importlib.util.spec_from_file_location(env_name, mod_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(f".{env_name}", package="alpha_seed.workers.agents.envs")

        env = module.create_from_env_str(env_str, **kwargs)
        envs.append(env)
    return envs
