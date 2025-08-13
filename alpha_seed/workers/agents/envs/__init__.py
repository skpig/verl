from typing import *
import os
from abc import ABC, abstractmethod
from alpha_seed.workers.agents import load_external_module
from alpha_seed.workers.agents.monitor_ctx import current_agent_tracker, current_agent
from alpha_seed.workers.agents.monitoring import AgentTaskTracker


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

    def __getattribute__(self, name):
        if name == "step":
            orig_call = super().__getattribute__(name)

            async def wrapped_step(*args, **kwargs):
                try:
                    tracker: AgentTaskTracker = current_agent_tracker.get()
                    agent_class_name = current_agent.get()
                except LookupError:
                    # skip if not tracker not set
                    return await orig_call(*args, **kwargs)

                with tracker.tool_call(agent_class_name, self.__class__.__name__):
                    result = await orig_call(*args, **kwargs)
                    return result

            return wrapped_step
        return super().__getattribute__(name)


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
