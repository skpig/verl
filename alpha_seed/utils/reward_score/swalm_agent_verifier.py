import torch

from .utils import Verifier


class SwalmAgentVerifier(Verifier, reward_style="swalm_agent_verifier"):

    @staticmethod
    def compute_score(*args, **kwargs) -> float:
        return get_agent_reward(*args, **kwargs)


def get_agent_reward(batch_info, **kwargs):
    return batch_info["swalm_agent_score"].item()
