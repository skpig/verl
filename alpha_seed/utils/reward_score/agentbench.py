from .utils import Verifier


class AgentbenchVerifier(Verifier, reward_style="agentbench"):

    @staticmethod
    def compute_score(*args, **kwargs) -> float:
        return agentbench_score(*args, **kwargs)


def agentbench_score(non_tensor_batch_info, **argv):
    return non_tensor_batch_info['reward_model']['agentbench_score']
