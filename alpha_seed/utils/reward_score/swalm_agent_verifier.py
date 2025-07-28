import torch


def get_agent_reward(batch_info, **kwargs):
    return batch_info["swalm_agent_score"].item()