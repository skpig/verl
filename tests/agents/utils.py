import json
from omegaconf import OmegaConf
from pathlib import Path
from alpha_seed.workers.agents.envs import create_agent_envs_from_str
import torch
import torch.distributed as dist
from functools import partial
from tests.test_utils import get_config, PytestXdistEnv


def get_plugin_config(override_config=None):
    default_conf_path = (Path(__file__).parent.parent.parent / "tasks/config/ppo_trainer.yaml")
    default_conf = OmegaConf.load(default_conf_path)
    default_plugin_config = default_conf.actor_rollout_ref.rollout.plugin
    if override_config is None:
        return default_plugin_config
    config = OmegaConf.merge(default_plugin_config, override_config)
    return config


def get_basic_example_env():
    kwargs = {'env_type': 'basic', 'env_args': {'round_ndigits': 2}}
    env = create_agent_envs_from_str(f'example_env@{json.dumps(kwargs)}')[0]
    return env


def setup_dist(rank, world_size, backend='nccl'):
    xdist_env = PytestXdistEnv()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://localhost:{12135 + xdist_env.worker_id}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )


def teardown_dist():
    dist.barrier()
    dist.destroy_process_group()


def dist_worker(rank, world_size, target):
    setup_dist(rank, world_size)
    try:
        target()
    finally:
        teardown_dist()
