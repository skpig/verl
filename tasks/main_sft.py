import hydra
from pprint import pprint
from omegaconf import OmegaConf

import os
import ray

from single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup

import torch
import torch.distributed as dist

from alpha_seed.trainer.sft import RaySFTTrainer

ENABLE_REDIS_TRITON_CACHE = int(os.getenv("ENABLE_REDIS_TRITON_CACHE", '1'))


def init_ray():
    if not ray.is_initialized():
        remote_cache_env = {
            'TRITON_CACHE_MANAGER': 'triton.runtime.cache:RemoteCacheManager',
            'TRITON_REMOTE_CACHE_BACKEND': 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend'
        }
        runtime_env = {
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': '0',
                'BPEX_NO_WARN_ON_UNTUNED_CASE': '1',
            }
        }
        if ENABLE_REDIS_TRITON_CACHE:
            runtime_env['env_vars'].update(remote_cache_env)

        ray.init(runtime_env=runtime_env)


@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
def main(config):
    init_ray()
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    pprint((config.trainer.n_gpus_per_node, config.trainer.nnodes))

    resource_pool = RayResourcePool([config.trainer.n_gpus_per_node] * config.trainer.nnodes, use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=ray.remote(RaySFTTrainer), config=config)
    worker_group = RayWorkerGroup(resource_pool, class_with_args, name_prefix="main_sft")

    worker_group.fit()


if __name__ == '__main__':
    main()
