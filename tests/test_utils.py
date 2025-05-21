import pytest
import os
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

from alpha_seed.workers.streaming_service.rollout_request_manager import RequestManager
from verl.utils.tracking import Tracking
from verl.utils.fs import copy_local_path_from_hdfs
from verl.single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from transformers import AutoTokenizer

import ray
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from alpha_seed.workers.actors.rollout_pool import RolloutPool
from alpha_seed.workers.streaming_service.streaming_rollout import RemoteAsyncXPerfGPTRollout
from alpha_seed.workers.streaming_service.rollout_manager import RolloutManager


class PytestXdistEnv:

    def __init__(self):
        self._worker = os.environ.get('PYTEST_XDIST_WORKER', 'master')
        self._worker_count = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', "1"))
        self._worker = 0 if self._worker == "master" else int(self._worker.replace('gw', ''))

    @property
    def worker_id(self):
        return self._worker

    @property
    def worker_count(self):
        return self._worker_count


@pytest.fixture(scope='function')
def ray_fixture():
    import ray
    ray.init()
    yield
    ray.shutdown()


@pytest.fixture(scope='function')
def gpu_allocator(request, monkeypatch):
    import torch
    avail_gpu_count = torch.cuda.device_count()

    gpu_count = request.param
    xdist_env = PytestXdistEnv()

    assert xdist_env.worker_count * gpu_count <= avail_gpu_count, (
        f"gpu_count={gpu_count} * xdist_env.worker_count={xdist_env.worker_count} > avail_gpu_count={avail_gpu_count}")

    gpu_idx_start = gpu_count * xdist_env.worker_id
    alloc_gpus = list(range(gpu_idx_start, gpu_idx_start + gpu_count))
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible_devices is not None:
        # adapt already set visible devices
        cuda_visible_devices = [int(i) for i in cuda_visible_devices.split(',')]
        assert len(cuda_visible_devices) == avail_gpu_count
        alloc_gpus = [cuda_visible_devices[i] for i in alloc_gpus]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', ','.join([str(i) for i in alloc_gpus]))
    yield


def set_common_envs(monkeypatch):
    monkeypatch.setenv('TOKENIZERS_PARALLELISM', "false")
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    monkeypatch.setenv("XPERF_DUMP_NAN", "0")


def get_config(override_config=None):
    default_conf_path = (Path(__file__).parent.parent / "tasks/config/ppo_trainer.yaml")
    default_conf = OmegaConf.load(default_conf_path)
    if override_config is None:
        return default_conf

    config = OmegaConf.merge(default_conf, override_config)
    return config


def get_logger(config):
    logger = Tracking(project_name=config.trainer.project_name,
                      experiment_name=config.trainer.experiment_name,
                      default_backend=config.trainer.logger,
                      config=OmegaConf.to_container(config, resolve=True))
    return logger


def get_tokenizer(config):
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    return tokenizer


def _create_rollout_wg_common(actor_rollout_ref_config, ngpus: int, role: str, name: str, is_server: bool):
    resource_pool = RayResourcePool(process_on_nodes=[ngpus], use_gpu=True, name_prefix=name)
    rollout_cls_with_init = RayClassWithInitArgs(
        cls=RemoteAsyncXPerfGPTRollout if is_server else AsyncActorRolloutRefWorker,
        config=actor_rollout_ref_config,
        role=role)
    class_dict = {name: rollout_cls_with_init}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
    wg_dict = wg.spawn(prefix_set=class_dict.keys())
    wg = wg_dict[name]

    ray.get(wg.init_model())
    return wg


def create_hybrid_wg(config):
    assert config.trainer.nnodes == 1
    return _create_rollout_wg_common(config.actor_rollout_ref,
                                     ngpus=config.trainer.n_gpus_per_node,
                                     role='rollout',
                                     name='actor_rollout_ref',
                                     is_server=False)


def create_streaming_rollout_wg(config):
    if config.streaming_rollout.nnodes == 0 or config.streaming_rollout.elastic.enable:
        return None
    is_server = config.actor_rollout_ref.rollout.mode == 'server'
    return _create_rollout_wg_common(config.actor_rollout_ref,
                                     ngpus=config.streaming_rollout.n_gpus_per_node,
                                     role='standalone_rollout',
                                     name='standalone_rollout',
                                     is_server=is_server)


def create_streaming_validator_wg(config):
    if config.streaming_validator.nnodes == 0:
        return None
    is_server = config.actor_rollout_ref.rollout.mode == 'server'
    return _create_rollout_wg_common(config.actor_rollout_ref,
                                     ngpus=config.streaming_validator.n_gpus_per_node,
                                     role='standalone_validator',
                                     name='standalone_validator',
                                     is_server=is_server)


def create_rollout_pool(config):
    rollout_pool = RolloutPool.get_or_create_actor(config)
    return rollout_pool


# store the global request manager objects to avoid being gc
request_managers = []


def create_request_manager(instance_name: str):
    remote_cls = ray.remote(RequestManager)
    request_manager = remote_cls.options(name=f'RequestManager/{instance_name}', max_concurrency=102400).remote()
    ray.wait([request_manager.ready.remote()])
    request_managers.append(request_manager)
    return request_manager


def create_rollout_manager(config):
    if config.actor_rollout_ref.rollout.mode == "server":
        create_request_manager('hybrid_rollout')
        create_request_manager('standalone_rollout')
        create_request_manager('validation')
        create_request_manager('hybrid_validation')

    logger = get_logger(config)
    tokenizer = get_tokenizer(config)

    hybrid_wg = create_hybrid_wg(config)
    streaming_rollout_wg = create_streaming_rollout_wg(config)
    streaming_validator_wg = create_streaming_validator_wg(config)
    rollout_pool = create_rollout_pool(config)
    rollout_manager = RolloutManager(config, logger=logger, tokenizer=tokenizer)

    rollout_manager.initialize(hybrid_wg,
                               rollout_pool=rollout_pool,
                               train_standalone_wg=streaming_rollout_wg,
                               val_standalone_wg=streaming_validator_wg)
    return rollout_manager
