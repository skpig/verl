import os
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from alpha_seed.utils.server_client import KVStore
from single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
from omegaconf import OmegaConf
import ray
import ipdb

os.environ["XPERF_DUMP_NAN"] = "0"

default_conf = OmegaConf.load("/opt/tiger/alpha-seed/tasks/config/ppo_trainer.yaml")
default_actor_rollout_ref_config = default_conf.actor_rollout_ref
config = OmegaConf.create({
    'model': {
        'path': "hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/liuxin.ai/rl/bs4k_merge_op_async_step80"
    },
    'rollout': {
        'name': 'xperf_gpt',
        'prompt_length': 2048,
        'response_length': 4096,
        'tensor_model_parallel_size': 8,
        'enable_paged_attention': True,
        'use_vllm': True,
        'num_slots': 256,
        'slot_block_size': 1024,
        'gpu_memory_utilization': 0.6,
        'quant_mode': "NO_QUANT",
        'model_type': "m8_14b",
        'train_generate_kwargs': {
            'stop_sequence_tokens': [[959, 39440, 157], [959, 35733, 157], [959, 2642, 26128, 1742, 157]]
        },
    },
})

config = OmegaConf.merge(default_actor_rollout_ref_config, config)

ray.init(namespace="alphaseed")
resource_pool = RayResourcePool(process_on_nodes=[8], use_gpu=True)
ray_cls_with_init = RayClassWithInitArgs(cls=AsyncActorRolloutRefWorker, config=config, role='rollout')
wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
ray.get(wg.init_model())
kv_store = ray.remote(KVStore).options(name=KVStore.name, lifetime="detached").remote()
kv_store.set_key_val.remote("worker_names", wg.worker_names)

ipdb.set_trace()
