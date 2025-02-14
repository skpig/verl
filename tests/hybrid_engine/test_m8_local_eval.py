import ray
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.utils.tracking import Tracking
from verl.utils.fs import copy_local_path_from_hdfs
from alpha_seed.utils.server_client import KVStore
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from alpha_seed.utils.validator.validation_manager import ValidateManager
from tasks.main_ppo import RewardManager
from single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

default_conf = OmegaConf.load("/opt/tiger/alpha-seed/tasks/config/ppo_trainer.yaml")
config = OmegaConf.create({
    'trainer': {
        'save_cases_to_hdfs': False,
        'code_sandbox_psm': 'seed.alphaseed.code_sandbox.service.yg'
    },
    'model': {
        'path': "hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/liuxin.ai/rl/bs4k_merge_op_async_step80"
    },
    "data": {
        "val_files": ["hdfs://haruna/home/byte_data_seed/ssd_hldy/user/jiangchengquan/rl/datasets/MBPP_eval.parquet"],
        "prompt_key": "prompt",
        "answer_key": "answer",
        "use_ref_answer": True,
        "max_prompt_length": 2048,
        "max_response_length": 4096,
        "shuffle": False,
        "truncation": "left",
        "val_batch_size": 32,
    }
})

config = OmegaConf.merge(default_conf, config)

local_path = copy_local_path_from_hdfs(config.model.path)
tokenizer = AutoTokenizer.from_pretrained(local_path)
logger = Tracking(
    project_name="alphaseed_debug",
    experiment_name="local_eval",
    default_backend=["console"],
    config=OmegaConf.to_container(config, resolve=True),
)
val_reward_fn = RewardManager(tokenizer=tokenizer, config=config, logger=logger, rm_name="val")

ray.init(namespace="alphaseed", address="auto")
kv_store = ray.get_actor(KVStore.name)
ray_cls_with_init = RayClassWithInitArgs(cls=AsyncActorRolloutRefWorker, config=None, role="rollout")
worker_names = ray.get(kv_store.get_by_key.remote("worker_names"))
rollout_wg = RayWorkerGroup(ray_cls_with_init=ray_cls_with_init, worker_names=worker_names)


def create_dataloader(config, tokenizer):
    from alpha_seed.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    val_dataset = RLHFDataset(
        parquet_files=config.data.val_files,
        tokenizer=tokenizer,
        prompt_key=config.data.prompt_key,
        answer_key=config.data.answer_key,
        use_ref_answer=config.data.use_ref_answer,
        max_prompt_length=config.data.max_prompt_length,
        filter_prompts=True,
        return_raw_chat=True,
        truncation=config.data.get("truncation", "error"),
        multi_prompts=config.data.get("multi_prompts", "none"),
        num_prompts_per_data=1,
        is_eval=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset),
        shuffle=config.data.shuffle,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return val_dataloader


val_dataloader = create_dataloader(config, tokenizer)

use_rm = False
validation_manager = ValidateManager(config, logger, val_dataloader, tokenizer, use_rm, val_reward_fn)
validation_manager.actor_rollout_wg = rollout_wg

validation_manager.validate(val_epoch=1, global_step=0, need_log=True)
