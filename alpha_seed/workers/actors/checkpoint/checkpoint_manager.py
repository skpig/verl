import os
import shutil
from filelock import FileLock
import tempfile

import ray
import omegaconf
import json
import torch
import torch.distributed
from transformers import PretrainedConfig, PreTrainedTokenizer
import numpy as np
import random
from ray.actor import ActorHandle


class BaseCheckpointManager:
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(self, model, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 hf_config: PretrainedConfig, tokenizer: PreTrainedTokenizer):
        self.previous_global_step = None
        self.previous_save_local_path = None

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.hf_config = hf_config
        self.tokenizer = tokenizer
        self.ray_actor_name = ray.get_runtime_context().get_actor_name()
        self.rank = torch.distributed.get_rank()

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def remove_previous_save_local_path(self):
        if not self.previous_save_local_path:
            return

        abs_path = os.path.abspath(self.previous_save_local_path)
        print(f'Checkpoint manager remove previous save local path: {abs_path}')
        if not os.path.exists(abs_path):
            return

        # remove previous local_path
        shutil.rmtree(abs_path, ignore_errors=True)

    def wait_previous_upload(self, role, ckpt_global_uploader_ref):
        if self.previous_global_step:
            ray.get(ckpt_global_uploader_ref.wait_by_role.remote(role, self.previous_global_step))
        torch.distributed.barrier()

    @staticmethod
    def local_mkdir(path):
        with FileLock(os.path.join(tempfile.gettempdir(), path + '.lock')):
            # make a new dir
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def get_rng_state():
        rng_state = {
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        }
        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state['cpu'])
        torch.cuda.set_rng_state(rng_state['cuda'])
        np.random.set_state(rng_state['numpy'])
        random.setstate(rng_state['random'])

    def save_hf_configs(self, local_path: str, hdfs_path: str, role: str, global_step: int,
                        ckpt_global_uploader_ref: ActorHandle):
        hf_local_path = os.path.join(local_path, 'huggingface')
        os.makedirs(hf_local_path, exist_ok=True)
        self.hf_config.save_pretrained(hf_local_path)
        self.tokenizer.save_pretrained(hf_local_path)
        if hdfs_path is not None:
            ray.get(
                ckpt_global_uploader_ref.register_upload_task.remote(role, global_step,
                                                                     ray.get_runtime_context().get_node_id(),
                                                                     hf_local_path, hdfs_path))
            print(f'[rank-{self.rank}]: register upload ckpt task of path {hf_local_path} to hdfs {hdfs_path} done')

    def save_megatron_configs(self, local_path: str, hdfs_path: str, role: str, global_step: int,
                              ckpt_global_uploader_ref: ActorHandle):
        model_config, megatron_config = self.get_megatron_configs_from_model()
        print(f'model config: {model_config}')
        print(f'megatron config: {megatron_config}')
        megatron_configs_local_path = os.path.join(local_path, 'megatron')
        os.makedirs(megatron_configs_local_path, exist_ok=True)
        self.save_json(model_config, os.path.join(megatron_configs_local_path, 'model_config.json'))
        self.save_json(megatron_config, os.path.join(megatron_configs_local_path, 'megatron_config.json'))
        if hdfs_path is not None:
            ray.get(
                ckpt_global_uploader_ref.register_upload_task.remote(role, global_step,
                                                                     ray.get_runtime_context().get_node_id(),
                                                                     megatron_configs_local_path, hdfs_path))
            print(
                f'[rank-{self.rank}]: register upload ckpt task of path {megatron_configs_local_path} to hdfs {hdfs_path} done'
            )
        return

    def get_megatron_configs_from_model(self):
        get_module = self.model[0]
        while hasattr(get_module, 'module'):
            get_module = get_module.module
        return self.convert_transformers_config(get_module.model_config), self.convert_transformers_config(
            get_module.megatron_config)

    @staticmethod
    def convert_transformers_config(config):

        def traverse_and_convert(obj):
            if isinstance(obj, dict):
                return {k: traverse_and_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [traverse_and_convert(item) for item in obj]
            elif isinstance(obj, (omegaconf.DictConfig, omegaconf.ListConfig)):
                return BaseCheckpointManager.omegaconf_config_to_py_obj(obj)
            return obj

        config_dict = config.to_dict()
        converted_dict = traverse_and_convert(config_dict)
        return converted_dict

    @staticmethod
    def omegaconf_config_to_py_obj(config):
        if isinstance(config, omegaconf.DictConfig):
            return {k: BaseCheckpointManager.omegaconf_config_to_py_obj(v) for k, v in config.items()}
        elif isinstance(config, omegaconf.ListConfig):
            return [BaseCheckpointManager.omegaconf_config_to_py_obj(item) for item in config]
        else:
            return config

    @staticmethod
    def save_json(config, local_path):
        with open(local_path, 'w') as f:
            json.dump(config, f)
