import ray
import os
import re
import inspect

import hdfs_io

import warnings
from packaging.version import Version

import torch
import torch.distributed

from transformers import PretrainedConfig, PreTrainedTokenizer
import importlib.metadata

from .checkpoint_manager import BaseCheckpointManager

from ray.actor import ActorHandle

REQUIRED_OMNISTORE_VERSION = '0.7.11'
ACTUAL_OMNISTORE_VERSION = None


def check_omnistore_version():
    try:
        global ACTUAL_OMNISTORE_VERSION
        ACTUAL_OMNISTORE_VERSION = importlib.metadata.version('byted-omnistore')

        assert Version(ACTUAL_OMNISTORE_VERSION) >= Version(REQUIRED_OMNISTORE_VERSION), \
            f'byted-omnistore version {ACTUAL_OMNISTORE_VERSION} is too old. Please upgrade to version ' \
            f'{REQUIRED_OMNISTORE_VERSION} or higher. Example command: pip3 install --upgrade byted-omnistore.'
    except importlib.metadata.PackageNotFoundError as e:
        print(f'byted-omnistore not installed. Please install it and upgrade to version {REQUIRED_OMNISTORE_VERSION} '
              f'or higher. Example command: pip3 install byted-omnistore=={REQUIRED_OMNISTORE_VERSION}.')
        raise e


check_omnistore_version()

import omnistore


def check_ckpt_is_omnistore(path):
    if path is None:
        return False
    target_file_path = os.path.join(path, "model")
    return hdfs_io.hexists(target_file_path)


class CheckpointManagerOmniStore(BaseCheckpointManager):
    """
    Diffs from V2: use OmniStore for saving ckpt

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
        super().__init__(model, optimizer, lr_scheduler, hf_config, tokenizer)
        if self.rank == 0:
            print(f'OmniStore ckpt manager initialized, byted-omnistore version: {ACTUAL_OMNISTORE_VERSION}')

    def load_checkpoint(self,
                        hdfs_path=None,
                        role: str = 'actor',
                        strategy: str = 'fsdp',
                        enable_shm: bool = False,
                        *args,
                        **kwargs):
        if hdfs_path is None:
            return

        if not check_ckpt_is_omnistore(hdfs_path):
            # be compatible with old ckpt folder format
            match = re.search(r'global_step_(\d+)', hdfs_path)
            if match:
                global_step = int(match.group(1))
            else:
                raise ValueError(f'[rank-{self.rank}] Invalid hdfs path: {hdfs_path}, no global step section found.')
            hdfs_path = os.path.join(hdfs_path, f'global_step_{global_step}')
        assert check_ckpt_is_omnistore(hdfs_path), f'{hdfs_path} is not in omnistore checkpoint format, resume failed'
        ckpt_state = {'model': self.model, 'extra_state': {}}
        if self.optimizer:
            if isinstance(self.optimizer, list):
                ckpt_state['optimizer'] = self.optimizer[0]
            else:
                ckpt_state['optimizer'] = self.optimizer
        if strategy == 'fsdp':
            sig = inspect.signature(omnistore.FSDPCheckpointer.load)
            if "kwargs" in sig.parameters:
                additional_kwargs_dict = {"metrics_supplement": {"ray_actor_name": self.ray_actor_name, "role": role}}
            else:
                additional_kwargs_dict = {}
            omnistore.FSDPCheckpointer.load(
                hdfs_path,
                ckpt_state,
                enable_shm_download_ckpt_tmp=enable_shm,
                role=role,
                **additional_kwargs_dict,
            )
        elif strategy == 'megatron':
            sig = inspect.signature(omnistore.MegatronCheckpointer.load)
            if "kwargs" in sig.parameters:
                additional_kwargs_dict = {"metrics_supplement": {"ray_actor_name": self.ray_actor_name, "role": role}}
            else:
                additional_kwargs_dict = {}
            omnistore.MegatronCheckpointer.load(
                hdfs_path,
                ckpt_state,
                enable_shm_download_ckpt_tmp=enable_shm,
                allow_extra_state_not_exists=True,
                allow_client_state_not_exists=True,
                role=role,
                **additional_kwargs_dict,
            )
        else:
            raise NotImplementedError(f'Alpha-seed OmniStore checkpointer does not support strategy {strategy}')

        # try loading lr scheduler state
        if 'lr_scheduler' in ckpt_state['extra_state']:
            if isinstance(self.lr_scheduler, list):
                self.lr_scheduler[0].load_state_dict(ckpt_state['extra_state']['lr_scheduler'])
            else:
                self.lr_scheduler.load_state_dict(ckpt_state['extra_state']['lr_scheduler'])
        else:
            print(f'[rank-{self.rank}]: lr_scheduler not found in extra_state, skip loading')
        if 'rng_state' in ckpt_state['extra_state']:
            self.load_rng_state(ckpt_state['extra_state']['rng_state'])
        print(f'[rank-{self.rank}]: finish loading checkpoint {hdfs_path}')

    def save_checkpoint(self, local_path: str, hdfs_path: str, role: str, strategy: str, global_step: int,
                        ckpt_global_uploader_ref: ActorHandle, enable_shm: bool, *args, **kwargs):
        path = os.path.abspath(local_path)
        print(f'[rank-{self.rank}]: start saving checkpoint {path}')
        # wait for previous upload to hdfs
        self.wait_previous_upload(role, ckpt_global_uploader_ref)
        self.previous_global_step = global_step

        # remove previous local_path
        self.remove_previous_save_local_path()
        self.local_mkdir(path)
        torch.distributed.barrier()

        file_path_list = [('model', os.path.join(path, 'model', f'__{self.rank}_0.distcp')),
                          ('extra_state', os.path.join(path, 'extra_state', f'extra_state_rank_{self.rank}.pt'))]
        if self.optimizer:
            file_path_list.append(('optimizer', os.path.join(path, 'optimizer', f'__{self.rank}_0.distcp')))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ckpt_state = {'model': self.model, 'extra_state': {'rng_state': self.get_rng_state(),}}
            if self.optimizer:
                if isinstance(self.optimizer, list):
                    ckpt_state['optimizer'] = self.optimizer[0]
                else:
                    ckpt_state['optimizer'] = self.optimizer
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, dict):
                    ckpt_state['extra_state']['lr_scheduler'] = self.lr_scheduler
                elif isinstance(self.lr_scheduler, list):
                    ckpt_state['extra_state']['lr_scheduler'] = self.lr_scheduler[0].state_dict()
                else:
                    ckpt_state['extra_state']['lr_scheduler'] = self.lr_scheduler.state_dict()

            print(f'[rank-{self.rank}]: Saving checkpoint to {os.path.abspath(path)} with omnistore')
            if strategy == 'fsdp':
                sig = inspect.signature(omnistore.FSDPCheckpointer.save)
                if "kwargs" in sig.parameters:
                    additional_kwargs_dict = {
                        "metrics_supplement": {
                            "ray_actor_name": self.ray_actor_name,
                            "role": role
                        }
                    }
                else:
                    additional_kwargs_dict = {}
                omnistore.FSDPCheckpointer.save(
                    path,
                    ckpt_state,
                    enable_shm_upload_ckpt_tmp=enable_shm,
                    async_fast_checkpoint=False,
                    enable_tree_topo=True,
                    global_steps=global_step,
                    role=role,
                    ignore_append_global_steps_to_folder=True,
                    **additional_kwargs_dict,
                )
            elif strategy == 'megatron':
                sig = inspect.signature(omnistore.MegatronCheckpointer.save)
                if "kwargs" in sig.parameters:
                    additional_kwargs_dict = {
                        "metrics_supplement": {
                            "ray_actor_name": self.ray_actor_name,
                            "role": role
                        }
                    }
                else:
                    additional_kwargs_dict = {}
                omnistore.MegatronCheckpointer.save(
                    path,
                    ckpt_state,
                    enable_shm_upload_ckpt_tmp=enable_shm,
                    async_fast_checkpoint=False,
                    enable_tree_topo=True,
                    global_steps=global_step,
                    role=role,
                    ignore_append_global_steps_to_folder=True,
                    **additional_kwargs_dict,
                )
            else:
                raise NotImplementedError(f'Alpha-seed OmniStore checkpointer does not support strategy {strategy}')

        if hdfs_path is not None:
            if self.rank == 0:
                print(f'[rank-{self.rank}]: prepare for uploading omnistore metadata')
                file_path_list.append(('model', os.path.join(path, 'model/.metadata')))
                if self.optimizer:
                    file_path_list.append(('optimizer', os.path.join(path, 'optimizer/.metadata')))
            for sub_folder_name, file_local_path in file_path_list:
                file_local_path = os.path.abspath(file_local_path)
                hdfs_path_sub_folder = os.path.join(hdfs_path, sub_folder_name)
                assert os.path.isfile(file_local_path), f'local path {file_local_path} does not exist'
                ray.get(
                    ckpt_global_uploader_ref.register_upload_task.remote(role, global_step,
                                                                         ray.get_runtime_context().get_node_id(),
                                                                         file_local_path, hdfs_path_sub_folder))
                print(
                    f'[rank-{self.rank}]: register upload ckpt task of path {file_local_path} to hdfs {hdfs_path_sub_folder} done'
                )
        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.rank == 0:
            self.save_hf_configs(path, hdfs_path, role, global_step, ckpt_global_uploader_ref)
            if strategy == 'megatron':
                self.save_megatron_configs(path, hdfs_path, role, global_step, ckpt_global_uploader_ref)
            if hdfs_path:
                ckpt_global_uploader_ref.start_uploading.remote(role, global_step)
                print(f'[rank-{self.rank}]: start uploading ckpt')
        torch.distributed.barrier()

        self.previous_save_local_path = path
        print(f'[rank-{self.rank}]: finish saving checkpoint {path}')
