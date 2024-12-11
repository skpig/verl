import ray
import os

import shutil

from filelock import FileLock

import hdfs_io

import tempfile
import warnings

import torch
import torch.distributed
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
import numpy as np
import random

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from transformers import PreTrainedTokenizer
from omnistore import RLFSDPCheckpointer

from .uploader import CkptGlobalUploader


def check_ckpt_is_omnistore(path):
    if path is None:
        return False
    target_file_path = os.path.join(path, "model")
    return hdfs_io.hexists(target_file_path)


def get_rng_state():
    rng_state = {
        'cpu': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state(),
        'numpy': np.random.get_state(),
        'random': random.getstate(),
    }
    return rng_state


def load_rng_state(rng_state):
    torch.set_rng_state(rng_state['cpu'])
    torch.cuda.set_rng_state(rng_state['cuda'])
    np.random.set_state(rng_state['numpy'])
    random.setstate(rng_state['random'])


class CheckpointManagerOmniStore:
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

    def __init__(self,
                 model: FSDP,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 tokenizer: PreTrainedTokenizer,
                 enable_flatten: bool = False):
        self.upload_future = None
        self.previous_save_local_path = None

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        self.enable_flatten = enable_flatten

        assert isinstance(self.model, FSDP)
        self.rank = torch.distributed.get_rank()

    def load_checkpoint(self, hdfs_path=None, device_mesh: DeviceMesh = None, role: str = 'actor'):
        if hdfs_path is None:
            return

        global_step = int(hdfs_path.split('global_step_')[-1].split('/')[0])
        hdfs_path = os.path.join(hdfs_path, f'global_step_{global_step}')
        assert check_ckpt_is_omnistore(hdfs_path), f'{hdfs_path} is not in omnistore checkpoint format, resume failed'
        ckpt_state = {'model': self.model, 'optimizer': self.optimizer, 'extra_state': {}}
        RLFSDPCheckpointer.load(hdfs_path,
                                ckpt_state,
                                enable_shm_download_ckpt_tmp=True,
                                allow_extra_states=True,
                                rl_role=role,
                                load_flatten_model_optimizer=self.enable_flatten)
        # try loading lr scheduler state
        if 'lr_scheduler' in ckpt_state['extra_state']:
            self.lr_scheduler.load_state_dict(ckpt_state['extra_state']['lr_scheduler'])
        else:
            print(f'[rank-{self.rank}]: lr_scheduler not found in extra_state, skip loading')
        if 'rng_state' in ckpt_state['extra_state']:
            load_rng_state(ckpt_state['extra_state']['rng_state'])
        print(f'[rank-{self.rank}]: finish loading checkpoint {hdfs_path}')

    def save_checkpoint(self, local_path: str, hdfs_path: str, device_mesh: DeviceMesh, role: str, global_step: int,
                        ckpt_global_uploader_ref: CkptGlobalUploader):
        path = os.path.abspath(local_path)
        print(f'[rank-{self.rank}]: start saving checkpoint {path}')
        # wait for previous upload to hdfs
        if self.upload_future is not None:
            ray.get(self.upload_future)
        torch.distributed.barrier()

        # remove previous local_path
        if self.previous_save_local_path is not None and self.rank == 0:
            shutil.rmtree(self.previous_save_local_path, ignore_errors=True)

        with FileLock(os.path.join(tempfile.gettempdir(), path + '.lock')):
            # make a new dir
            os.makedirs(path, exist_ok=True)

        torch.distributed.barrier()

        file_path_list = [(f'global_step_{global_step}/model',
                           os.path.join(path, f'global_step_{global_step}/model', f'__{self.rank}_0.distcp')),
                          (f'global_step_{global_step}/optimizer',
                           os.path.join(path, f'global_step_{global_step}/optimizer', f'__{self.rank}_0.distcp')),
                          (f'global_step_{global_step}/extra_state',
                           os.path.join(path, f'global_step_{global_step}/extra_state',
                                        f'extra_state_rank_{self.rank}.pt'))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ckpt_state = {
                'model': self.model,
                'optimizer': self.optimizer,
                'extra_state': {
                    'lr_scheduler':
                        self.lr_scheduler if isinstance(self.lr_scheduler, dict) else self.lr_scheduler.state_dict(),
                    'rng_state':
                        get_rng_state(),
                }
            }

            print(f'[rank-{self.rank}]: Saving checkpoint to {os.path.abspath(path)} with omnistore FSDP')
            RLFSDPCheckpointer.save(
                path,
                ckpt_state,
                async_fast_checkpoint=False,
                enable_tree_topo=True,
                global_steps=global_step,
                rl_role=role,
                save_flatten_model_optimizer=self.enable_flatten,
            )

        if hdfs_path is not None:
            if self.rank == 0:
                print(f'[rank-{self.rank}]: prepare for uploading omnistore metadata')
                file_path_list.append(
                    (f'global_step_{global_step}/model', os.path.join(path,
                                                                      f'global_step_{global_step}/model/.metadata')))
                file_path_list.append((f'global_step_{global_step}/optimizer',
                                       os.path.join(path, f'global_step_{global_step}/optimizer/.metadata')))
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
            hf_local_path = os.path.join(path, 'huggingface')
            os.makedirs(hf_local_path, exist_ok=True)
            self.model._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
            self.tokenizer.save_pretrained(hf_local_path)
            if hdfs_path is not None:
                ray.get(
                    ckpt_global_uploader_ref.register_upload_task.remote(role, global_step,
                                                                         ray.get_runtime_context().get_node_id(),
                                                                         hf_local_path, hdfs_path))
                print(f'[rank-{self.rank}]: register upload ckpt task of path {hf_local_path} to hdfs {hdfs_path} done')
                self.upload_future = ckpt_global_uploader_ref.start_uploading.remote(role, global_step)
                print(f'[rank-{self.rank}]: start uploading ckpt')

        torch.distributed.barrier()

        self.previous_save_local_path = path
        print(f'[rank-{self.rank}]: finish saving checkpoint {path}')
