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
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.utils.fs import copy_local_path_from_hdfs

from transformers import PreTrainedTokenizer
from torch.distributed._tensor.api import DTensor, Shard, Replicate

from .uploader import CkptGlobalUploader


def remove_replicate_in_dtensor(state_dict, device_mesh: DeviceMesh):
    if device_mesh.ndim == 1:
        return state_dict
    for k in state_dict.keys():
        if isinstance(state_dict[k], DTensor):
            v: DTensor = state_dict[k]
            # NOTE(jianyujiang): we must save the shape and stride, as
            # from_local assumes evenly sharded across ranks w/o them
            shape, stride = v.shape, v.stride()
            state_dict[k] = DTensor.from_local(v._local_tensor,
                                               device_mesh['fsdp'], [Shard(dim=0)],
                                               shape=shape,
                                               stride=stride)
    return state_dict


def replicate_in_dtensor(state_dict, device_mesh: DeviceMesh):
    replicate_mesh = [Replicate()] * (device_mesh.ndim - 1)
    for k in state_dict.keys():
        if isinstance(state_dict[k], DTensor):
            v: DTensor = state_dict[k]
            shape, stride = v.shape, v.stride()
            state_dict[k] = DTensor.from_local(state_dict[k]._local_tensor,
                                               device_mesh,
                                               replicate_mesh + [Shard(dim=0)],
                                               shape=shape,
                                               stride=stride)
    return state_dict


class CheckpointManagerV2:
    """
    Diffs from V1: use LocalStateDict for saving ckpt

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

    def __init__(self, model: FSDP, optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler, tokenizer: PreTrainedTokenizer):
        self.upload_future = None
        self.previous_save_local_path = None

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        assert isinstance(self.model, FSDP)
        self.rank = torch.distributed.get_rank()

    def load_checkpoint(self, hdfs_path=None, device_mesh: DeviceMesh = None, role: str = 'actor'):
        if hdfs_path is None:
            return

        # NOTE (jianyujiang): v2 ckpt must have device_mesh
        state_idx = device_mesh.get_local_rank(device_mesh.ndim - 1)
        remote_path = os.path.join(hdfs_path, f'model_optim_rank_{state_idx}.pt')
        print(f'[rank-{self.rank}]: Loading from {remote_path}')
        local_path = copy_local_path_from_hdfs(remote_path)

        state_dict = torch.load(local_path)

        model_state_dict = replicate_in_dtensor(state_dict['model'], device_mesh)
        optimizer_state_dict = replicate_in_dtensor(state_dict['optimizer'], device_mesh)
        lr_scheduler_state_dict = state_dict['lr_scheduler']

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def save_checkpoint(self, local_path: str, hdfs_path: str, device_mesh: DeviceMesh, role: str, global_step: int,
                        ckpt_global_uploader_ref: CkptGlobalUploader):
        # wait for previous upload to hdfs
        if self.upload_future is not None:
            ray.get(self.upload_future)
        torch.distributed.barrier()

        # remove previous local_path
        if self.previous_save_local_path is not None:
            previous_save_local_path = os.path.join(self.previous_save_local_path, f'model_optim_rank_{self.rank}.pt')
            if os.path.isfile(previous_save_local_path):
                os.remove(previous_save_local_path)
            else:
                shutil.rmtree(previous_save_local_path, ignore_errors=True)

        with FileLock(os.path.join(tempfile.gettempdir(), local_path + '.lock')):
            # make a new dir
            os.makedirs(local_path, exist_ok=True)

        torch.distributed.barrier()

        # NOTE (jianyujiang): v2 ckpt must have device_mesh
        should_save_ckpt = device_mesh.ndim > 1 and device_mesh.get_local_rank(0) == 0  # HSDP's first FSDP group
        should_save_ckpt = should_save_ckpt or (device_mesh.ndim == 1)  # FSDP
        state_idx = device_mesh.get_local_rank(device_mesh.ndim - 1)

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state = self.model.state_dict()
                optimizer_state_dict = self.optimizer.state_dict()

                # we have to remove replication to ease merging
                remove_replicate_in_dtensor(model_state, device_mesh)
                remove_replicate_in_dtensor(optimizer_state_dict, device_mesh)
                state_dict = {
                    'model': model_state,
                    'optimizer': optimizer_state_dict,
                    'lr_scheduler': self.lr_scheduler.state_dict()
                }
                path = os.path.join(local_path, f'model_optim_rank_{state_idx}.pt')

                if should_save_ckpt:
                    print(f'[rank-{self.rank}]: Saving checkpoint to {os.path.abspath(path)}')
                    torch.save(state_dict, path)
                else:
                    print(f'[rank-{self.rank}]: Skip saving checkpoint to {os.path.abspath(path)}')

        if hdfs_path is not None and should_save_ckpt:
            ray.get(
                ckpt_global_uploader_ref.register_upload_task.remote(role, global_step,
                                                                     ray.get_runtime_context().get_node_id(), path,
                                                                     hdfs_path))
            print(f'[rank-{self.rank}]: register upload ckpt task of path {path} to hdfs {hdfs_path} done')
        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.rank == 0:
            hf_local_path = os.path.join(local_path, 'huggingface')
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

        self.previous_save_local_path = local_path
