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


@ray.remote
def upload_ckpt(local_path, hdfs_path):
    local_path = os.path.abspath(local_path)
    print(f'Start uploading checkpoint from {local_path} to {hdfs_path}')
    hdfs_io.copy(src=local_path, dst=hdfs_path)
    print(f'Finish uploading checkpoint from {local_path} to {hdfs_path}')


class CheckpointManagerV1:
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

    def __init__(self, model: FSDP, optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler, tokenizer: PreTrainedTokenizer):
        self.upload_sharded_future = None
        self.upload_merged_future = None
        self.previous_save_local_path = None

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        assert isinstance(self.model, FSDP)
        self.rank = torch.distributed.get_rank()

    def load_checkpoint(self, hdfs_path=None, device_mesh: DeviceMesh = None):
        if hdfs_path is None:
            return

        # every rank download its own checkpoint
        state_idx = device_mesh.get_local_rank(device_mesh.ndim - 1)
        remote_path = os.path.join(hdfs_path, f'model_optim_rank_{state_idx}.pt')
        print(f'[rank-{self.rank}]: Loading from {remote_path}')
        local_path = copy_local_path_from_hdfs(remote_path)

        state_dict = torch.load(local_path)

        model_state_dict = state_dict['model']
        optimizer_state_dict = state_dict['optimizer']
        lr_scheduler_state_dict = state_dict['lr_scheduler']

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def save_checkpoint(self, local_path: str, hdfs_path: str, device_mesh: DeviceMesh):
        # wait for previous upload to hdfs
        if self.upload_sharded_future is not None:
            ray.get(self.upload_sharded_future)
        if self.upload_merged_future is not None:
            ray.get(self.upload_merged_future)

        # remove previous local_path
        if self.previous_save_local_path is not None:
            previous_save_local_path = os.path.join(self.previous_save_local_path, f'model_optim_rank_{self.rank}.pt')
            shutil.rmtree(previous_save_local_path, ignore_errors=True)

        with FileLock(os.path.join(tempfile.gettempdir(), local_path + '.lock')):
            # make a new dir
            os.makedirs(local_path, exist_ok=True)

        torch.distributed.barrier()

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

        if self.rank == 0:
            hdfs_io.makedirs(hdfs_path, exist_ok=True)
        # wait for everyone to dump to local
        torch.distributed.barrier()

        # Very important: We must specify NodeAffinitySchedulingStrategy so that upload_ckpt will be
        # scheduled on the same node as the caller. Otherwise, it may be scheduled on another node that
        # causes local_dir not found error.

        # upload to hdfs
        if hdfs_path is not None and should_save_ckpt:
            # everyone upload its own checkpoint
            assert os.path.isfile(path), f'local path {path} does not exist'
            self.upload_sharded_future = upload_ckpt.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            )).remote(path, hdfs_path)

        if self.rank == 0:
            hf_local_path = os.path.join(local_path, 'huggingface')
            os.makedirs(hf_local_path, exist_ok=True)
            self.model._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
            self.tokenizer.save_pretrained(hf_local_path)
            if hdfs_path is not None:
                self.upload_merged_future = upload_ckpt.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )).remote(hf_local_path, hdfs_path)

        torch.distributed.barrier()

        self.previous_save_local_path = local_path
