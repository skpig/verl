import ray
import os

import warnings

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig

from verl.utils.fs import copy_local_path_from_hdfs

from transformers import PreTrainedTokenizer

from .checkpoint_manager import BaseCheckpointManager
from .uploader import CkptGlobalUploader


class CheckpointManagerV1(BaseCheckpointManager):
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
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        super().__init__(model, optimizer, lr_scheduler, tokenizer)

    def load_checkpoint(self, hdfs_path=None, *args, **kwargs):
        if hdfs_path is None:
            return

        # every rank download its own checkpoint
        remote_path = os.path.join(hdfs_path, f'model_optim_rank_{self.rank}.pt')
        print(f'[rank-{self.rank}]: Loading from {remote_path}')
        local_path = copy_local_path_from_hdfs(remote_path)

        state_dict = torch.load(local_path)
        try:
            os.remove(local_path)
        except Exception as e:
            print(f'[rank-{self.rank}]: remove local load ckpt file failed, exception {e} will be ignored')

        model_state_dict = state_dict['model']
        optimizer_state_dict = state_dict['optimizer']
        lr_scheduler_state_dict = state_dict['lr_scheduler']

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if 'rng' in state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(state_dict['rng'])

        self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def save_checkpoint(self, local_path: str, hdfs_path: str, role: str, global_step: int,
                        ckpt_global_uploader_ref: CkptGlobalUploader, *args, **kwargs):
        # wait for previous upload to hdfs
        self.wait_previous_upload(role, ckpt_global_uploader_ref)
        self.previous_global_step = global_step

        # remove previous local_path
        self.remove_previous_save_local_path()
        self.local_mkdir(local_path)
        torch.distributed.barrier()

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
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'rng': self.get_rng_state(),
                }
                path = os.path.join(local_path, f'model_optim_rank_{self.rank}.pt')

                print(f'[rank-{self.rank}]: Saving checkpoint to {os.path.abspath(path)}')
                torch.save(state_dict, path)

        if hdfs_path is not None:
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
                ckpt_global_uploader_ref.start_uploading.remote(role, global_step)

        torch.distributed.barrier()

        self.previous_save_local_path = local_path
