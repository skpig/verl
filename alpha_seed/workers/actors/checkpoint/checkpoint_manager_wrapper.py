from .checkpoint_v1 import CheckpointManagerV1
from .checkpoint_v2 import CheckpointManagerV2
from .checkpoint_omnistore import CheckpointManagerOmniStore


class CheckpointManagerWrapper:

    def __init__(self, strategy: str = 'fsdp', *args, **kwargs):
        if strategy in ('fsdp', 'vescale-fsdp2'):
            self.checkpoint_manager_map = {
                'v1': CheckpointManagerV1(*args, **kwargs),
                'v2': CheckpointManagerV2(*args, **kwargs),
                'omnistore': CheckpointManagerOmniStore(*args, **kwargs)
            }
        elif strategy == 'megatron':
            self.checkpoint_manager_map = {'omnistore': CheckpointManagerOmniStore(*args, **kwargs)}
        else:
            raise NotImplementedError(f'No checkpoint manager supports strategy {strategy}')

    def load_checkpoint(self, version, *args, **kwargs):
        if version not in self.checkpoint_manager_map:
            raise ValueError(
                f'Checkpoint manager wrapper got invalid version {version}. Please use {self.checkpoint_manager_map.keys()}'
            )
        else:
            self.checkpoint_manager_map[version].load_checkpoint(*args, **kwargs)

    def save_checkpoint(self, version, *args, **kwargs):
        if version not in self.checkpoint_manager_map:
            raise ValueError(
                f'Checkpoint manager wrapper got invalid version {version}. Please use {self.checkpoint_manager_map.keys()}'
            )
        else:
            self.checkpoint_manager_map[version].save_checkpoint(*args, **kwargs)
