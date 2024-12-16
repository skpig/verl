from .checkpoint_v1 import CheckpointManagerV1
from .checkpoint_v2 import CheckpointManagerV2
from .checkpoint_omnistore import CheckpointManagerOmniStore


class CheckpointManagerWrapper:

    def __init__(self, *args, **kwargs):
        self.checkpoint_manager_map = {
            'v1': CheckpointManagerV1(*args, **kwargs),
            'v2': CheckpointManagerV2(*args, **kwargs),
            'omnistore': CheckpointManagerOmniStore(*args, **kwargs)
        }

    def load_checkpoint(self, version, *args, **kwargs):
        if version not in self.checkpoint_manager_map:
            raise ValueError(
                f'Checkpoint manager wrapper got invalid version number {version}. Please use v1, v2 or omnistore')
        else:
            self.checkpoint_manager_map[version].load_checkpoint(*args, **kwargs)

    def save_checkpoint(self, version, *args, **kwargs):
        if version not in self.checkpoint_manager_map:
            raise ValueError(
                f'Checkpoint manager wrapper got invalid version number {version}. Please use v1, v2 or omnistore')
        else:
            self.checkpoint_manager_map[version].save_checkpoint(*args, **kwargs)