from .checkpoint_v1 import CheckpointManagerV1
from .checkpoint_v2 import CheckpointManagerV2
from .checkpoint_omnistore import CheckpointManagerOmniStore


class CheckpointManager:

    def __init__(self, *args, **kwargs):
        self.checkpoint_manager_omnistore = CheckpointManagerOmniStore(*args, **kwargs)
        kwargs.pop('enable_flatten', False)
        self.checkpoint_manager_v1 = CheckpointManagerV1(*args, **kwargs)
        self.checkpoint_manager_v2 = CheckpointManagerV2(*args, **kwargs)

    def load_checkpoint(self, version, *args, **kwargs):
        if version == "v1":
            return self.checkpoint_manager_v1.load_checkpoint(*args, **kwargs)
        elif version == "v2":
            return self.checkpoint_manager_v2.load_checkpoint(*args, **kwargs)
        elif version == "omnistore":
            return self.checkpoint_manager_omnistore.load_checkpoint(*args, **kwargs)
        else:
            raise ValueError("Invalid version number. Use 'v1', 'v2' or 'omnistore.")

    def save_checkpoint(self, version, *args, **kwargs):
        if version == "v1":
            return self.checkpoint_manager_v1.save_checkpoint(*args, **kwargs)
        elif version == "v2":
            return self.checkpoint_manager_v2.save_checkpoint(*args, **kwargs)
        elif version == "omnistore":
            return self.checkpoint_manager_omnistore.save_checkpoint(*args, **kwargs)
        else:
            raise ValueError("Invalid version number. Use 'v1', 'v2' or 'omnistore'.")
