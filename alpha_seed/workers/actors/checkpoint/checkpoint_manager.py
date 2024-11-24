from .checkpoint_v1 import CheckpointManagerV1
from .checkpoint_v2 import CheckpointManagerV2


class CheckpointManager:

    def __init__(self, *args, **kwargs):
        self.checkpoint_manager_v1 = CheckpointManagerV1(*args, **kwargs)
        self.checkpoint_manager_v2 = CheckpointManagerV2(*args, **kwargs)

    def load_checkpoint(self, version, *args, **kwargs):
        if version == "v1":
            return self.checkpoint_manager_v1.load_checkpoint(*args, **kwargs)
        elif version == "v2":
            return self.checkpoint_manager_v2.load_checkpoint(*args, **kwargs)
        else:
            raise ValueError("Invalid version number. Use 'v1' or 'v2'.")

    def save_checkpoint(self, version, *args, **kwargs):
        if version == "v1":
            return self.checkpoint_manager_v1.save_checkpoint(*args, **kwargs)
        elif version == "v2":
            return self.checkpoint_manager_v2.save_checkpoint(*args, **kwargs)
        else:
            raise ValueError("Invalid version number. Use 'v1' or 'v2'.")
