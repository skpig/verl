import os
from omnistore.utilities.io import bfile
from omnistore.api.meta_type import _DIRECTORY_FORMAT


def validate_ckpt(path, iteration, async_rollout=False, server_mode=None):
    ckpt_path = os.path.join(path, _DIRECTORY_FORMAT.format(iteration))
    if not bfile.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return
    if async_rollout and server_mode != 'server':
        standalone_path = os.path.join(ckpt_path, "standalone_gen_batch_output.batch.pt")
        if not bfile.exists(standalone_path):
            print(f"standalone_gen_batch_output.batch.pt does not exist: {standalone_path}")
            return
    return ckpt_path


def find_latest_ckpt_path_(path, async_rollout=False, server_mode=None):
    if path is None:
        return None

    tracker_file = os.path.join(path, "latest_checkpointed_iteration.txt")
    if not bfile.exists(tracker_file):
        print("Checkpoint does not exist: %s", tracker_file)
        return None

    with bfile.BFile(tracker_file, "rb", skip_encryption=True) as f:
        iteration = int(f.read().decode())
    while iteration >= 0:
        ckpt_path = validate_ckpt(path, iteration, async_rollout, server_mode)
        if ckpt_path or not async_rollout:
            return ckpt_path
        iteration -= 1
