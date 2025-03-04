import os
from verl.utils.fs import copy_local_path_from_hdfs, md5_encode
from hdfs_io import copy
from filelock import FileLock

cache_dir = "/var/tmp"


def download_minimal_required_files(model_path, from_scratch, rank, world_size):
    if not from_scratch:
        return download_config_and_tokenizer(model_path)
    else:
        return download_limited_chunks(model_path, rank, world_size)


def download_limited_chunks(model_path, rank, world_size):
    return copy_local_path_from_hdfs(model_path, cache_dir=cache_dir)


def download_config_and_tokenizer(model_path):
    download_files = ['config.json', 'tokenizer.json', 'special_tokens_map.json', 'tokenizer_config.json']
    local_path = copy_local_path_from_hdfs_files(model_path, download_files, cache_dir)
    return local_path


def get_local_dir(hdfs_path: str, cache_dir: str) -> str:
    """Return a local temp cache_dir
    Args:
        hdfs_path:
        cache_dir:
    """
    # make a base64 encoding of hdfs_path to avoid directory conflict
    encoded_hdfs_path = md5_encode(hdfs_path)
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)
    return temp_dir


def copy_local_path_from_hdfs_files(src: str, files: list, cache_dir=None, filelock='.file.lock', verbose=False) -> str:
    assert src[-1] != '/', f'Make sure the last char in src is not / because it will cause error. Got {src}'
    os.makedirs(cache_dir, exist_ok=True)
    assert os.path.exists(cache_dir)

    joint_path = "".join([os.path.join(src, fn) for fn in files])
    local_folder_path = get_local_dir(joint_path, cache_dir)

    # get a specific lock
    filelock = md5_encode(src) + '.lock'
    lock_file = os.path.join(cache_dir, filelock)
    with FileLock(lock_file=lock_file):
        if not os.path.exists(local_folder_path):
            os.makedirs(local_folder_path, exist_ok=True)
            if verbose:
                print(f'Copy from {src} to {local_folder_path}')
            for file_name in files:
                remote_path = os.path.join(src, file_name)
                print(f"copying file {remote_path} to {local_folder_path}")
                copy(remote_path, local_folder_path)
    return local_folder_path
