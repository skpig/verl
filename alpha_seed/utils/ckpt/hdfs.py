import os
from verl.utils.fs import copy_local_path_from_hdfs, md5_encode
from hdfs_io import copy, hexists
from filelock import FileLock
from seed_models.utils.envs import SeedModelsEnvs
from omnistore.utilities.io.bfile import is_local_path

cache_dir = "/var/tmp"


def download_minimal_required_files(model_path, from_scratch, rank, world_size):
    if not from_scratch:
        return download_config_and_tokenizer(model_path)
    else:
        return download_limited_chunks(model_path, rank, world_size)


def download_limited_chunks(model_path, rank, world_size):
    return copy_local_path_from_hdfs(model_path, cache_dir=cache_dir)


def download_config_and_tokenizer(model_path):
    download_files = [
        'config.json', 'tokenizer.json', 'special_tokens_map.json', 'tokenizer_config.json', 'preprocessor_config.json'
    ]
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
                if hexists(remote_path):
                    print(f"copying file {remote_path} to {local_folder_path}")
                    copy(remote_path, local_folder_path)
    return local_folder_path


def prepare_hdfs_copy_kwargs():
    """default in SeedModelsEnvs:
    HDFS_THREAD_NUM = int(os.getenv('HDFS_THREAD_NUM', '32'))
    HDFS_CHUNK_THREAD_NUM = int(os.environ.get('HDFS_CHUNK_THREAD_NUM', '32'))
    """
    hdfs_kwargs = {
        'thread_num': SeedModelsEnvs.HDFS_THREAD_NUM,
        'chunk_thread_num': SeedModelsEnvs.HDFS_CHUNK_THREAD_NUM,
    }
    return hdfs_kwargs


def hdfs_path_map2_mount_path(hdfs_path: str, rw: bool = False) -> str:
    if not hdfs_path.startswith("hdfs://"):
        return ""
    fuse_mount_maps = os.getenv("ARNOLD_HDFSFUSE_VOLUMES", None)
    if not fuse_mount_maps:
        return ""
    try:
        fuse_mount_maps = eval(fuse_mount_maps)
    except Exception as e:
        print(f'fuse_mount_maps={fuse_mount_maps} eval error: {e}')
        return ""
    hdfs_path = hdfs_path if hdfs_path.endswith("/") else hdfs_path + "/"
    longest_matched = None
    for record in fuse_mount_maps:
        if "roles" in record and len(record["roles"]) > 0 and os.getenv("ARNOLD_ROLE", "NONE") not in record["roles"]:
            continue
        if rw and record["access_mode"] != "RW":
            continue
        record_hdfs_path = record.get("hdfs_path", "NONE")
        record_hdfs_path = record_hdfs_path if record_hdfs_path.endswith("/") else record_hdfs_path + "/"
        if (hdfs_path.startswith(record_hdfs_path) and
            (not longest_matched or len(record_hdfs_path) > len(longest_matched["hdfs_path"]))):
            longest_matched = record
    if longest_matched:
        sub_path = hdfs_path[len(longest_matched["hdfs_path"]):].strip("/")
        return os.path.join(longest_matched.get("mount_path"), sub_path)
    return ""


def mount_path_map2_hdfs_path(mount_path: str) -> str:
    if not is_local_path(mount_path):
        return ""
    fuse_mount_maps = os.getenv("ARNOLD_HDFSFUSE_VOLUMES", None)
    if not fuse_mount_maps:
        return ""
    try:
        fuse_mount_maps = eval(fuse_mount_maps)
    except Exception as e:
        print(f'fuse_mount_maps={fuse_mount_maps} eval error: {e}')
        return ""
    mount_path = mount_path if mount_path.endswith("/") else mount_path + "/"
    longest_matched = None
    for record in fuse_mount_maps:
        if "roles" in record and len(record["roles"]) > 0 and os.getenv("ARNOLD_ROLE", "NONE") not in record["roles"]:
            continue
        record_mount_path = record.get("mount_path", "NONE")
        record_mount_path = record_mount_path if record_mount_path.endswith("/") else record_mount_path + "/"
        if (mount_path.startswith(record_mount_path) and
            (not longest_matched or len(record_mount_path) > len(longest_matched["mount_path"]))):
            longest_matched = record
    if longest_matched:
        sub_path = mount_path[len(longest_matched["mount_path"]):].strip("/")
        return os.path.join(longest_matched.get("hdfs_path"), sub_path)
    return ""
