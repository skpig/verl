import os
import json
from typing import Dict, List, Optional
import bytedredis
from triton.runtime.cache import RedisRemoteCacheBackend, RemoteCacheBackend, RemoteCacheManager
from concurrent.futures import ThreadPoolExecutor
import pickle
import uuid
import logging

logger = logging.getLogger(__name__)

ARNOLD_REGION = os.getenv("ARNOLD_REGION", "CN")
if ARNOLD_REGION == "CN":
    BYTED_SEED_KV_PSM = "toutiao.redis.seed_kv"
else:
    BYTED_SEED_KV_PSM = "toutiao.redis.seed_kv.service.maliva"


class BytedRedisRemoteCacheBackend(RedisRemoteCacheBackend):

    def __init__(self, key):
        super().__init__(key)
        self._redis = bytedredis.Client.from_url(
            f"redis://?db=0&redis_psm={BYTED_SEED_KV_PSM}&socket_connect_timeout=25&socket_timeout=30")

    def get(self, filenames: List[str]) -> Dict[str, str]:
        try:
            results = self._redis.mget([self._get_key(f) for f in filenames])
        except Exception as e:
            print(e)
            results = [None] * len(filenames)

        return {filename: result for filename, result in zip(filenames, results) if result is not None}


class HdfsRemoteCacheBackend(RemoteCacheBackend):

    def __init__(self, key):
        self._key = key
        self._key_fmt = os.environ.get("TRITON_REDIS_KEY_FORMAT", "triton:{key}:{filename}")
        assert 'HDFS_REMOTE_CACHE_DIR' in os.environ
        self.root = os.environ['HDFS_REMOTE_CACHE_DIR']
        os.makedirs(self.root, exist_ok=True)
        # 创建线程池，控制并发写入的线程数量
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.memory_cache = {}

    def _get_key(self, filename: str) -> str:
        return self._key_fmt.format(key=self._key, filename=filename)

    def _hget(self, filename):
        key = self._get_key(filename)
        if key in self.memory_cache:
            return self.memory_cache[key]
        cache_path = os.path.join(self.root, key)
        if not os.path.exists(cache_path):
            return
        try:
            with open(cache_path, 'rb') as f:
                loaded = pickle.load(f)
                self.memory_cache[key] = loaded
                return loaded
        except:
            return

    def get(self, filenames: List[str]) -> Dict[str, str]:
        results = [self._hget(f) for f in filenames]
        return {filename: result for filename, result in zip(filenames, results) if result is not None}

    def _write_to_hdfs(self, key, data):
        cache_path = os.path.join(self.root, key)
        tmp_path = cache_path + "_tmp_" + str(uuid.uuid4())
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(data, f)
            os.rename(tmp_path, cache_path)
        except Exception as e:
            logger.warning(f"[HDFS Cache WARN] write {cache_path} failed: {e}")

    def put(self, filename: str, data: bytes) -> Dict[str, bytes]:
        key = self._get_key(filename)
        self.memory_cache[key] = data
        self.executor.submit(self._write_to_hdfs, key, data)


class HdfsRemoteCacheManager(RemoteCacheManager):

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.get_group(filename)

        grp_filename = f"__grp__{filename}"
        grp_filepath = self.get_file(grp_filename)
        if grp_filepath is None:
            return None
        with open(grp_filepath) as f:
            grp_data = json.load(f)
        child_paths = grp_data.get("child_paths", None)

        result = None

        # Found group data.
        if child_paths is not None:
            result = {}
            for child_path, data in self._backend.get(child_paths).items():
                result[child_path] = self._materialize(child_path, data)
        # ['*.cubin', '*.json', '*.llir', '*.ptx', '*.ttgir', '*.ttir']
        if len(result) != 6:
            logger.warning(f"[HDFS Cache WARN] {grp_filename} keys={list(result.keys())}, expect 6 files")
            return None
        return result
