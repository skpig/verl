import warnings
import cloudpickle
import lmdb

from alpha_seed.utils.store.kv import KVDB


class LMDBKV(KVDB):

    # 注意，如果共用了path，则会互相看到keys，key不相同的话问题不大
    def __init__(self, lmdb_path='/tmp/lmdb.db', map_size=1000 * 2**30):
        """
        Args:
            lmdb_path: LMDB数据库路径
            map_size: 数据库最大大小（字节），默认10GB
        """
        self.lmdb_path = lmdb_path
        # 创建LMDB环境，map_size指定最大数据库大小
        self.env = lmdb.open(lmdb_path, map_size=map_size)

    def get(self, key: str, default=None):
        """
        get key if exists, otherwise return default
        """
        with self.env.begin() as txn:
            value_bytes = txn.get(key.encode('utf-8'))
            if value_bytes is None:
                return default
            try:
                return cloudpickle.loads(value_bytes)
            except Exception:
                return default

    def put(self, key: str, value):
        """
        put key-value pair. evict previous if exists
        如果空间满了，会发出警告并静默丢弃数据
        """
        try:
            with self.env.begin(write=True) as txn:
                value_bytes = cloudpickle.dumps(value)
                txn.put(key.encode('utf-8'), value_bytes)
        except lmdb.MapFullError:
            # map_size已满，发出警告并丢弃数据
            warnings.warn(
                f"LMDB map_size limit reached. Current map_size: {self.env.info()['map_size'] / (1024**3):.2f}GB. "
                f"Discarding data for key '{key}'. Consider increasing map_size.",
                RuntimeWarning,
                stacklevel=2)
        except OSError as e:
            # 磁盘空间不足或其他IO错误
            if e.errno == 28:  # ENOSPC: No space left on device
                warnings.warn(
                    f"Disk space full. LMDB failed to write to {self.lmdb_path}. "
                    f"Discarding data for key '{key}'. Free up disk space or use a different path.",
                    RuntimeWarning,
                    stacklevel=2)
            else:
                # 其他IO错误仍然抛出
                raise

    def delete(self, key: str):
        """
        delete key if exists
        """
        with self.env.begin(write=True) as txn:
            txn.delete(key.encode('utf-8'))

    def __len__(self):
        with self.env.begin() as txn:
            return txn.stat()['entries']

    def close(self):
        """关闭LMDB环境"""
        if self.env:
            self.env.close()

    def __del__(self):
        """析构时关闭环境"""
        self.close()
