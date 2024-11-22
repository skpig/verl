import bytedredis
from triton.runtime.cache import RedisRemoteCacheBackend

BYTED_SEED_KV_PSM = "toutiao.redis.seed_kv"


class BytedRedisRemoteCacheBackend(RedisRemoteCacheBackend):

    def __init__(self, key):
        super().__init__(key)
        self._redis = bytedredis.Client.from_url(
            f"redis://?db=0&redis_psm={BYTED_SEED_KV_PSM}&socket_connect_timeout=0.25&socket_timeout=0.3")
