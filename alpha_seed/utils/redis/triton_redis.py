import os
import bytedredis
from triton.runtime.cache import RedisRemoteCacheBackend

ARNOLD_REGION = os.getenv("ARNOLD_REGION", "CN")
if ARNOLD_REGION == "CN":
    # NOTE(zr): we could not use the default psm in CN, because 'lf' does not exit.
    BYTED_SEED_KV_PSM = "toutiao.redis.seed_kv.service.lq"
else:
    BYTED_SEED_KV_PSM = ""


class BytedRedisRemoteCacheBackend(RedisRemoteCacheBackend):

    def __init__(self, key):
        super().__init__(key)
        self._redis = bytedredis.Client.from_url(
            f"redis://?db=0&redis_psm={BYTED_SEED_KV_PSM}&socket_connect_timeout=25&socket_timeout=30")
