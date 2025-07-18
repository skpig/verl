import json
import os
import time
import random
import requests
from bytedtcc import ClientV2


class TccV3(object):
    _instance = None

    arnold_region = os.getenv("ARNOLD_REGION", "CN")
    if arnold_region == "CN":
        host = "paas-gw.byted.org"
        secret = "a4d276bcb60ca262ef52597c30feaffc"
        region = "CN"
    else:
        host = "paas-gw-i18n.byted.org"
        secret = "443b1bf49d6d24f8c29475c6fd428bbf"
        region = "US-East"
    print(f"tcc host:{host}")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_agent_config(cls):
        if os.environ.get("ARNOLD_RL_ENV", '') == '1':
            tcc_client = ClientV2('mlsys.arnold.metis', 'default')
            return json.loads(tcc_client.get("agent_conf"))
        else:
            data = {"ns_name": "data.aml.seed_eval", "region": cls.region, "dir": "/default", "conf_name": "agent_conf"}
            return cls.get_config_with_retrying(data)

    @classmethod
    def get_config_with_retrying(cls, data: dict, skip_retry=False):
        url = f"http://{cls.host}/bcc/open/config/get"
        headers = {"Domain": "tcc_v3_openapi", "Authorization": f"Bearer {cls.secret}"}
        tcc_config = None
        for i in range(5):
            try:
                r = requests.get(url=url, headers=headers, params=data)
                payload = r.json()
                if payload["base_resp"]["error_code"] != 0:
                    print(f"get config from tcc failed: {payload['base_resp']['error_message']}")
                tcc_config = payload["data"]["version_data"]["data"]
                break
            except Exception as e:
                print(f"get config from tcc failed, retry times:{i}, Exception: {e}, data: {data}")
                if skip_retry:
                    break
                sleep_time = random.randint(10, 30)
                time.sleep(sleep_time)
        if tcc_config is None:
            print(f"get config from tcc failed, raise Exception, data: {data}")
            raise Exception("get config from tcc failed")
        return json.loads(tcc_config)


if __name__ == '__main__':
    cfg = TccV3().get_agent_config()
    print('cfg', cfg)
    registered_bench_hosts = cfg.get("swe", {}).get("bench_hosts", [])
    print('registered_bench_hosts', registered_bench_hosts)
    registered_bench_repo2hosts = cfg.get("swe", {}).get("bench_repo2hosts", {})
    print('registered_bench_repo2hosts', registered_bench_repo2hosts)
