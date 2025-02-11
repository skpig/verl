import ray
import omegaconf
from omegaconf import OmegaConf, DictConfig


class KVStore:
    name = "kv_store"

    def __init__(self):
        self.kwargs = dict()

    def set_key_val(self, k, v):
        self.kwargs[k] = v

    def get_by_key(self, k):
        return self.kwargs.get(k, None)


class TaskRunnerBase:

    def main(self, func, *args, **kwargs):
        return func(*args, **kwargs)


@ray.remote
class TaskRunner(TaskRunnerBase):
    name = "task_runner"


@ray.remote
class ClientTaskRunner(TaskRunnerBase):
    name = "client_task_runner"


@ray.remote(num_gpus=0, num_cpus=1)
class ServerHealthCheck:
    name = "server_health_check"

    def __init__(self):
        self._is_ready = False

    def set_ready(self, ready):
        self._is_ready = ready

    def is_ready(self):
        return self._is_ready


def validate_client_config(config: DictConfig, ref_server_config: DictConfig, ref_common_config: DictConfig):
    kv_store = ray.get_actor(KVStore.name)
    server_config = ray.get(kv_store.get_by_key.remote("config"))
    assert server_config is not None, "get server config failed"

    def filter_config(target, ref):
        filtered = omegaconf.OmegaConf.create({})
        for key in target:
            if isinstance(target[key], DictConfig):
                if (key in ref) and isinstance(ref[key], DictConfig):
                    filtered[key] = filter_config(target[key], ref[key])
            elif key in ref:
                if not isinstance(ref[key], DictConfig):
                    filtered[key] = target[key]
        return filtered

    server_config = filter_config(server_config, ref_server_config)
    common_config_server = filter_config(server_config, ref_common_config)
    common_config_client = filter_config(config, ref_server_config)

    def assert_config_match(prefix, server_conf, client_conf):
        if (not isinstance(server_conf, DictConfig)) or (not isinstance(client_conf, DictConfig)):
            if server_conf != client_conf:
                raise ValueError(f"Common config [{prefix}] mismatch: {server_conf}(server) != {client_conf}(client)")
            return
        for key in server_conf:
            if key in client_conf:
                assert_config_match(prefix + "." + key, server_conf[key], client_conf[key])

    assert isinstance(common_config_server, DictConfig)
    assert isinstance(common_config_client, DictConfig)
    assert_config_match("", common_config_server, common_config_client)

    def override_with_warning(prefix, target, ref):
        for key in ref:
            nxt_prefix = f"{prefix}.{key}"
            assert key in target, f"{nxt_prefix} not in target config"
            if isinstance(ref[key], DictConfig):
                assert isinstance(target[key], DictConfig), f"{nxt_prefix} is dict in ref while not in target"
                override_with_warning(nxt_prefix, target[key], ref[key])
            else:
                if target[key] != ref[key]:
                    print(f"[WARN]:config{nxt_prefix} overrided by server config: [{target[key]}]->[{ref[key]}]")
                    target[key] = ref[key]

    # overwrite configs that should be server config but passed by different value by client
    # config = omegaconf.OmegaConf.merge(config, server_config)
    override_with_warning("", config, server_config)

    print(f"Client script final config: {omegaconf.OmegaConf.to_yaml(config)}")
    return config


def check_all_workers_alive(workers):
    from ray.experimental.state.api import get_actor
    for worker in workers:
        worker_state_dict = get_actor(worker._actor_id.hex())
        if worker_state_dict is None:
            return False
        if worker_state_dict.get("state", "undefined") != "ALIVE":
            return False
    return True


def recreate_actor(actor_cls, name, *args, **kwargs):
    try:
        # kill existing
        actor = ray.get_actor(name)
        ray.kill(actor)
    except Exception:
        pass

    if not hasattr(actor_cls, "options"):
        # wrap by ray.remote if the class is a plain class
        actor_cls = ray.remote(actor_cls)
    actor = actor_cls.options(name=name, *args, **kwargs).remote()
    return actor
