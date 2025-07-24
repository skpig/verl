import os

if os.getenv('AGENTBENCH_ENABLE'):
    try:
        if os.getenv('AGENTBENCH_STORAGE_MODE', 'ray') == 'ray':
            import ray
            if os.getenv('AGENTBENCH_DEBUG_MODE'):
                _ray_init = ray.init

                def _spawn_agentbench_proxy_after_ray_init(*args, **kwargs):
                    rsp = _ray_init(*args, **kwargs)
                    if ray.get_runtime_context().get_actor_name() is None:
                        from .proxy import get_proxy_server
                        get_proxy_server()
                    return rsp

                ray.init = _spawn_agentbench_proxy_after_ray_init
            else:
                actor_name = ray.get_runtime_context().get_actor_name()
                if actor_name is None or actor_name == 'task_runner':
                    from .proxy import get_proxy_server
                    get_proxy_server()
        else:
            from .proxy import get_proxy_server
            get_proxy_server()
    except importerror:
        pass
    except exception:
        pass
