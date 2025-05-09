import asyncio
import json
import os
from alpha_seed.workers.agents.envs import create_agent_envs_from_str


def test_add():
    kwargs = {'env_type': 'basic', 'env_args': {'round_ndigits': 2}}
    env = create_agent_envs_from_str(f'example_env@{json.dumps(kwargs)}')[0]

    res = asyncio.run(env.step(action="Add(x=1.02592, y=3.1415926)"))
    assert res == "4.17"


def test_pwd():
    kwargs = {'env_type': 'subprocess', 'env_args': {}}
    env = create_agent_envs_from_str(f'example_env@{json.dumps(kwargs)}')[0]

    res = asyncio.run(env.step(action="Pwd()"))
    import os

    assert os.path.isdir(res)


def test_now():
    kwargs = {'env_type': 'basic', 'env_args': {'strftime_format': r"%Y-%m-%d %H:%M:%S"}}
    env = create_agent_envs_from_str(f'example_env@{json.dumps(kwargs)}')[0]

    res = asyncio.run(env.step(action="Now()"))
    print(res)
    import datetime

    _ = datetime.datetime.strptime(res, kwargs['env_args']['strftime_format'])


def test_sleep():
    kwargs = {'env_type': 'basic', 'env_args': {}}
    env = create_agent_envs_from_str(f'example_env@{json.dumps(kwargs)}')[0]

    res = asyncio.run(env.step(action="Sleep(seconds=0.5)"))
    assert res == "Slept for 0.5 seconds."


def test_append():
    kwargs = {'env_type': 'stateful', 'env_args': {}}
    env = create_agent_envs_from_str(f'example_env@{json.dumps(kwargs)}')[0]

    async def main():
        tasks = [asyncio.create_task(env.step(action=f"Add_(val={i})")) for i in range(10)]
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert env.sum == 45
