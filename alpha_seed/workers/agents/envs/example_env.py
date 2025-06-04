from typing import *
import asyncio
import json
import subprocess
from alpha_seed.workers.agents.envs import BaseEnv
from alpha_seed.workers.agents.envs.utils import parse_func_call_kwargs
"""An Example for how to write an env class.
Supported actions:
BasicExampleEnv:
- Now(): print current date and time
- Add(x, y): calculation and print result
- Sleep(seconds) : sleep for specified seconds

SubprocessExampleEnv:
- Pwd() : print current directory path

StatefulExampleEnv:
- Add_(val): add value to self.sum, this illustrates how to prevent arbitrary order of async __call__
"""


class BasicExampleEnv(BaseEnv):

    def __init__(self, round_ndigits: int = 3, strftime_format: str = r'%Y-%m-%d %H:%M:%S', **kwargs):
        self.round_ndigits = round_ndigits
        self.strftime_format = strftime_format
        self.values = []

    def action_supported(self, action: str) -> bool:
        try:
            func_name, _ = parse_func_call_kwargs(action)
            return func_name in ['Now', 'Add', 'Sleep']
        except:
            pass
        return False

    def _now(self) -> str:
        import datetime
        return f"{datetime.datetime.now().strftime(self.strftime_format)}"

    def _add(self, x, y):
        res = round(x + y, self.round_ndigits)
        return f"{res}"

    async def _sleep(self, seconds):
        await asyncio.sleep(seconds)
        return f"Slept for {seconds} seconds."

    async def step(self, action: str) -> str:
        func_name, kwargs = parse_func_call_kwargs(action)
        if func_name == 'Now':
            return self._now(**kwargs)
        elif func_name == 'Add':
            return self._add(**kwargs)
        elif func_name == 'Sleep':
            return await self._sleep(**kwargs)
        else:
            raise NotImplementedError(f"unexpected func_name={func_name}")


class SubprocessExampleEnv(BaseEnv):

    def __init__(self, **kwargs):
        pass

    def action_supported(self, action: str) -> bool:
        try:
            func_name, _ = parse_func_call_kwargs(action)
            return func_name in ['Pwd']
        except:
            pass
        return False

    async def _pwd(self):
        p = await asyncio.create_subprocess_exec('pwd', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await p.communicate()
        return stdout.decode().strip()

    async def step(self, action: str) -> str:
        func_name, kwargs = parse_func_call_kwargs(action)
        if func_name == "Pwd":
            return await self._pwd(**kwargs)
        else:
            raise NotImplementedError(f"unexpected func_name={func_name}")


class StatefulExampleEnv(BaseEnv):

    def __init__(self, **kwargs):
        self._lock = asyncio.Lock()
        self.sum = 0
        self._val = None

    def action_supported(self, action: str) -> bool:
        try:
            func_name, _ = parse_func_call_kwargs(action)
            return func_name in ['Add_', 'Set_', 'Get']
        except:
            pass
        return False

    async def step(self, action: str) -> str:
        func_name, kwargs = parse_func_call_kwargs(action)
        if func_name == "Add_":
            val = kwargs['val']
            async with self._lock:
                # This example demonstrate when it is necessary to use an asyncio.Lock.
                # - The lock is not necessary in most cases because asyncio is single-threaded.
                # - The lock should be used in a minimum scope to prevent the tasks' execution being serialized.

                # A lock is required for the following case:
                # a stateful value is read and cached in local variable
                # -> await something (which yields control to the event loop, the value maybe updated by another coroutine)
                # -> the value is updated based on the cached value.
                prev_sum = self.sum
                await asyncio.sleep(0.1 / (round(val) + 1))  # prev_sum becomes stale
                new_sum = prev_sum + val
                self.sum = new_sum
                return f'prev_sum:{prev_sum} + val:{val} = new_sum:{new_sum}'
        elif func_name == "Set_":
            val = kwargs['val']
            await asyncio.sleep(0.1)  # mock overhead
            self._val = val
            return f"set value={val}"
        elif func_name == 'Get':
            return f"get value={self._val}"
        else:
            raise NotImplementedError(f"unexpected func_name={func_name}")


def create_from_env_str(env_str: str, **kwargs):
    prefix = 'example_env@'
    assert env_str.startswith(prefix)
    kwargs = json.loads(env_str[len(prefix):])
    env_type = kwargs['env_type']
    env_args = kwargs['env_args']
    if env_type.lower() == 'basic':
        return BasicExampleEnv(**env_args)
    elif env_type.lower() == 'subprocess':
        return SubprocessExampleEnv(**env_args)
    elif env_type.lower() == 'stateful':
        return StatefulExampleEnv(**env_args)
    else:
        raise NotImplementedError(f"unsupported env_type={env_type}")
