try:
    import ujson as json
except ImportError:
    import json
import functools
import os
import socket
import time
from typing import List
import ray
from torch.profiler import profile, schedule, ProfilerActivity

from alpha_seed.utils.profile.timeline import make_pid
import torch
import torch.nn as nn
from torch import distributed as dist


class TorchProfiler(object):

    def __init__(self, start_step, end_step=None):
        if start_step <= 2:
            raise ValueError('start_step must be greater than 2 to ensure the warmup')
        if end_step < start_step:
            raise ValueError(f'end_step({end_step}) must be greater equal start_step({start_step})')
        self.start_step = start_step
        self.end_step = end_step or start_step

        self.warmup_steps = 2
        self.wait_steps = start_step - self.warmup_steps
        self.active_steps = self.end_step - self.start_step + 1  # end是闭区间

        self.prof = None
        self.step_count = 0  # step start from 1
        self._saved = False  # 用来记录是否调用过export_trace
        self._started = False  # 用来记录是否调用过start
        self._stopped = False  # 用来记录是否调用过stop

    def started(self):
        return self._started

    def stopped(self):
        return self._stopped

    def start(self):
        if self._started:
            return
        self._started = True

        # initialize here
        self.prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=self.wait_steps, warmup=self.warmup_steps, active=self.active_steps, repeat=1),
            with_stack=True,
            with_modules=True,
        )

        self.prof.start()
        print(f'start(id={id(self.prof)}), step=', self.step_count)

    def step(self):
        if self.reached_end_step():
            self.stop()
        else:
            # step start from 1, increase first
            self.step_count += 1
            self.prof.step()
            print('step, step=', self.step_count)

    def stop(self):
        if not self._stopped:
            self.prof.stop()
            print('stop, step=', self.step_count)
            self._stopped = True

    def export_trace(self, filename_tpl):
        if self._saved:
            return
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()
        env = {
            'rank': rank,
            'start_step': self.start_step,
            'end_step': self.end_step,
            'pid': os.getpid(),
            'ts': int(time.time()),
        }
        filename = filename_tpl.format(**env)
        self.prof.export_chrome_trace(filename)
        print(f'torch profile trace saved at {filename}')
        self._saved = True

    def reached_end_step(self):
        return self.step_count > self.end_step

    def profile_saved(self):
        return self._saved

    def __enter__(self):
        # 进入上下文时开始分析
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出上下文时结束分析
        self.stop()


def torch_trace_this(start_step, end_step):
    profiler = TorchProfiler(start_step, end_step)

    # 不能在这里就调用start，因为func可能是一个remote call，这里不一定是GPU runtime

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # 这里才是真的GPU runtime，虽然这样写有点丑
            if not profiler.started():
                profiler.start()

            ret = func(*args, **kwargs)
            profiler.step()

            if profiler.reached_end_step() and profiler.stopped() and not profiler.profile_saved():
                tpl = f'torch_trace_{func.__name__}_' + 'r{rank}_s{start_step}-{end_step}_p{pid}_t{ts}.json'
                profiler.export_trace(tpl)

            return ret

        return wrapper

    return decorator


# Define a simple neural network layer
class SimpleLayer(nn.Module):

    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.fc(x)


def test_profiler_context_manager():
    # Initialize the neural network layer
    model = SimpleLayer()
    model.cuda()

    # Initialize the TorchProfiler for steps 5 to 6
    profiler = TorchProfiler(start_step=5, end_step=6, filename_tpl="torch_trace.json")

    with profiler:
        for i in range(10):
            # Generate some random input data
            input_data = torch.randn(10, 1024).to('cuda')
            # Perform a forward pass through the model
            output = model(input_data).to('cpu')
            # Step the profiler
            profiler.step()

    # Print the profiling results
    print(profiler.prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    profiler.export_trace("torch_trace.json")


@torch_trace_this(5, 6)
def test_compute_layer(data):
    model = SimpleLayer()
    model.cuda()
    input_data = data.to('cuda')
    output = model(input_data).to('cpu')
    return output


def test_profiler_decorator():
    for i in range(10):
        # Generate some random input data
        input_data = torch.randn(10, 1024)
        test_compute_layer(input_data)


def test_ray_remote_actor():
    import ray

    ray.init(num_gpus=1)

    @ray.remote(num_gpus=1)
    class RayRemoteSimpleLayer(SimpleLayer):

        def __init__(self):
            super().__init__()
            self.cuda()

        @torch_trace_this(5, 6)
        def forward(self, x):
            x = x.cuda()
            return super().forward(x).to('cpu')

    @ray.remote
    def another_remote_func():
        # Initialize the neural network layer
        model = RayRemoteSimpleLayer.remote()
        for i in range(10):
            # Generate some random input data
            input_data = torch.randn(10, 1024)
            # Perform a forward pass through the model
            output = ray.get(model.forward.remote(input_data))
            # Step the profiler
            print(output.shape)

    ray.get(another_remote_func.remote())


if __name__ == '__main__':
    test_ray_remote_actor()
