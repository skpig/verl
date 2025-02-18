import os
import sys
import time
import logging
from typing import Any

# 记录跟踪信息
last_times = {}
last_lineno = {}

logger = logging.getLogger("trace")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("trace.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def local_line_trace(frame, event, arg):
    """局部跟踪器：计算代码行执行耗时"""
    fid = id(frame)

    if event == 'line':
        current_time = time.perf_counter()
        current_lineno = frame.f_lineno

        if fid in last_times:
            elapsed = current_time - last_times[fid]
            caller = frame.f_back.f_code.co_name if frame.f_back else "None"
            callee = frame.f_code.co_name
            class_name = frame.f_locals["self"].__class__.__name__ if "self" in frame.f_locals else ""
            callee = (class_name if "self" in frame.f_locals else "") + "." + callee
            logger.info(
                f"[LineProfiler] {caller} -> {callee} in {frame.f_code.co_filename}:{last_lineno[fid]} took {elapsed:.6f} seconds"
            )

        last_lineno[fid] = current_lineno  # 记录上一行
        last_times[fid] = current_time  # 更新时间

    elif event == 'return':
        last_times.pop(fid, None)  # 清理记录

    return local_line_trace


class TracerContextManager:

    def __init__(self) -> None:
        """初始化时，从环境变量获取要跟踪的函数"""
        self.trace_methods = set(filter(None, os.getenv("TRACING_FUNCTIONS", "").split(",")))
        self.conflict = os.getenv('DISABLE_BGCP') != '1'
        self.switch = True if len(self.trace_methods) > 0 else False

    def __enter__(self):
        """进入上下文管理器时，启动全局跟踪"""
        if self.switch:
            sys.settrace(self.global_trace)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """退出上下文管理器时，停止跟踪"""
        if self.switch:
            sys.settrace(None)

    def global_trace(self, frame, event, arg):
        """
        全局跟踪器：当进入目标函数时返回局部跟踪器，
        只跟踪在 `TRACING_FUNCTIONS` 环境变量中指定的函数。
        """
        if event == 'call':
            func_name = frame.f_code.co_name

            if func_name in self.trace_methods:
                return local_line_trace

            if func_name == '__init__' and "self" in frame.f_locals:
                class_name = frame.f_locals["self"].__class__.__name__
                full_method_name = f"{class_name}.__init__"

                if full_method_name in self.trace_methods:
                    return local_line_trace
        return None


# 测试上下文管理器
if __name__ == "__main__":
    os.environ["TRACING_FUNCTIONS"] = "init_workers,worker_function"  # 设置要跟踪的函数

    with TracerContextManager():

        def worker_function():
            time.sleep(1.2)

        def init_workers():
            worker_function()

        init_workers()
