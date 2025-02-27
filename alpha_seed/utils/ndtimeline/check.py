import os
from packaging.version import Version


def version_checker() -> str:
    NDTIMELINE_BASE_VERSION = "2.2.14"
    NDTIMELINE_HIGH_VERSION = "3.0.0"
    err_msg = f"bytedance.ndtimeline is not installed or not proper. Try to " + \
    f"set env ALPHASEED_USE_NDTIMELINE=0 disable it or install bytedance.ndtimeline >={NDTIMELINE_BASE_VERSION} <{NDTIMELINE_HIGH_VERSION}." +\
    f"\n Training will not exit and bytedance.ndtimeline will be automatically disabled."
    try:
        from bytedance.ndtimeline import __version__
        if Version(__version__) < Version(NDTIMELINE_BASE_VERSION) or Version(__version__) >= Version(
                NDTIMELINE_HIGH_VERSION):
            print(err_msg)
            return err_msg
        return ""
    except (ImportError, ModuleNotFoundError):
        print(err_msg)
        return err_msg


USE_CUDA_TIMER = os.getenv("ALPHASEED_USE_NDTIMELINE", "0") == "1" and len(version_checker()) == 0


def use_cuda_timer():
    global USE_CUDA_TIMER
    if USE_CUDA_TIMER and "NDTIMELINE_LOG_LEVEL" not in os.environ:
        os.environ["NDTIMELINE_LOG_LEVEL"] = "WARNING"
    return USE_CUDA_TIMER
