# this is a null monitor, in order to drop in aiomonitor to disable the feature,
# as well as keep the code working wo/ aiomonitor
class NullMonitor:

    def __init__(self, loop, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_aiomonitor_cls(enabled):
    if enabled:
        try:
            import aiomonitor
            return aiomonitor.Monitor
        except ImportError:
            return NullMonitor
    else:
        return NullMonitor
