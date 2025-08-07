import warnings
import logging
import os


class LogFilter(logging.Filter):

    def __init__(self):
        self.patterns = []

    def add_pattern(self, msg: str):
        if msg not in self.patterns:
            self.patterns.append(msg)

    def filter(self, record):
        for pattern in self.patterns:
            if pattern in record.getMessage():
                return False
        return True


log_filter = LogFilter()
logging.root.addFilter(log_filter)

orig_getLogger = logging.getLogger


def getLogger(name=None):
    logger = orig_getLogger(name)
    logger.addFilter(log_filter)
    return logger


logging.getLogger = getLogger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def refine_log():

    # - bytedmetric
    warnings.filterwarnings("ignore", "bytedmetrics is renamed", category=UserWarning)

    # - verl
    warnings.filterwarnings("ignore", "Open-source MegatronPPOActor", category=UserWarning)
    warnings.filterwarnings("ignore", "MegatronPPOActor", category=UserWarning)
    warnings.filterwarnings("ignore", "MegatronPPOCritic", category=UserWarning)

    # - seed_models
    log_filter.add_pattern("npu is not supported")
    log_filter.add_pattern("Seed Models will run on")
    log_filter.add_pattern("Flash Attention 2.0")
    log_filter.add_pattern("`rope_scaling`")

    # - dist_attn
    log_filter.add_pattern("Failed to find")
    log_filter.add_pattern("Unable to import")

    # - bumi
    log_filter.add_pattern("Bumi is running")
    log_filter.add_pattern("byted-triton is not installed properly")

    # - set logging level
    try:
        logging.getLogger().setLevel(os.getenv("LOGGING_LEVEL", logging.getLogger().getEffectiveLevel()))
    except:
        pass
