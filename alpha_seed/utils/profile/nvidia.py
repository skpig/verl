import glob

from typing import List


def get_my_gpu_ids() -> List[str]:
    ret = []
    for dev in glob.glob('/dev/nvidia[0-9]'):
        gpu_id = dev[len('/dev/nvidia'):]
        ret.append(gpu_id)
    return ret
