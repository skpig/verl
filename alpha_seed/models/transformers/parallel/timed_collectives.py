from typing import Optional
import torch.distributed as dist
import torch

# if ndtimeline not init, use logic in this file
# otherwise, use ndtimeline patched logic


class TimedDistOP:
    EP_AR = "ep-ar"
    TP_ARI = "tp-ari"
    TP_IAR = "tp-iar"
    EP_ARI = "ep-ari"
    EP_IAR = "ep-iar"

    @staticmethod
    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False, name: Optional[str] = None):
        return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)
