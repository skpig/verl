import torch


def clip_by_value_preserve_gradient(t, min=None, max=None):
    """
    Copied from https://github.com/vermouth1992/rlutils-python/blob/aab1d670860cfdf53fd8805528ec11ef0bfdf015/rlutils/pytorch/functional.py#L46
    """
    clip_t = torch.clip(t, min=min, max=max)
    return t + (clip_t - t).detach()
