import torch


def clip_by_value_preserve_gradient(t, min=None, max=None):
    """
    Copied from https://github.com/vermouth1992/rlutils-python/blob/aab1d670860cfdf53fd8805528ec11ef0bfdf015/rlutils/pytorch/functional.py#L46
    """
    clip_t = torch.clip(t, min=min, max=max)
    return t + (clip_t - t).detach()


def get_text_config(config):
    if hasattr(config, "text_config"):
        return config.text_config
    else:
        return config


def get_text_model_type(config):
    return get_text_config(config).model_type
