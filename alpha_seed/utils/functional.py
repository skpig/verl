import torch
from mono_rl import DataProto
from verl.utils.seqlen_balancing import rearrange_micro_batches


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


def rearrange_micro_data_proto(max_token_len, mini_batch):
    micro_batches, indices = rearrange_micro_batches(mini_batch.batch, max_token_len=max_token_len)
    non_tensor_batches = []
    for indice_group in indices:
        non_tensor_batch = {}
        for idx in indice_group:
            for key in mini_batch.non_tensor_batch.keys():
                if key not in non_tensor_batch:
                    non_tensor_batch[key] = []
                non_tensor_batch[key].append(mini_batch.non_tensor_batch[key][idx])
        non_tensor_batches.append(non_tensor_batch)
    return indices, micro_batches, non_tensor_batches


def update_model_config(module_config, override_config_kwargs):
    for key, val in override_config_kwargs.items():
        if isinstance(val, dict):
            update_model_config(getattr(module_config, key), val)
        else:
            if not hasattr(module_config, key):
                print(f"WARN: {key} not exists in {module_config}", flush=True)
            setattr(module_config, key, val)


def print_dataproto_size(data: DataProto, head):
    size_of_tensordict = 0
    for key, tensor in data.batch.items():
        size_of_tensordict += tensor.element_size() * tensor.numel()
    size_of_numpy_array = 0
    for key, numpy_array in data.non_tensor_batch.items():
        size_of_numpy_array += numpy_array.nbytes

    size_of_numpy_array /= 1024**3
    size_of_tensordict /= 1024**3
    print(f'{head}, Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB')
