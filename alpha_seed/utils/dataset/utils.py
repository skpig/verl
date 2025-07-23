import numpy as np
import torch


def convert_tensor_to_numpy(tensor):
    if tensor is None or isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, list):
        return [convert_tensor_to_numpy(t) for t in tensor]
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy()
    return tensor.numpy()


def convert_numpy_to_tensor(numpy_array, dtype=None):
    if numpy_array is None or isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if isinstance(numpy_array, list):
        return [convert_numpy_to_tensor(t, dtype) for t in numpy_array]
    if numpy_array.dtype == np.uint16:
        return torch.from_numpy(numpy_array).view(torch.bfloat16)
    if dtype is not None:
        return torch.from_numpy(numpy_array.astype(dtype))
    return torch.from_numpy(numpy_array)
