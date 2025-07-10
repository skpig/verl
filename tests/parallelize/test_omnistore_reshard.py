import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import torch.distributed as dist
from tests.launch import torchrun
from functools import partial
from torch.distributed._tensor import DTensor

from tests.parallelize.test_save_load import build_model, train_one_step, save_omnistore, load_omnistore
from omnistore.utilities.ckpt_format.merge_tool import omnistore_ckpt_to_pytorch_ckpt
from omnistore.planner.common import GLOBAL_PLAN_CACHE


def diff(x1, x2, prefix=()):
    """Recursive diff of dicts.

    Args:
        x1 (object): left dict
        x2 (object): right dict
        prefix (tuple): tracks recursive calls. Used for reporting differing keys.

    Returns:
        Tuple[list, list, list]: tuple of:
            - only_left: Prefixes present only in left dict
            - only_right: Prefixes present only in right dict
            - mismatch: values present in both dicts but not equal across dicts.
                For tensors equality of all elems is checked.
                Each element is a tuple (prefix, type of left value, type of right value).
    """
    mismatch = []
    if isinstance(x1, dict) and isinstance(x2, dict):
        only_left = [prefix + (k,) for k in x1.keys() - x2.keys()]
        only_right = [prefix + (k,) for k in x2.keys() - x1.keys()]
        for k in x2.keys() & x1.keys():
            _left, _right, _mismatch = diff(x1[k], x2[k], prefix + (k,))
            only_left.extend(_left)
            only_right.extend(_right)
            mismatch.extend(_mismatch)
    elif isinstance(x1, list) and isinstance(x2, list):
        only_left = list(range(len(x1) - 1, len(x2) - 1, -1))
        only_right = list(range(len(x1) - 1, len(x2) - 1, -1))
        for i, (v1, v2) in enumerate(zip(x1, x2)):
            _left, _right, _mismatch = diff(v1, v2, prefix + (i,))
            only_left.extend(_left)
            only_right.extend(_right)
            mismatch.extend(_mismatch)
    else:
        only_left = []
        only_right = []
        if isinstance(x1, DTensor) and isinstance(x2, DTensor):
            _is_mismatch = not torch.allclose(
                x1._local_tensor, x2._local_tensor, equal_nan=True, rtol=1e-16, atol=1e-16)
        elif isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            _is_mismatch = not torch.allclose(x1, x2, equal_nan=True, rtol=1e-16, atol=1e-16)
        else:
            try:
                _is_mismatch = bool(x1 != x2)
            except RuntimeError:
                _is_mismatch = True

        if _is_mismatch:
            mismatch.append((prefix, x1, x2))

    return only_left, only_right, mismatch


def model_save_load_fsdp_hsdp_tp_omnistore_reshard(fsdp_size_save: int,
                                                   tp_size_save: int,
                                                   fsdp_size_load: int,
                                                   tp_size_load: int,
                                                   optimizer_type: str = 'adam'):

    model_save, optim_save, meshes = build_model(fsdp_size=fsdp_size_save,
                                                 tp_size=tp_size_save,
                                                 optimizer_type=optimizer_type)
    train_one_step(model_save, optim_save)
    # save
    save_omnistore(model_save, optim_save, f"/tmp/ckpt/fsdp_{fsdp_size_save}_tp_{tp_size_save}")

    train_one_step(model_save, optim_save)
    model_load, optim_load, meshes = build_model(fsdp_size=fsdp_size_load,
                                                 tp_size=tp_size_load,
                                                 optimizer_type=optimizer_type)
    GLOBAL_PLAN_CACHE.clear()
    # load
    load_omnistore(model_load, optim_load, f"/tmp/ckpt/fsdp_{fsdp_size_save}_tp_{tp_size_save}")
    # save again
    save_omnistore(model_load, optim_load, f"/tmp/ckpt/fsdp_{fsdp_size_load}_tp_{tp_size_load}")

    if dist.get_rank() == 0:
        state_dict_for_compare_save = omnistore_ckpt_to_pytorch_ckpt(
            f"/tmp/ckpt/fsdp_{fsdp_size_save}_tp_{tp_size_save}",
            os.path.join(f"/tmp/ckpt/fsdp_{fsdp_size_save}_tp_{tp_size_save}", 'merged'),
            'fsdp',
            fsdp_save_flatten_model=False,
            return_dict=True,
        )
        state_dict_for_compare_load = omnistore_ckpt_to_pytorch_ckpt(
            f"/tmp/ckpt/fsdp_{fsdp_size_load}_tp_{tp_size_load}",
            os.path.join(f"/tmp/ckpt/fsdp_{fsdp_size_load}_tp_{tp_size_load}", 'merged'),
            'fsdp',
            fsdp_save_flatten_model=False,
            return_dict=True,
        )
        model_diffs = diff(state_dict_for_compare_save["model"], state_dict_for_compare_load["model"])
        assert not any(map(bool, model_diffs)), model_diffs
        optimizer_diffs = diff(state_dict_for_compare_save["optimizer"], state_dict_for_compare_load["optimizer"])
        assert not any(map(bool, optimizer_diffs)), optimizer_diffs
        print(
            f"test done, no diff in model / optimizer, model diffs: {model_diffs}, optimizer diffs: {optimizer_diffs}")


test_model_save_load_fsdp_hsdp_tp_omnistore_reshard = partial(torchrun, 8,
                                                              model_save_load_fsdp_hsdp_tp_omnistore_reshard, 8, 1, 2,
                                                              4)
# test_model_save_load_fsdp_hsdp_tp_omnistore_reshard = partial(torchrun, 8,
#                                                               model_save_load_fsdp_hsdp_tp_omnistore_reshard, 8, 1, 2,
#                                                               4, 'lion')
