import os
import argparse
from omnistore.utilities.ckpt_format.merge_tool import omnistore_ckpt_to_pytorch_ckpt
from typing import Any, Tuple
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor


def diff(x1: Any, x2: Any, prefix: Tuple = ()) -> Tuple[list, list, list]:
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
            print(f"key: {k}")
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
            print(f"tensor1: {x1} shape {x1.shape}; tensor2: {x2} shape {x2.shape}")
            _is_mismatch = not torch.allclose(x1, x2, equal_nan=True, rtol=1e-16, atol=1e-16)
        else:
            try:
                _is_mismatch = bool(x1 != x2)
            except RuntimeError:
                _is_mismatch = True

        if _is_mismatch:
            print("prefix: ", prefix, " x1:", x1, " x2:", x2)
            mismatch.append((prefix, type(x1), type(x2)))

    return only_left, only_right, mismatch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir1', required=True)
    parser.add_argument('--ckpt-dir2', required=True)

    parser.add_argument('--flatten-ckpt1',
                        action='store_true',
                        help='should be set explicitly if config trainer.ckpt_enable_flatten is enabled when training',
                        default=False)
    parser.add_argument('--flatten-ckpt2',
                        action='store_true',
                        help='should be set explicitly if config trainer.ckpt_enable_flatten is enabled when training',
                        default=False)
    parser.add_argument('--untie-embeddings1', action='store_true', default=False)
    parser.add_argument('--untie-embeddings2', action='store_true', default=False)
    parser.add_argument('--model', action='store_true', default=False)
    parser.add_argument('--optimizer', action='store_true', default=False)
    args = parser.parse_args()

    local_dir = '/opt/tiger/.cache/src_model'
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(os.path.join(local_dir, 'ckpt1'), exist_ok=True)
    os.makedirs(os.path.join(local_dir, 'ckpt2'), exist_ok=True)

    state_dict1 = omnistore_ckpt_to_pytorch_ckpt(
        args.ckpt_dir1,
        os.path.join(local_dir, 'ckpt1'),
        'fsdp',
        model_only=True if args.model and not args.optimizer else False,
        optimizer_only=True if args.optimizer and not args.model else False,
        fsdp_save_flatten_model=args.flatten_ckpt1,
        return_dict=True,
        untie_embeddings=args.untie_embeddings1,
    )
    state_dict2 = omnistore_ckpt_to_pytorch_ckpt(
        args.ckpt_dir2,
        os.path.join(local_dir, 'ckpt2'),
        'fsdp',
        model_only=True if args.model and not args.optimizer else False,
        optimizer_only=True if args.optimizer and not args.model else False,
        fsdp_save_flatten_model=args.flatten_ckpt2,
        return_dict=True,
        untie_embeddings=args.untie_embeddings2,
    )

    if args.model:
        print(f"model 1: {state_dict1['model']}")
        print(f"model 2: {state_dict2['model']}")
        model_diff = diff(state_dict1['model'], state_dict2['model'])
        print(f'model diff: {model_diff}')
    if args.optimizer:
        print(f"optimizer 1: {state_dict1['optimizer']}")
        print(f"optimizer 2: {state_dict2['optimizer']}")
        optimizer_diff = diff(state_dict1['optimizer'], state_dict2['optimizer'])
        print(f'optimizer diff: {optimizer_diff}')
