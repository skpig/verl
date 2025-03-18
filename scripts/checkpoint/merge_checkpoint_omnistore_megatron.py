"""
Example usage: python3 scripts/checkpoint/merge_checkpoint_omnistore_megatron.py
--load-dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangmofan/test/new_omnistore_ckpt_folder_format/checkpoints/global_step_1/actor/
"""

import argparse
import os
import time

from omnistore.utilities.ckpt_format.merge_tool import omnistore_ckpt_to_pytorch_ckpt
from transformers import PretrainedConfig
from verl.utils.fs import copy_local_path_from_hdfs


def prepare_megatron_merge_kwargs(megatron_configs_dir):
    model_config = PretrainedConfig.from_json_file(os.path.join(megatron_configs_dir, 'model_config.json'))
    model_config = model_config.to_dict()
    megatron_merge_kwargs = {
        'noop_layer_ids': model_config.get('noop_transformer_layers', None),
        'untie_embeddings': not model_config.get('tie_weight', True),
        'q_scale_factor': model_config.get('query_head_scale_factor', None),
        'compatible_with_megatron_inference': True,
    }
    print(f'Megatron merge kwargs: {megatron_merge_kwargs}')
    return megatron_merge_kwargs


if __name__ == '__main__':
    total_time_begin = time.time()
    print('Step1: prepare args and folders')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load-dir',
        required=True,
        help='the HDFS directory in the form of default_hdfs_dir/checkpoints/global_step_1/actor/, and the directory '
        'contains the omnistore model/ subdirectory and huggingface/ subdirectory')
    parser.add_argument('--save-path', required=False)
    parser.add_argument('--hf-dir',
                        required=False,
                        help='the directory of the corresponding HuggingFace configs. If not specified, it is assumed '
                        'that the HuggingFace subdirectory is under the load-dir directory, in the form of '
                        '${load-dir}/huggingface/.')
    args, unknown = parser.parse_known_args()
    print(f'Ignore unknown arguments: {unknown}')

    if not args.save_path:
        args.save_path = args.load_dir
    print(f'Complete save dir path for merge checkpoint: {args.save_path}')

    megatron_path = copy_local_path_from_hdfs(os.path.join(args.load_dir, 'megatron'))
    merge_kwargs = prepare_megatron_merge_kwargs(megatron_path)

    print('Step2: merge omnistore ckpt and upload converted xperf model')
    time_begin = time.time()
    _ = omnistore_ckpt_to_pytorch_ckpt(
        args.load_dir,
        os.path.join(args.save_path, 'megatron'),
        'megatron',
        model_only=True,
        safetensors_format=False,
        return_dict=False,
        **merge_kwargs,
    )
    print(f'Merge omnistore checkpoint successfully! cost time: {time.time() - time_begin}s')
