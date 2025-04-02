"""
Example usage: python3 scripts/checkpoint/merge_checkpoint_omnistore_megatron.py
--load-dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangmofan/test/new_omnistore_ckpt_folder_format/checkpoints/global_step_1/actor/
"""

import argparse
import os
import time
import tempfile
import torch
import threading

from alpha_seed.utils.ckpt.hdfs import prepare_hdfs_copy_kwargs

import hdfs_io
from omnistore.utilities.ckpt_format.merge_tool import omnistore_ckpt_to_pytorch_ckpt
from packaging.version import Version
from transformers import PretrainedConfig
from verl.utils.fs import copy_local_path_from_hdfs

REQUIRED_SEED_MODELS_VERSION = '1.2.3'


def check_seed_models_version():
    from seed_models import __version__
    assert Version(__version__) >= Version(REQUIRED_SEED_MODELS_VERSION), \
        (f'seed_models version {__version__} is too old. Please upgrade to version {REQUIRED_SEED_MODELS_VERSION} '
         'or higher.')


check_seed_models_version()
from seed_models.utils.ckpt.checkpoint_utils import LLM_CONVERT_FUNC


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


def save_local_and_upload_merged_megatron_ckpt(pt_state_dict, merged_model_save_path):
    time_begin = time.time()
    with tempfile.TemporaryDirectory() as local_tmp_dir:
        local_megatron_merge_path = os.path.join(local_tmp_dir, 'megatron_merge_states.pt')
        torch.save(pt_state_dict, local_megatron_merge_path)

        if not hdfs_io.exists(merged_model_save_path):
            hdfs_io.makedirs(merged_model_save_path)
        hdfs_io.copy(local_megatron_merge_path, merged_model_save_path, **prepare_hdfs_copy_kwargs())
    print(f'Save to local and upload merged megatron ckpt cost time: {time.time() - time_begin}s')


def infer_model_type_and_convert_to_hf(pt_state_dict, hf_path, save_path):
    time_begin = time.time()
    hf_config = PretrainedConfig.from_pretrained(hf_path)
    parts = hf_config.architectures[0].split("For")
    if not parts:
        print(
            f'Cannot infer model type from huggingface config, where architectures field is {hf_config.architectures}')
        return
    model_type = parts[0]
    if model_type in LLM_CONVERT_FUNC:
        LLM_CONVERT_FUNC[model_type](pt_path=None,
                                     pt_state_dict=pt_state_dict,
                                     hf_path=hf_path,
                                     safe_serialization=True)

    hdfs_io.copy(hf_path, save_path, **prepare_hdfs_copy_kwargs())
    print(f'Infer model type and convert to hf cost time: {time.time() - time_begin}')


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

    if args.hf_dir:
        hf_path = copy_local_path_from_hdfs(args.hf_dir)
    else:
        hf_path = copy_local_path_from_hdfs(os.path.join(args.load_dir, 'huggingface'))

    megatron_path = copy_local_path_from_hdfs(os.path.join(args.load_dir, 'megatron'))
    merge_kwargs = prepare_megatron_merge_kwargs(megatron_path)

    print('Step2: merge omnistore ckpt and convert to xperf model')
    time_begin = time.time()
    state_dict = omnistore_ckpt_to_pytorch_ckpt(
        args.load_dir,
        os.path.join(args.save_path, 'megatron'),
        'megatron',
        model_only=True,
        safetensors_format=False,
        return_dict=True,
        **merge_kwargs,
    )
    print(f'Merge omnistore checkpoint successfully! cost time: {time.time() - time_begin}s')

    thread_map = {}
    print('Step3: async save and upload converted xperf model')
    thread_map['xperf'] = threading.Thread(target=save_local_and_upload_merged_megatron_ckpt,
                                           args=(state_dict['model'], os.path.join(args.save_path, 'megatron')))
    thread_map['xperf'].start()

    print('Step4: async convert to hf format and save')
    thread_map['hf'] = threading.Thread(
        target=infer_model_type_and_convert_to_hf,
        args=(state_dict['model'], hf_path, args.save_path),
    )
    thread_map['hf'].start()

    print('Step5: wait async upload hf and xperf merged ckpt')
    time_begin = time.time()
    for k, v in thread_map.items():
        v.join()
        print(f'Wait async upload for {k} finished!')
    print(f'Wait async upload cost time: {time.time() - time_begin}s')
    print(f'Total time cost: {time.time() - total_time_begin}s')
