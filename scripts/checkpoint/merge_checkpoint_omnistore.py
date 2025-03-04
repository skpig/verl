"""
Example usage: python3 scripts/checkpoint/merge_checkpoint_omnistore.py
--load-dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangmofan/test/new_omnistore_ckpt_folder_format/checkpoints/global_step_1/actor/
--save_hf
"""

import os
import argparse
import time
import hdfs_io
import torch
from verl.utils.fs import copy_local_path_from_hdfs
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification
from omnistore.utilities.ckpt_format.merge_tool import omnistore_ckpt_to_pytorch_ckpt
from seed_models.commands.convert_to_megatron import convert_seed_models_to_megatron

if __name__ == '__main__':
    print('Step1: prepare args and folders')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load-dir',
        required=True,
        help='the HDFS directory in the form of default_hdfs_dir/checkpoints/global_step_1/actor/, and the directory '
        'contains the omnistore model/ subdirectory and huggingface/ subdirectory')
    parser.add_argument('--save-path', required=False)
    # for compatibility with merlin auto eval
    parser.add_argument('--cruise-config', required=False)
    parser.add_argument('--dtype', required=False)
    parser.add_argument('--save_hf', action='store_true')
    args = parser.parse_args()

    if not args.save_path:
        args.save_path = args.load_dir
    print(f'Complete save dir path for merge checkpoint: {args.save_path}')

    local_dir = '/opt/tiger/.cache/src_model'
    os.makedirs(local_dir, exist_ok=True)
    hf_path = copy_local_path_from_hdfs(os.path.join(args.load_dir, 'huggingface'))

    print('Step2: merge omnistore ckpt to get state_dict')
    time_begin = time.time()
    state_dict = omnistore_ckpt_to_pytorch_ckpt(
        args.load_dir,
        local_dir,
        'fsdp',
        model_only=True,
        fsdp_save_flatten_model=args.flatten_ckpt,
        safetensors_format=True,
        return_dict=True,
    )
    print(f'Merge omnistore checkpoint successfully! cost time: {time.time() - time_begin}s')
    config = AutoConfig.from_pretrained(hf_path)

    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    else:
        raise NotImplementedError(f'Unknown architecture {config["architectures"]}')

    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Step3: saving merged model to local {hf_path}')
    model.save_pretrained(hf_path, state_dict=state_dict['model'])

    del state_dict
    del model

    if args.save_hf:
        print(f'Upload huggingface model from {hf_path} to {args.load_dir}')
        hdfs_io.copy(hf_path, args.load_dir)

    # upload back to hdfs
    print(f'Step4: convert model to xperf format and upload to {args.save_path}')
    # convert to megatron for autoeval
    megatron_save_path = os.path.join(args.save_path, 'megatron')
    convert_seed_models_to_megatron(hf_path=hf_path,
                                    local_path=local_dir,
                                    output_path=megatron_save_path,
                                    validate=False)
