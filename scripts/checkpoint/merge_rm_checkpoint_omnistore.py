import os
import argparse
import time
import re
import hdfs_io
import torch
from verl.utils.fs import copy_local_path_from_hdfs
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification
from omnistore.utilities.ckpt_format.merge_tool import omnistore_ckpt_to_pytorch_ckpt
from seed_models.commands.convert_to_megatron import convert_seed_models_to_megatron

if __name__ == '__main__':
    print('Warning: This script is going to be deprecated and will be removed in the near future. Please use '
          'scripts/checkpoint/merge_checkpoint_omnistore.py instead.')
    print('Step1: prepare args and folders')
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', required=True)
    parser.add_argument('--save-path', required=False)
    args = parser.parse_args()

    match = re.search(r'global_step_(\d+)', args.load_dir)
    if not match:
        assert False, "load-dir must ends with global_step_xxx"

    if not args.save_path:
        args.save_path = args.load_dir
    print(f'Complete save dir path for merge checkpoint: {args.save_path}')

    print(f"Downloading ckpt from {args.load_dir}")
    local_dir = copy_local_path_from_hdfs(args.load_dir)
    hf_load_dir = os.path.join(args.load_dir, "../../huggingface")
    print(f"Downloading hf path from {hf_load_dir}")
    hf_path = copy_local_path_from_hdfs(hf_load_dir)

    print('Step2: merge omnistore ckpt to get state_dict')
    time_begin = time.time()
    state_dict = omnistore_ckpt_to_pytorch_ckpt(
        args.load_dir,
        local_dir,
        'fsdp',
        model_only=True,
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

    print(f'Upload huggingface model from {hf_path} to {args.load_dir}')
    hdfs_io.copy(hf_path, args.load_dir)

    # upload back to hdfs
    print(f'Step4: convert model to megatron/xperf format and upload to {args.save_path}')
    # convert to megatron for autoeval
    megatron_save_path = os.path.join(args.load_dir, "megatron")
    convert_seed_models_to_megatron(hf_path=hf_path,
                                    local_path=local_dir,
                                    output_path=megatron_save_path,
                                    validate=False)
