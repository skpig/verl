import os
import argparse
import time
import re
import hdfs_io
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification
from omnistore.utilities.ckpt_format.merge_tool import omnistore_ckpt_to_pytorch_ckpt
from seed_models.commands.convert_to_megatron import convert_seed_models_to_megatron

if __name__ == '__main__':
    print('Step1: prepare args and folders')
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', required=True)
    parser.add_argument('--save-path', required=False)
    parser.add_argument('--flatten-ckpt',
                        action='store_true',
                        help='should be set explicitly if config trainer.ckpt_enable_flatten is enabled when training',
                        default=False)
    # for compatibility with merlin auto eval
    parser.add_argument('--cruise-config', required=False)
    parser.add_argument('--dtype', required=False)
    parser.add_argument('--save_hf', action='store_true')
    args = parser.parse_args()

    if not args.load_dir.endswith('actor') and not args.load_dir.endswith('critic'):
        args.load_dir = os.path.join(args.load_dir, 'actor')

    if not args.save_path:
        args.save_path = args.load_dir
    print(f'Complete save dir path for merge checkpoint: {args.save_path}')

    local_dir = '/opt/tiger/.cache/src_model'
    os.makedirs(local_dir, exist_ok=True)
    hdfs_io.copy(os.path.join(args.load_dir, 'huggingface'), os.path.join(local_dir, 'huggingface'))
    hf_path = os.path.join(local_dir, 'huggingface')

    # prepare omnistore ckpt folder
    match = re.search(r'global_step_(\d+)', args.load_dir)
    if match:
        global_step = match.group(0)
        print(f'Extracted global step: {global_step}')
        args.load_dir = os.path.join(args.load_dir, global_step)
    print(f'Complete load dir path for merge checkpoint: {args.load_dir}')

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
    convert_seed_models_to_megatron(hf_path=hf_path,
                                    local_path=local_dir,
                                    output_path=os.path.dirname(args.save_path),
                                    validate=False)
