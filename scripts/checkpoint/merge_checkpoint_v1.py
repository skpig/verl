from verl.utils.fs import copy_local_path_from_hdfs
import seed_models
import os

import torch
import torch.distributed

import argparse

from transformers import AutoConfig, AutoModelForCausalLM

import hdfs_io

from tqdm.auto import trange

from seed_models.commands.convert_to_megatron import convert_seed_models_to_megatron

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_path', required=True)
    parser.add_argument('--fsdp_size', required=False, type=int, default=-1)
    args = parser.parse_args()

    print('Downloading model shards')

    local_path = copy_local_path_from_hdfs(args.hdfs_path)

    # find how many shards
    files = [filename for filename in os.listdir(local_path) if filename.startswith('model_optim_rank')]
    # to support HSDP with v1 ckpt
    total_shards = len(files) if args.fsdp_size == -1 else args.fsdp_size

    print('Processing model shards')

    model_state_dict_lst = []
    for rank in trange(total_shards, desc='Loading model shards'):
        model_path = os.path.join(local_path, f'model_optim_rank_{rank}.pt')
        state_dict = torch.load(model_path, map_location='cpu')
        model_state_dict_lst.append(state_dict['model'])

    # reorder model_state_dict based on keys

    state_dict = {}

    keys = set(model_state_dict_lst[0].keys())

    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            tensor = model_state_dict.pop(key)
            if isinstance(tensor, torch.distributed._tensor.DTensor):
                state_dict[key].append(tensor._local_tensor)
            else:
                state_dict[key] = tensor

    for key in sorted(state_dict):
        # FSDP is shard-0 dtensor
        if isinstance(state_dict[key], list):
            print(f'Concat key {key}')
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print('Writing to local disk')

    hf_path = os.path.join(local_path, 'huggingface')
    config = AutoConfig.from_pretrained(hf_path)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)

    print(f'Saving model to {hf_path}')
    model.save_pretrained(hf_path, state_dict=state_dict)

    print(f'Upload merged huggingface model from {hf_path} to {args.hdfs_path}')

    # upload back to hdfs
    hdfs_io.copy(hf_path, args.hdfs_path)

    print(f'Upload merged megatron model from {hf_path} to {args.hdfs_path}')

    # convert to megatron for autoeval
    convert_seed_models_to_megatron(hf_path=hf_path, local_path=hf_path, output_path=args.hdfs_path)
