from verl.utils.fs import copy_local_path_from_hdfs
import seed_models
import os

import torch
import torch.distributed

import argparse

from transformers import AutoConfig, AutoModelForCausalLM

import hdfs_io

from tqdm.auto import trange

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdfs_path', required=True)
    args = parser.parse_args()

    print('Downloading model shards')

    local_path = copy_local_path_from_hdfs(args.hdfs_path)

    # find how many shards
    files = [filename for filename in os.listdir(local_path) if filename.startswith('model_optim_rank')]
    total_shards = len(files)

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

    for key in state_dict:
        # FSDP is shard-0 dtensor
        if isinstance(state_dict[key], list):
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print('Writing to local disk')

    config_path = os.path.join(local_path, 'huggingface')
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)

    print(f'Saving model to {config_path}')
    model.save_pretrained(config_path, state_dict=state_dict)

    print(f'Upload from {config_path} to {args.hdfs_path}')

    # upload back to hdfs
    hdfs_io.copy(config_path, args.hdfs_path)
