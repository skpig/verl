import seed_models
import os
import torch
import argparse
import seed_models
from transformers import AutoConfig, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import hdfs_io
from tqdm.auto import trange
from seed_models.commands.convert_to_megatron import convert_seed_models_to_megatron
from torch.distributed._tensor import DTensor, Replicate, Shard

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', required=True)
    parser.add_argument('--save-path', required=False)
    # for compatibility with merlin auto eval
    parser.add_argument('--cruise-config', required=False)
    parser.add_argument('--dtype', required=False)
    args = parser.parse_args()

    print('Downloading model shards')
    local_dir = '/opt/tiger/.cache/src_model'
    os.makedirs(local_dir, exist_ok=True)

    if not args.load_dir.endswith('actor'):
        args.load_dir = os.path.join(args.load_dir, 'actor')

    if not args.save_path:
        args.save_path = os.path.join(args.load_dir, 'megatron_merge_states.pt')

    # hdfs_io.copy(args.load_dir, local_dir)
    hdfs_io.copy(os.path.join(args.load_dir, 'huggingface'), os.path.join(local_dir, 'huggingface'))

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    hdfs_io.copy(os.path.join(args.load_dir, f'model_optim_rank_{rank}.pt'),
                 os.path.join(local_dir, f'model_optim_rank_{rank}.pt'),
                 chunk_thread_num=16)
    state_dict = torch.load(os.path.join(local_dir, f'model_optim_rank_{rank}.pt'), map_location='cpu')['model']
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]
    assert isinstance(weight, torch.distributed._tensor.DTensor)
    # get sharding info
    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    print(f'Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}')

    assert mesh_dim_names in (('fsdp',), ('dp', 'fsdp'))

    total_shards = mesh.shape[-1]

    print(f'Processing model shards with {total_shards} in total')

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        hdfs_io.copy(os.path.join(args.load_dir, f'model_optim_rank_{rank}.pt'),
                     os.path.join(local_dir, f'model_optim_rank_{rank}.pt'),
                     chunk_thread_num=16)
        model_path = os.path.join(local_dir, f'model_optim_rank_{rank}.pt')
        state_dict = torch.load(model_path, map_location='cpu')
        model_state_dict_lst[rank] = state_dict['model']
        os.remove(model_path)

    with ThreadPoolExecutor(max_workers=32) as executor:
        for rank in trange(1, total_shards, desc='Loading model shards'):
            executor.submit(process_one_shard, rank)

    # reorder model_state_dict based on keys
    state_dict = {}
    shard_dim = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            tensor = model_state_dict.pop(key)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                if key in shard_dim:
                    assert shard_dim[key] == tensor.placements[-1]
                else:
                    shard_dim[key] = tensor.placements[-1]
            else:
                state_dict[key] = tensor.bfloat16()

    del model_state_dict_lst

    for key in sorted(state_dict):
        if isinstance(state_dict[key], list):
            if isinstance(shard_dim[key], Shard):
                sdim = shard_dim[key].dim
                print(f"Merging sharded tensor {key} at dimension {sdim}")
                state_dict[key] = torch.cat(state_dict[key], dim=sdim)
            elif isinstance(shard_dim[key], Replicate):
                print(f"Unexpected replicated tensor {key}. Only take the first one")
                state_dict[key] = state_dict[key][0]
            else:
                raise ValueError(f'Unknown shard dim {shard_dim[key]}')
            print(f'Merged {key} shape: {state_dict[key].size()}')
        else:
            print(f'No need to concat key {key}')
    print('Writing to local disk')
    hf_path = os.path.join(local_dir, 'huggingface')
    config = AutoConfig.from_pretrained(hf_path)

    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving model to {hf_path}')
    model.save_pretrained(hf_path, state_dict=state_dict)

    del state_dict
    del model

    # print(f'Upload merged huggingface model from {hf_path} to {args.hdfs_path}')
    # upload back to hdfs
    # hdfs_io.copy(hf_path, args.hdfs_path)
    print(f'Upload merged megatron model from {hf_path} to {args.save_path}')
    # convert to megatron for autoeval
    convert_seed_models_to_megatron(hf_path=hf_path,
                                    local_path=local_dir,
                                    output_path=os.path.dirname(args.save_path),
                                    validate=False)
