from typing import List, Tuple, Dict
import shutil
import seed_models
import os
import torch
import argparse
import seed_models
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq
from concurrent.futures import ThreadPoolExecutor
import hdfs_io
from tqdm.auto import trange
from seed_models.commands.convert_to_megatron import convert_seed_models_to_megatron
from torch.distributed._tensor import DTensor, Shard, Placement


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', required=True)
    parser.add_argument('--save-path', required=False)
    # for compatibility with merlin auto eval
    parser.add_argument('--cruise-config', required=False)
    parser.add_argument('--dtype', required=False)
    parser.add_argument('--save_hf', action='store_true')
    args = parser.parse_args()

    print('Downloading model shards')
    local_dir = '/opt/tiger/.cache/src_model'
    shutil.rmtree(local_dir, ignore_errors=True)
    os.makedirs(local_dir, exist_ok=True)

    # if not args.load_dir.endswith('actor'):
    #     args.load_dir = os.path.join(args.load_dir, 'actor')

    if not args.save_path:
        args.save_path = os.path.join(args.load_dir, 'megatron_merge_states.pt')

    hdfs_hf_path = os.path.join(args.load_dir, 'huggingface')
    # hdfs_io.copy(args.load_dir, local_dir)
    hdfs_io.copy(hdfs_hf_path, os.path.join(local_dir, 'huggingface'))

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

    assert mesh_dim_names in (
        ('fsdp',),
        ('dp', 'fsdp'),
        ('fsdp', 'tp'),
        ('dp', 'fsdp', 'tp'),
    ), f'Unsupported mesh_dim_names {mesh_dim_names}'

    if 'tp' in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f'Processing model shards with {total_shards} {mesh_shape} in total')

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        hdfs_io.copy(os.path.join(args.load_dir, f'model_optim_rank_{rank}.pt'),
                     os.path.join(local_dir, f'model_optim_rank_{rank}.pt'),
                     chunk_thread_num=16)
        model_path = os.path.join(local_dir, f'model_optim_rank_{rank}.pt')
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        model_state_dict_lst[rank] = state_dict['model']
        os.remove(model_path)

    with ThreadPoolExecutor(max_workers=32) as executor:
        for rank in trange(1, total_shards, desc='Loading model shards'):
            executor.submit(process_one_shard, rank)

    # reorder model_state_dict based on keys
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            tensor = model_state_dict.pop(key)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == 'dp':
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key] = tensor.bfloat16()

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        # merge shards
        placements: Tuple[Shard] = param_placements[key]
        if len(mesh_shape) == 1:
            # 1-D list, FSDP without TP
            assert len(placements) == 1
            shards = state_dict[key]
            state_dict[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D list, FSDP + TP
            assert len(placements) == 2 and len(mesh_shape) == 2
            flatten = state_dict[key]
            shards = []
            for i in range(mesh_shape[0]):
                tp_shards = [flatten[i * mesh_shape[1] + j] for j in range(mesh_shape[1])]
                shards.append(tp_shards)
            # merge at tp dimension
            tp_placement = placements[1]
            fsdp_shards = [merge_by_placement(tp_shards, tp_placement) for tp_shards in shards]
            # merge at fsdp dimension
            fsdp_placement = placements[0]
            state_dict[key] = merge_by_placement(fsdp_shards, fsdp_placement)
        print(f'Merged {key} shape: {state_dict[key].size()}')

    print('Writing to local disk')
    hf_path = os.path.join(local_dir, 'huggingface')
    config = AutoConfig.from_pretrained(hf_path)

    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif 'ForConditionalGeneration' in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f'Unknown architecture {config["architectures"]}')

    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving model to {hf_path}')
    model.save_pretrained(hf_path, state_dict=state_dict)

    del state_dict
    del model

    # print(f'Upload merged huggingface model from {hf_path} to {args.hdfs_path}')
    # upload back to hdfs
    if args.save_hf:
        print(f'Upload huggingface model from {hf_path} to {args.load_dir}')
        hdfs_io.copy(hf_path, args.load_dir)

    # convert to megatron for autoeval
    # if 'ForCausalLM' in config.architectures[0]:
    # only save ForCausalLM
    print(f'Upload merged megatron model from {hf_path} to {args.save_path}')
    convert_seed_models_to_megatron(hf_path=hf_path,
                                    local_path=local_dir,
                                    output_path=os.path.dirname(args.save_path),
                                    validate=False)
