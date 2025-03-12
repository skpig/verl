"""
Example usage: python3 scripts/checkpoint/merge_checkpoint_omnistore.py
--load-dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangmofan/test/new_omnistore_ckpt_folder_format/checkpoints/global_step_1/actor/
"""

import argparse
import os
import tempfile
import threading
import time
from typing import Optional

import hdfs_io
import torch
from omnistore.utilities.ckpt_format.merge_tool import omnistore_ckpt_to_pytorch_ckpt
from seed_models.utils.envs import SeedModelsEnvs
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
)
from verl.utils.fs import copy_local_path_from_hdfs


def prepare_hdfs_copy_kwargs():
    """default in SeedModelsEnvs:
    HDFS_THREAD_NUM = int(os.getenv("HDFS_THREAD_NUM", "32"))
    HDFS_CHUNK_THREAD_NUM = int(os.environ.get("HDFS_CHUNK_THREAD_NUM", "32"))
    """
    hdfs_kwargs = {
        "thread_num": SeedModelsEnvs.HDFS_THREAD_NUM,
        "chunk_thread_num": SeedModelsEnvs.HDFS_CHUNK_THREAD_NUM,
    }
    return hdfs_kwargs


def hdfs_upload(local_path, remote_path, log_text):
    time_begin = time.time()
    hdfs_io.copy(local_path, remote_path, **prepare_hdfs_copy_kwargs())
    print(f'{log_text} cost time: {time.time() - time_begin}s')


def simple_convert_seed_models_to_megatron(
    model,
    local_path: str,
    output_path: Optional[str] = None,
):
    converted_ckpt = model.get_xperf_compatible_state_dict()

    local_output_path = f"{local_path}/megatron_merge_states.pt"
    torch.save(converted_ckpt, local_output_path)
    if output_path is not None:
        print(f'Start upload model from {local_output_path} to hdfs path {output_path}')
        if not hdfs_io.hexists(local_output_path):
            raise ValueError(f"{local_output_path} is not found")

        upload_thread = threading.Thread(
            target=hdfs_upload,
            args=(local_output_path, output_path, 'Async upload converted xperf model'),
        )
        upload_thread.start()
        return upload_thread


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

    with tempfile.TemporaryDirectory() as local_tmp_dir:
        print('Step2: merge omnistore ckpt to get state_dict')
        time_begin = time.time()
        state_dict = omnistore_ckpt_to_pytorch_ckpt(
            args.load_dir,
            local_tmp_dir,
            'fsdp',
            model_only=True,
            safetensors_format=True,
            return_dict=True,
        )
        print(f'Merge omnistore checkpoint successfully! cost time: {time.time() - time_begin}s')

        print('Step3: load state_dict to huggingface model')
        time_begin = time.time()
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
        print(f'Load state_dict to huggingface model cost time: {time.time() - time_begin}s')

        print(f'Step4: save merged huggingface model to local {hf_path}')
        time_begin = time.time()
        model.save_pretrained(hf_path, state_dict=state_dict['model'])
        print(f'Save merged huggingface model to local cost time: {time.time() - time_begin}s')

        del state_dict

        thread_map = {}
        # upload hf folder with configs and safetensors to hdfs by default
        print(f'Step5: async save merged huggingface model and configs from {hf_path} to remote {args.save_path}')
        thread_map['hf'] = threading.Thread(
            target=hdfs_upload,
            args=(hf_path, args.save_path, 'Async save merged huggingface model and configs to remote'),
        )
        thread_map['hf'].start()

        # upload merged megatron ckpt to hdfs
        print(f'Step6: convert model to xperf format and async upload to {args.save_path}')
        # convert to megatron for autoeval
        megatron_save_path = os.path.join(args.save_path, 'megatron')
        thread_map['xperf'] = simple_convert_seed_models_to_megatron(model, local_tmp_dir, megatron_save_path)

        print(f'Step7: wait async upload hf and xperf merged ckpt')
        time_begin = time.time()
        for k, v in thread_map.items():
            v.join()
            print(f'Wait async upload for {k} finished!')
        print(f'Wait async upload cost time: {time.time() - time_begin}s')
        print(f'Total time cost: {time.time() - total_time_begin}s')
