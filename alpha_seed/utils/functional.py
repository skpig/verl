import psutil
import datetime
import importlib
import logging
import torch
import json
import os
import shutil
from mono_rl import DataProto
from verl.utils.seqlen_balancing import rearrange_micro_batches
from typing import Dict, Any
from hdfs_io import makedirs, hput, hcopy, hexists
import contextlib

logger = logging.getLogger(__name__)


# Megavision ETTR Logger, Ensure no error if training metrics is not installed
class SafeStageLogger:

    def __init__(self):
        self._logger = None
        try:
            from bytedance.trainingmetrics.logger import get_stage_logger
            self._logger = get_stage_logger()
        except:
            pass

    def __getattr__(self, name):
        if name == '_logger':
            return None

        if self._logger is not None:
            return getattr(self._logger, name)

        if name.endswith('_context'):
            return lambda *a, **k: contextlib.nullcontext()
        else:
            return lambda *a, **k: (lambda f: f)


def clip_by_value_preserve_gradient(t, min=None, max=None):
    """
    Copied from https://github.com/vermouth1992/rlutils-python/blob/aab1d670860cfdf53fd8805528ec11ef0bfdf015/rlutils/pytorch/functional.py#L46
    """
    clip_t = torch.clip(t, min=min, max=max)
    return t + (clip_t - t).detach()


def get_text_config(config):
    if hasattr(config, "text_config"):
        return config.text_config
    else:
        return config


def get_text_model_type(config):
    return get_text_config(config).model_type


def rearrange_micro_data_proto(max_token_len, mini_batch, dp_group=None):
    micro_batches, indices = rearrange_micro_batches(mini_batch.batch, max_token_len=max_token_len, dp_group=dp_group)
    non_tensor_batches = []
    for indice_group in indices:
        non_tensor_batch = {}
        for idx in indice_group:
            for key in mini_batch.non_tensor_batch.keys():
                if key not in non_tensor_batch:
                    non_tensor_batch[key] = []
                non_tensor_batch[key].append(mini_batch.non_tensor_batch[key][idx])
        non_tensor_batches.append(non_tensor_batch)
    return indices, micro_batches, non_tensor_batches


def update_model_config(module_config, override_config_kwargs):
    logger.info(f'!!!!!!!!!!! module_config: {module_config}, ###### override_config_kwargs: {override_config_kwargs}')
    for key, val in override_config_kwargs.items():
        if isinstance(val, dict):
            update_model_config(getattr(module_config, key), val)
        else:
            if not hasattr(module_config, key):
                logger.warning(f"WARN: {key} not exists in {module_config}")
            setattr(module_config, key, val)


def print_dataproto_size(data: DataProto, head):
    size_of_tensordict = 0
    for key, tensor in data.batch.items():
        size_of_tensordict += tensor.element_size() * tensor.numel()
    size_of_numpy_array = 0
    for key, numpy_array in data.non_tensor_batch.items():
        size_of_numpy_array += numpy_array.nbytes

    size_of_numpy_array /= 1024**3
    size_of_tensordict /= 1024**3
    logger.info(
        f'{head}, Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB')
    log_cpu_memory_usage(head)


def log_cpu_memory_usage(key):
    process = psutil.Process()
    mem = psutil.virtual_memory()
    logger.info(
        f"{key} {datetime.datetime.now()} cpu memory usage: process rss{process.memory_info().rss / 1024**3}GB, sys memory used: {mem.used / 1024**3}GB"
    )


def append_dict_items_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        if isinstance(val, list):
            data[key].extend(val)
        else:
            data[key].append(val)


def import_from_string(import_str: str) -> Any:
    if '.' in import_str:
        module_name, obj_name = import_str.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, obj_name)
    else:
        return importlib.import_module(import_str)


def save_simple_train_data_to_hdfs(batch, tokenizer, step, config):
    file_base_path = config.trainer.default_hdfs_dir
    max_prompt_length = config.data.max_prompt_length
    file_path = f"{file_base_path}/simple_train_data"
    local_file_path = f'train_simple_{step}.jsonl'
    if file_path.startswith("hdfs://"):
        makedirs(file_path, exist_ok=True)
    else:
        os.makedirs(file_path, exist_ok=True)
    # open file
    with open(local_file_path, "w") as f:

        if "index" in batch.non_tensor_batch:
            dataset_indexs = batch.non_tensor_batch['index']
        else:
            dataset_indexs = [None] * batch.batch.batch_size[0]

        if 'prompt' in batch.non_tensor_batch:
            prompts = batch.non_tensor_batch['prompt']
        else:
            prompts = [""] * batch.batch.batch_size[0]

        raw_scores = batch.batch['raw_scores'].sum(dim=-1)
        token_level_rewards = batch.batch['token_level_rewards'].sum(dim=-1)

        decode_batch_response = []
        for i in range(batch.batch['responses'].shape[0]):
            valid_response_length = batch.batch['attention_mask'][i, max_prompt_length:].sum().item()
            valid_response_idx = batch.batch['responses'][i, :valid_response_length]
            # remove potential special tokens(-100)
            valid_response_idx = valid_response_idx[valid_response_idx != -100]
            decode_batch_response.append(valid_response_idx)
        responses = tokenizer.batch_decode(decode_batch_response, skip_special_tokens=False)

        if 'raw_rm_score' in batch.batch:
            raw_rm_scores = batch.batch['raw_rm_score']
        if 'raw_verifier_score' in batch.batch:
            raw_verifier_scores = batch.batch['raw_verifier_score']
        soi = config.data.special_tokens.soi
        eoi = config.data.special_tokens.eoi

        for i in range(len(raw_scores)):
            data = {
                "index_id": dataset_indexs[i],
                "raw_score": raw_scores[i].item(),
                "token_level_reward": token_level_rewards[i].item(),
                "prompt": prompts[i],
                "response": responses[i].replace(f"{soi}{eoi}", f"{soi}<ImageHere>{eoi}"),
            }
            if 'raw_rm_score' in batch.batch:
                data['raw_rm_score'] = raw_rm_scores[i].item()
            if 'raw_verifier_score' in batch.batch:
                data['raw_verifier_score'] = raw_verifier_scores[i].item()
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()

    if file_path.startswith("hdfs://"):
        hput(local_file_path, file_path)
    else:
        shutil.copy(local_file_path, file_path)
