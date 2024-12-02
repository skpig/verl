import torch
import random
import numpy as np
from tensordict import TensorDict
from collections import defaultdict
from verl import DataProto


def league_training_filter_prompt(batch, strategy, config):
    num_bon = config.actor_rollout_ref.rollout.num_bon
    bsz = config.data.train_batch_size
    buffer_size = config.trainer.league_training_config.buffer_size
    # mean score per prompt
    mean_scores = batch.batch['token_level_scores'].sum(-1).reshape(bsz * buffer_size,
                                                                    num_bon).mean(-1)  # (num_bon * buffer_size, )
    if strategy == "hard":
        sort_idex = torch.argsort(mean_scores, dim=0)
    elif strategy == "easy":
        sort_idex = torch.argsort(mean_scores, dim=0, descending=True)
    elif strategy == "filter":
        min_score = config.trainer.league_training_config.min_score
        max_score = config.trainer.league_training_config.max_score
        mask = torch.logical_and(torch.ge(mean_scores, min_score), torch.le(mean_scores, max_score)).long()
        sort_idex = torch.argsort(mask, dim=0, descending=True)
        filter_num = max(mask.sum().tolist(), 1)
        sort_idex = sort_idex[:filter_num]
        while sort_idex.size(0) < bsz * buffer_size:
            sort_idex = torch.cat([sort_idex, sort_idex], dim=0)
        sort_idex = sort_idex[:bsz * buffer_size]
    else:
        raise NotImplemented

    # gather corresponding tensor
    tensors = {}
    for key, tensor in batch.batch.items():
        seq_len = tensor.shape[-1]
        cur_idx = sort_idex.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, num_bon, seq_len)[:bsz, :, :]
        tensor = tensor.reshape(bsz * buffer_size, num_bon, -1)
        tensor = torch.gather(tensor, dim=0, index=cur_idx).reshape(-1, seq_len)
        assert tensor.shape == (bsz * num_bon, seq_len)
        tensors[key] = tensor
    final_batch = TensorDict(
        source=tensors,
        batch_size=(batch.batch.batch_size[0] // buffer_size),
    )

    final_non_tensor_batch = {}
    for key, val in batch.non_tensor_batch.items():
        val = val.reshape(bsz * buffer_size, num_bon)
        cur_idx = sort_idex.unsqueeze(dim=1).repeat(1, num_bon)[:bsz, :].numpy()
        val = np.take_along_axis(val, cur_idx, axis=0).reshape(-1)
        assert len(val) == bsz * num_bon
        final_non_tensor_batch[key] = val

    return DataProto(
        batch=final_batch,
        non_tensor_batch=final_non_tensor_batch,
        meta_info=batch.meta_info,
    )


def league_training_filter_prompt_v2(batch, strategy, config):
    # step1, 处理成id2feat的格式
    id2feat = defaultdict(list)
    cur_bsz = batch.batch.batch_size[0]
    buffer_size = config.trainer.league_training_config.buffer_size
    mini_bsz = config.actor_rollout_ref.actor.ppo_mini_batch_size
    final_bsz = cur_bsz // buffer_size // mini_bsz * mini_bsz
    tensor_keys = set()
    non_tensor_keys = set()
    for i in range(cur_bsz):
        cur_dict = {}
        for key, tensor in batch.batch.items():
            cur_dict[key] = tensor[i]
            tensor_keys.add(key)
        for key, val in batch.non_tensor_batch.items():
            cur_dict[key] = val[i:i + 1]
            non_tensor_keys.add(key)
        scores = batch.batch['token_level_scores'][i].sum().numpy()
        cur_dict["scores"] = scores
        index = batch.non_tensor_batch['index'][i]
        id2feat[index].append(cur_dict)

    # step2, 根据score大小排序
    key_score_pairs = []
    for key, val in id2feat.items():
        scores = list(map(lambda x: x["scores"], val))
        key_score_pairs.append((key, np.mean(scores)))
    if strategy == "hard":
        key_score_pairs = sorted(key_score_pairs, key=lambda x: x[1], reverse=True)
    elif strategy == "easy":
        key_score_pairs = sorted(key_score_pairs, key=lambda x: x[1], reverse=True)
    elif strategy == "filter":
        min_score = config.trainer.league_training_config.min_score
        max_score = config.trainer.league_training_config.max_score
        key_score_pairs = list(filter(lambda x: x[1] >= min_score and x[1] <= max_score, key_score_pairs))
    else:
        raise NotImplemented
    final_samples = []
    cur_len = 0
    for key, _ in key_score_pairs:
        final_samples.extend(id2feat[key])
        cur_len += len(id2feat[key])
        if cur_len >= final_bsz:
            break
    final_samples = final_samples[:final_bsz]
    uniq_bsz = len(final_samples)
    while len(final_samples) < final_bsz:
        random.shuffle(final_samples)
        final_samples = final_samples + final_samples
    final_samples = final_samples[:final_bsz]

    # step3, 处理成DataProto格式
    tensors = {}
    final_non_tensor_batch = {}
    for sample in final_samples:
        for key in sample:
            if key == "scores":
                continue
            if key in tensor_keys:
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(sample[key])
            elif key in non_tensor_keys:
                if key not in final_non_tensor_batch:
                    final_non_tensor_batch[key] = []
                final_non_tensor_batch[key].append(sample[key])
    for key in tensor_keys:
        tensors[key] = torch.stack(tensors[key], dim=0)
    final_batch = TensorDict(
        source=tensors,
        batch_size=(final_bsz,),
    )
    for key in final_non_tensor_batch:
        final_non_tensor_batch[key] = np.concatenate(final_non_tensor_batch[key], axis=0)

    # 统计metrics
    prompt_num = len(id2feat)
    response_num_per_prompt = list(map(lambda x: len(x), id2feat.values()))
    metrics = {
        "league_training/prompt_num": prompt_num,
        "league_training/response_num": sum(response_num_per_prompt),
        "league_training/response_num_mean": sum(response_num_per_prompt) / max(1, prompt_num),
        "league_training/response_num_max": max(response_num_per_prompt),
        "league_training/response_num_min": min(response_num_per_prompt),
        "league_training/response_num_std": np.std(response_num_per_prompt),
        "league_training/final_bsz": final_bsz,
        "league_training/uniq_bsz": uniq_bsz,
    }

    return DataProto(
        batch=final_batch,
        non_tensor_batch=final_non_tensor_batch,
        meta_info=batch.meta_info,
    ), metrics
