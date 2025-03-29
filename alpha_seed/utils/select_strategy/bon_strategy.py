import torch
import random
import numpy as np
from tensordict import TensorDict
from collections import defaultdict
from verl import DataProto


def select_training_samples(batch, strategy, config):
    # strategy:
    #   - all: use all responses to train policy and value
    #   - best: use BoN to train policy and value
    #   - best_mix_random: use BoN and random-choice-one to train policy and value
    #   - best_worst: use BoN and WoN to train policy and value
    num_bon = config.actor_rollout_ref.rollout.num_bon
    bsz = config.data.train_batch_size
    # calc select ids
    scores = batch.batch['token_level_scores'].sum(-1).reshape(bsz, num_bon)
    if strategy == "all":
        final_idx = torch.range(0, num_bon - 1).unsqueeze(dim=0).tile([bsz, 1]).to(torch.int64).unsqueeze(dim=2)
        response_num_per_prompt = num_bon
    elif strategy == "best":
        final_idx = torch.argmax(scores, dim=1).unsqueeze(dim=1).unsqueeze(dim=2)
        response_num_per_prompt = 1
    elif strategy == "best_mix_random":
        random_idx = torch.randint(0, num_bon, (bsz,)).unsqueeze(dim=1).unsqueeze(dim=2)
        best_idx = torch.argmax(scores, dim=1).unsqueeze(dim=1).unsqueeze(dim=2)
        final_idx = torch.cat([random_idx, best_idx], dim=1)
        response_num_per_prompt = 2
    elif strategy == "best_worst":
        worst_idx = torch.argmin(scores, dim=1).unsqueeze(dim=1).unsqueeze(dim=2)
        best_idx = torch.argmax(scores, dim=1).unsqueeze(dim=1).unsqueeze(dim=2)
        final_idx = torch.cat([worst_idx, best_idx], dim=1)
        response_num_per_prompt = 2
    else:
        raise NotImplemented

    # gather corresponding tensor
    tensors = {}
    for key, tensor in batch.batch.items():
        seq_len = tensor.shape[-1]
        cur_idx = final_idx.repeat(1, 1, seq_len)
        tensor = tensor.reshape(bsz, num_bon, -1)
        tensor = torch.gather(tensor, dim=1, index=cur_idx).reshape(-1, seq_len)
        assert tensor.shape == (bsz * response_num_per_prompt, seq_len)
        tensors[key] = tensor
    final_batch = TensorDict(
        source=tensors,
        batch_size=(batch.batch.batch_size[0] // num_bon * response_num_per_prompt,),
    )

    final_non_tensor_batch = {}
    for key, val in batch.non_tensor_batch.items():
        val = val.reshape(bsz, num_bon)
        cur_idx = final_idx.squeeze(dim=2).numpy()
        val = np.take_along_axis(val, cur_idx, axis=1).reshape(-1)
        assert len(val) == bsz * response_num_per_prompt
        final_non_tensor_batch[key] = val

    return DataProto(
        batch=final_batch,
        non_tensor_batch=final_non_tensor_batch,
        meta_info=batch.meta_info,
    )


def select_training_samples_v2(batch, strategy, config):
    """
    batch: [bsz * bon, ...]
    """
    # strategy:
    #   - all: use all responses to train policy and value
    #   - best: use BoN to train policy and value
    #   - best_mix_random: use BoN and random-choice-one to train policy and value
    #   - best_worst: use BoN and WoN to train policy and value

    # step1, 处理成id2feat的格式
    id2feat = defaultdict(list)
    id2acc = {}
    total_samples = []
    cur_bsz = batch.batch.batch_size[0]
    mini_bsz = config.actor_rollout_ref.actor.ppo_mini_batch_size
    num_bon = config.actor_rollout_ref.rollout.num_bon
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
        total_samples.append(cur_dict)

    # step2, 根据格式，从每个id里挑选样本
    final_samples = []
    for key, val in id2feat.items():
        scores = list(map(lambda x: x["scores"], val))
        acc = (np.mean(scores).item() + 1) / 2
        id2acc[key] = acc
        if config.algorithm.bon_filter_high_acc_data and acc > 0.8:
            continue
        scores_noise = np.random.uniform(
            -0.01, 0.01, size=len(scores))  # add noise to avoid always choosing the first one, avoiding potential bias
        scores += scores_noise
        if strategy == "all":
            final_idx = list(range(len(val)))
            response_num_per_prompt = num_bon
        elif strategy == "best":
            final_idx = [np.argmax(scores)]
            response_num_per_prompt = 1
        elif strategy == "best_mix_random":
            final_idx = [np.argmax(scores)]
            random_idx = np.random.randint(0, len(val))
            while random_idx in final_idx and len(val) != 1:
                random_idx = np.random.randint(0, len(val))
            final_idx.append(random_idx)
            response_num_per_prompt = 2
        elif strategy == "best_worst":
            final_idx = [np.argmin(scores), np.argmax(scores)]
            response_num_per_prompt = 2
        else:
            raise NotImplemented
        for index in final_idx:
            final_samples.append(val[index])

    # step3, 不够的补，多的随机挑
    final_bsz = cur_bsz // num_bon * response_num_per_prompt // mini_bsz * mini_bsz
    if config.algorithm.bon_filter_high_acc_data:
        if len(final_samples) < final_bsz:
            # rank: acc低的里面的正确的 -> acc低的里面的错误的
            correct_pool_under80 = []
            false_pool_under80 = []
            # 按正确率排序，筛掉正确率大于0.8的
            id2acc_rank = [item for item in sorted(id2acc.items(), key=lambda x: x[1]) if item[1] < 0.8]
            for idx, acc in id2acc_rank:
                correct_pool_under80 += [item for item in id2feat[idx] if item["scores"] == 1]
                false_pool_under80 += [item for item in id2feat[idx] if item["scores"] == -1]
            selection_pool = correct_pool_under80 + false_pool_under80
            weights = [i for i in range(len(selection_pool), 0, -1)]
            # weights = [i / sum(weights) for i in weights]
            final_samples += random.choices(selection_pool, weights=weights, k=final_bsz - len(final_samples))
        elif len(final_samples) > final_bsz:
            random.shuffle(final_samples)
            final_samples = final_samples[:final_bsz]
    else:
        if len(final_samples) < final_bsz:
            random.shuffle(total_samples)
            remain_len = final_bsz - len(final_samples)
            final_samples = final_samples + total_samples[:remain_len]
        elif len(final_samples) > final_bsz:
            random.shuffle(final_samples)
            final_samples = final_samples[:final_bsz]
    final_samples = sorted(final_samples, key=lambda x: x["index"])

    if config.algorithm.group_shuffle:
        index_list = list(map(lambda x: x["index"][0], final_samples))
        from itertools import groupby
        indices = np.arange(len(index_list))
        grouped_indices = [
            list(map(lambda x: x[0], group)) for key, group in groupby(enumerate(index_list), key=lambda x: x[1])
        ]
        np.random.shuffle(grouped_indices)
        shuffled_indices = np.concatenate(grouped_indices)
        final_samples = [final_samples[i] for i in shuffled_indices]

    # step4, 处理成DataProto格式
    tensors = defaultdict(list)
    final_non_tensor_batch = defaultdict(list)
    for sample in final_samples:
        for key in sample:
            if key == "scores":
                continue
            if key in tensor_keys:
                tensors[key].append(sample[key])
            elif key in non_tensor_keys:
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
        "bon/prompt_num": prompt_num,
        "bon/response_num": sum(response_num_per_prompt),
        "bon/response_num_mean": sum(response_num_per_prompt) / max(1, prompt_num),
        "bon/response_num_max": max(response_num_per_prompt),
        "bon/response_num_min": min(response_num_per_prompt),
        "bon/response_num_std": np.std(response_num_per_prompt),
        "bon/final_bsz": final_bsz,
        "bon/acc_80+": len(list(filter(lambda x: x > 0.8, id2acc.values()))) / len(id2acc),
        "bon/acc_50+": len(list(filter(lambda x: x > 0.5, id2acc.values()))) / len(id2acc),
        "bon/acc_20+": len(list(filter(lambda x: x > 0.2, id2acc.values()))) / len(id2acc),
        "bon/acc_10+": len(list(filter(lambda x: x > 0.1, id2acc.values()))) / len(id2acc),
        "bon/acc_0": len(list(filter(lambda x: x == 0, id2acc.values()))) / len(id2acc),
    }

    return DataProto(
        batch=final_batch,
        non_tensor_batch=final_non_tensor_batch,
        meta_info=batch.meta_info,
    ), metrics, id2acc
