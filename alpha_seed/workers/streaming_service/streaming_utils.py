import ray
import torch
import numpy as np
import torch.nn.functional as F


def rmpad(item):
    start_idx = torch.nonzero(item.batch['attention_mask'].flatten())[0]
    end_idx = start_idx + item.batch['attention_mask'].sum(-1)
    item.batch['input_ids'] = item.batch['input_ids'][:, start_idx:end_idx]
    item.batch['attention_mask'] = item.batch['attention_mask'][:, start_idx:end_idx]
    return item


def pad(item, max_standalone_len, tokenizer):
    pad_len = max_standalone_len - item.batch['attention_mask'].sum(-1)
    item.batch['input_ids'] = F.pad(item.batch['input_ids'], (pad_len, 0), value=tokenizer.pad_token_id)
    item.batch['attention_mask'] = F.pad(item.batch['attention_mask'], (pad_len, 0), value=0)
    return item


def process_output(input_batch, output_batch, tokenizer, ready_batch, pending_batch, config, standalone=False):
    is_finished = output_batch.pop(batch_keys=['is_finished']).batch['is_finished']
    finished_num = is_finished.sum().int().item()
    if not standalone:
        output_batch.union(input_batch)
        for i, item in enumerate(output_batch.chunk(len(output_batch))):
            if is_finished[i]:
                ready_batch.append(item)
            else:
                item.pop(batch_keys=['responses'])
                pending_batch.append(rmpad(item))
    else:
        max_new_tokens = output_batch.meta_info.get('generation_kwargs').get('max_new_tokens',
                                                                             config.data.max_response_length)
        if config.streaming_rollout.force_eos:
            need_eos = is_finished == 0
            is_finished = torch.ones_like(is_finished)
        # output_batch.pop(batch_keys=['prompts', 'responses'])
        output_batch.union(input_batch)

        # rearrange...
        #   prompts layout: [00111111] left-padding only, shape [bs, max_prompt_length]
        #   input_ids layout: [00111111 11111111100] prompts's left-padding + response's right-padding, shape [bs, max_prompt_length + max_response_length]
        for i, item in enumerate(output_batch.chunk(len(output_batch))):
            if is_finished[i]:
                left_pad_len = (item.batch['prompts'] != tokenizer.pad_token_id).int().argmax(dim=1)
                start_idx = torch.nonzero(item.batch['attention_mask'].flatten())[0]
                real_len = item.batch['attention_mask'].sum(-1)
                total_len = config.data.max_prompt_length + max_new_tokens
                right_pad_len = total_len - left_pad_len - real_len
                item.batch['attention_mask'] = F.pad(item.batch['attention_mask'][:, start_idx:start_idx + real_len],
                                                     (left_pad_len, right_pad_len),
                                                     value=0)
                item.batch['input_ids'] = F.pad(item.batch['input_ids'][:, start_idx:start_idx + real_len],
                                                (left_pad_len, right_pad_len),
                                                value=tokenizer.pad_token_id)
                item.batch['responses'] = item.batch['input_ids'][:, item.batch['prompts'].shape[1]:]
                if config.streaming_rollout.force_eos and need_eos[i]:
                    item.batch['input_ids'][:, -1 if left_pad_len + real_len >= total_len else left_pad_len +
                                            real_len] = tokenizer.eos_token_id
                    gen_len = item.batch['attention_mask'][:, item.batch['prompts'].shape[1]:].sum(-1)
                    item.batch['responses'][:, -1 if gen_len >=
                                            config.data.max_response_length else gen_len] = tokenizer.eos_token_id
                    item.batch['attention_mask'][:, -1 if item.batch['prompts'].shape[1] +
                                                 gen_len >= total_len else item.batch['prompts'].shape[1] + gen_len] = 1
                ready_batch.append(item)
            else:
                pending_batch.append(rmpad(item))
    return finished_num, ready_batch, pending_batch


def record_xperf_metrics(batch_info, metrics, logger, global_step, prefix=''):
    xperf_metrics = batch_info.meta_info['xperf_metrics']
    metrics[f'rollout/{prefix}/steps'] = len(xperf_metrics.get('finished_tokens_by_step', []))
    # sampling tokens
    sample_token_num = xperf_metrics.get('sample_token_num', 0)
    metrics[f'rollout/{prefix}/prob_mean'] = xperf_metrics.get('prob_mean', 0) / (sample_token_num + 1e-6)
    metrics[f'rollout/{prefix}/prob_lt_0.0001_ratio'] = xperf_metrics.get('prob_lt_0.0001',
                                                                          0) / (sample_token_num + 1e-6)
    metrics[f'rollout/{prefix}/prob_lt_1e-5_ratio'] = xperf_metrics.get('prob_lt_1e-5', 0) / (sample_token_num + 1e-6)
    metrics[f'rollout/{prefix}/prob_lt_1e-6_ratio'] = xperf_metrics.get('prob_lt_1e-6', 0) / (sample_token_num + 1e-6)
    metrics[f'rollout/{prefix}/page_swap_out_bs'] = xperf_metrics.get('page_swap_out_bs', 0)
    metrics[f'rollout/{prefix}/page_swap_out_token'] = xperf_metrics.get('page_swap_out_token', 0)
    metrics[f'rollout/{prefix}/max_off_policy_steps'] = max(max(xperf_metrics.get('off_policy_steps', [[0]])))

    # context + decode tokens
    tokens_num = xperf_metrics.get('tokens_num', [])
    per_token_latency = xperf_metrics.get('per_token_latency', [])
    total_tokens = sum(tokens_num)
    metrics[f'rollout/{prefix}/per_token_latency_avg'] = 0 if len(per_token_latency) == 0 else (sum(per_token_latency) /
                                                                                                len(per_token_latency))
    metrics[f'rollout/{prefix}/tps'] = total_tokens / (sum(per_token_latency) + 1e-6) * 1000
    metrics[f'rollout/{prefix}/bs_avg'] = 0 if len(tokens_num) == 0 else (total_tokens / len(tokens_num))
    batch_info.meta_info.pop('xperf_metrics')
    return


def get_gpus_per_node():
    gpu_per_node = 0
    for node in ray.nodes():
        if "Resources" not in node or "GPU" not in node['Resources']:
            continue
        gpu_per_node = int(node['Resources']['GPU'])
        break
    return gpu_per_node


def is_multihost_model(model_parallel_size: int) -> bool:
    if model_parallel_size <= 8:
        return False
    gpu_per_node = get_gpus_per_node()
    print(f'find gpu_per_node: {gpu_per_node}, model_parallel_size: {model_parallel_size}')
    # assuming that all nodes have the same number of GPUs
    return gpu_per_node < model_parallel_size
