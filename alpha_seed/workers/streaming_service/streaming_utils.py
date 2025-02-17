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


def process_output(input_batch,
                   output_batch,
                   tokenizer,
                   ready_batch_queue,
                   pending_batch_queue,
                   config,
                   standalone=False):
    is_finished = output_batch.pop(batch_keys=['is_finished']).batch['is_finished']
    finished_num = is_finished.sum().int().item()
    if not standalone:
        output_batch.union(input_batch)
        for i, item in enumerate(output_batch.chunk(len(output_batch))):
            if is_finished[i]:
                ready_batch_queue.put(item)
            else:
                item.batch['off_policy_steps'] += 1
                item.pop(batch_keys=['responses'])
                pending_batch_queue.put(rmpad(item))
    else:
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
                total_len = config.data.max_prompt_length + config.data.max_response_length
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
                ready_batch_queue.put(item)
            else:
                item.batch['off_policy_steps'] += 1
                pending_batch_queue.put(rmpad(item))
    return finished_num, ready_batch_queue, pending_batch_queue


def record_xperf_metrics(batch_info, metrics, logger, global_step, prefix=''):
    import wandb
    if 'xperf_metrics' in batch_info.meta_info:
        for name, x_metric in batch_info.meta_info['xperf_metrics'].items():
            if isinstance(x_metric, list):
                logger.log(data={"rollout/{}/{}".format(prefix, name): wandb.Histogram(x_metric)}, step=global_step)
        try:
            sample_token_num = batch_info.meta_info['xperf_metrics']['sample_token_num']
            metrics[
                f'rollout/{prefix}/prob_mean'] = batch_info.meta_info['xperf_metrics']['prob_mean'] / sample_token_num
            metrics[f'rollout/{prefix}/prob_lt_0.0001_ratio'] = batch_info.meta_info['xperf_metrics'][
                'prob_lt_0.0001'] / sample_token_num
            metrics[f'rollout/{prefix}/prob_lt_1e-5_ratio'] = batch_info.meta_info['xperf_metrics'][
                'prob_lt_1e-5'] / sample_token_num
            metrics[f'rollout/{prefix}/prob_lt_1e-6_ratio'] = batch_info.meta_info['xperf_metrics'][
                'prob_lt_1e-6'] / sample_token_num
            metrics[f'rollout/{prefix}/page_swap_out_bs'] = batch_info.meta_info['xperf_metrics']['page_swap_out_bs']
            metrics[f'rollout/{prefix}/page_swap_out_token'] = batch_info.meta_info['xperf_metrics'][
                'page_swap_out_token']
            metrics[f'rollout/{prefix}/max_off_policy_steps'] = max(
                batch_info.meta_info['xperf_metrics']['off_policy_steps'])
            per_token_latency = batch_info.meta_info['xperf_metrics']['per_token_latency']
            tokens_num = batch_info.meta_info['xperf_metrics']['tokens_num']
            total_tokens = sum(tokens_num)
            metrics[f'rollout/{prefix}/per_token_latency_avg'] = 0 if len(per_token_latency) == 0 else (
                sum(per_token_latency) / len(per_token_latency))
            metrics[f'rollout/{prefix}/tps'] = total_tokens / (sum(per_token_latency) + 1e-6) * 1000
            metrics[f'rollout/{prefix}/bs_avg'] = 0 if len(tokens_num) == 0 else (total_tokens / len(tokens_num))
        except Exception as e:
            # some xperf metrics is not ready on lower version
            pass
        batch_info.meta_info.pop('xperf_metrics')
    return
