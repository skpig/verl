import torch
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
        output_batch.pop(batch_keys=['prompts', 'responses'])
        output_batch.union(input_batch)
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
                    item.batch['input_ids'][:, left_pad_len + real_len] = tokenizer.eos_token_id
                    gen_len = item.batch['attention_mask'][:, item.batch['prompts'].shape[1]:].sum(-1)
                    item.batch['responses'][:, gen_len] = tokenizer.eos_token_id
                ready_batch_queue.put(item)
            else:
                item.batch['off_policy_steps'] += 1
                pending_batch_queue.put(rmpad(item))
    return is_finished.sum().int().item(), ready_batch_queue, pending_batch_queue
