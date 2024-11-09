import torch


def rearrange_micro_batches(batch, ppo_max_token_len):
    ## bin packing problem: NP-Complete probelm
    seq_len_effective = batch['attention_mask'].cuda().sum(dim=1)
    bucket_index = seq_len_effective.cumsum(dim=0) // ppo_max_token_len
    ## bucket_index is continuous
    bucket_segment_point = (bucket_index[1:] - bucket_index[:-1]).nonzero() + 1

    num_micro_batches = len(bucket_segment_point) + 1
    max_num_micro_batches = torch.tensor([num_micro_batches]).cuda()
    torch.distributed.all_reduce(max_num_micro_batches, op=torch.distributed.ReduceOp.MAX)

    micro_batches = []
    last_seg_point = 0
    for seg_point in bucket_segment_point:
        micro_batches.append(batch[last_seg_point:seg_point])
        last_seg_point = seg_point
    micro_batches.append(batch[last_seg_point:])

    if max_num_micro_batches > num_micro_batches:
        # fake data
        for i in range(num_micro_batches, max_num_micro_batches):
            micro_batches.append(batch[-1:])
    return micro_batches, num_micro_batches
