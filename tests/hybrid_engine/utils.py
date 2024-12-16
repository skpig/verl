from verl.utils.fs import copy_local_path_from_hdfs
from flash_attn.bert_padding import index_first_axis, rearrange
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
import torch
import torch.distributed as dist

data_sample = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/haibin.lin/rl/ppo_actor_rmpad_v3.pt'


def prepare_data():

    local_path = copy_local_path_from_hdfs(data_sample)

    device = f"cuda:{torch.cuda.current_device()}"
    rmpad_data = torch.load(local_path, map_location=device, weights_only=False)

    response_length = rmpad_data['response_length']
    attention_mask = rmpad_data['attention_mask']
    input_ids_rmpad = rmpad_data['input_ids']
    position_ids_rmpad = rmpad_data['position_ids']
    indices = rmpad_data['indices']

    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

    # hidden_states: (nnz, hidden_size) after rmpad
    full_response_mask = attention_mask.clone()
    full_response_mask[:, :-response_length] = 0  # set the prompt part to zero
    full_response_mask_rmpad = index_first_axis(rearrange(full_response_mask.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                indices).squeeze(-1)  # (total_nnz,)

    return input_ids_rmpad, input_ids_rmpad_rolled, full_response_mask_rmpad, position_ids_rmpad


def to_random(input_ids, input_ids_rolled, mask, position_ids):
    device = input_ids.device
    random_input_ids = torch.randint(0, 1000, input_ids.size(), device=device)
    random_input_ids_rolled = torch.roll(random_input_ids, shifts=-1, dims=1)
    random_input_ids_rolled = random_input_ids_rolled.squeeze(0)
    return random_input_ids, random_input_ids_rolled, mask, position_ids


def ref_loss_fn(output, input_ids_rmpad_rolled, full_response_mask_rmpad):
    # input_ids_rmpad_rolled: (total_nnz,)
    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
    # logits_rmpad.div_(temperature)
    full_log_probs_rmpad = -cross_entropy_loss(logits_rmpad, input_ids_rmpad_rolled,
                                               inplace_backward=False)[0]  # (total_nnz,)

    # compute entropy loss
    # logits_rmpad: (total_nnz, vocab_size)
    # full_response_mask_rmpad: (total_nnz,)
    pd = torch.nn.functional.softmax(logits_rmpad, dim=-1)
    entropy = torch.logsumexp(logits_rmpad, dim=-1) - torch.sum(pd * logits_rmpad, dim=-1)
    entropy_loss = (entropy * full_response_mask_rmpad).sum(axis=None) / full_response_mask_rmpad.sum(axis=None)
    # entropy_loss: scaler
    return entropy_loss, full_log_probs_rmpad


def print_each_rank(*msg: str):
    for rank in range(dist.get_world_size()):
        if rank == dist.get_rank():
            print(f"rank [{dist.get_rank()}]:", *msg)
        dist.barrier()
