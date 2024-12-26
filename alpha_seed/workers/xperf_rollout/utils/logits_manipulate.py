import torch
import numpy as np


def logits_manipulate_fn_core(logits, history_ids, manipulate_args):
    LARGE = max(1.0, torch.max(logits))
    soft_interval = manipulate_args['soft_interval']
    eot = manipulate_args['eothink']
    total_token = torch.tensor([len(l) for l in history_ids], device=logits.device)
    # eot_pos = torch.tensor([l.index(eot) if eot in l else -1 for l in history_ids], device=logits.device)
    no_eot = torch.tensor([float(eot not in l) for l in history_ids], device=logits.device)
    eot_thres = manipulate_args['response_length'] - manipulate_args['summary_min_space']
    soft_rate = torch.relu((total_token - (eot_thres - soft_interval)) / soft_interval)
    logits[:, eot] += LARGE * soft_rate * no_eot
    return logits


def logits_manipulate_fn_minp(logits, history_ids, manipulate_args):
    """
    logits: [batch_size, vocab_size]

    reference: https://github.com/huggingface/transformers/issues/27670
    """
    filter_value = -float("Inf")
    probs = torch.nn.functional.softmax(logits, dim=-1)
    min_p = manipulate_args['min_p']
    probs_max = torch.max(probs, dim=-1, keepdim=True)[0]
    probs_clip = probs_max * min_p
    min_p_mask = probs < probs_clip
    logits[min_p_mask] = filter_value
    return logits


def logits_manipulate_fn_eta(logits, history_ids, manipulate_args):
    """
    logits: [batch_size, vocab_size]

    paper: https://arxiv.org/abs/2210.15191
    """
    filter_value = -float("Inf")
    probs = torch.nn.functional.softmax(logits, dim=-1)
    min_p_mask = probs < manipulate_args['eta_epsilon']
    logits[min_p_mask] = filter_value
    return logits
