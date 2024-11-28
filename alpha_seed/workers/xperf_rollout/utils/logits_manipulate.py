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
