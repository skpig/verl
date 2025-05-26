import os
import random
import socket

import ray
import torch
import numpy as np
import torch.nn.functional as F
from typing import *

from verl import DataProto


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
    # output_batch = hybrid_rollout.forward(input_batch)
    # if is_finished[i]:
    #     ready_batch.append(output_batch[i])
    # else:
    #     pending_batch.append(output_batch[i])

    is_finished = output_batch.pop(batch_keys=['is_finished']).batch['is_finished']
    finished_num = is_finished.sum().int().item()
    # TODO: issue in comparing non_tensor_batches
    # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
    same_keys = input_batch.non_tensor_batch.keys() & output_batch.non_tensor_batch.keys()
    input_batch.pop(non_tensor_batch_keys=list(same_keys))
    output_batch.union(input_batch)
    if not standalone:
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

        force_eos = config.streaming_rollout.force_eos
        max_response_length = config.data.max_response_length
        max_prompt_length = config.data.max_prompt_length

        # rearrange...
        #   prompts layout: [00111111] left-padding only, shape [bs, max_prompt_length]
        #   input_ids layout: [00111111 11111111100] prompts's left-padding + response's right-padding, shape [bs, max_prompt_length + max_response_length]
        for i, item in enumerate(output_batch.chunk(len(output_batch))):
            if is_finished[i]:
                left_pad_len = (item.batch['prompts'] != tokenizer.pad_token_id).int().argmax(dim=1)
                start_idx = torch.nonzero(item.batch['attention_mask'].flatten())[0]
                real_len = item.batch['attention_mask'].sum(-1)
                total_len = max_prompt_length + max_new_tokens
                right_pad_len = total_len - left_pad_len - real_len
                item.batch['attention_mask'] = F.pad(item.batch['attention_mask'][:, start_idx:start_idx + real_len],
                                                     (left_pad_len, right_pad_len),
                                                     value=0)
                item.batch['input_ids'] = F.pad(item.batch['input_ids'][:, start_idx:start_idx + real_len],
                                                (left_pad_len, right_pad_len),
                                                value=tokenizer.pad_token_id)
                item.batch['responses'] = item.batch['input_ids'][:, item.batch['prompts'].shape[1]:]
                if force_eos and need_eos[i]:
                    item.batch['input_ids'][:, -1 if left_pad_len + real_len >= total_len else left_pad_len +
                                            real_len] = tokenizer.eos_token_id
                    gen_len = item.batch['attention_mask'][:, item.batch['prompts'].shape[1]:].sum(-1)
                    item.batch['responses'][:,
                                            -1 if gen_len >= max_response_length else gen_len] = tokenizer.eos_token_id
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

    # plugin metrics
    for key, val in xperf_metrics.items():
        if not key.startswith('plugin/'):
            continue
        metrics_key = f"rollout/{prefix}/{key}"
        if isinstance(val, list):
            import wandb
            metrics[metrics_key] = wandb.Histogram(val)
        else:
            metrics[metrics_key] = val
    batch_info.meta_info.pop('xperf_metrics')
    return


def get_gpus_per_node():
    if not ray.is_initialized():
        return 8
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


def _postprocess(off_p_list, on_p_list, target_length, pad_token=-1, mode="off_policy_step"):
    assert (
        len(off_p_list) == len(on_p_list)
    ), f"off-policy and on-policy list should have the same length, but got {len(off_p_list)} and {len(on_p_list)}, mode = {mode}"
    list_padded = []
    for i, on_p_list_i in enumerate(on_p_list):
        off_p_list_i = off_p_list[i]
        prev_index = torch.nonzero(off_p_list_i == pad_token)
        if prev_index.numel() == 0:
            prev_index = -1
        else:
            prev_index = prev_index[0]
        cur_list_i = off_p_list_i[:prev_index].tolist() + on_p_list_i
        if mode == "off_policy_step":
            cur_list_i = [x + 1 for x in cur_list_i]
        # off-policy + on-policy might exceeds the target length
        if len(cur_list_i) > target_length:
            padded_list = cur_list_i[:target_length]
        else:
            padded_list = cur_list_i + [pad_token] * (target_length - len(cur_list_i))
        list_padded.append(padded_list)
    t_padded = torch.tensor(list_padded)
    return t_padded


from dataclasses import dataclass


@dataclass
class DataPack:
    response_log_probs: list
    response_probs_gt_threshold_num: list
    response_probs_lt_threshold_sum: list
    this_turn_off_policy_steps: list
    response_outputs: list
    response_model_output_mask: list
    is_finished: list
    metrics: dict
    extra_data: Optional[list] = None

    @classmethod
    def create_from_completion(cls, message):
        data_pack = DataPack(response_outputs=[message.raw_output_ids],
                             response_log_probs=[message.response_log_probs],
                             response_probs_gt_threshold_num=[message.response_probs_gt_threshold_num],
                             response_probs_lt_threshold_sum=[message.response_probs_lt_threshold_sum],
                             response_model_output_mask=[message.model_output_mask],
                             this_turn_off_policy_steps=[[-1 for _ in range(len(message.raw_output_ids))]],
                             is_finished=[message.is_finished],
                             extra_data=[message.extra_data],
                             metrics=message.metrics)
        return data_pack

    @classmethod
    def create_from_completion_dict(cls, message):
        data_pack = DataPack(response_outputs=[message['raw_output_ids']],
                             response_log_probs=[message['response_log_probs']],
                             response_probs_gt_threshold_num=[message['response_probs_gt_threshold_num']],
                             response_probs_lt_threshold_sum=[message['response_probs_lt_threshold_sum']],
                             response_model_output_mask=[message['model_output_mask']],
                             this_turn_off_policy_steps=[[-1 for _ in range(len(message['raw_output_ids']))]],
                             is_finished=[message['is_finished']],
                             extra_data=[message['extra_data']],
                             metrics=message['metrics'])
        return data_pack


def pack_to_dataproto(prompts, tokenizer, data_pack: DataPack, config) -> DataProto:
    max_new_tokens = prompts.meta_info.get('generation_kwargs').get('max_new_tokens', config.response_length)
    prompts.batch = prompts.batch.cpu()
    prompt_ids = prompts.batch['input_ids']  # (bs, prompt_length)
    # left-padded attention_mask
    attention_mask = prompts.batch['attention_mask']
    off_policy_model_output_mask = prompts.batch.get('model_output_mask', None)
    off_turn_off_policy_steps = prompts.batch["off_policy_steps"]
    off_policy_response_log_probs = prompts.batch["rollout_log_probs"]
    off_policy_probs_gt_threshold_num = prompts.batch["probs_gt_threshold_num"]
    off_policy_probs_lt_threshold_sum = prompts.batch["probs_lt_threshold_sum"]

    from unittest.mock import patch
    # remove warning
    tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True
    with patch.object(tokenizer, "padding_side", "right"):
        truncated_response_outputs = [seq[:max_new_tokens] for seq in data_pack.response_outputs]
        response_outputs = tokenizer.pad(dict(input_ids=truncated_response_outputs),
                                         padding="max_length",
                                         max_length=max_new_tokens,
                                         return_tensors="pt")

    response_log_probs = _postprocess(off_policy_response_log_probs,
                                      data_pack.response_log_probs,
                                      max_new_tokens,
                                      mode="log_prob")
    response_probs_gt_threshold_num = _postprocess(off_policy_probs_gt_threshold_num,
                                                   data_pack.response_probs_gt_threshold_num,
                                                   max_new_tokens,
                                                   mode="probs_gt_threshold_num")
    response_probs_lt_threshold_sum = _postprocess(off_policy_probs_lt_threshold_sum,
                                                   data_pack.response_probs_lt_threshold_sum,
                                                   max_new_tokens,
                                                   mode="probs_lt_threshold_sum")
    response_off_policy = _postprocess(off_turn_off_policy_steps,
                                       data_pack.this_turn_off_policy_steps,
                                       max_new_tokens,
                                       mode="off_policy_step")
    if off_policy_model_output_mask is not None:
        response_model_output_mask = _postprocess(off_policy_model_output_mask,
                                                  data_pack.response_model_output_mask,
                                                  max_new_tokens,
                                                  mode='model_output_mask',
                                                  pad_token=0)
    else:
        response_model_output_mask = None
    response_ids = response_outputs["input_ids"][:, :max_new_tokens].to(torch.int32)
    response_attention_mask = response_outputs["attention_mask"][:, :max_new_tokens].to(torch.int8)

    attention_mask = torch.hstack((attention_mask, response_attention_mask))
    input_ids = torch.hstack((prompt_ids, response_ids))

    # all the tp ranks should contain the same data here. data in all ranks are valid
    batch = {
        'rollout_log_probs': response_log_probs.to(torch.bfloat16),
        'probs_gt_threshold_num': response_probs_gt_threshold_num.to(torch.bfloat16),
        'probs_lt_threshold_sum': response_probs_lt_threshold_sum.to(torch.bfloat16),
        'input_ids': input_ids.to(torch.int32),  # here input_ids become the whole sentences
        'attention_mask': attention_mask.to(torch.int8),
        'is_finished': torch.Tensor(data_pack.is_finished).to(torch.int8),
        'off_policy_steps': response_off_policy.to(torch.int8),
    }
    if response_model_output_mask is not None:
        batch['model_output_mask'] = response_model_output_mask.to(torch.int8)
    from verl import DataProto
    out = DataProto.from_dict(batch)
    data_pack.metrics["off_policy_steps"] = response_off_policy.tolist()
    out.meta_info["xperf_metrics"] = data_pack.metrics
    out.meta_info["generation_kwargs"] = prompts.meta_info['generation_kwargs']
    out.non_tensor_batch = prompts.non_tensor_batch
    out.non_tensor_batch['extra_data'] = np.array(data_pack.extra_data, dtype=object)
    return out


def get_local_ip():
    ip_list = []

    # IPv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # 设置套接字选项，避免实际连接
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            # 随便广播一个地址
            s.connect(("255.255.255.255", 0))
            ip_list.append(s.getsockname()[0])
    except:
        pass

    # IPv6
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
            # 设置套接字选项，避免实际连接
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            # google public DNS
            s.connect(("2001:4860:4860::8888", 53))
            ip_list.append(s.getsockname()[0])
    except:
        pass

    # 解析主机名获取 IP
    try:
        hostname = socket.gethostname()
        addrinfos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_DGRAM)
        for addrinfo in addrinfos:
            ip = addrinfo[4][0]
            if ip not in ip_list:
                ip_list.append(ip)
    except:
        pass

    # 返回第一个可用的非回环地址，优先 IPv4
    for ip in ip_list:
        if not (ip.startswith("127.") or ip in ["::1", "::", "0.0.0.0"]):
            return ip

    return None


def get_node_ip():

    def get_node_ip_by_sdk():
        if os.getenv("WG_BACKEND", None) == "ray":
            import ray
            return ray._private.services.get_node_ip_address()
        elif os.getenv("WG_BACKEND", None) == "torch_rpc":
            from verl_ext.single_controller.torchrpc.k8s_client import get_ip_addr
            return get_ip_addr()
        return None

    host_ipv4 = os.getenv("MY_HOST_IP", None)
    host_ipv6 = os.getenv("MY_HOST_IPV6", None)
    host_ip_by_env = host_ipv4 or host_ipv6
    host_ip_by_if = get_local_ip()
    host_ip_by_sdk = get_node_ip_by_sdk()

    host_ip = host_ip_by_env or host_ip_by_if or host_ip_by_sdk
    return host_ip


def get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


# 给于torch.classes.XGPT.NCCLPrimitive()专用的找端口方法
def get_free_port_for_nccl_primitive():
    ports = []
    count = 10
    while count > 0:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            # 绑定到一个随机的可用端口
            s.bind(('::', 0))
            # 获取绑定的端口号
            port = s.getsockname()[1]
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s2:
                try:
                    # 由于torch.classes.XGPT.NCCLPrimitive()里自动会给传进去的端口+1，
                    # 所以还要测试+1后的端口是否空闲，才算真的空闲端口
                    s2.bind(('::', port + 1))
                    port2 = s2.getsockname()[1]
                    ports.append(port2)
                    count -= 1
                except OSError:
                    continue
            ports.append(port)
    return random.choice(ports)


def is_ipv6(ip):
    return ':' in ip
