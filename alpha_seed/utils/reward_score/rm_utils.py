import re
import time
import asyncio
import math
import torch
import random
import traceback
import pandas as pd
import ray
import logging
import numpy as np
from uuid import uuid4
from alpha_seed.workers.xperf_rollout.component.query import Query
from mono_rl.utils.infer.client import GRMServingClient
from mono_rl.utils.dataset.dist_data_util import get_dist_data_manager
from typing import Dict, List
from langchain.schema import HumanMessage, SystemMessage
from bytedagi.model_io import InferenceRequest, ModelIO
from langchain.schema import HumanMessage
from bytedance import servicediscovery
from servicediscovery import ServiceDiscoveryError


def wait_remote_server_ready(psm: str):
    fail_time = 0
    while fail_time < 20:
        try:
            sd_result = servicediscovery.get_one(psm, address_family="dual-stack")
            print(f"[RM SERVER INFO] psm {psm} ready !!!")
            return
        except ServiceDiscoveryError:
            print(f"[RM SERVER WARNING] waitting psm {psm} ready, cnt={fail_time}, begin sleep 60s")
            time.sleep(60)
            fail_time += 1
    raise ServiceDiscoveryError(f"psm {psm} not ready after 30 times retry, please check remote rm log")


def check_nan(v) -> bool:
    import math
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def _extract_conversation(prompts_lst):
    """
    prompts_lst: np.ndarray / list, like ["U0", "A0", "U1", "A1", ...]
    return: list[{"role": ..., "content": ...}]
    """
    if isinstance(prompts_lst, np.ndarray):
        prompts_lst = prompts_lst.tolist()
    roles = ("user", "assistant")
    conversation = []
    for i, text in enumerate(prompts_lst):
        conversation.append({"role": roles[i % 2], "content": str(text)})
    assert len(conversation) % 2 == 1, f"invalid conversation: {prompts_lst}"
    return conversation[:-1], conversation[-1]


def process_grm_history(tokenizer, history, base_length, max_total):
    """
    style:
        <对话历史>
        user\nxxx
        assistant\nxxx
        </对话历史>
    """
    history_tag_ids = tokenizer("<对话历史>\n")["input_ids"]
    history_end_tag_ids = tokenizer("\n</对话历史>\n\n")["input_ids"]
    if history is None or len(history) == 0:
        return tokenizer("<对话历史>\n无\n</对话历史>\n\n")["input_ids"]

    available = (max_total - base_length - len(history_tag_ids) - len(history_end_tag_ids))
    buffer = []
    current_len = 0

    # 逆向处理历史对话
    for message in reversed(history):
        role = message["role"]
        content = message["content"]
        new_content = f"{role}\n{content}\n"
        new_tokens = tokenizer(new_content)["input_ids"]

        if current_len + len(new_tokens) > available:
            break

        buffer.append(new_tokens)
        current_len += len(new_tokens)

    if buffer:
        history_tokens = sum(reversed(buffer), [])
        return (history_tag_ids + history_tokens + history_end_tag_ids)
    else:
        return []


def process_qrm_history(tokenizer, history, base_length, max_total):
    """
    style:
        [BOS]user\nXXXXX[EOS]
        [BOS]assistant\nXXXXX[EOS]
    """
    if history is None or len(history) == 0:
        return []
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    available = max_total - base_length
    buffer = []
    current_len = 0

    # 逆向处理历史对话
    for message in reversed(history):
        role = message["role"]
        content = message["content"]
        new_content = f"{bos_token}{role}\n{content}\n{eos_token}"
        new_tokens = tokenizer(new_content)["input_ids"]

        if current_len + len(new_tokens) > available:
            break

        buffer.append(new_tokens)
        current_len += len(new_tokens)

    if buffer:
        history_tokens = sum(reversed(buffer), [])
        return history_tokens
    else:
        return []


def decode_with_image_tag(tokenizer, ids, skip_special_tokens=True, image_tag="<|image|>"):
    if isinstance(ids, (list, tuple)):
        seq = ids
    else:
        seq = torch.as_tensor(ids, dtype=torch.long, device="cpu").tolist()
    if -100 not in seq:  # pure text
        return tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)
    out, i, n = [], 0, len(seq)
    while i < n:
        if seq[i] == -100:
            while i < n and seq[i] == -100:
                i += 1
            out.append(image_tag)
        else:
            j = i
            while j < n and seq[j] != -100:
                j += 1
            out.append(tokenizer.decode(seq[i:j], skip_special_tokens=skip_special_tokens))
            i = j

    return "".join(out)


def replace_image_tag(prompt, images_bytes_lst=None, img_tag="<|image|>", tokenizer=None, use_token_ids=False):
    pattern = rf"({re.escape(img_tag)})"
    prompt_chunks = re.split(pattern, prompt)
    image_tag_count = sum(1 for chunk in prompt_chunks if chunk == img_tag)
    if image_tag_count != len(images_bytes_lst):
        raise ValueError("Mismatch between image tags "
                         f"({image_tag_count}) and provided images ({len(images_bytes_lst)})\n"
                         "=== PROMPT RAW START ===\n"
                         f"{prompt}\n"
                         "=== PROMPT RAW END ===")
    content = []
    image_idx = 0
    for chunk in prompt_chunks:
        if len(chunk) == 0:
            continue
        if chunk == img_tag:
            content.append({"type": "image_binary", "image_binary": {"binary": images_bytes_lst[image_idx]}})
            image_idx += 1
        else:
            if use_token_ids:
                content.append({"type": "input_ids", "input_ids": tokenizer(chunk)["input_ids"]})
            else:
                content.append({"type": "text", "text": chunk})
    return content


def get_query_imgs(dist_data_manager, **kwargs):
    images_bytes_lst = kwargs.get("images_bytes_lst", [])
    images_bytes_ref = kwargs.get('images_bytes_ref', [])
    if images_bytes_lst:
        return images_bytes_lst
    if images_bytes_ref:
        refs = ray.get(dist_data_manager.get_refs.remote([images_bytes_ref]))  # -> List[ObjectRef]
        image_lst = ray.get(refs)  # -> List[List[bytes]]
        return image_lst[0]  # np.array
    return []


def swap_std_ans(text: str,
                 std_open: str = "<回答1>",
                 std_close: str = "</回答1>",
                 ans_open: str = "<回答2>",
                 ans_close: str = "</回答2>") -> str:
    std_pat = re.compile(re.escape(std_open) + r'(.*?)' + re.escape(std_close), flags=re.DOTALL)
    ans_pat = re.compile(re.escape(ans_open) + r'(.*?)' + re.escape(ans_close), flags=re.DOTALL)

    std_m = std_pat.search(text)
    ans_m = ans_pat.search(text)
    if not (std_m and ans_m):
        return text  # 任一缺失就原样返回

    std, ans = std_m.group(1), ans_m.group(1)

    placeholder = f"__SWAP_PLACEHOLDER_{uuid4().hex}__"

    text = re.sub(rf"({re.escape(std_open)})(.*?)({re.escape(std_close)})",
                  lambda m: m.group(1) + placeholder + m.group(3),
                  text,
                  count=1,
                  flags=re.DOTALL)

    text = re.sub(rf"({re.escape(ans_open)})(.*?)({re.escape(ans_close)})",
                  lambda m: m.group(1) + std + m.group(3),
                  text,
                  count=1,
                  flags=re.DOTALL)

    return text.replace(placeholder, ans)
