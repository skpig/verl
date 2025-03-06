# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer, PreTrainedTokenizer
from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask


def label_str2dict(label):
    label_count = 0
    try:
        res = {}
        ranks = label.split(",")
        rank_list = []
        for i in ranks:
            cur_ranks = i.split("=")
            label_count += len(cur_ranks)
            rank_list.append(list(map(lambda x: int(x), cur_ranks)))
        for i in range(len(rank_list)):
            for j in rank_list[i]:
                res[j] = i
        label_num = len(rank_list)
    except:
        res = {}
        label_num = 0
    return res, label_num, label_count


class RMDataset(Dataset):
    """
    This is an in-memory RMDataset
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 key='session',
                 max_length=1024,
                 max_response_num=6,
                 truncation='error'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.key = key
        self.max_length = max_length
        self.max_response_num = max_response_num

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.sessions = self.dataframe[self.key].tolist()

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, item):
        '''
        session: {'prompt': array([]), 'response': array([{'content':, }, {'content':, }, ...]), 
                  'vote_label': '2=1=4,3', 'prompt_id': '9e719224288a25505f7911e0f8d8cb7b'}
        '''
        tokenizer = self.tokenizer

        prompt = self.sessions[item]['prompt'][0]
        responses = self.sessions[item]['response']
        response_num = len(responses)
        vote_label = self.sessions[item]['vote_label']
        prompt_id = self.sessions[item]['prompt_id']
        label_dict, label_num, label_count = label_str2dict(vote_label)  # {rank: score}, score lower better

        # prompt process
        prompt_chat = [{'role': 'user', 'content': prompt}]
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]
        prompt_length = prompt_ids.shape[0]

        input_ids_lst, attention_mask_lst, response_labels = [], [], []
        for index, response in enumerate(responses, start=1):
            response_label = label_num - 1 - label_dict[index]  # higher better
            response_chat_str = response['content'] + tokenizer.eos_token
            response_ids_output = tokenizer(response_chat_str, return_tensors='pt', add_special_tokens=False)
            response_ids = response_ids_output['input_ids'][0]
            response_attention_mask = response_ids_output['attention_mask'][0]

            input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
            attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
            # padding to max length
            sequence_length = input_ids.shape[0]
            if sequence_length < self.max_length:
                padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                              dtype=input_ids.dtype) * self.tokenizer.pad_token_id
                padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,),
                                                    dtype=attention_mask.dtype)
                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
            elif sequence_length > self.max_length:
                if self.truncation == 'left':
                    # actually, left truncation may not be reasonable
                    input_ids = input_ids[-self.max_length:]
                    attention_mask = attention_mask[-self.max_length:]
                elif self.truncation == 'right':
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                elif self.truncation == 'error':
                    raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
                else:
                    raise NotImplementedError(f'Unknown truncation method {self.truncation}')
            input_ids_lst.append(input_ids)
            attention_mask_lst.append(attention_mask)
            response_labels.append(response_label)
        while response_num < self.max_response_num:
            response_num += 1
            input_ids_lst.append(input_ids_lst[-1])
            attention_mask_lst.append(attention_mask_lst[-1])
            response_labels.append(response_labels[-1])
        input_ids = torch.stack(input_ids_lst, dim=0)
        attention_mask = torch.stack(attention_mask_lst, dim=0)
        response_labels = torch.tensor(response_labels)
        response_mask = attention_mask.clone()  # (N, max_length)
        response_mask[:, :prompt_length] = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'response_mask': response_mask,
            'scores': response_labels,
            'response_num': torch.tensor(len(responses)),
        }


if __name__ == '__main__':
    local_model_path = copy_local_path_from_hdfs(
        'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/models/gemma-2b-it')
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    dataset = RMDataset(
        parquet_files='hdfs://haruna/home/byte_data_seed/lf_lq/user/caizhao/rm_0812_data/stage1.p0.parquet',
        tokenizer=tokenizer,
        key='session',
        max_length=512)

    data = dataset[0]['input_ids']
    output = tokenizer.batch_decode([data])[0]
    print(output)
