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

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from alpha_seed.prompts.load import random_transform, load_prompts


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 answer_key='answer',
                 use_ref_answer=False,
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 multi_prompts="none",
                 num_prompts_per_data=1):

        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.use_ref_answer = use_ref_answer
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.multi_prompts = multi_prompts
        self.num_prompts_per_data = num_prompts_per_data

        self._download()
        self._read_files_and_tokenize()
        self._initialize_prompts()

    def _initialize_prompts(
        self,
    ):
        self.prompts = load_prompts(self.multi_prompts)

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # # filter out too long prompts
        # tokenizer = self.tokenizer
        # prompt_key = self.prompt_key
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        if self.multi_prompts == "none":
            # chat[0] is dict({'content': '', 'role': ''})
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat,
                                                                           add_generation_prompt=True,
                                                                           tokenize=False)
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                             tokenizer=self.tokenizer,
                                                                             max_length=self.max_prompt_length,
                                                                             pad_token_id=self.tokenizer.pad_token_id,
                                                                             left_pad=True,
                                                                             truncation=self.truncation)
            row_dict['input_ids'] = input_ids[0]
            row_dict['attention_mask'] = attention_mask[0]
        else:
            all_input_ids = []
            all_attention_mask = []
            for i in range(self.num_prompts_per_data):
                data = random_transform(self.prompts, chat[0]['content'])  # -> str
                prompt_with_chat_template = self.tokenizer.apply_chat_template(data,
                                                                               add_generation_prompt=True,
                                                                               tokenize=False)
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=prompt_with_chat_template,
                    tokenizer=self.tokenizer,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation)
                all_input_ids.append(input_ids[0])
                all_attention_mask.append(attention_mask[0])

            row_dict['input_ids'] = torch.cat(all_input_ids)
            row_dict['attention_mask'] = torch.cat(all_attention_mask)

        # 添加answer
        if self.use_ref_answer:
            answer = row_dict.get(self.answer_key, "")
        else:
            answer = ""
        if answer and not pd.isna(answer):
            sp = "请参考以下内容进行回答: \n\n" + answer
            if chat[0]["role"] == "system":
                prompt_with_chat_template = prompt_with_chat_template.split(self.tokenizer.eos_token)[1]
            prompt_with_chat_template = self.tokenizer.bos_token + "system\n" + sp + self.tokenizer.eos_token + prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        row_dict['answer_input_ids'] = input_ids[0]
        row_dict['answer_attention_mask'] = attention_mask[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        # type cast to save memory
        row_dict['input_ids'] = row_dict['input_ids'].to(torch.int32)
        row_dict['attention_mask'] = row_dict['attention_mask'].to(torch.int8)
        row_dict['answer_input_ids'] = row_dict['answer_input_ids'].to(torch.int32)
        row_dict['answer_attention_mask'] = row_dict['answer_attention_mask'].to(torch.int8)
        row_dict['off_policy_steps'] = torch.zeros([1]).to(torch.int8)
        return row_dict


if __name__ == '__main__':
    from transformers import AutoTokenizer

    from torch.utils.data import DataLoader

    local_path = "p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf"
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    from verl.utils.seed import CHAT_TEMPLATE
    tokenizer.chat_template = CHAT_TEMPLATE

    dataset = RLHFDataset(parquet_files='combine_math7k_aime800_mathv2_repeat10.parquet',
                          tokenizer=tokenizer,
                          prompt_key='prompt',
                          answer_key='answer',
                          use_ref_answer=True,
                          max_prompt_length=4096,
                          multi_prompts="all",
                          num_prompts_per_data=1)

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    from verl import DataProto

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    data = dataset[0]['input_ids']
    output = tokenizer.batch_decode([data])[0]
    print(f'\n\noutput: {output.replace(tokenizer.pad_token, "")}')

    data = dataset[0]['answer_input_ids']
    output = tokenizer.batch_decode([data])[0]
    print(f'\n\noutput: {output.replace(tokenizer.pad_token, "")}')
