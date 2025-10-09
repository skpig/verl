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
import copy

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from hdfs_io import hlist_files, hisdir

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from alpha_seed.prompts.load import random_transform, load_prompts, ith_transform
from alpha_seed.utils.reward_score import select_prepare_rm_input_fn


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
                 remote_rm_type=None,
                 max_prompt_length=1024,
                 max_response_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 multi_prompts="none",
                 num_prompts_per_data=1,
                 is_eval=False,
                 total_epochs=1,
                 shuffle_per_epoch=False,
                 data_auto_repeat=False):

        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        # Check if any of the paths are HDFS directories and expand them
        expanded_files = []
        for file_path in parquet_files:
            if hisdir(file_path):
                # If it's an HDFS directory, list all files in it
                files_in_dir = hlist_files([file_path])
                parquet_files_in_dir = [f for f in files_in_dir if f.endswith('.parquet')]
                expanded_files.extend(parquet_files_in_dir)
                print(f"Expanded HDFS directory {file_path} to {len(parquet_files_in_dir)} parquet files")
            else:
                # If it's not a directory, keep it as is
                expanded_files.append(file_path)

        parquet_files = expanded_files

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.use_ref_answer = use_ref_answer
        self.remote_rm_type = remote_rm_type
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.multi_prompts = multi_prompts
        self.num_prompts_per_data = num_prompts_per_data
        self.is_eval = is_eval
        self.new_dataset_flag = True

        # New parameters for epoch replication
        self.total_epochs = total_epochs
        self.shuffle_per_epoch = shuffle_per_epoch
        self.data_auto_repeat = data_auto_repeat

        self._download()
        self._read_files_and_tokenize()
        self._initialize_prompts()

        if self.is_eval:
            self.num_prompts_per_data = len(self.prompts)  # every prompt need eval

    def _initialize_prompts(
        self,
    ):
        self.prompts = load_prompts(self.multi_prompts)

    def _download(self, origin=False):
        from verl.utils.fs import copy_local_path_from_hdfs
        parquet_files = self.parquet_files if not origin else copy.deepcopy(self.original_parquet_files)
        for i, parquet_file in enumerate(parquet_files):
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

        # Apply epoch replication if needed
        if self.shuffle_per_epoch:
            self.dataframe = self.dataframe.sample(frac=1.0).reset_index(drop=True)
            print('dataset shuffled')
        if hasattr(self, 'data_auto_repeat') and self.data_auto_repeat:
            self._replicate_for_epochs()

    def _replicate_for_epochs(self):
        """
        Replicates the dataset for the specified number of epochs.
        If shuffle_per_epoch is True, each epoch's data will be shuffled before concatenation.
        """
        original_df = self.dataframe.copy()
        all_dataframes = [original_df]

        for i in range(1, self.total_epochs):
            if self.shuffle_per_epoch:
                # Shuffle the dataframe
                epoch_df = original_df.sample(frac=1.0).reset_index(drop=True)
            else:
                epoch_df = original_df.copy()

            # Add an epoch identifier for easier tracking if needed
            epoch_df['_epoch_id'] = i
            all_dataframes.append(epoch_df)

        # Concatenate all dataframes
        self.dataframe = pd.concat(all_dataframes, ignore_index=True)
        print(f'Dataset replicated for {self.total_epochs} epochs, new len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.new_dataset_flag = True if hasattr(self, 'original_parquet_files') else False
        # resume dataframe if not it's serialized in data.pt
        if self.new_dataset_flag:
            self._download(origin=True)
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_names = []
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
            prompt_names.append("")
        else:
            all_input_ids = []
            all_attention_mask = []
            for i in range(self.num_prompts_per_data):
                if not self.is_eval:  # random
                    data, prompt_name = random_transform(self.prompts, chat[0]['content'])  # -> str
                else:
                    data, prompt_name = ith_transform(self.prompts, chat[0]['content'], idx=i)  # -> str

                prompt_names.append(prompt_name)
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

        if self.remote_rm_type is not None:
            prepare_rm_input = select_prepare_rm_input_fn(self.remote_rm_type)
            rm_input = prepare_rm_input(chat,
                                        answer,
                                        self.tokenizer,
                                        max_prompt_len=self.max_prompt_length,
                                        max_resp_len=self.max_response_length)
            row_dict['reward_model']['rm_pre_ids'] = rm_input['rm_pre_ids'].to(torch.int32).tolist()
            row_dict['reward_model']['rm_post_ids'] = rm_input['rm_post_ids'].to(torch.int32).tolist()
            row_dict['reward_model']['rm_required_type'] = self.remote_rm_type

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        extra_info = row_dict.get("extra_info")
        if extra_info is None or pd.isna(extra_info) or 'index' not in extra_info:
            index = item
        else:
            index = extra_info['index']
        row_dict["index"] = index
        row_dict['prompt_names'] = prompt_names

        # type cast to save memory
        row_dict['input_ids'] = row_dict['input_ids'].to(torch.int32)
        row_dict['attention_mask'] = row_dict['attention_mask'].to(torch.int8)
        row_dict['answer_input_ids'] = row_dict['answer_input_ids'].to(torch.int32)
        row_dict['answer_attention_mask'] = row_dict['answer_attention_mask'].to(torch.int8)

        row_dict['max_new_tokens'] = self.max_response_length

        return row_dict

    def __getstate__(self):
        if self.new_dataset_flag:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()


if __name__ == '__main__':
    from transformers import AutoTokenizer

    from torch.utils.data import DataLoader

    local_path = "p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf"
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    from mono_rl.utils.seed import CHAT_TEMPLATE
    tokenizer.chat_template = CHAT_TEMPLATE

    dataset = RLHFDataset(parquet_files='combine_math7k_aime800_mathv2_repeat10.parquet',
                          tokenizer=tokenizer,
                          prompt_key='prompt',
                          answer_key='answer',
                          use_ref_answer=True,
                          max_prompt_length=4096,
                          max_response_length=24576,
                          multi_prompts="all",
                          num_prompts_per_data=1)

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    from mono_rl import DataProto

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
