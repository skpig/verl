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
import logging
from typing import List, Union

import torch
from transformers import PreTrainedTokenizer

from alpha_seed.utils.dataset.rl_dataset import RLHFDataset
from alpha_seed.utils.dataset.vlm_rl_dataset import RLHFDatasetVL


class RLHFDatasetSwalm(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ability_key = kwargs.pop('ability_key', 'ability')

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()
        ability = row_dict[self.ability_key]

        if ability == "swalm_env":
            prompt = row_dict.pop(self.prompt_key)
            extra_info = row_dict.get("extra_info", {})
            if len(prompt):
                extra_info.update({"prompt": prompt})
            task_type = extra_info.get("task_type", "")
            dataset_id = extra_info.get("dataset_id", "")
            instance_id = extra_info.get("instance_id", "")
            index = extra_info.get("index", item)

            if self.return_raw_chat:
                if len(prompt):
                    row_dict['raw_prompt'] = prompt
                else:
                    row_dict['raw_prompt'] = []

            row_dict["index"] = index
            row_dict['max_new_tokens'] = self.max_response_length
            row_dict['prompt_names'] = [""]

            logging.info(
                f"agent task {index} ->  task_type: {task_type}, dataset_id: {dataset_id}, instance_id: {instance_id}")

            # fake input_ids as placeholder
            row_dict['input_ids'] = torch.ones(self.max_prompt_length, dtype=torch.int32) * self.tokenizer.pad_token_id
            row_dict['attention_mask'] = torch.zeros(self.max_prompt_length, dtype=torch.int32)
            row_dict['answer_input_ids'] = torch.ones(self.max_prompt_length,
                                                      dtype=torch.int32) * self.tokenizer.pad_token_id
            row_dict['answer_attention_mask'] = torch.zeros(self.max_prompt_length, dtype=torch.int32)
            extra_info.update({"is_eval": self.is_eval})
            row_dict['extra_info'] = extra_info
        else:
            row_dict = super().__getitem__(item)
            extra_info = row_dict.get("extra_info", {})
            row_dict['extra_info'] = extra_info
        return row_dict


class RLHFDatasetVLSwalm(RLHFDatasetSwalm, RLHFDatasetVL):

    def __init__(self, *args, **kwargs):
        RLHFDatasetVL.__init__(self, *args, **kwargs)
        self.ability_key = kwargs.pop('ability_key', 'ability')

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()
        ability = row_dict[self.ability_key]

        if ability == "swalm_env":
            row_dict = RLHFDatasetSwalm.__getitem__(self, item)
        else:
            row_dict = RLHFDatasetVL.__getitem__(self, item)
        return row_dict
