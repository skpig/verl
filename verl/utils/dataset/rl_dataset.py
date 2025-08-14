# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
from hmac import new
import logging
import os
import json
import re
from collections import defaultdict
from typing import Optional

from networkx import ancestors

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.protocol import DataProto
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


system_prompt0 = """When tackling complex reasoning tasks, you should first thinks about the reasoning process in the mind and then provides the answer. 

You should strictly follow the format below:

<think>
Your reasoning process step 1 here
</think>
<think>
Your reasoning process step 2 here
</think>
<think>
Your reasoning process step 3 here
</think>
...
<think>
Your reasoning process step N here
</think>
<answer>
Put your final answer within \\boxed{}.
</answer>
"""
system_prompt1 = """Your task is to solve the user's math problem. You should first thinks about the reasoning process in the mind and then provides the answer. 

Your reasoning must be broken down into 3~5 distinct steps, each enclosed in `<think>` tags. An ideal step represents the completion of **a clear sub-goal**. It should be a self-contained paragraph that explains how you achieved one milestone in the overall solution. Group related calculations and logic together.

You should strictly follow the format below:

<think>
Reasoning step 1 here
</think>
<think>
Reasoning step 2 here
</think>
...
<think>
Final reasoning step here
</think>
<answer>
Put your final answer within \\boxed{}.
</answer>
"""
system_prompt2 = """When tackling complex reasoning tasks, you should first thinks about the reasoning process step by step in the mind and then provides the answer. 
An ideal reasoning step represents the completion of **a clear sub-goal**. It should be a self-contained paragraph that explains how you achieved one milestone in the overall solution. Group related calculations and logic together.
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., 

<think> reasoning process here </think> <answer> answer here </answer>.
"""
system_prompt3 = """Please reason step by step, put your reasoning process within <think> </think> tags, and put your final answer within <answer> </answer> tags, respectively, i.e., 
<think> reasoning process here </think> <answer> answer here </answer>.
"""
system_prompt4 = """Please reason step by step, and put your final answer within <answer> </answer> tags"""
# NOTE: always wrap answer within <answer> </answer> tags, related to "math_verify.extract_answer()"
# system_prompt3 = """Please reason step by step.  Put your final answer within <answer> </answer> tags, i.e., <answer> your answer here </answer>"""
all_prompts = [
    system_prompt0,
    system_prompt1,
    system_prompt2,
    system_prompt3,
    system_prompt4,
]



class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig, # config.data
        processor: Optional[ProcessorMixin] = None,
        is_thinking_tokenizer: bool = False,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.is_thinking_tokenizer = is_thinking_tokenizer

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_response_length = config.get("max_response_length", 1024 * 5)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)
        
    def _add_system_prompt_to_doc(self, doc: dict):
        """
        Add system prompt to the document.
        """
        assert doc['prompt'][0]['role'] != "system", "The first message should not be a system message."
        doc['prompt'].insert(0, {"role": "system", "content": all_prompts[self.config.prompt_id]})
        if self.is_thinking_tokenizer:
            doc['prompt'].pop(-1) # remove the last message, which is the "<think>" prefix

        return doc

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        # DEBUG:
        # self.dataframe = self.dataframe.select(range(50))

        print(f"dataset len: {len(self.dataframe)}")

        # add back system prompt
        self.dataframe = self.dataframe.map(
            self._add_system_prompt_to_doc,
            num_proc=self.num_workers,
            desc="Adding system prompt",
        )

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=self.is_thinking_tokenizer, continue_final_message=not self.is_thinking_tokenizer, tokenize=False))
                <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")
        
        print("==== Example Dataset ====")
        print(json.dumps(self.dataframe[0], indent=2, ensure_ascii=False))

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import (process_image,
                                                         process_video)

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=self.is_thinking_tokenizer, continue_final_message=not self.is_thinking_tokenizer, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=self.is_thinking_tokenizer, continue_final_message=not self.is_thinking_tokenizer, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        # TODO: maybe buggy when the input_ids contains previous response
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            # TODO: implement position ID computation for other processors
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = torch.tensor(index, dtype=torch.int)
        row_dict['item'] = int(item)  # indicate the index of the item in the dataset
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()



class TreeNode:
    """
    A class representing a node in a tree structure.
    """

    def __init__(self, 
                 item,
                 father_node: Optional['TreeNode'] = None,
                 partial_rollout: Optional[list[int]] = None,
                 step_num=0):
        """
        Initialize the TreeNode with the given data.
        """
        self.item = item # a unique identifier for the node, also the one used to access the node in the dataset
        self.father_node = father_node  # the parent node of this node, None if it's the root node
        self.children = []

        self.step_num = step_num  # the number of steps in the training process

        self.partial_rollout = partial_rollout  # the length of the partial rollout

        assert step_num <= 0 or partial_rollout is not None and father_node is not None
    
    @property
    def partial_rollout_len(self):
        """
        The length of the partial rollout.
        If partial_rollout is None, return 0.
        """
        return len(self.partial_rollout)
    
    @property
    def depth(self):
        """
        The depth of the node in the tree.
        """
        if self.step_num == 0:
            return 0
        return self.father_node.depth + 1
    
    def get_original_ancestor_item(self):
        """
        Get the original ancestor of this node.
        The original ancestor is the root node of the tree.
        """
        if self.step_num == 0:
            return self.item
        return self.father_node.get_original_ancestor_item()

    def add_child(self, child_node: 'TreeNode'):
        """
        Add a child node to this node.
        """
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode(item={self.item})"

class TreeDataset(RLHFDataset):
    """
    A dataset class that represents a tree structure.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the TreeDataset with the given arguments.
        """
        super().__init__(*args, **kwargs)
        # breakpoint()
        self.original_datalength = len(self.dataframe)

        # Initialize an empty dataset for new data
        self.new_dataframe = datasets.Dataset.from_dict({})

        self.root = TreeNode(item=-1, father_node=None, step_num=-1)
        self.item2node = {-1: self.root}
        self.next_item = 0

        for i in range(self.original_datalength):
            node = TreeNode(item=i, father_node=self.root, step_num=0)
            self.root.add_child(node)
            self.item2node[i] = node
            self.next_item += 1
    
    def __len__(self):
        return self.next_item


    def __getitem__(self, item):
        if item < self.original_datalength:
            row_dict = super().__getitem__(item)
            row_dict['partial_rollout_len'] = 0  # no partial rollout for original data
            return row_dict
        
        # If the index is greater than the original data length, it is a newly generated item.
        node = self.item2node.get(item, None)
        original_item = node.get_original_ancestor_item()

        # Get the original row dict from the dataframe
        original_row_dict = super().__getitem__(original_item)

        # TODO: Replace some important fields with the new data
        # input_ids
        # attention_mask
        # position_ids
        # 上述这三个都是有严格的长度限制的，考虑不修改
        # raw_prompt_ids改为original prompt + partial rollout，这个是用于rollout作为input的
        original_row_dict["item"] = item
        original_row_dict["raw_prompt_ids"] = original_row_dict["raw_prompt_ids"] + node.partial_rollout

        # 加上一个partial rollout len (int) OR partial rollout mask (tensor[response_len])
        original_row_dict["partial_rollout_len"] = node.partial_rollout_len
        # original_row_dict["rollout_kwargs"] = {
        #     'max_new_tokens': self.max_response_length - node.partial_rollout_len,
        #     }


        # 并在rollout: 
        # 1. 更新max_new_tokens
        # 2. 更新input_ids, attention_mask, position_ids
        # 还需要更新对应的response_mask
        return original_row_dict
        

    def update(self, batch: DataProto, step_num: int) -> None:
        """
        Update the dataset with the current batch.
        This method is called after each training batch.
        """
        items = torch.tensor(batch.non_tensor_batch['item'].astype(int))

        unique_indices, inverse_indices = torch.unique(items, return_inverse=True)

        all_scores = torch.tensor(batch.non_tensor_batch['score']) # raw score
        all_response_mask = batch.batch['response_mask_w_partial_rollouts'].bool()
        all_response_len = all_response_mask.sum(dim=-1).tolist()
        all_responses = batch.batch["responses"]
        all_values = batch.batch["values"]
        all_entropys = batch.batch["entropys"]

        # We can select the item with highest score as the new node
        assert len(unique_indices) == len(set(items)), "Currently, items should be unique in the batch."
        # assert self.use_critic, "Currently only support use_critic=True for TreeDataset"

        # breakpoint()
        for i, index in enumerate(inverse_indices):
            item = unique_indices[index].item()
            father_node = self.item2node.get(item, None)
            assert father_node is not None, f"Item {item} not found in the dataset."

            # Currently only keep the correct response and the original data item
            # DEBUG:
            if all_scores[i] == 0 or father_node.depth > 0:
                continue

            valid_position = int(all_response_len[i] * 0.5) # only use the first half of the response as partial rollout

            # V1: Use the index with highest value as the new node, should assert critic_lam == 1
            values = all_values[i, :valid_position]
            max_value_index = torch.argmax(values).item()

            partial_rollout_len = max_value_index # the index with highest value should be excluded, since V[i] is the value of the previous token
            partial_rollout = all_responses[i, :partial_rollout_len].tolist()


            # Create new node
            new_item = self.next_item
            self.item2node[new_item] = TreeNode(
                item=new_item,
                father_node=father_node,
                partial_rollout=partial_rollout,
                step_num=step_num
            )
            father_node.add_child(self.item2node[new_item])
            self.next_item += 1
            
        # Remove some old rollouts if log_prob of partial rollout is too low under current policy


    def state_dict(self):
        """
        Return the state dict of the dataset.
        """
        # TODO:
        pass

    def load_state_dict(self, state_dict):
        """
        Load the state dict of the dataset.
        """
        # TODO:
        pass

