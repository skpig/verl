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
VLM dataset
"""

import io
import re

import numpy as np
import pandas as pd
import torch
import verl.utils.torch_functional as verl_F
from PIL import Image

from alpha_seed.prompts.load import random_transform, ith_transform
from alpha_seed.utils.dataset.rl_dataset import RLHFDataset


class BytesDecoder:

    def __call__(self, image_bytes):
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
                image = image.convert("RGBA")
                white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
                white.paste(image, mask=image.split()[3])
                image = white
            else:
                image = image.convert("RGB")
        return image


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor) and key not in ['pixel_values', 'image_grid_hw', 'num_image_tokens']:
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
        non_tensors[key] = np.fromiter(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


def postprocess_data(input_ids, attention_mask, max_length: int, pad_token_id: int, left_pad=True, truncation='error'):
    """
    input_data is the output from tokenizer.
    """
    assert truncation in ['left', 'right', 'error']
    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = verl_F.pad_sequence_to_length(input_ids,
                                                  max_seq_len=max_length,
                                                  pad_token_id=pad_token_id,
                                                  left_pad=left_pad)
        attention_mask = verl_F.pad_sequence_to_length(attention_mask,
                                                       max_seq_len=max_length,
                                                       pad_token_id=0,
                                                       left_pad=left_pad)
    elif sequence_length > max_length:
        if truncation == 'left':
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == 'right':
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == 'error':
            raise NotImplementedError(f'{sequence_length=} is larger than {max_length=}')
        else:
            raise NotImplementedError(f'Unknown truncation method {truncation}')

    return input_ids, attention_mask


class RLHFDatasetVL(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self, *args, **kwargs):
        self.processor = kwargs.pop('processor', None)
        self.num_limit = kwargs.pop('num_limit', None)
        self.image_key = kwargs.pop('image_key', 'image')
        self.bytes_decoder = BytesDecoder()
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        if 'session' in self.dataframe.iloc[item]:
            row_dict = self.dataframe.iloc[item]['session']
        else:
            row_dict = self.dataframe.iloc[item].to_dict()

        row_dict_ret = {}

        chat = row_dict[self.prompt_key]
        image = row_dict[self.image_key]

        prompt_names = []
        if self.multi_prompts == "none":
            pil_images = [self.bytes_decoder(img) for img in image] if image is not None and len(image) > 0 else None
            user_contents = [{"type": "text", "text": f"{self.tokenizer.bos_token}user\n "}]
            prompt_chunks = re.split(r"(<image>)", chat[0])
            for chunk in prompt_chunks:
                if not chunk:
                    continue
                if chunk == '<image>':
                    user_contents.append({"type": "image"})
                else:
                    user_contents.append({"type": "text", "text": chunk})
            user_contents.append({
                "type": "text",
                "text": f"{self.tokenizer.eos_token}{self.tokenizer.bos_token}assistant\n"
            })

            conversation = [{
                "role":
                    "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{self.tokenizer.bos_token}system\n"
                    },
                    {
                        "type": "text",
                        "text": row_dict['system_prompt']
                    },
                    {
                        "type": "text",
                        "text": self.tokenizer.eos_token,
                    },
                ]
            }, {
                "role": "user",
                "content": user_contents
            }]
            inputs = self.processor(images=pil_images,
                                    conversation=conversation,
                                    return_tensors="pt",
                                    return_prompt=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            prompt_with_chat_template = inputs['prompt']
            input_ids, attention_mask = postprocess_data(input_ids,
                                                         attention_mask,
                                                         max_length=self.max_prompt_length,
                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                         left_pad=True,
                                                         truncation=self.truncation)

            row_dict_ret['input_ids'] = input_ids[0]
            row_dict_ret['prompt'] = prompt_with_chat_template
            row_dict_ret['attention_mask'] = attention_mask[0]
            if image is not None and len(image) > 0:
                row_dict_ret['raw_image'] = []
                row_dict_ret['pixel_values'] = inputs['pixel_values']
                row_dict_ret['image_grid_hw'] = torch.tensor(inputs['image_grid_hw'])
            else:
                row_dict_ret['raw_image'] = []
                row_dict_ret['pixel_values'] = None
                row_dict_ret['image_grid_hw'] = None

            # reward_model is required
            row_dict_ret['reward_model'] = {}
            row_dict_ret['reward_model']['style'] = row_dict['ability']
            row_dict_ret['reward_model']['ground_truth'] = row_dict['verifier_feature']
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

            row_dict_ret['input_ids'] = torch.cat(all_input_ids)
            row_dict_ret['attention_mask'] = torch.cat(all_attention_mask)

        # 添加answer
        if self.use_ref_answer:
            answer = row_dict.get(self.answer_key, "")
        else:
            answer = ""
        if answer and not pd.isna(answer):
            raise NotImplementedError("ref answer is not supported now")

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict_ret['raw_prompt'] = [{"role": "user", "content": chat[0]}]

        index = row_dict.get("extra_info", {}).get("index", item)  ## important for grpo to group info
        row_dict_ret["index"] = index
        row_dict_ret['prompt_names'] = prompt_names

        def cast_type(key, dtype):
            if key in row_dict_ret:
                row_dict_ret[key] = row_dict_ret[key].to(dtype)

        # type cast to save memory
        cast_type('input_ids', torch.int32)
        cast_type('attention_mask', torch.int8)
        row_dict_ret['data_source'] = row_dict['data_source']
        row_dict_ret['off_policy_steps'] = torch.zeros([1]).to(torch.int8)
        return row_dict_ret

    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()
        if self.num_limit:
            self.dataframe = self.dataframe[:self.num_limit]


class RLHFDatasetGUI(RLHFDatasetVL):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        if 'session' in self.dataframe.iloc[item]:
            row_dict = self.dataframe.iloc[item]['session'].copy()
        else:
            row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict[self.prompt_key]
        system_prompt = row_dict["system_prompt"]
        image = row_dict[self.image_key]

        prompt_names = []
        if self.multi_prompts == "none":
            pil_images = [self.bytes_decoder(img) for img in image]
            content = [
                {
                    "type":
                        "text",
                    "text":
                        f"<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>user\n{system_prompt}<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"
                },
            ]

            last_response = None
            num_img = 0
            for item in chat:
                if item == "<image>":
                    num_img += 1
                    content[-1]["text"] += f"<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>user\n"
                    content.append({
                        "type": "image",
                    })
                    last_response = "image"
                else:
                    if last_response == "image":
                        content.append({
                            "type":
                                "text",
                            "text":
                                f"<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant\n{item}<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
                        })
                    else:
                        content.append({
                            "type":
                                "text",
                            "text":
                                f"<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant\n{item}<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
                        })
                    last_response = "thought"
            content.append({
                "type":
                    "text",
                "text":
                    f"<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]><[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>assistant\n",
            })

            conversation = [{
                "role": "user",
                "content": content,
            }]

            assert num_img == len(image)

            inputs = self.processor(images=pil_images,
                                    conversation=conversation,
                                    return_tensors="pt",
                                    return_prompt=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            prompt_with_chat_template = inputs['prompt']
            input_ids, attention_mask = postprocess_data(input_ids,
                                                         attention_mask,
                                                         max_length=self.max_prompt_length,
                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                         left_pad=True,
                                                         truncation=self.truncation)

            row_dict['input_ids'] = input_ids[0]
            row_dict['prompt'] = prompt_with_chat_template
            row_dict['attention_mask'] = attention_mask[0]
            row_dict['pixel_values'] = inputs['pixel_values']
            row_dict['image_grid_hw'] = torch.tensor(inputs['image_grid_hw'])

            # reward_model is required
            row_dict['reward_model'] = {}
            row_dict['reward_model']['style'] = row_dict['ability']
            row_dict['reward_model']['ground_truth'] = row_dict['verifier_feature']
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
            raise NotImplementedError("use ref answer is not supported now")

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = [{"role": "user", "content": chat[0]}]

        index = row_dict.get("extra_info", {}).get("index", item)  ## important for grpo to group info
        row_dict["index"] = index
        row_dict['prompt_names'] = prompt_names

        def cast_type(key, dtype):
            if key in row_dict:
                row_dict[key] = row_dict[key].to(dtype)

        # type cast to save memory
        cast_type('input_ids', torch.int32)
        cast_type('attention_mask', torch.int8)
        row_dict['off_policy_steps'] = torch.zeros([1]).to(torch.int8)
        return row_dict

    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()
        if self.num_limit:
            self.dataframe = self.dataframe[:self.num_limit]
