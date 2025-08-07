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

from typing import Dict, List, Optional, Union
import io
import re
import json

import ray
import copy
import numpy as np
import pandas as pd
import torch
import verl.utils.torch_functional as verl_F
from PIL import Image

from alpha_seed.utils.dataset.rl_dataset import RLHFDataset
from alpha_seed.utils.dataset.dist_data_util import DistImageLoader, get_image_manager, get_local_inputs, \
    save_dataproto_image_data_dist, init_or_get_image_manager

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.utils import TensorType

from alpha_seed.utils.server_client import is_local_ray_instance
import pyarrow as pa


def convert_prompts_into_input_ids(prompts,
                                   tokenizer,
                                   image_processor,
                                   max_prompt_length,
                                   num_image_tokens,
                                   truncation='error'):
    input_ids_list = []
    attention_mask_list = []
    for i in range(prompts.batch.batch_size[0]):
        prompt = prompts.non_tensor_batch['prompt'][i]
        input_ids, attention_mask = convert_single_prompt_to_input_ids(prompt,
                                                                       tokenizer=tokenizer,
                                                                       image_processor=image_processor,
                                                                       num_image_tokens=num_image_tokens[i])
        input_ids, attention_mask = postprocess_data(input_ids,
                                                     attention_mask,
                                                     max_length=max_prompt_length,
                                                     pad_token_id=tokenizer.pad_token_id,
                                                     left_pad=True,
                                                     truncation=truncation)
        input_ids_list.append(input_ids[0])
        attention_mask_list.append(attention_mask[0])
    input_ids = torch.stack(input_ids_list, dim=0)
    attention_mask = torch.stack(attention_mask_list, dim=0)
    prompts.batch['input_ids'] = input_ids
    prompts.batch['attention_mask'] = attention_mask


def decode_bytes_to_rgb_image(image_bytes):
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


def convert_conversation_to_prompt(conversation):
    prompt = ""
    for turn in conversation:
        turn_prompt = ""
        for content in turn["content"]:
            if content["type"] == "image":
                turn_prompt += "[SOI]<ImageHere>[EOI]"
            elif content["type"] == "text":
                turn_prompt += content["text"]
            else:
                raise NotImplementedError
        # turn_prompt = f"{self.tokenizer.bos_token} {turn_prompt}"
        prompt += turn_prompt
    return prompt


def convert_single_prompt_to_input_ids(prompt,
                                       tokenizer,
                                       image_processor,
                                       num_image_tokens,
                                       padding=False,
                                       truncation=False,
                                       max_length=None):
    image_token_id = (-100 if not hasattr(tokenizer, "image_token_id") else tokenizer.image_token_id)
    chunks = prompt.split("<ImageHere>")

    input_ids = []
    attention_mask = []
    img_idx = 0
    for chunk in chunks:
        if chunk == "":
            continue
        text_inputs = tokenizer(chunk, padding=padding, truncation=truncation, max_length=max_length)
        input_ids += text_inputs["input_ids"]
        attention_mask += text_inputs["attention_mask"]
        if num_image_tokens is not None and img_idx < len(num_image_tokens):
            if image_processor.use_navit:
                num_img_token = num_image_tokens[img_idx]
            else:
                num_img_token = image_processor.num_img_token
            input_ids += [image_token_id] * num_img_token
            attention_mask += [1] * num_img_token
        img_idx += 1

    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    return input_ids, attention_mask


def process_images(images, image_processor):
    if images is not None:
        if not isinstance(images, List):
            images = [images]
        image_inputs = image_processor(images=images)
        return image_inputs


def get_reward_model(row_dict: dict) -> dict:
    # Use AlphaSeed LLM's reward system if `reward_model` is specified:
    if row_dict.get('reward_model'):
        reward_model = {
            'style': row_dict['reward_model']['style'],
            'ground_truth': row_dict['reward_model']['ground_truth'],
        }
        return reward_model

    # Use VLM's reward system, i.e., row_dict['verifier_feature'], if row_dict['reward_model'] does not exist:
    verifier_feature = json.loads(row_dict['verifier_feature'])
    verifier_name = verifier_feature.get('verifier_name', '')

    # These if-else statements are for backward compatibility. To be deprecated in the future.
    if not verifier_name:
        if row_dict['ability'] == 'verifier_math':
            verifier_name = 'math'
        elif row_dict['ability'] == 'verifier_boxed_str':
            verifier_name = 'boxed_str'
        elif row_dict['ability'] == 'verifier_service':
            verifier_name = 'math_verifier_service'
        elif row_dict['ability'] == 'verifier_visual_functionCall_answer':
            verifier_name = 'visual_cot_verifier'
        elif row_dict['ability'] == 'verifier_vstar':
            verifier_name = 'verifier_vstar'
        elif row_dict['ability'] == 'verifier_zerobench':
            verifier_name = 'verifier_zerobench'
        else:
            data_source = row_dict['data_source']
            raise ValueError(f'Please specify verifier_name in verifier_feature! {data_source} {verifier_feature}')
        verifier_feature['verifier_name'] = verifier_name

    # Use the VLM verifier router to route to the corresponding VLM specialized verifier.
    reward_model = {
        'style': 'vlm_verifier_router',
        'ground_truth': json.dumps(verifier_feature),
    }
    return reward_model


class RLHFDatasetVL(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self, *args, **kwargs):
        self.processor = kwargs.pop('processor', None)
        self.num_limit = kwargs.pop('num_limit', None)
        self.image_key = kwargs.pop('image_key', 'image')
        self.tokenizer_file = kwargs.pop('tokenizer_file', None)
        self.dist_image = kwargs.pop('dist_image', True)
        self.stable_pool_names = kwargs.pop('stable_pool_names', [])
        stable_pool_name = self.stable_pool_names[0] if self.stable_pool_names else ''
        self.image_manager = init_or_get_image_manager(stable_pool_name)
        super().__init__(*args, **kwargs)

    def _read_files_and_tokenize_dist(self):
        pa.set_memory_pool(pa.system_memory_pool())
        is_local_ray = is_local_ray_instance()

        def satisfied(node):
            # 这个约束是为了不要让DistImageLoader调度到spot资源上
            res = node['Resources']
            if res.get('GPU') is None or res.get('GPU') <= 0:
                return False
            if is_local_ray or not self.stable_pool_names:
                # local或没有pool约束则不检查是否有对应的pool资源
                return True
            # 否则满足任意一个pool的都要
            return any(res.get(pool, 0) > 0 for pool in self.stable_pool_names)

        nodes = [node for node in ray.nodes() if node["Alive"] and satisfied(node)]
        assert len(nodes) > 0, (f"should have at least one satisfied node in the cluster. "
                                f"total {len(ray.nodes())} nodes. check the criteria whether too strict")
        self.image_loaders = []
        image_keys = [self.image_key]
        for i, node in enumerate(nodes):
            image_loader = DistImageLoader.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                )).remote(self.original_parquet_files, self.image_key, self.tokenizer_file, len(nodes), i)
            self.image_loaders.append(image_loader)
        ray.get(self.image_manager.set_image_loaders.remote(self.image_loaders))
        offsets = []
        refs = []
        for img_loader in self.image_loaders:
            refs.append(img_loader.process_all_images.remote())

            offsets.append(img_loader.get_offset.remote())

        self.offsets = ray.get(offsets)
        images_bytes_refs = ray.get(refs)
        images_bytes_refs_list = []
        indices = []
        for refs in images_bytes_refs:
            images_bytes_refs_list.extend(refs[0])
            indices.extend(refs[1])
        # make sure no duplicate index
        assert len(indices) == len(set(indices))

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            if 'session' in dataframe:
                dataframe = pd.DataFrame(list(dataframe['session']))
            # dataframe.drop(columns=image_keys, inplace=True)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.dataframe['images_bytes_ref'] = images_bytes_refs_list
        self.dataframe['dataset_index'] = indices

        print(f'original dataset len: {len(self.dataframe)}')

        pool = pa.default_memory_pool()
        pool.release_unused()

        # Apply epoch replication if needed
        if hasattr(self, 'data_auto_repeat') and self.data_auto_repeat:
            assert not self.dist_image
            self._replicate_for_epochs()

    def _read_files_and_tokenize(self):
        if self.dist_image:
            return self._read_files_and_tokenize_dist()
        else:
            return super()._read_files_and_tokenize()

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        if 'session' in self.dataframe.iloc[item]:
            row_dict = self.dataframe.iloc[item]['session']
        else:
            row_dict = self.dataframe.iloc[item].to_dict()

        row_dict_ret = {}
        row_dict_ret["agent_handler"] = row_dict.get("agent_handler", "")

        chat = row_dict[self.prompt_key]
        # encode prompts without chat template
        if self.return_raw_chat:
            if isinstance(chat[0], str):
                row_dict_ret['raw_prompt'] = [{"role": "user", "content": chat[0]}]
            elif isinstance(chat[0], dict):
                row_dict_ret['raw_prompt'] = copy.deepcopy(chat)
            else:
                raise NotImplementedError("raw_prompt must be str or dict")
        if isinstance(chat[0], dict):
            chat = [c["content"] for c in chat]

        user_contents = [{"type": "text", "text": f"{self.tokenizer.bos_token}user\n"}]
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
        system_prompt = row_dict.get('system_prompt', '').strip()
        conversation = []
        if system_prompt:
            conversation.append({
                "role":
                    "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{self.tokenizer.bos_token}system\n"
                    },
                    {
                        "type": "text",
                        "text": system_prompt
                    },
                    {
                        "type": "text",
                        "text": self.tokenizer.eos_token,
                    },
                ]
            })
        conversation.append({"role": "user", "content": user_contents})

        prompt = convert_conversation_to_prompt(conversation)

        # reward_model is required
        row_dict_ret['reward_model'] = get_reward_model(row_dict)

        if 'prompt_id' in row_dict:
            row_dict_ret['prompt_id'] = row_dict['prompt_id']

        # 添加answer
        if self.use_ref_answer:
            answer = row_dict.get(self.answer_key, "")
        else:
            answer = ""
        if answer and not pd.isna(answer):
            raise NotImplementedError("ref answer is not supported now")

        index = row_dict.get("extra_info", {}).get("index", item)  ## important for grpo to group info
        row_dict_ret["index"] = index
        row_dict_ret['prompt_names'] = [""]
        row_dict_ret['prompt'] = prompt

        def cast_type(key, dtype):
            if key in row_dict_ret:
                row_dict_ret[key] = row_dict_ret[key].to(dtype)

        # Add grm input on VLM dataset
        if self.use_grm:
            grm_input = self._prepare_grm_input(conversation,
                                                answer,
                                                max_prompt_len=self.max_prompt_length,
                                                max_resp_len=self.max_response_length)
            row_dict_ret.update(grm_input)
            # type cast for grm input
            cast_type('grm_input_ids', torch.int32)
            cast_type('grm_attention_mask', torch.int8)

        row_dict_ret['data_source'] = row_dict['data_source']
        row_dict_ret['off_policy_steps'] = torch.zeros([1]).to(torch.int8)
        row_dict_ret['images_bytes_ref'] = row_dict['images_bytes_ref']
        return row_dict_ret

    def __getstate__(self):
        if self.new_dataset_flag:
            state = self.__dict__.copy()
            if 'dataframe' in state:
                del state['dataframe']
            if 'image_manager' in state:
                del state['image_manager']
            if 'image_loaders' in state:
                del state['image_loaders']
            return state
        return self.__dict__.copy()

    def resume_dataset_state(self):
        self.new_dataset_flag = True if hasattr(self, 'original_parquet_files') else False
        # resume dataframe if not it's serialized in data.pt
        if self.new_dataset_flag:
            stable_pool_name = self.stable_pool_names[0] if self.stable_pool_names else ''
            self.image_manager = init_or_get_image_manager(stable_pool_name)
            self._download(origin=True)
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')


def convert_input_ids_to_chat(prompt_ids, tokenizer):
    first_non_one_indices = (prompt_ids != tokenizer.pad_token_id).int().argmax(dim=1)
    rmv_padding_prompt_ids = [row[index:].tolist() for row, index in zip(prompt_ids, first_non_one_indices)]
    chat = []
    for input_ids in rmv_padding_prompt_ids:
        processed_ids = []
        i = 0
        n = len(input_ids)
        while i < n:
            if input_ids[i] == -100:
                # 检查连续的-100
                start = i
                while i < n and input_ids[i] == -100:
                    i += 1
                # 替换为一个<image>标记
                processed_ids.append(tokenizer.convert_tokens_to_ids("<image>"))
            else:
                processed_ids.append(input_ids[i])
                i += 1
        # 将处理后的ids转换为字符串
        text = tokenizer.decode(processed_ids, skip_special_tokens=True)
        chat.append([text])
    return chat


def transform_image(prompt, images_bytes, tokenizer, processor, truncation, max_prompt_length=None):
    row_dict_ret = {}
    pil_images = [decode_bytes_to_rgb_image(img) for img in images_bytes
                 ] if images_bytes is not None and len(images_bytes) > 0 else None
    inputs = process_images(pil_images, processor.image_processor)
    if pil_images is not None:
        num_image_tokens = inputs['num_image_tokens']
    else:
        num_image_tokens = []
    input_ids, attention_mask = convert_single_prompt_to_input_ids(prompt, tokenizer, processor.image_processor,
                                                                   num_image_tokens)

    if max_prompt_length is not None:
        input_ids, attention_mask = postprocess_data(input_ids,
                                                     attention_mask,
                                                     max_length=max_prompt_length,
                                                     pad_token_id=tokenizer.pad_token_id,
                                                     left_pad=True,
                                                     truncation=truncation)

    row_dict_ret['input_ids'] = input_ids[0]
    row_dict_ret['prompt'] = prompt
    row_dict_ret['attention_mask'] = attention_mask[0]
    if images_bytes is not None and len(images_bytes) > 0:
        row_dict_ret['raw_image'] = []
        pixel_values = inputs['pixel_values']
        row_dict_ret['image_data'] = {
            "pixel_values": pixel_values,
            "image_grid_hw": torch.tensor(inputs['image_grid_hw'])
        }
        row_dict_ret['num_image_tokens'] = inputs['num_image_tokens']
    else:
        row_dict_ret['raw_image'] = []
        row_dict_ret['image_data'] = None
        row_dict_ret['num_image_tokens'] = None
    return row_dict_ret


def load_and_transform_save_image(prompts,
                                  tokenizer,
                                  processor,
                                  image_manager,
                                  max_prompt_length=None,
                                  truncation="left"):
    if 'input_ids' not in prompts.batch:
        image_bytes = get_local_inputs(prompts.non_tensor_batch, 'images_bytes_ref', image_manager)
        processed_list = []
        from alpha_seed.utils.dataset.vlm_rl_dataset import collate_fn
        for i in range(len(image_bytes)):
            processed = transform_image(prompts.non_tensor_batch['prompt'][i],
                                        image_bytes[i],
                                        tokenizer,
                                        processor,
                                        truncation=truncation,
                                        max_prompt_length=max_prompt_length)
            processed_list.append(processed)
        processed_dict = collate_fn(processed_list)
        prompt_ids = processed_dict.pop('input_ids')  # (bs, prompt_length)
        prompts.batch['input_ids'] = prompt_ids
        prompts.batch['attention_mask'] = processed_dict.pop('attention_mask')
        prompts.non_tensor_batch['image_data'] = processed_dict['image_data']
        prompts.non_tensor_batch['num_image_tokens'] = processed_dict['num_image_tokens']
        save_dataproto_image_data_dist(prompts, image_manager)
    return prompts
