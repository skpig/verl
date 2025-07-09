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

import ray
import numpy as np
import pandas as pd
import torch
import verl.utils.torch_functional as verl_F
from PIL import Image

from alpha_seed.utils.dataset.rl_dataset import RLHFDataset
from transformers import AutoImageProcessor

from alpha_seed.models.transformers.modeling_vlm import convert_tensor_to_numpy
from alpha_seed.utils.ckpt.hdfs import download_config_and_tokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.utils import TensorType


def load_image_data_dist(non_tensor_batch):
    if not 'pixel_values' in non_tensor_batch:
        return
    image_manager = get_image_manager()
    data_indices = []
    for i, pixel_values in enumerate(non_tensor_batch['pixel_values']):
        if pixel_values is not None:
            data_indices.append(non_tensor_batch['dataset_index'][i])

    image_refs = ray.get(image_manager.get_ref_ids_by_indices.remote(data_indices))
    for i, pixel_values in enumerate(non_tensor_batch['pixel_values']):
        if pixel_values is not None:
            non_tensor_batch['pixel_values'][i] = image_refs[i]


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


@ray.remote
class DistImageLoader:

    def __init__(self, parquet_files, image_key, tokenizer_file, n_partition, node_rank):
        self.image_key = image_key
        self.tokenizer_file = tokenizer_file
        self.bytes_decoder = BytesDecoder()
        self.node_rank = node_rank
        self.n_partition = n_partition
        if tokenizer_file.startswith('hdfs'):
            self.tokenizer_file = download_config_and_tokenizer(tokenizer_file)
        self.processor = AutoImageProcessor.from_pretrained(self.tokenizer_file)
        image_keys = [image_key]
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(parquet_files):
            parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file)
        dataframes = []
        for fn in parquet_files:
            ds = pd.read_parquet(fn)

            for i, row_dict in ds.iterrows():
                if 'session' in row_dict:
                    row_dict = row_dict['session']
                else:
                    row_dict = row_dict.to_dict()
                image_data = {}
                for key in image_keys:
                    image_data[key] = row_dict.get(key)
                dataframes.append(image_data)
        ds = pd.DataFrame.from_dict(dataframes)
        partitions = np.array_split(ds, n_partition)
        self.ds = partitions[node_rank]
        self.partition_counts = [len(d) for d in partitions]
        self.offset = 0
        for i in range(node_rank):
            self.offset += self.partition_counts[i]
        self.image_manager = get_image_manager()

    def get_offset(self):
        return self.offset

    def process_image(self, row, idx):
        if 'session' in row:
            row_dict = row['session']
        else:
            row_dict = row
        if self.image_key in row_dict:
            images = row_dict[self.image_key]
            if isinstance(images, dict):
                assert 'pixel_values_ref' in images
            else:
                pil_images = [self.bytes_decoder(img) for img in images
                             ] if images is not None and len(images) > 0 else None
                if pil_images is not None:
                    inputs = self.processor(images=pil_images)
                    img_token_num = inputs['pixel_values'].shape[0]
                    pixel_values = convert_tensor_to_numpy(inputs['pixel_values'])
                    inputs = {
                        "image_grid_hw": inputs["image_grid_hw"],
                        "num_image_tokens": inputs["num_image_tokens"],
                        "img_token_num": img_token_num
                    }
                    # TODO: put the whole inputs to ray object store
                    ref = ray.put(pixel_values)
                    ray.get(self.image_manager.add_refs.remote({idx: ref}))
                    inputs['pixel_values_ref'] = ref.hex()
                    row_dict[self.image_key] = inputs
                    images = inputs
        return images

    def get_item(self, idx):
        idx = idx - self.offset
        row_dict = self.ds.iloc[idx]
        return self.process_image(row_dict, idx)


@ray.remote
class ImageManager:

    def __init__(self):
        self.image_refs: Dict[str, ray.ObjectRef] = dict()
        self.index_to_image_refs = dict()

    def get_refs(self, ids: List[str]) -> List[ray.ObjectRef]:
        refs = []
        for i in ids:
            ref = None
            if i is not None:
                ref = self.image_refs[i]
            refs.append(ref)
        return refs

    def get_ref_ids_by_indices(self, indices: List[int]) -> List[str]:
        return [self.index_to_image_refs[idx].hex() for idx in indices]

    def add_refs(self, id2refs):
        for idx, ref in id2refs.items():
            self.image_refs[ref.hex()] = ref
            self.index_to_image_refs[idx] = ref


def get_image_manager():
    return ImageManager.options(name="ImageManager", get_if_exists=True).remote()


class RLHFDatasetVL(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self, *args, **kwargs):
        self.processor = kwargs.pop('processor', None)
        self.num_limit = kwargs.pop('num_limit', None)
        self.image_key = kwargs.pop('image_key', 'image')
        self.tokenizer_file = kwargs.pop('tokenizer_file', None)
        self.bytes_decoder = BytesDecoder()
        self.dist_image = kwargs.pop('dist_image', False)
        super().__init__(*args, **kwargs)

    def process(self,
                image_inputs: ImageInput = None,
                conversation: List[Dict] = None,
                padding=False,
                truncation=None,
                max_length: int = None,
                return_tensors=TensorType.PYTORCH,
                return_prompt: bool = False,
                return_num_image_tokens: bool = False) -> BatchFeature:
        if image_inputs is None or len(image_inputs) == 0:
            image_inputs = {}
            num_image_tokens = None
        else:
            num_image_tokens = image_inputs['num_image_tokens']

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

        chunks = prompt.split("<ImageHere>")

        input_ids = []
        attention_mask = []
        img_idx = 0
        for chunk in chunks:
            if chunk == "":
                continue
            text_inputs = self.tokenizer(chunk, padding=padding, truncation=truncation, max_length=max_length)
            input_ids += text_inputs["input_ids"]
            attention_mask += text_inputs["attention_mask"]
            if num_image_tokens is not None and img_idx < len(num_image_tokens):
                if self.processor.image_processor.use_navit:
                    num_img_token = num_image_tokens[img_idx]
                else:
                    num_img_token = self.processor.image_processor.num_img_token
                input_ids += [self.processor.image_token_id] * num_img_token
                attention_mask += [1] * num_img_token
            img_idx += 1

        input_ids = torch.tensor(input_ids).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)
        if return_prompt:
            image_inputs["prompt"] = prompt

        return BatchFeature(data={"input_ids": input_ids, "attention_mask": attention_mask, **image_inputs})

    def _read_files_and_tokenize(self):
        if self.dist_image:
            nodes = [node for node in ray.nodes() if node["Alive"] and node['Resources'].get('GPU', 0) > 0]
            self.image_loaders = []
            image_keys = [self.image_key]
            for i, node in enumerate(nodes):
                image_loader = DistImageLoader.options(
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"],
                        soft=False,
                    )).remote(self.original_parquet_files, self.image_key, self.tokenizer_file, len(nodes), i)
                self.image_loaders.append(image_loader)
            offsets = []
            for img_loader in self.image_loaders:
                offsets.append(img_loader.get_offset.remote())

            self.offsets = ray.get(offsets)

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            if 'session' in dataframe:
                dataframe = pd.DataFrame(list(dataframe['session']))
            if self.dist_image:
                dataframe.drop(columns=image_keys, inplace=True)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # Apply epoch replication if needed
        if hasattr(self, 'data_auto_repeat') and self.data_auto_repeat:
            assert not self.dist_image
            self._replicate_for_epochs()
        if self.num_limit:
            assert not self.dist_image
            self.dataframe = self.dataframe[:self.num_limit]

    def get_remote_loader(self, idx):
        assert self.dist_image
        loader = self.image_loaders[0]
        for i, offset in enumerate(self.offsets):
            if idx >= offset:
                loader = self.image_loaders[i]
            else:
                break

        return loader

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
        image = None
        if self.image_key in row_dict and not self.dist_image:
            image = row_dict[self.image_key]
        image_inputs = None
        if self.dist_image:
            if '__image_inputs__' in row_dict:
                image_inputs = row_dict['__image_inputs__']
            else:
                image_inputs = ray.get(self.get_remote_loader(item).get_item.remote(item))
                row_dict['__image_inputs__'] = image_inputs

        prompt_names = []
        if self.multi_prompts == "none":
            if not self.dist_image:
                pil_images = [self.bytes_decoder(img) for img in image
                             ] if image is not None and len(image) > 0 else None
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
            if self.dist_image:
                inputs = self.process(image_inputs=image_inputs,
                                      conversation=conversation,
                                      return_tensors="pt",
                                      return_prompt=True)
            else:
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
            if self.dist_image:
                assert image_inputs is not None
                if len(image_inputs) > 0:
                    row_dict_ret['pixel_values_ref'] = image_inputs['pixel_values_ref']
                    row_dict_ret['image_grid_hw'] = np.array(image_inputs['image_grid_hw'])
                    row_dict_ret['img_token_num'] = image_inputs['img_token_num']
                else:
                    row_dict_ret['pixel_values_ref'] = None
                    row_dict_ret['image_grid_hw'] = None
                    row_dict_ret['img_token_num'] = 0
            elif image is not None and len(image) > 0:
                row_dict_ret['pixel_values'] = inputs['pixel_values']
                row_dict_ret['image_grid_hw'] = np.array(inputs['image_grid_hw'])
            else:
                row_dict_ret['pixel_values'] = None
                row_dict_ret['image_grid_hw'] = None

            # reward_model is required
            row_dict_ret['reward_model'] = {}
            row_dict_ret['reward_model']['style'] = row_dict['ability']
            row_dict_ret['reward_model']['ground_truth'] = row_dict['verifier_feature']
            prompt_names.append("")
        else:
            assert NotImplementedError

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
        row_dict_ret['dataset_index'] = item
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
