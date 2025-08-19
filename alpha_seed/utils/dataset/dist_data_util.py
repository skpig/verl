import os
import pickle
from hdfs_io import hcopy
import ray
from typing import Dict, List
from collections import defaultdict

import ray
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor

from alpha_seed.utils.ckpt.hdfs import download_config_and_tokenizer
from .utils import convert_tensor_to_numpy
from verl.utils.fs import copy_local_path_from_hdfs
from ..server_client import is_local_ray_instance

import pyarrow as pa

import logging

logger = logging.getLogger(__name__)


def write_objects_to_bin(objects_list, filename="data.bin"):
    """
    将一个对象列表写入二进制文件，并返回每个对象的偏移量和大小。

    Args:
        objects_list (list): 要写入文件的对象列表。
        filename (str): 要写入的二进制文件名。

    Returns:
        list: 包含每个对象 (offset, size) 元组的列表。
    """
    offsets_info = []
    with open(filename, 'wb') as f:
        for obj in objects_list:
            # 记录当前写入前的偏移量
            current_offset = f.tell()

            # 序列化对象
            serialized_obj = pickle.dumps(obj)

            # 获取序列化后对象的大小
            obj_size = len(serialized_obj)

            # 写入二进制数据
            f.write(serialized_obj)

            # 存储偏移量和大小信息
            offsets_info.append((current_offset, obj_size))
            # print(f"写入对象: {obj}, 偏移量: {current_offset}, 大小: {obj_size} 字节")
    return offsets_info


def load_objects_from_bin(filename="data.bin", offsets_to_load=None):
    """
    从二进制文件中根据提供的偏移量和大小加载对象。

    Args:
        filename (str): 要读取的二进制文件名。
        offsets_to_load (list): 一个列表，包含要加载对象的 (offset, size) 元组。
                                如果为None，则尝试从文件开头读取所有对象直到文件末尾
                                (但这需要知道每个对象的大小，因此强烈建议传入 offsets_to_load)。

    Returns:
        list: 加载的对象的列表。
    """
    loaded_objects = []
    if not os.path.exists(filename):
        print(f"错误: 文件 '{filename}' 不存在。")
        return []

    with open(filename, 'rb') as f:
        if offsets_to_load:
            for offset, size in offsets_to_load:
                try:
                    f.seek(offset)  # 移动文件指针到指定偏移量
                    serialized_data = f.read(size)  # 读取指定大小的字节
                    obj = pickle.loads(serialized_data)  # 反序列化
                    loaded_objects.append(obj)
                    # print(f"成功加载对象: {obj} (偏移量: {offset}, 大小: {size})")
                except Exception as e:
                    print(f"加载对象失败 (偏移量: {offset}, 大小: {size}): {e}")
        else:
            print("警告: 未提供 offsets_to_load。此方法不适用于加载所有对象，因为不知道每个对象的大小。")
            print("请提供准确的 (offset, size) 元组列表来加载特定对象。")
            # 如果真的想读所有，需要更复杂的逻辑，比如写入对象前先写入其长度
            # 但那会增加文件复杂度，不如直接存储offsets_info
    return loaded_objects


def release_object(image_manager, non_tensor_batch=None, keys=None):
    refs_to_release = None
    if non_tensor_batch is not None:
        refs_to_release = []
        for key in keys:
            if key in non_tensor_batch:
                refs_to_release.extend(list(non_tensor_batch[key]))
            non_tensor_batch[key] = None
    ray.get(image_manager.release_refs.remote(refs_to_release))


def _get_image_refs(data_list):
    refs = []
    for key in ['image_data_ref', 'images_bytes_ref']:
        for data in data_list:
            if key in data.non_tensor_batch:
                refs.extend(list(data.non_tensor_batch[key]))
    return refs


def add_ref_counts(image_manager, data_list):
    refs = _get_image_refs(data_list)
    if refs:
        ray.get(image_manager.add_refs_counts.remote(refs))


def release_ref_counts(image_manager, data_list):
    refs = _get_image_refs(data_list)
    if refs:
        ray.get(image_manager.release_refs_counts.remote(refs))


def split_dict_avg(d, n):
    """将字典按键的顺序平均分割为n个子字典"""
    keys = list(d.keys())
    k, m = divmod(len(keys), n)  # k为商，m为余数
    result = []

    start = 0
    for i in range(n):
        # 计算当前子字典的键数量
        current_size = k + (1 if i < m else 0)
        end = start + current_size

        # 提取对应的键值对
        sub_keys = keys[start:end]
        sub_dict = {key: d[key] for key in sub_keys}
        result.append(sub_dict)

        start = end

    return result


@ray.remote
class DistImageLoader:

    def __init__(self, parquet_file, image_key, tokenizer_file, n_partition, node_rank):
        self.image_key = image_key
        self.tokenizer_file = tokenizer_file
        self.node_rank = node_rank
        self.n_partition = n_partition
        pa.set_memory_pool(pa.system_memory_pool())
        if tokenizer_file.startswith('hdfs'):
            self.tokenizer_file = download_config_and_tokenizer(tokenizer_file)
        self.processor = AutoImageProcessor.from_pretrained(self.tokenizer_file)
        image_keys = [image_key]
        if isinstance(parquet_file, str):
            parquet_file = [parquet_file]
        dataframes = []
        for fn in parquet_file:
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
        pool = pa.default_memory_pool()
        pool.release_unused()
        self.image_manager = get_image_manager()

    def load_data_from_file(self, bin_file, fn_dict):
        refs = {}
        binfile2offset = defaultdict(list)
        for idx, data in fn_dict.items():
            if data is None:
                continue
            if bin_file.startswith('hdfs:'):
                bin_file = copy_local_path_from_hdfs(bin_file)
            offset = data[0]
            size = data[1]
            binfile2offset[bin_file].append((idx, offset, size))
        for bin_file, offsets in binfile2offset.items():
            offsets.sort(key=lambda x: x[0])
            indices = [x[0] for x in offsets]
            offsets = [x[1:] for x in offsets]
            obj_list = load_objects_from_bin(bin_file, offsets)
            for i, idx in enumerate(indices):
                ref = ray.put(obj_list[i])
                refs[idx] = ref
        return refs

    def get_offset(self):
        return self.offset

    def process_image(self, row):
        if self.image_key in row:
            images = row[self.image_key]
            if images is not None and len(images) > 0:
                image_ref = ray.put(images)
                return image_ref

    def process_all_images(self):
        image_refs = []
        indices = []
        image_ref_str = []
        for index, row in self.ds.iterrows():
            ref = self.process_image(row)
            indices.append(self.offset + index)
            if ref is not None:
                image_ref_str.append(ref.hex())
                image_refs.append(ref)
            else:
                image_ref_str.append(None)
                image_refs.append(None)
        ray.get(self.image_manager.add_refs.remote(image_refs, indices, 'images_bytes', persistent=True))
        return image_ref_str, indices

    def get_item(self, idx):
        idx = idx - self.offset
        row_dict = self.ds.iloc[idx]
        return self.process_image(row_dict, idx)


@ray.remote
class ImageManager:

    def __init__(self):
        self.uid2refs: Dict[str, ray.ObjectRef] = dict()
        self.index2refs: Dict[int, Dict[str, ray.ObjectRef]] = dict()
        self.persistent_refs = set()
        self.ref_counts = defaultdict(int)
        self.history_refs = set()
        self.image_loaders = None

    def get_refs(self, uids: List[str], indices=None, key=None) -> List[ray.ObjectRef]:
        refs = []
        if indices is not None:
            assert len(uids) == len(indices)
            assert key is not None
        for idx, uid in enumerate(uids):
            if isinstance(uid, list):
                ref = self.get_refs(uid, indices, key)
            else:
                index = indices[idx] if indices is not None else None
                ref = None
                if uid is not None:
                    if uid in self.uid2refs:
                        ref = self.uid2refs[uid]
                    elif index is not None and index in self.index2refs and key in self.index2refs[index]:
                        ref = self.index2refs[index][key]
                    else:
                        in_history = uid in self.history_refs
                        raise RuntimeError(f"{uid} not found in current pool, in_history: {in_history}")
            refs.append(ref)
        return refs

    def add_refs(self, refs: List[ray.ObjectRef], indices=None, key=None, persistent=False):
        if indices is not None:
            assert len(refs) == len(indices)
            assert key is not None
            assert persistent
        for i, ref in enumerate(refs):
            if ref is None:
                continue
            self.uid2refs[ref.hex()] = ref
            if persistent:
                self.persistent_refs.add(ref.hex())
            self.history_refs.add(ref.hex())
            if indices is not None:
                if indices[i] not in self.index2refs:
                    self.index2refs[indices[i]] = {}
                self.index2refs[indices[i]][key] = ref

    def add_refs_counts(self, refs):
        for r in refs:
            if r is None:
                continue
            if isinstance(r, list):
                self.add_refs_counts(r)
            else:
                assert isinstance(r, str), type(r)
                self.ref_counts[r] += 1

    def release_refs_counts(self, refs):
        for r in refs:
            if r is None:
                continue
            if isinstance(r, list):
                self.release_refs_counts(r)
            else:
                assert isinstance(r, str), type(r)
                if r in self.ref_counts:
                    self.ref_counts[r] -= 1

    def release_refs(self, refs=None):
        if refs is not None and len(refs) > 0 and isinstance(refs[0], list):
            for ref in refs:
                self.release_refs(ref)
            return
        for ref, count in self.ref_counts.items():
            if count <= 0 and ref not in self.persistent_refs and ref in self.uid2refs:
                res = self.uid2refs.pop(ref)
                logger.info(f"release ref {ref}")
                del res
        if refs is not None:
            for ref in refs:
                if ref not in self.persistent_refs and ref in self.uid2refs:
                    res = self.uid2refs.pop(ref)
                    logger.info(f"release ref {ref}")
                    del res

    def set_image_loaders(self, image_loaders):
        self.image_loaders = image_loaders

    def resume_refs(self, resume_list):
        data = {}
        refs = [None] * len(resume_list)
        for idx, resume in enumerate(resume_list):
            if resume is None:
                continue
            index, bin_file, offset, size = resume
            if bin_file not in data:
                data[bin_file] = {}
            data[bin_file][idx] = (offset, size)
        for bin_file, values in data.items():
            parts = split_dict_avg(values, len(self.image_loaders))
            load_refs = []
            for i, image_loader in enumerate(self.image_loaders):
                load_ref = image_loader.load_data_from_file.remote(bin_file, parts[i])
                load_refs.append(load_ref)
            load_refs = ray.get(load_refs)
            merged_refs = {}
            for load_ref in load_refs:
                merged_refs.update(load_ref)
            load_refs = merged_refs
            for idx, load_ref in load_refs.items():
                refs[idx] = load_ref
            self.add_refs(refs)
        return refs


def init_or_get_image_manager(stable_pool_name: str):
    resources = {}
    if stable_pool_name and not is_local_ray_instance():
        resources = {stable_pool_name: 1}
    return ImageManager.options(name="ImageManager", get_if_exists=True, resources=resources).remote()


def get_image_manager():
    return ray.get_actor("ImageManager")


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


def save_image_data_dist(data):
    if data is None or len(data) == 0:
        return
    for key, value in data.items():
        # check if the data is changed or new
        if isinstance(value, torch.Tensor):
            data[key] = convert_tensor_to_numpy(value)
    data_ref = ray.put(data)
    return data_ref


def save_query_image_data_dist(query, image_manager):
    if query.image_data is not None:
        image_data_ref = save_image_data_dist(query.image_data)
        ray.get(image_manager.add_refs.remote([image_data_ref]))
        query.image_data_ref = image_data_ref.hex()


def save_dataproto_image_data_dist(output, image_manager):
    all_refs = []
    # set image to ref here
    import numpy as np
    if 'image_data' not in output.non_tensor_batch:
        return
    if 'image_data_ref' not in output.non_tensor_batch:
        output.non_tensor_batch['image_data_ref'] = np.array([None] * output.batch.batch_size[0], dtype=object)

    for i, image_data in enumerate(output.non_tensor_batch['image_data']):
        if image_data is None:
            continue
        data_ref = save_image_data_dist(image_data)
        all_refs.append(data_ref)
        output.non_tensor_batch['image_data_ref'][i] = data_ref.hex()
    output.non_tensor_batch.pop('image_data')
    if len(all_refs) > 0:
        ray.get(image_manager.add_refs.remote(all_refs))


def get_local_inputs(non_tensor_batch, key, image_manager):
    ref_list = non_tensor_batch[key]
    image_refs = ray.get(image_manager.get_refs.remote(ref_list))
    result_flag = [i != None for i in image_refs]
    image_refs = [i for i in image_refs if i is not None]
    images = ray.get(image_refs)
    result = []
    i = 0
    for flag in result_flag:
        if flag:
            result.append(images[i])
            i += 1
        else:
            result.append(None)
    return result


def load_and_resume_image_data(image_manager, non_tensor_batch, path, prefix):
    image_meta_info = f"{path}/{prefix}.image_meta_info.pt"
    image_meta_info = copy_local_path_from_hdfs(image_meta_info)
    image_meta_info = torch.load(image_meta_info)
    if image_meta_info:
        for key, values in image_meta_info.items():
            ref = image_manager.resume_refs.remote(values)
            refs = ray.get(ref)
            assert len(refs) == len(non_tensor_batch[key])
            for i, ref in enumerate(refs):
                if ref is None:
                    assert non_tensor_batch[key][i] is None
                else:
                    non_tensor_batch[key][i] = ref.hex()


def load_and_save_image_data(data, image_manager, path, prefix):
    image_meta_info = defaultdict(list)
    for key in ['images_bytes_ref', 'pixel_values_ref']:
        if key in data.non_tensor_batch:
            refs_to_get = [i for i in data.non_tensor_batch[key] if i is not None]
            if len(refs_to_get) == 0:
                continue
            refs_to_get = ray.get(image_manager.get_refs.remote(refs_to_get))
            real_data = ray.get(refs_to_get)
            bin_file = f"{prefix}.{key}.bin"
            if path.startswith('/'):
                bin_file = os.path.join(path, bin_file)

            offsets = write_objects_to_bin(real_data, bin_file)

            if path.startswith('hdfs:'):
                hcopy(bin_file, path)
                bin_file = os.path.join(path, bin_file)
            i = 0
            for idx, ref in enumerate(data.non_tensor_batch[key]):
                if ref is None:
                    image_meta_info[key].append(None)
                    continue
                image_meta_info[key].append([idx, bin_file, offsets[i][0], offsets[i][1]])
                i += 1
    save_fn = f"{prefix}.image_meta_info.pt"
    if path.startswith('/'):
        save_fn = os.path.join(path, save_fn)
    torch.save(image_meta_info, save_fn)
    if path.startswith('hdfs:'):
        hcopy(f"{prefix}.image_meta_info.pt", path)
