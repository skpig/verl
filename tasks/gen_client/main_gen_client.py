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
import time
import yaml
import warnings
import contextlib
from typing import Union, List
from collections import defaultdict
import tempfile
import random

from verl import DataProto
import torch
from verl.utils.tracking import Tracking
import wandb
import os
import pandas as pd
import hdfs_io
from datetime import datetime
from multiprocessing import Process

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from alpha_seed.trainer.ppo import RayPPOTrainer
from alpha_seed.utils.duplicate import para_dup
from alpha_seed.utils.dataset.rl_dataset import collate_fn
from alpha_seed.workers.actors.checkpoint import CkptGlobalUploader
from tasks.main_ppo import validate_config, RewardManager
from omegaconf import OmegaConf, ListConfig
import ray
import hydra
from verl.utils.fs import copy_local_path_from_hdfs
from math_verifier import compute_score
from hdfs_io.hdfs_io import hcopy, hmkdir


class SimpleDataset(Dataset):

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 cache_dir="~/.cache/alphaseed/gen_cli",
                 truncation='error',
                 preprocess_mode='RAW'):

        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.preprocess_mode = preprocess_mode

        self._download()
        self._read_files()

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)
            print(i, self.parquet_files[i])

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        import verl.utils.torch_functional as verl_F
        row_dict = self.dataframe.iloc[item].to_dict()
        chat = row_dict[self.prompt_key]
        # Apply chat template here, align with seed/cook
        if self.preprocess_mode == 'RAW':
            pass
        elif self.preprocess_mode == 'CHATML_SESSION':
            chat = f"{self.tokenizer.bos_token}user\n{chat}{self.tokenizer.eos_token}{self.tokenizer.bos_token}assistant\n"
        else:
            raise ValueError(f"unsupported preprocess_mode: {self.preprocess_mode}")

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=chat,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
        row_dict['input_ids'] = input_ids[0].to(torch.int32)
        row_dict['attention_mask'] = attention_mask[0].to(torch.int8)
        row_dict['off_policy_steps'] = torch.zeros([1]).to(torch.int8)
        return row_dict


def compute_score_by_rule(data):
    output_score = {}
    for k, conts in data.items():
        scores = []
        for i in range(len(conts)):
            if isinstance(conts[i]['output'], str):
                outputs = [conts[i]['output']]
            else:
                outputs = conts[i]['output']
            for pred in outputs:
                answer = conts[i]['answer']
                score = compute_score(pred, str(answer))
                scores.append(score)
        output_score[k] = scores
    return output_score


def sample_and_compute_score(df: pd.DataFrame, bon_list: List[int], sample_num: int):
    df = df.sample(frac=1)
    print(len(df))
    data = defaultdict(list)

    for _, line in df.iterrows():
        id = line['id']
        data[id].append(line)
    print(len(data))

    rule_scores = compute_score_by_rule(data)

    random.seed(2024)
    bok_list = []
    for k in bon_list:
        assert sample_num >= k, f"{sample_num} < {k}"
        bok = 0
        for i in range(100):
            for idx, scores in rule_scores.items():
                select_score = random.sample(scores[:sample_num], k)
                if sum(select_score) >= 1:
                    bok += 1
        bok_list.append(bok / 100 / 30)

    print(bok_list)


def slice_data_proto(batch: DataProto, slice_num: int):
    sliced = batch[:slice_num]
    return DataProto(batch=sliced.batch, non_tensor_batch=sliced.non_tensor_batch, meta_info=sliced.meta_info)


class GenClient:

    def __init__(self, config, kv_store_name="kv_store"):
        self.config = config
        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(self.config.data.tokenizer)
        # instantiate tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)

        is_ready = False
        for i in range(200):
            try:
                server_health_check = ray.get_actor("server_health_check")
                is_ready = ray.get(server_health_check.is_ready.remote())
                if is_ready:
                    break
            except:
                print(f"waiting for server to be ready (iter #{i})...")
                time.sleep(5)

        if not is_ready:
            raise RuntimeError("wait for server ready timeout")

        self.kv_store = ray.get_actor(kv_store_name)
        server_config = ray.get(self.kv_store.get_by_key.remote('config'))
        validate_config(server_config)
        server_tokenizer = ray.get(self.kv_store.get_by_key.remote('tokenizer'))

        resource_pool_manager = ray.get(self.kv_store.get_by_key.remote('resource_pool_manager'))
        ray_worker_group_cls = ray.get(self.kv_store.get_by_key.remote('ray_worker_group_cls'))

        role_worker_mapping = ray.get(self.kv_store.get_by_key.remote('role_worker_mapping'))
        available_roles = list(role_worker_mapping.keys())

        print(f"The server provides roles: {', '.join([str(role) for role in available_roles])}")

        self.trainer = RayPPOTrainer(config=server_config,
                                     tokenizer=server_tokenizer,
                                     role_worker_mapping=role_worker_mapping,
                                     resource_pool_manager=resource_pool_manager,
                                     ray_worker_group_cls=ray_worker_group_cls,
                                     reward_fn=None,
                                     val_reward_fn=None,
                                     logger=None)

    def init_workers(self):
        self.trainer.init_workers(ckpt_global_uploader=None)
        self.actor_rollout_wg = self.trainer.actor_rollout_wg

    def gen(self, input_files, output_file, preprocess_mode):
        dataset = SimpleDataset(parquet_files=input_files,
                                tokenizer=self.tokenizer,
                                prompt_key=self.config.data.prompt_key,
                                max_prompt_length=self.config.data.max_prompt_length,
                                truncation=self.config.data.truncation,
                                preprocess_mode=preprocess_mode)

        from torch.utils.data import SequentialSampler

        sampler = SequentialSampler(data_source=dataset)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.config.gen.batch_size,
                                shuffle=None,
                                drop_last=False,
                                collate_fn=collate_fn,
                                sampler=sampler)

        total_iters = len(dataloader)
        data = []
        gen_bs = self.config.gen.batch_size
        for iter, batch_dict in enumerate(dataloader):
            print(f"Running iter #{iter}/{total_iters}...")
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            origin_bs = len(batch)

            if origin_bs < gen_bs:
                # padding to batch_size
                repeat_num = (gen_bs + origin_bs - 1) // origin_bs
                batch = batch.repeat(repeat_num, interleave=False)
                batch = slice_data_proto(batch, gen_bs)

            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'off_policy_steps'])
            gen_batch.meta_info.update({'generation_kwargs': self.config.gen.generate_kwargs, 'complete_ratio': 1.0})
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            if len(gen_batch_output) > origin_bs:
                # remove padding
                batch = slice_data_proto(batch, origin_bs)
                gen_batch_output = slice_data_proto(gen_batch_output, origin_bs)

            input_ids = gen_batch_output.batch['input_ids']
            prompt_ids = input_ids[:, :self.config.data.max_prompt_length]
            response_ids = input_ids[:, self.config.data.max_prompt_length:]

            first_non_one_indices = (prompt_ids != self.tokenizer.pad_token_id).int().argmax(dim=1)
            rmv_padding_prompt_ids = [row[index:].tolist() for row, index in zip(prompt_ids, first_non_one_indices)]

            for i in range(len(batch)):
                item = {
                    self.config.data.prompt_key: batch.non_tensor_batch[self.config.data.prompt_key][i],
                    'id': batch.non_tensor_batch['id'][i],
                    'index': batch.non_tensor_batch['index'][i],
                    'eval_prompt': batch.non_tensor_batch['eval_prompt'][i],
                    'answer': batch.non_tensor_batch['answer'][i],
                    'prompt': self.tokenizer.decode(rmv_padding_prompt_ids[i]),
                    'output': self.tokenizer.decode(response_ids[i, :], skip_special_tokens=True),
                }
                data.append(item)
        df = pd.DataFrame(data)
        with tempfile.NamedTemporaryFile(mode='w', suffix=".parquet") as f:
            df.to_parquet(f.name)
            hmkdir(os.path.dirname(output_file))
            hcopy(f.name, output_file)
        return df

    def gen_and_eval(self):
        df = self.gen(self.config.data.input_files, self.config.data.output_file, preprocess_mode="CHATML_SESSION")
        sample_and_compute_score(df, self.config.eval.bon_list, self.config.eval.sample_num)


@hydra.main(config_path='config', config_name='gen_client', version_base=None)
def main(config):
    if config.gen.skip:
        local_path = copy_local_path_from_hdfs(config.data.output_file)
        df = pd.read_parquet(local_path)
        sample_and_compute_score(df, config.eval.bon_list, config.eval.sample_num)
        return

    alpha_seed_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    with open(os.path.join(alpha_seed_root, "tasks/runtime_env/runtime_env.yaml")) as fin:
        runtime_env = yaml.safe_load(fin)
    for _ in range(600):
        try:
            ray.init(namespace="alphaseed", address=config.ray.server_addr, runtime_env=runtime_env)
            break
        except:
            print("waiting for ray server init...")
            time.sleep(1)
    gen_cli = GenClient(config)
    gen_cli.init_workers()
    gen_cli.gen_and_eval()


if __name__ == '__main__':
    main()
