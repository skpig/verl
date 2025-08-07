from typing import List
import torch, os
from omegaconf import OmegaConf, open_dict, ListConfig
from alpha_seed.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.fs import copy_local_path_from_hdfs
from omnistore.utilities.io.bfile import is_local_path
from alpha_seed.utils.functional import log_cpu_memory_usage


class DataLoaderMgr:

    def __init__(self, config, tokenizer, is_vlm=False, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.is_vlm = is_vlm
        self.processor = processor
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.total_training_steps = 0

        if self.is_vlm:
            from alpha_seed.utils.dataset.vlm_rl_dataset import collate_fn
            if config.data.get('task_type') == 'VLM_GUI':
                from alpha_seed.utils.dataset.vlm_rl_dataset import RLHFDatasetGUI as RLHFDataset
            else:
                if self.config.data.get("enable_swalm_agent", False):
                    from alpha_seed.utils.dataset.rl_dataset_swalm import RLHFDatasetVLSwalm as RLHFDataset
                else:
                    from alpha_seed.utils.dataset.vlm_rl_dataset import RLHFDatasetVL as RLHFDataset
        else:
            from alpha_seed.utils.dataset.rl_dataset import collate_fn
            if self.config.data.get("enable_swalm_agent", False):
                from alpha_seed.utils.dataset.rl_dataset_swalm import RLHFDatasetSwalm as RLHFDataset
            else:
                from alpha_seed.utils.dataset.rl_dataset import RLHFDataset

        self.RLHFDataset = RLHFDataset
        self.collate_fn = collate_fn

        self._create_datasets()
        self._create_dataloaders()

        self._compute_total_training_steps()

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
            self.config.critic.optim.total_training_steps = self.total_training_steps

    def _create_datasets(self):

        kwargs = {
            "processor": self.processor,
            'image_key': self.config.data.image_key,
            'tokenizer_file': self.config.actor_rollout_ref.model.path,
            'dist_image': self.config.data.dist_image,
            'stable_pool_names': self.config.elastic.resource_pools.stable_pool_names,
        } if self.is_vlm else {}
        data_auto_repeat = self.config.data.get('data_auto_repeat', False)

        log_cpu_memory_usage('before create train_dataset')

        self.train_dataset = self.RLHFDataset(parquet_files=self.config.data.train_files,
                                              tokenizer=self.tokenizer,
                                              prompt_key=self.config.data.prompt_key,
                                              answer_key=self.config.data.answer_key,
                                              use_ref_answer=self.config.data.use_ref_answer,
                                              max_prompt_length=self.config.data.max_prompt_length,
                                              filter_prompts=True,
                                              return_raw_chat=self.config.data.get('return_raw_chat', False),
                                              truncation=self.config.data.get('truncation', 'error'),
                                              multi_prompts=self.config.data.get("multi_prompts", "none"),
                                              num_prompts_per_data=self.config.data.get("num_prompts_per_data", 1),
                                              total_epochs=self.config.trainer.total_epochs,
                                              shuffle_per_epoch=self.config.data.shuffle,
                                              data_auto_repeat=data_auto_repeat,
                                              use_grm=self.config.trainer.use_grm,
                                              max_response_length=self.config.data.max_response_length,
                                              **kwargs)

        log_cpu_memory_usage('after create train_dataset')

        self.val_dataset = self.RLHFDataset(parquet_files=self.config.data.val_files,
                                            tokenizer=self.tokenizer,
                                            prompt_key=self.config.data.prompt_key,
                                            answer_key=self.config.data.answer_key,
                                            use_ref_answer=self.config.data.use_ref_answer,
                                            max_prompt_length=self.config.data.max_prompt_length,
                                            filter_prompts=True,
                                            return_raw_chat=True,
                                            truncation=self.config.data.get('truncation', 'error'),
                                            multi_prompts=self.config.data.get("multi_prompts", "none"),
                                            num_prompts_per_data=1,
                                            is_eval=True,
                                            use_grm=self.config.trainer.use_grm,
                                            max_response_length=self.config.data.max_response_length,
                                            **kwargs)
        log_cpu_memory_usage('after create val_dataset')

    def _create_dataloaders(self):
        from torch.utils.data import DataLoader

        train_batch_size = self.config.data.train_batch_size if (
            not self.config.algorithm.priority_sample
        ) else self.config.data.train_batch_size * self.config.algorithm.get('priority_buffer_size', 4)
        assert not (self.config.trainer.league_training_config.enable and
                    self.config.trainer.queued_rollout_config.enable)
        if self.config.trainer.league_training_config.enable:
            train_batch_size *= self.config.trainer.league_training_config.buffer_size
        if self.config.trainer.queued_rollout_config.enable:
            assert not self.config.algorithm.priority_sample, "queued rollout incompatible with priority sample"
            train_batch_size = self.config.trainer.queued_rollout_config.chunk_size

        sampler = self._get_train_sampler()
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=train_batch_size,
                                           sampler=sampler,
                                           drop_last=True,
                                           collate_fn=self.collate_fn,
                                           num_workers=self.config.data.num_workers)

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=self.config.data.shuffle,
                                         drop_last=True,
                                         collate_fn=self.collate_fn,
                                         num_workers=self.config.data.num_workers)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

    def _get_train_sampler(self):
        data_auto_repeat = self.config.data.get('data_auto_repeat', False)
        if self.config.data.BITWISE_RESUME:
            from alpha_seed.utils.dataset.sampler import RandomSampler, SequentialSampler
        else:
            from torch.utils.data import RandomSampler, SequentialSampler

        if self.config.data.shuffle and not data_auto_repeat:
            generator = torch.Generator()
            generator.manual_seed(self.config.data.get('seed', 1))
            return RandomSampler(self.train_dataset, generator=generator)
        else:
            return SequentialSampler(self.train_dataset)

    def _compute_total_training_steps(self):
        self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_steps is not None:
            self.total_training_steps = self.config.trainer.total_steps

    def _load_dataloader(self, remote_global_step_folder, donot_resume_data):
        # load dataloader
        dataloader_remote_path = os.path.join(remote_global_step_folder, 'data.pt')
        dataloader_local_path = copy_local_path_from_hdfs(dataloader_remote_path)
        train_dataloader = torch.load(dataloader_local_path)

        if donot_resume_data:
            resume_dataset_name = train_dataloader.dataset.parquet_files  # cached name like ['/home/tiger/.cache/verl/rlhf/b0e4ef4425c3409d3c7a19350c3a3e43/train_with_ref_ans.parquet']
            if not isinstance(
                    self.config.data.train_files, (List, ListConfig)
            ):  # user provided name like ['hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet']
                train_files = [self.config.data.train_files]

            resume_dataset_set = set([s.split('/')[-1] for s in resume_dataset_name])
            train_files_set = set([s.split('/')[-1] for s in train_files])
            if resume_dataset_set != train_files_set:
                print('the resume dataset is different from the train dataset, will not resume data')
                return self.train_dataloader

        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            train_dataloader.dataset.resume_dataset_state()

        if not is_local_path(dataloader_remote_path):
            print(f'dataloader_remote_path: {dataloader_remote_path} is not a local or fuse dir, '
                  f'try to remove dataloader_local_path={dataloader_local_path}')
            try:
                os.remove(dataloader_local_path)
            except Exception as e:
                print(f'remove local dataloader ckpt file after loading failed, exception {e} will be ignored')
        return train_dataloader
