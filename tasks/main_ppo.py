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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.tracking import Tracking
import wandb
import os
import pandas as pd
import hdfs_io

# rule-based reward score
from alpha_seed.utils.reward_score import gsm8k, math, math_v2, model_score_fn, logic_puzzle, oj_utils
from concurrent.futures import ThreadPoolExecutor, as_completed


def _select_rm_score_fn(reward_style):
    if reward_style == "model-raw_score":
        return model_score_fn.raw_score
    elif reward_style == "model-raw_score_reflection_penalty":
        return model_score_fn.raw_score_reflection_penalty
    elif reward_style == "code-sandbox":
        return oj_utils.compute_score
    elif reward_style == 'rule-openai/gsm8k':
        return gsm8k.compute_score
    elif reward_style == 'rule-lighteval/MATH':
        return math.compute_score
    elif reward_style == 'rule-lighteval/MATH_v2':
        return math_v2.compute_score
    else:
        if reward_style.startswith("rule-logic_puzzle"):
            return logic_puzzle.compute_score
        raise NotImplementedError


class RewardManager():

    def __init__(self, tokenizer, config, logger: Tracking, rm_name="train") -> None:
        self.tokenizer = tokenizer
        self.logger = logger
        self.log_table = []
        self.rm_name = rm_name
        self.config = config
        if self.config.trainer.save_cases_to_hdfs:
            self.case_study_dir = config.trainer.default_local_dir + "/cases/"
            os.makedirs(self.case_study_dir, exist_ok=True)
        self.rm_req_executor = ThreadPoolExecutor(max_workers=128)

    def __call__(self, data: DataProto, global_step=None):
        """We will expand this function gradually based on the available datasets"""
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        save_to_hdfs = []
        rm_res_future_list = []
        if global_step is not None and global_step % self.config.trainer.logger_step_interval == 0:
            self.log_table = []  # 清空self.log_table

        def get_rm_score(idx):
            data_item = data[idx]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            solution_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # select rm_score
            reward_style = data_item.non_tensor_batch['reward_model']['style']
            compute_score_fn = _select_rm_score_fn(reward_style)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            score_fn_inputs = {
                "batch_info": data_item.batch,
                "tokenizer": self.tokenizer,
                "solution_str": solution_str,
                "ground_truth": ground_truth,
                "config": self.config
            }
            score = compute_score_fn(**score_fn_inputs)
            return prompt_str, solution_str, ground_truth, reward_style, valid_response_length, score, idx

        for i in range(len(data)):
            rm_res_future_list.append(self.rm_req_executor.submit(get_rm_score, i))
        for res in as_completed(rm_res_future_list):
            prompt_str, solution_str, ground_truth, reward_style, valid_response_length, score, idx = res.result()

            reward_tensor[idx, valid_response_length - 1] = score

            if reward_style not in already_print_data_sources:
                already_print_data_sources[reward_style] = 0

            if already_print_data_sources[reward_style] < self.config.trainer.num_cases_to_wandb:
                already_print_data_sources[reward_style] += 1
                if reward_style == "code-sandbox":
                    ground_truth = {}  # 对于OJ问题，ground_truth会比较大，扛不住
                self.log_table.append([global_step, prompt_str, solution_str, ground_truth, score])
            save_to_hdfs.append([global_step, prompt_str, solution_str, ground_truth, score])

        if self.config.trainer.num_cases_to_wandb > 0:
            logger_step = global_step - global_step % self.config.trainer.logger_step_interval
            self.logger.log(
                {
                    f"gen&score_{self.rm_name}_{logger_step}":
                        wandb.Table(columns=["Step", "Prompt", "Gen Sequence", "GroundTruth", "Score"],
                                    data=self.log_table)
                },
                step=global_step,
                backend='tracking')
        if self.config.trainer.save_cases_to_hdfs:
            df = pd.DataFrame(columns=["Step", "Prompt", "Gen Sequence", "GroundTruth", "Score"], data=save_to_hdfs)
            df.to_parquet(f"{self.rm_name}.{str(global_step)}.parquet")
            hdfs_io.hput(f"{self.rm_name}.{str(global_step)}.parquet", self.case_study_dir)
            os.remove(f"{self.rm_name}.{str(global_step)}.parquet")
        return reward_tensor


import ray
import hydra

from alpha_seed.trainer.ppo import RayPPOTrainer


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'BPEX_NO_WARN_ON_UNTUNED_CASE': '1'
            }
        })

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    logger = Tracking(project_name=config.trainer.project_name,
                      experiment_name=config.trainer.experiment_name,
                      default_backend=config.trainer.logger,
                      config=OmegaConf.to_container(config, resolve=True))

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    if config.data.get('chat_template', None) == 'seed':
        from verl.utils.seed import CHAT_TEMPLATE
        tokenizer.chat_template = CHAT_TEMPLATE

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from alpha_seed.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    from alpha_seed.trainer.ppo import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ActorRolloutRefWorker,
        Role.Critic: CriticWorker,
        Role.RefPolicy: ActorRolloutRefWorker
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        from alpha_seed.workers.fsdp_workers import RewardModelWorker
        role_worker_mapping[Role.RewardModel] = RewardModelWorker
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, config=config, logger=logger, rm_name="train")

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, config=config, logger=logger, rm_name="val")

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            logger=logger)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
