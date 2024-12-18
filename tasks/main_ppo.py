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

import time
import warnings
import contextlib

import ray
from verl import DataProto
import torch
from verl.utils.tracking import Tracking
import wandb
import os
import pandas as pd
import hdfs_io
from datetime import datetime
from multiprocessing import Process

# rule-based reward score
from alpha_seed.utils.reward_score import gsm8k, math, math_v2, model_score_fn, logic_puzzle, oj_utils, math_verifier, response_post_proc, gpqa_verifier
from alpha_seed.utils.duplicate import para_dup
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from alpha_seed.workers.actors.critic_worker import CriticWorker
from alpha_seed.utils.alarm.lark_util import send_message_to_employee
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from bytedance.trainingmetrics.rl_metrics_client_context_manager import RLMetricsClientContextManager
except ImportError:
    RLMetricsClientContextManager = None

user_email = os.getenv('ARNOLD_LARK_RECEIVER', '')
task_url = os.getenv('ARNOLD_ORIGIN_PLATFORM_URL', '')
ARNOLD_REGION = os.getenv("ARNOLD_REGION", "CN")
ENABLE_REDIS_TRITON_CACHE = int(os.getenv("ENABLE_REDIS_TRITON_CACHE", '1'))


def _select_rm_score_fn(reward_style):
    if reward_style == "model-raw_score":
        return model_score_fn.raw_score
    elif reward_style == "model-raw_score_reflection_penalty":
        return model_score_fn.raw_score_reflection_penalty
    elif reward_style == "code-sandbox":
        return oj_utils.compute_score_client
    elif reward_style == 'rule-openai/gsm8k':
        return gsm8k.compute_score
    elif reward_style == 'rule-lighteval/MATH':
        return math.compute_score
    elif reward_style == 'rule-lighteval/MATH_v2':
        return math_v2.compute_score
    elif reward_style == "rule-math_verifier":
        return math_verifier.compute_score
    elif reward_style == "rule-boxed_gpqa":
        return gpqa_verifier.compute_score
    else:
        if reward_style.startswith("rule-logic_puzzle"):
            return logic_puzzle.compute_score
        raise NotImplementedError


def post_process_solution_str(config, solution_str):
    if config.reward_model.use_last_response == 'summarize':
        solution_str_post_proc = response_post_proc.summary_postprocess(
            solution_str,
            last_response_sep=config.reward_model.last_response_sep,
            last_response_strict=config.reward_model.last_response_strict)
    elif config.reward_model.use_last_response == 'lastcodeblock':
        solution_str_post_proc = response_post_proc.last_codeblock_postprocess(
            solution_str,
            codeblock_seps=config.reward_model.last_response_sep,
            last_response_strict=config.reward_model.last_response_strict)
    else:
        solution_str_post_proc = solution_str
    return solution_str_post_proc


@ray.remote(num_cpus=1)
def call_oj(solution_str, ground_truth, code_sandbox_psm):
    result = oj_utils.compute_score(solution_str=solution_str,
                                    ground_truth=ground_truth,
                                    code_sandbox_psm=code_sandbox_psm)
    return result


@ray.remote(num_cpus=1)
class SandboxClient:

    def __init__(self, config, tokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.results = {}

    def clear(self):
        # for some cases, the results won't be claimed. So we need to clear the results.
        self.results = {}

    def get_num_pending_outputs(self):
        """Return the number of outputs, whose result is not claimed"""
        return len(self.results)

    def add_requests(self, req_id, input_ids, ground_truth):
        solution_str = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        solution_str_post_proc = post_process_solution_str(self.config, solution_str)
        result_future = call_oj.remote(solution_str_post_proc, ground_truth, self.config.trainer.code_sandbox_psm)
        assert req_id not in self.results, f"{req_id} already exists"
        self.results[req_id] = result_future

    def get_results(self, req_id):
        if req_id not in self.results:
            return None

        assert req_id in self.results, f"{req_id} not found"
        result_future = self.results.pop(req_id)
        return ray.get(result_future)


class RewardManager():

    def __init__(self, tokenizer, config, logger: Tracking, rm_name="train") -> None:
        self.tokenizer = tokenizer
        self.logger = logger
        self.log_table = []
        self.rm_name = rm_name
        self.config = config
        self.case_study_dir = config.trainer.default_hdfs_dir + "/cases/"
        self.rm_req_executor = ThreadPoolExecutor(
            max_workers=int(self.config.reward_model.get('reward_executor_maxnum', 128)))
        self.mean = self.config.reward_model.mean
        self.std = self.config.reward_model.std
        self.need_punish_duplicate = self.config.reward_model.get('need_punish_duplicate', False)
        self.punish_score = self.config.reward_model.get('punish_score', 'rule-lighteval/MATH_v2:-1,code-sandbox:0')
        self.punish_score = dict(map(lambda x: (x.split(':')[0], float(x.split(':')[1])), self.punish_score.split(',')))
        self.need_punish_trunc = self.config.reward_model.get('need_punish_trunc', False)
        self.trunc_punish_score = self.config.reward_model.get('trunc_punish_score', -5)

    def __call__(self, data: DataProto, global_step=None, need_norm=True, is_validation=False):
        """We will expand this function gradually based on the available datasets"""
        response_ids = data.batch['input_ids'][:, self.config.data.max_prompt_length:]

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        already_print_data_sources = {}
        save_to_hdfs = []
        rm_res_future_list = []
        if global_step is not None and global_step % self.config.trainer.logger_step_interval == 0:
            self.log_table = []  # 清空self.log_table

        def get_rm_score(idx):
            data_item = data[idx]  # DataProtoItem

            prompt_ids = data_item.batch['input_ids'][:self.config.data.max_prompt_length]
            response_ids = data_item.batch['input_ids'][self.config.data.max_prompt_length:]

            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            solution_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            solution_str_post_proc = post_process_solution_str(config=self.config, solution_str=solution_str)

            # get prompt uuid
            data_uid = data_item.non_tensor_batch['uid']

            # select rm_score
            reward_style = data_item.non_tensor_batch['reward_model']['style']
            compute_score_fn = _select_rm_score_fn(reward_style)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            score_fn_inputs = {
                "batch_info": data_item.batch,
                "tokenizer": self.tokenizer,
                "solution_str": solution_str_post_proc,
                "ground_truth": ground_truth,
                "config": self.config,
                'data_uid': data_uid
            }
            if reward_style == "code-sandbox":
                score_fn_inputs["code_sandbox_psm"] = self.config.trainer.code_sandbox_psm
            score = compute_score_fn(**score_fn_inputs)
            is_para_dup = para_dup.find_single_turn_duplicate(solution_str)[0]
            is_trunc = (response_length == valid_response_length).item() and score == -1
            return prompt_str, solution_str, ground_truth, reward_style, valid_response_length, score, is_para_dup, is_trunc, idx, solution_str_post_proc

        for i in range(len(data)):
            rm_res_future_list.append(self.rm_req_executor.submit(get_rm_score, i))
        fail_cnt = 0
        total_cnt = 0
        dup_cnt = 0
        dup_lens = []
        not_dup_lens = []
        from tqdm import tqdm
        for res in tqdm(as_completed(rm_res_future_list), total=len(data), desc="get_rm_score"):
            prompt_str, solution_str, ground_truth, reward_style, valid_response_length, score, is_para_dup, is_trunc, idx, solution_str_post_proc = res.result(
            )
            if reward_style == "code-sandbox":
                total_cnt += 1
                # 访问失败的score现在设置成-2，用来计数，但是训练的时候还是当做没做对来处理
                if score == -2:
                    score = 0
                    fail_cnt += 1
            # train的时候做这个norm，但是打点的时候恢复，打原始值
            # eval的时候不做这个norm
            if need_norm:
                score = (score - self.mean) / self.std
            if is_para_dup:
                dup_cnt += 1
                dup_lens.append(valid_response_length)
                if self.need_punish_duplicate and not is_validation:
                    score = self.punish_score.get(reward_style, -1)
            else:
                not_dup_lens.append(valid_response_length)
            if self.need_punish_trunc and is_trunc and not is_validation:
                score = self.trunc_punish_score
            reward_tensor[idx, valid_response_length - 1] = score

            if reward_style not in already_print_data_sources:
                already_print_data_sources[reward_style] = 0

            if already_print_data_sources[reward_style] < self.config.trainer.num_cases_to_wandb:
                already_print_data_sources[reward_style] += 1
                if reward_style == "code-sandbox":
                    ground_truth = ''  # 对于OJ问题，ground_truth会比较大，扛不住
                self.log_table.append(
                    [global_step, prompt_str, solution_str, ground_truth, score, solution_str_post_proc])
            save_to_hdfs.append([
                idx, global_step, prompt_str, solution_str, ground_truth, score, solution_str_post_proc, is_para_dup,
                is_trunc,
                valid_response_length.item()
            ])

        prefix = "" if not is_validation else "val/"
        self.logger.log(data={
            prefix + "oj/fail_rate": fail_cnt / total_cnt if total_cnt > 0 else -1,
            prefix + "dup/para_dup": dup_cnt / len(data),
            prefix + "dup/dup_response_len": sum(dup_lens) / max(1, len(dup_lens)),
            prefix + "dup/not_dup_response_len": sum(not_dup_lens) / max(1, len(not_dup_lens)),
        },
                        step=global_step)

        if total_cnt > 0 and fail_cnt / total_cnt >= 0.01:
            send_message_to_employee("alpha seed任务oj失败率过高", f"任务链接: {task_url}, 失败率: {round(fail_cnt/total_cnt, 2)}",
                                     user_email)

        if self.config.trainer.num_cases_to_wandb > 0:
            logger_step = global_step - global_step % self.config.trainer.logger_step_interval
            self.logger.log(
                {
                    f"gen&score_{self.rm_name}_{logger_step}":
                        wandb.Table(
                            columns=["Step", "Prompt", "Gen Sequence", "GroundTruth", "Score", "Gen Sequence PostProc"],
                            data=self.log_table)
                },
                step=global_step,
                backend='tracking')

        if self.config.trainer.save_cases_to_hdfs:
            print(f"reward_fn begin hput: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            dir_name = self.case_study_dir
            file_name = f"{self.rm_name}.{str(global_step)}.parquet"

            def async_hput(save_to_hdfs, dir_name, file_name):
                df = pd.DataFrame(columns=[
                    "idx", "step", "prompt", "gen", "groundtruth", "score", "gen_postproc", "is_dup", "is_trunc", 'len'
                ],
                                  data=save_to_hdfs)
                df.to_parquet(f"{dir_name}{file_name}")

            p = Process(target=async_hput, args=(save_to_hdfs, dir_name, file_name))
            p.start()
            print(f"reward_fn end hput: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return reward_tensor


import ray
import hydra

from alpha_seed.trainer.ppo import RayPPOTrainer


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    training_duration_metrics_collector = None
    if RLMetricsClientContextManager:
        training_duration_metrics_collector = RLMetricsClientContextManager()

    metric_collection_context = training_duration_metrics_collector.collect_init_ray_cluster_duration() \
        if training_duration_metrics_collector else contextlib.nullcontext()

    with metric_collection_context:
        init_ray()
        check_arnold_resources(config=config)
    runner = TaskRunner.remote()
    ray.get(runner.main.remote(main_task, config=config))


def get_total_gpus_in_ray_cluster():
    total_gpus = 0
    for node in ray.nodes():
        gpus = node['Resources'].get('GPU', 0)
        total_gpus += gpus
    return total_gpus


def wait_till_nodes_ready(total_required_gpus: int, try_time=100):
    while True:
        try:
            available_gpus = ray.available_resources().get('GPU', 0)
            print(f"Checking nodes ready, {available_gpus} GPUs available, {total_required_gpus} GPUs expected")
            if available_gpus >= total_required_gpus:
                break
            time.sleep(15)
            try_time -= 1
        except Exception as e:
            print(f"ray nodes not ready yet, {e}")
            continue


def check_arnold_resources(config):
    """Check the arnold resources before running"""
    num_gpu_nodes = int(os.getenv('ARNOLD_WORKER_NUM', '0'))
    num_gpus_per_node = int(os.getenv('ARNOLD_WORKER_GPU', '0'))

    total_gpus = num_gpu_nodes * num_gpus_per_node
    if total_gpus <= 0:
        # maybe not on arnold environment? skip the check
        return

    total_required_gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node + \
        config.streaming_rollout.nnodes * config.streaming_rollout.n_gpus_per_node + \
        config.streaming_validator.nnodes * config.streaming_validator.n_gpus_per_node

    assert total_required_gpus <= total_gpus, f'Require {total_required_gpus} GPUs, but only have {total_gpus} GPUs'

    if total_required_gpus < total_gpus:
        warnings.warn(
            f'The total gpus {total_gpus} is larger than total required GPUs {total_required_gpus}. There might be a waste.'
        )

    # wait for all the GPUs to be ready before training
    wait_till_nodes_ready(total_required_gpus)


def init_ray():
    if not ray.is_initialized():
        # this is for local ray cluster
        remote_cache_env = {
            'TRITON_CACHE_MANAGER': 'triton.runtime.cache:RemoteCacheManager',
            'TRITON_REMOTE_CACHE_BACKEND': 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend'
        }
        runtime_env = {
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'BPEX_NO_WARN_ON_UNTUNED_CASE': '1'
            }
        }
        if ENABLE_REDIS_TRITON_CACHE:
            runtime_env['env_vars'].update(remote_cache_env)

        ray.init(runtime_env=runtime_env)


def validate_config(config):
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    # data
    real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.num_bon
    assert real_train_batch_size % n_gpus == 0

    # rollout
    # assert real_train_batch_size % config.actor_rollout_ref.rollout.micro_batch_size == 0
    complete_ratio = config.actor_rollout_ref.rollout.get("complete_ratio", 1.0)
    if config.streaming_rollout.nnodes == 0:
        assert complete_ratio == 1.0, f'When streaming rollout is not enabled, complete_ratio must be 1. Got {complete_ratio}'
    else:
        assert complete_ratio < 1.0, f'When streaming rollout is enabled, complete_ratio must be smaller than 1. Got {complete_ratio}.'

    # actor
    assert real_train_batch_size % config.actor_rollout_ref.actor.ppo_mini_batch_size == 0
    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        ulysses = config.actor_rollout_ref.actor.ulysses_sequence_parallel_size
        assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
        assert config.actor_rollout_ref.actor.ppo_micro_batch_size * ulysses >= n_gpus

    # critic
    assert real_train_batch_size % config.critic.ppo_mini_batch_size == 0
    if not config.critic.use_dynamic_bsz:
        ulysses = config.critic.ulysses_sequence_parallel_size
        assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
        assert config.critic.ppo_micro_batch_size * ulysses >= n_gpus

    min_required_seq_len = config.data.max_prompt_length + config.data.max_response_length
    if config.actor_rollout_ref.actor.use_dynamic_bsz:
        if min_required_seq_len > config.actor_rollout_ref.actor.ppo_max_token_len:
            config.actor_rollout_ref.actor.ppo_max_token_len = min_required_seq_len
            print(
                f"Warning: config.actor_rollout_ref.actor.ppo_max_token_len is set to {config.actor_rollout_ref.actor.ppo_max_token_len}"
            )
    if config.actor_rollout_ref.ref.use_dynamic_bsz:
        if min_required_seq_len > config.actor_rollout_ref.ref.max_token_len:
            config.actor_rollout_ref.ref.max_token_len = min_required_seq_len
            print(
                f"Warning: config.actor_rollout_ref.ref.max_token_len is set to {config.actor_rollout_ref.ref.max_token_len}"
            )
    if config.actor_rollout_ref.rollout.use_dynamic_bsz:
        if min_required_seq_len > config.actor_rollout_ref.rollout.max_token_len:
            config.actor_rollout_ref.rollout.max_token_len = min_required_seq_len
            print(
                f"Warning: config.actor_rollout_ref.rollout.max_token_len is set to {config.actor_rollout_ref.rollout.max_token_len}"
            )
    if config.critic.use_dynamic_bsz:
        if min_required_seq_len > config.critic.ppo_max_token_len:
            config.critic.ppo_max_token_len = min_required_seq_len
            print(f"Warning: config.critic.ppo_max_token_len is set to {config.critic.ppo_max_token_len}")
    if config.reward_model.use_dynamic_bsz:
        if min_required_seq_len > config.reward_model.max_token_len:
            config.reward_model.max_token_len = min_required_seq_len
            print(f"Warning: config.reward_model.max_token_len is set to {config.reward_model.max_token_len}")


@ray.remote
class TaskRunner:

    def main(self, func, *args, **kwargs):
        return func(*args, **kwargs)


def main_task(config):

    training_duration_metrics_collector = None

    if RLMetricsClientContextManager:
        training_duration_metrics_collector = RLMetricsClientContextManager()

    metric_collection_context = training_duration_metrics_collector.collect_setup_trainer_duration() \
        if training_duration_metrics_collector else contextlib.nullcontext()

    with metric_collection_context:
        validate_config(config=config)

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
            from single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        from alpha_seed.trainer.ppo import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRolloutRef: AsyncActorRolloutRefWorker,
            Role.Critic: CriticWorker,
            Role.Rollout: AsyncActorRolloutRefWorker,
            Role.Validator: AsyncActorRolloutRefWorker,
        }

        global_pool_id = 'global_pool'
        standalone_pool_id = 'standalone_pool'
        validation_pool_id = 'validation_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            standalone_pool_id: [config.streaming_rollout.n_gpus_per_node] * config.streaming_rollout.nnodes,
            validation_pool_id: [config.streaming_validator.n_gpus_per_node] * config.streaming_validator.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
            Role.Rollout: standalone_pool_id,
            Role.Validator: validation_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            from alpha_seed.workers.actors.reward_worker import RewardModelWorker
            role_worker_mapping[Role.RewardModel] = RewardModelWorker
            mapping[Role.RewardModel] = global_pool_id

        reward_fn = RewardManager(tokenizer=tokenizer, config=config, logger=logger, rm_name="train")

        # Note that we always use function-based RM for validation
        val_reward_fn = RewardManager(tokenizer=tokenizer, config=config, logger=logger, rm_name="val")

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.trainer.use_remote_sandbox:
            sandbox_client = SandboxClient.options(name='sandbox_client').remote(config=config, tokenizer=tokenizer)
        else:
            sandbox_client = None

        trainer = RayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn,
                                logger=logger,
                                sandbox_client=sandbox_client)

    metric_collection_context = training_duration_metrics_collector.collect_init_worker_duration() \
        if training_duration_metrics_collector else contextlib.nullcontext()
    with metric_collection_context:
        trainer.init_workers()
    send_message_to_employee("alpha seed任务开始训练", f"任务链接: {task_url}", user_email)
    if config.convert_ckpt_to_omnistore_task.enable:
        trainer.convert_ckpt_to_omnistore()
        send_message_to_employee("alpha seed任务转换ckpt到omnistore完成，任务结束", f"任务链接: {task_url}", user_email)
    else:
        trainer.fit()


if __name__ == '__main__':
    main()
