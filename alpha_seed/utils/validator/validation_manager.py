import os
import time
import asyncio
from collections import defaultdict
import uuid
import json
import torch
import wandb
import queue
import threading
import numpy as np
from pprint import pprint
from mono_rl import DataProto
import random
import ray

try:
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
except ImportError:
    print('Cannot find pad_dataproto_to_divisor. Please use latest verl master')
    raise
from mono_rl.utils.dataset.dist_data_util import release_object, get_dist_data_manager, get_local_inputs


class ValidateManager(object):
    """
    This is a standalone validator that runs in a single thread. It controls a SPMD workergroup that performs generation.
    The workergroup fetches latest weights from main task when it finishes the last iteration of validation
    """

    def __init__(self, config, logger, val_dataloader, tokenizer, use_rm, val_reward_fn, rollout_manager,
                 dist_data_manager) -> None:
        self.config = config
        self.is_vlm = self.config.data['image_key'] is not None
        self.logger = logger
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.use_rm = use_rm
        self.val_reward_fn = val_reward_fn
        self.rm_wg = None
        self.val_thread = None
        self.val_result_queue = queue.Queue()
        self.fast_result = os.getenv('WANDB_IGNORE_STEP_ORDER') == '1'
        if self.fast_result:
            print('Using fast result on wandb mode.')
        assert len(self.val_dataloader) == 1, "for bon metrics computation"
        self.rollout_manager = rollout_manager
        self.dist_data_manager = dist_data_manager

    def _save_val_data(self, reward_tensor_before_select, prompts, responses, f):
        for reward, prompt, response in zip(reward_tensor_before_select, prompts, responses):
            data = {"reward": reward.item(), "prompt": prompt, "response": response}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()

    def _save_val_data_vlm(self, test_batch, prompt_ids, reward_tensor_before_select, prompts, responses, val_epoch_idx,
                           val_idx, f):
        rollout_ids = test_batch.non_tensor_batch['rollout_id']
        if "prompt_id" in test_batch.non_tensor_batch:
            ori_prompt_indexs = test_batch.non_tensor_batch['prompt_id']
        elif "index" in test_batch.non_tensor_batch:
            ori_prompt_indexs = test_batch.non_tensor_batch['index']
        else:
            ori_prompt_indexs = [None] * prompt_ids.shape[0]
        if self.config.actor_rollout_ref.rollout.vlm.return_raw_output:
            raw_outputs = get_local_inputs(test_batch.non_tensor_batch, 'raw_output_ref', self.dist_data_manager)
        else:
            raw_outputs = [''] * prompt_ids.shape[0]
        if "index" in test_batch.non_tensor_batch:
            dataset_indexs = test_batch.non_tensor_batch['index']
        else:
            dataset_indexs = [None] * prompt_ids.shape[0]
        for reward, prompt, response, rollout_id, ori_prompt_index, raw_output, dataset_index in zip(
                reward_tensor_before_select, prompts, responses, rollout_ids, ori_prompt_indexs, raw_outputs,
                dataset_indexs):
            data = {
                "val_epoch_idx": val_epoch_idx,
                "val_idx": val_idx,
                "prompt_id": ori_prompt_index,
                "index_id": dataset_index,
                "ori_prompt_id": dataset_index,
                "rollout_id": rollout_id,
                "reward": reward.item(),
                "prompt": prompt,
                "response": response.replace("[SOI][EOI]", "[SOI]<ImageHere>[EOI]"),
                "raw_output": raw_output
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()

    def validate(self,
                 val_epoch=1,
                 need_log=False,
                 log_file="/opt/tiger/alpha-seed/log.jsonl",
                 is_async=False,
                 global_step=0):

        if is_async:
            assert not self.use_rm, "Async validation is not supported with RM yet."
        if self.val_thread is not None:
            self.val_thread.join()
            if self.val_result_queue.qsize() > 0:
                val_metrics, val_log_lst, val_step = self.val_result_queue.get()
                if not self.fast_result:
                    if wandb.run is not None:
                        for metric in val_metrics.keys():
                            wandb.define_metric(metric, step_metric="val_step")
                    val_metrics["val_step"] = val_step
                    self.logger.log(data=val_metrics, step=global_step)
                    for val_log in val_log_lst:
                        if val_log is not None:
                            self.logger.log(data=val_log, step=global_step, backend="tracking")

        self.val_thread = threading.Thread(target=self._validate,
                                           args=(val_epoch, need_log, log_file, is_async, global_step))
        self.val_thread.start()
        self.rollout_manager.wait_nccl_comm_threadsafe()

        if is_async:
            return
        else:
            self.val_thread.join()

            while True:
                try:
                    val_metrics, val_log_lst, val_step = self.val_result_queue.get(timeout=1)
                    break
                except Exception:
                    assert self.val_thread.is_alive()

            self.val_thread = None
            if wandb.run is not None:
                for metric in val_metrics.keys():
                    wandb.define_metric(metric, step_metric="val_step")
            val_metrics["val_step"] = val_step
            self.logger.log(data=val_metrics, step=global_step)
            for val_log in val_log_lst:
                if val_log is not None:
                    self.logger.log(data=val_log, step=global_step, backend="tracking")
        return

    def _validate(self, val_epoch, need_log, log_file, is_async, global_step):
        print(f'{time.time()} start validate with fast_result={self.fast_result}')
        metric_dict = {}
        reward_tensor_lst = []
        data_source_lst = []
        prompt_name_lst = []
        bopxn_lst = []
        val_log_lst = []
        if need_log:
            f = open(log_file, "w")
        for val_epoch_idx in range(val_epoch):
            for val_idx, test_data in enumerate(self.val_dataloader):
                test_batch = DataProto.from_single_dict(test_data)
                if 'images_bytes_ref' in test_batch.non_tensor_batch:
                    test_batch_padded, pad_size = pad_dataproto_to_divisor(
                        test_batch, size_divisor=self.rollout_manager.hybrid_wg.world_size)
                    test_batch_padded = self.rollout_manager.hybrid_wg.load_and_transform_save_image(test_batch_padded)
                    if pad_size > 0:
                        test_batch = test_batch_padded.slice(end=-pad_size)
                    else:
                        test_batch = test_batch_padded

                prompt_names = test_batch.non_tensor_batch['prompt_names'][0]
                num_prompts_per_data = len(prompt_names)

                if num_prompts_per_data > 1:
                    test_batch = test_batch.unfold_column_chunks(
                        num_prompts_per_data, split_keys=['input_ids', 'attention_mask', 'prompt_names'])

                eval_bon = self.config.actor_rollout_ref.rollout.get("eval_bon", 1)
                test_batch.non_tensor_batch['rollout_id'] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch))], dtype=object)
                bon_ids = list(range(eval_bon)) * (len(test_batch))
                if eval_bon != 1:
                    test_batch = test_batch.repeat(eval_bon)
                    test_batch.non_tensor_batch['bon_id'] = np.array(bon_ids, dtype=object)
                test_batch.meta_info['epoch_id'] = val_epoch_idx

                # create a uid for each data inside the batch
                test_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch))],
                                                              dtype=object)
                input_batch = test_batch
                test_batch = self.rollout_manager.val_generate(test_batch, step=global_step, is_async=is_async)

                print(
                    f'{val_epoch_idx + 1}-th/{val_epoch} {val_idx + 1}-th/{len(self.val_dataloader)} validation generation end'
                )
                if test_batch is None:
                    release_object(self.dist_data_manager, input_batch.non_tensor_batch,
                                   ['image_data_ref', 'images_bytes_ref'])
                    continue

                if self.use_rm:
                    # we first compute reward model score
                    test_batch_padded, pad_size = pad_dataproto_to_divisor(test_batch,
                                                                           size_divisor=self.rm_wg.world_size)
                    reward_tensor = self.rm_wg.compute_rm_score(test_batch_padded)
                    reward_tensor = unpad_dataproto(reward_tensor, pad_size=pad_size)

                    test_batch = test_batch.union(reward_tensor)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor, val_log = self.val_reward_fn(test_batch,
                                                            global_step=global_step,
                                                            need_norm=False,
                                                            is_validation=True)
                val_log_lst.append(val_log)

                reward_tensor_before_select = reward_tensor.clone()  # (B x bon, seqlen)
                if eval_bon > 1 and global_step % self.config.actor_rollout_ref.rollout.get("eval_bon_every", 20) == 0:
                    print("begin compute bon")
                    from alpha_seed.utils.reward_score.bootstrap_bon import bootstrap_bon_metric
                    nxm_mat = reward_tensor_before_select.sum(-1).reshape(-1, eval_bon)
                    bopxn_mat = reward_tensor_before_select.sum(-1).reshape(-1, num_prompts_per_data * eval_bon)

                    bon_matrix, bon_metric = bootstrap_bon_metric(nxm_mat)  #  nxm
                    bopxn, _ = bootstrap_bon_metric(bopxn_mat)
                    bopxn_lst.append(bopxn)
                    # metric_dict.update({f"diversity/eval_bo{k}": v for k, v in bon_metric.items()})
                    # metric_dict['diversity/eval_bon_hist'] = wandb.Histogram(np_histogram=np.histogram(
                    #     np.arange(0, eval_bon) + 0.5, bins=eval_bon, weights=bon_matrix.mean(0)))
                    reward_tensor = bon_matrix  #[:, 0]  # bo1 as reward
                else:
                    reward_tensor = reward_tensor.sum(-1).unsqueeze(-1)  # sum over seqlen

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(
                    test_batch.non_tensor_batch.get('data_source',
                                                    ['unknown'] * reward_tensor.shape[0]).reshape(-1, eval_bon)[:, 0])
                prompt_name_lst.append(
                    test_batch.non_tensor_batch.get('prompt_names',
                                                    ['unknown'] * reward_tensor.shape[0]).reshape(-1, eval_bon)[:, 0])
                if need_log:
                    input_ids = test_batch.batch['input_ids'].cpu().numpy()
                    prompt_ids = input_ids[:, :self.config.data.max_prompt_length]
                    decode_batch_prompt = []
                    for i in range(prompt_ids.shape[0]):
                        valid_prompt_idx = prompt_ids[i]
                        # remove potential special tokens(-100)
                        valid_prompt_idx = valid_prompt_idx[valid_prompt_idx != -100]
                        decode_batch_prompt.append(valid_prompt_idx)
                    decode_batch_response = []
                    for i in range(test_batch.batch['responses'].shape[0]):
                        valid_response_length = test_batch.batch['attention_mask'][
                            i, self.config.data.max_prompt_length:].sum().item()
                        valid_response_idx = test_batch.batch['responses'][i, :valid_response_length]
                        # remove potential special tokens(-100)
                        valid_response_idx = valid_response_idx[valid_response_idx != -100]
                        decode_batch_response.append(valid_response_idx)
                    prompts = self.tokenizer.batch_decode(decode_batch_prompt, skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(decode_batch_response, skip_special_tokens=False)
                    reward_tensor_before_select = reward_tensor_before_select.sum(-1).cpu()
                    if 'image_bytes_ref' in test_batch.non_tensor_batch or \
                            'image_data_ref' in test_batch.non_tensor_batch:
                        self._save_val_data_vlm(test_batch, prompt_ids, reward_tensor_before_select, prompts, responses,
                                                val_epoch_idx, val_idx, f)
                    else:
                        self._save_val_data(reward_tensor_before_select, prompts, responses, f)
                release_object(self.dist_data_manager, test_batch.non_tensor_batch,
                               ['image_data_ref', 'images_bytes_ref'])

        if len(reward_tensor_lst) == 0:
            self.val_result_queue.put(({}, val_log_lst, global_step))
            return

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).cpu()  # (valsize*num_prompt_per_data, eval_bon)
        reward_tensor = torch.clamp(reward_tensor, min=0)
        bopxn = torch.cat(bopxn_lst, dim=0).cpu(
        ) if eval_bon > 1 and num_prompts_per_data > 0 else None  # (valsize, num_prompt_per_data*eval_bon)

        def compute_metric(reward_tensor, bopxn, metric_dict, data_source="all"):
            '''reward_tensor : (datasize*num_prompt_per_data, eval_bon)
            bobon_reward: (datasize, num_prompt_per_data*eval_bon)
            '''
            logN = int(np.log(eval_bon) / np.log(2))
            power_index = torch.LongTensor([2**i for i in range(logN)] + [eval_bon]) - 1
            format_fn = lambda lst: ",\t".join("{:.3f}".format(x) for x in lst)
            if num_prompts_per_data > 0:
                reward_tensor_per_prompt = [
                    row for row in reward_tensor.reshape(-1, num_prompts_per_data, eval_bon).transpose(0, 1).mean(1)
                ]
            avgp_bon = reward_tensor.mean(0)
            print("{}, avgpbon\t\t {}".format(data_source, format_fn(avgp_bon[power_index])))
            if bopxn is not None:
                # bopboN = reward_tensor.reshape(-1, num_prompts_per_data, eval_bon).max(dim=1).values.mean(0)
                bopxn = bopxn.mean(0)[(num_prompts_per_data - 1)::num_prompts_per_data]
                # print("{}, bopboN\t\t {}".format(data_source, format_fn(bopboN[power_index])))
                print("{}, bopxn\t\t {}".format(data_source, format_fn(bopxn[power_index])))

            if num_prompts_per_data > 0:
                for pid in range(num_prompts_per_data):
                    print("{} BoN (prompt {}):\t {}".format(data_source, prompt_names[pid],
                                                            format_fn(reward_tensor_per_prompt[pid][power_index])))

            for N in power_index:
                metric_dict[f'test_score/{data_source}_avgpbo{N}'] = avgp_bon[N]
                if bopxn is not None:
                    # metric_dict[f'test_score/{data_source}_bopbo{N}'] = bopboN[N]
                    metric_dict[f'test_score/{data_source}_bopxn{N}'] = bopxn[N]
                if num_prompts_per_data > 0:
                    for pid in range(num_prompts_per_data):
                        metric_dict[f'test_score/all_{prompt_names[pid]}_bo{N}'] = reward_tensor_per_prompt[pid][N]
            metric_dict[f'test_cnt/{data_source}'] = len(
                reward_tensor) // num_prompts_per_data if num_prompts_per_data > 0 else len(reward_tensor)

        compute_metric(reward_tensor, bopxn, metric_dict, data_source="all")

        # group by data source metrics
        data_sources = np.concatenate(data_source_lst, axis=0)
        prompt_names_per_sample = np.concatenate(prompt_name_lst, axis=0)  # not useful for now

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i])

        if bopxn is not None:
            data_source_bopxn = {}
            for i in range(bopxn.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_bopxn:
                    data_source_bopxn[data_source] = []
                data_source_bopxn[data_source].append(bopxn[i])

        prompt2rwd = defaultdict(list)
        prompt2source = {}
        source2rwd = defaultdict(list)
        for i, item in enumerate(test_batch.non_tensor_batch['raw_prompt']):
            prompt2rwd[item[0]['content']].append(reward_tensor[i].item())
            prompt2source[item[0]['content']] = data_sources[i]
        for prompt, rwd in prompt2rwd.items():
            source2rwd[prompt2source[prompt]].append(rwd)
        for source, rwds in source2rwd.items():
            if len(rwds[0]) == 32:
                for n in [4, 8, 16, 32]:
                    total = len(rwds) * 128
                    correct = 0
                    if n > len(rwds[0]):
                        continue
                    for rwd in rwds:
                        for _ in range(128):
                            sample_n = random.sample(rwd, k=n)
                            if max(sample_n) == 1:
                                correct += 1
                    acc = correct / total
                    metric_dict[f'test_score/{source}_bo{n}'] = acc
                # worst of n
                for n in [4, 8, 16, 32]:
                    total = len(rwds) * 128
                    correct = 0
                    if n > len(rwds[0]):
                        continue
                    for rwd in rwds:
                        for _ in range(128):
                            sample_n = random.sample(rwd, k=n)
                            if min(sample_n) == 1:
                                correct += 1
                    acc = correct / total
                    metric_dict[f'test_score/{source}_wo{n}'] = acc

        for data_source, rewards in data_source_reward.items():
            rewards_tensor_data_source = torch.vstack(rewards)
            bopxn_data_source = torch.vstack(data_source_bopxn[data_source]) if bopxn is not None else None
            compute_metric(rewards_tensor_data_source, bopxn_data_source, metric_dict, data_source=data_source)
        if need_log:
            f.close()

        # upload validation metrics
        val_metrics = {
            f'val/{key}': val.item() if isinstance(val, torch.Tensor) else val for key, val in metric_dict.items()
        }
        if global_step == 0:
            pprint(f'Initial validation metrics: {val_metrics}')

        if self.fast_result:
            for metric in val_metrics.keys():
                wandb.define_metric(metric, step_metric="val_step")
            val_metrics["val_step"] = global_step
            self.logger.log(data=val_metrics, step=global_step)
            for val_log in val_log_lst:
                if val_log is not None:
                    self.logger.log(data=val_log, step=global_step, backend="tracking")
        print(f'{time.time()} end validate with fast_result={self.fast_result}')
        self.val_result_queue.put((val_metrics, val_log_lst, global_step))
