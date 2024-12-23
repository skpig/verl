from collections import defaultdict
import uuid
import json
import torch
import wandb
import queue
import threading
import numpy as np
from pprint import pprint
from verl import DataProto
import random

try:
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
except ImportError:
    print('Cannot find pad_dataproto_to_divisor. Please use latest verl master')
    raise


class ValidateManager(object):
    """
    This is a standalone validator that runs in a single thread. It controls a SPMD workergroup that performs generation.
    The workergroup fetches latest weights from main task when it finishes the last iteration of validation
    """

    def __init__(self, config, logger, val_dataloader, tokenizer, use_rm, val_reward_fn) -> None:
        self.config = config
        self.logger = logger
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.use_rm = use_rm
        self.val_reward_fn = val_reward_fn
        self.actor_rollout_wg = None
        self.rm_wg = None
        self.standalone_validator_wg = None
        self.val_thread = None
        self.val_result_queue = queue.Queue()

        assert len(self.val_dataloader) == 1, "for bon metrics computation"

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
                for metric in val_metrics.keys():
                    wandb.define_metric(metric, step_metric="val_step")
                val_metrics["val_step"] = val_step
                self.logger.log(data=val_metrics, step=global_step)
                for val_log in val_log_lst:
                    if val_log is not None:
                        self.logger.log(data=val_log, step=global_step, backend="tracking")

        if is_async:
            self.actor_rollout_wg.update_standalone_worker("standalone_validator")
            self.standalone_validator_wg.update_standalone_worker("standalone_validator")
            validator_wg = self.standalone_validator_wg
        else:
            validator_wg = self.actor_rollout_wg

        self.val_thread = threading.Thread(target=self._validate,
                                           args=(val_epoch, need_log, log_file, is_async, global_step, validator_wg))
        self.val_thread.start()

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
            for metric in val_metrics.keys():
                wandb.define_metric(metric, step_metric="val_step")
            val_metrics["val_step"] = val_step
            self.logger.log(data=val_metrics, step=global_step)
            for val_log in val_log_lst:
                if val_log is not None:
                    self.logger.log(data=val_log, step=global_step, backend="tracking")
        return

    def _validate(self, val_epoch, need_log, log_file, is_async, global_step, validator_wg):
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

                prompt_names = test_batch.non_tensor_batch['prompt_names'][0]
                num_prompts_per_data = len(prompt_names)

                if num_prompts_per_data > 1:
                    test_batch = test_batch.unfold_column_chunks(
                        num_prompts_per_data, split_keys=['input_ids', 'attention_mask', 'prompt_names'])

                eval_bon = self.config.actor_rollout_ref.rollout.get("eval_bon", 1)
                test_batch = test_batch.repeat(eval_bon)

                # create a uid for each data inside the batch
                test_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch))],
                                                              dtype=object)

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'off_policy_steps'])
                # copy relevant non-tensor info
                non_tensor_infos = ['uid', 'reward_model']
                for key in non_tensor_infos:
                    test_gen_batch.non_tensor_batch[key] = test_batch.non_tensor_batch[key]

                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'validate': True,
                    'complete_ratio': 1,  # validation does not need timeout
                }
                test_gen_batch.meta_info[
                    'generation_kwargs'] = self.config.actor_rollout_ref.rollout.val_generate_kwargs

                # pad test_gen_batch to divisible by world_size. TODO(zhangchi.usc1992): shall we move this logic to dispatch?
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, validator_wg.world_size)

                # mark the paddig data uid to None
                for i in range(pad_size):
                    test_gen_batch_padded.non_tensor_batch['uid'][i] = None

                test_output_gen_batch = validator_wg.generate_sequences(test_gen_batch_padded)
                test_output_gen_batch.batch['prompts'] = test_output_gen_batch.batch['input_ids'][:, :self.config.data.
                                                                                                  max_prompt_length]
                test_output_gen_batch.batch['responses'] = test_output_gen_batch.batch['input_ids'][:, self.config.data.
                                                                                                    max_prompt_length:]

                test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)

                print(
                    f'{val_epoch_idx + 1}-th/{val_epoch} {val_idx + 1}-th/{len(self.val_dataloader)} validation generation end'
                )

                test_batch = test_batch.union(test_output_gen_batch)

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
                    input_ids = test_output_gen_batch.batch['input_ids'].cpu().numpy()
                    prompt_ids = input_ids[:, :self.config.data.max_prompt_length]
                    response_ids = input_ids[:, self.config.data.max_prompt_length:]
                    prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                    reward_tensor_before_select = reward_tensor_before_select.sum(-1).cpu()
                    for reward, prompt, response in zip(reward_tensor_before_select, prompts, responses):
                        data = {"reward": reward.item(), "prompt": prompt, "response": response}
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        f.flush()

        # validation release memory
        if not is_async:
            validator_wg.release_param_and_cache()

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).cpu()  # (valsize*num_prompt_per_data, eval_bon)
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
                    total = len(rwds) * 5
                    correct = 0
                    for rwd in rwds:
                        for _ in range(5):
                            sample_n = random.sample(rwd, k=n)
                            if max(sample_n) == 1:
                                correct += 1
                    acc = correct / total
                    metric_dict[f'test_score/{source}_bo{n}'] = acc

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
        self.val_result_queue.put((val_metrics, val_log_lst, global_step))
