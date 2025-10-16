import ray
import wandb
import torch
from codetiming import Timer
from alpha_seed.trainer.ppo import RayPPOTrainer
from alpha_seed.workers.xperf_rollout.profiler.visualizer import visualize_standalone_usage
from alpha_seed.utils.observility.pretty_print import pprint
from alpha_seed.utils.functional import save_simple_train_data_to_hdfs
from mono_rl.utils.dataset.dist_data_util import load_image_data_dist, init_or_get_dist_data_manager, release_object
from mono_rl import DataProto


class VLMRayPPOTrainer(RayPPOTrainer):

    def fit(self):
        self.setup()

        self.validate_before_training()
        if self.config.trainer.val_only:
            return

        # Note that we start from step 1. After resume, we increment step by 1 to start next step
        self.global_step += 1
        start_step = self.global_step

        while True:
            metrics = {}
            with Timer(name='step', logger=None) as step_timer:
                batch, status = self._train_generate(metrics, start_step)
                if status == 'continue':
                    continue

                with Timer(name='train', logger=None) as train_timer:
                    batch.meta_info['global_step'] = self.global_step
                    self.update_len_per_query(batch, metrics)
                    batch = self._rollout_log_probs(batch, metrics)
                    batch = self._rm_score(batch, metrics)
                    raw_scores_log = self._reward_fn(batch, metrics)

                    if self.config.algorithm.priority_sample:
                        self.sample_pool.update_priority_dict(batch)

                    batch, status = self._dynamic_sampling(batch, metrics)
                    if status == 'continue':
                        continue

                    batch = self._mask_overlong(batch, metrics, raw_scores_log)

                    batch, use_async_gen = self._league_training(batch, metrics)

                    batch = self._select_bon_samples(batch, metrics, use_async_gen)

                    batch = self._shuffle_sample_batch(batch)

                    # perform sequence balancing.
                    # Very important: Note that this reorders data globally.
                    # So anything that requires ordering below this line will cause incorrect results
                    self._balance_batch(batch=batch, metrics=metrics, logging_prefix='global_seqlen')

                    metrics.setdefault('timing/train_mem_offload', 0)

                    batch = self.compute_reference(batch, metrics)

                    input_batch = batch
                    if self.enable_actor_critic_spatial_mux:
                        input_batch = input_batch.repeat(2, interleave=False)

                    # compute actor
                    actor_future = self.actor_rollout_wg.old_log_probs(input_batch)
                    critic_future = None
                    # compute values
                    if self.use_critic and self.enable_actor_critic_spatial_mux:
                        critic_future = self.critic_wg.compute_values(input_batch)

                    batch = self._compute_old_log_probs(actor_future, batch, metrics)
                    if self.config.actor_rollout_ref.actor.reuse_old_experts:
                        ray.get(
                            self.dist_data_manager.release_refs.remote(
                                batch.non_tensor_batch['old_experts_ref'].tolist()))
                        batch.non_tensor_batch.pop('old_experts_ref')
                    if self.use_critic:
                        batch, critic_future = self._compute_values(batch, critic_future, input_batch, metrics)

                    batch = self._compute_adv(batch, metrics, use_async_gen)
                    self.compute_metrics(batch, metrics)
                    if 'ability_idx' in batch.batch:
                        self.compute_metrics_per_ability(batch, metrics)
                    with Timer(name='save_output_batch', logger=None) as timer:
                        save_simple_train_data_to_hdfs(batch, self.tokenizer, self.global_step, self.config)
                    metrics['timing/save_output_batch'] = timer.last
                    if self.global_step == 1:
                        print('Debugging', batch.batch)

                    input_batch = batch
                    if self.enable_actor_critic_spatial_mux:
                        input_batch = input_batch.repeat(2, interleave=False)

                    # update actor
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_step and self.global_step % self.config.trainer.actor_update_freq == 0:
                        actor_future = self.actor_rollout_wg.update_actor(input_batch)

                    # update critic
                    if self.use_critic:
                        critic_future = self.critic_wg.update_critic(input_batch)

                    self._update_actor(actor_future, batch, metrics)

                    self._update_critic(critic_future, batch, metrics)

                    # update ref ema
                    with Timer(name='update_ref_ema', logger=None) as timer:
                        self.ref_policy_wg.update_ref_ema(self.global_step)
                    metrics['timing/update_ref_ema'] = timer.last

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_step % self.config.trainer.test_freq == 0:
                        with Timer(name='testing', logger=None) as timer:
                            self.validation_manager.validate(is_async=self.use_standalone_validator,
                                                             global_step=self.global_step)
                        metrics['timing/testing'] = timer.last

                    self._save_checkpoint(metrics)

                    # collect sandbox client remaining results
                    if self.config.trainer.use_remote_sandbox:
                        remote_client = ray.get_actor('remote_client_0')
                        num_remaining_results = ray.get(remote_client.get_num_pending_outputs.remote())
                        metrics['remote_client/remaining_results'] = num_remaining_results
                metrics['timing/train'] = train_timer.last

            metrics['timing/step'] = step_timer.last

            visualize_standalone_usage(self.config, metrics)

            # TODO: make a canonical logger that supports various backend
            self.logger.log(data=metrics, step=self.global_step)
            release_object(self.dist_data_manager)

            self.global_step += 1
            if self.global_step >= self.total_training_steps:

                # perform validation after training
                if self.val_reward_fn is not None:
                    val_metrics = self.validation_manager.validate(is_async=False, global_step=self.global_step)
                    pprint(f'Final validation metrics: {val_metrics}')

                # wait for the last ckpt to finish uploading if there are any
                ray.get(self.ckpt_global_uploader.final_wait_all_steps.remote())

                # wait for async tracking
                for t in self.async_tracking_running_tasks:
                    t.result()  # call this to collect the result(including error traceback)
                wandb.finish()
                return

    def compute_metrics_per_ability(self, batch, metrics):
        with Timer(name='compute_metrics_per_ability', logger=None) as timer:
            batch.meta_info['use_critic'] = self.use_critic
            batch.meta_info["ability_dict"] = self.reward_fn.ability_dict
            data_metrics: DataProto = self.actor_rollout_wg.execute_with_func_generator(
                compute_data_metrics_per_ability, batch)
            data_metrics = data_metrics.meta_info['metrics']
            metrics.update(data_metrics)

        metrics['timing/compute_metrics_per_ability'] = timer.last


def compute_data_metrics_per_ability(self, batch: DataProto):

    def allreduce_sum(x: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def allreduce_var(x: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return x.std(unbiased=unbiased)

        local_sum = x.sum()
        local_sqsum = (x**2).sum()
        local_count = torch.tensor(x.numel(), device=x.device, dtype=torch.long)

        global_sum = local_sum.clone()
        global_sqsum = local_sqsum.clone()
        global_count = local_count.clone()

        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_sqsum, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)

        global_sum = global_sum.to(torch.float32)
        global_sqsum = global_sqsum.to(torch.float32)
        global_count = global_count.to(torch.float32)

        mean = global_sum / global_count
        var = global_sqsum / global_count - mean**2

        if unbiased and global_count > 1:
            var *= global_count / (global_count - 1)
        return var.to(x.dtype)

    metrics = {}
    use_critic = batch.meta_info['use_critic']
    ability_dict = batch.meta_info["ability_dict"]

    batch = batch.to('cuda')
    old_log_probs = batch.batch['old_log_probs']  # (bs, s)
    old_entropy = batch.batch['old_entropy']  # (bs, s)
    ability_idx = batch.batch['ability_idx']  # (bs)
    response_length = batch.batch['responses'].shape[-1]  # (bs)
    advantages = batch.batch['advantages']  # (bs, s)
    response_mask = batch.batch['attention_mask'][:, -response_length:]  # (bs, s)
    if use_critic:
        returns = batch.batch['returns']  # (bs, s)
        values = batch.batch['values']

    if batch.meta_info['use_model_output_mask']:
        model_output_mask = batch.batch['model_output_mask'][:, -response_length:]
    else:
        model_output_mask = response_mask
    model_output_mask_bool = model_output_mask.bool()  # (bs, s)

    old_prob = old_log_probs.exp()  # 提前做完exp计算

    response_length = response_mask.sum(-1)  # (bs)
    eos_idx = torch.clamp(response_length.long(), min=1) - 1  # (bs)
    eos_adv = torch.gather(advantages, dim=1, index=eos_idx.unsqueeze(dim=1).long()).reshape(-1)  # (bs)

    # 求和过程中，通信shape一直为(bs,s)
    for ab_idx, ab_name in ability_dict.items():
        token_mask = model_output_mask_bool & (ability_idx.unsqueeze(-1) == ab_idx
                                              )  # (bs,s) 有效回复位置+对应ability的位置，按每个bs进行mask

        ent_sum_local = (old_entropy.masked_fill(~token_mask, 0.0)).sum()  # (bs, s)
        ent_cnt_local = token_mask.sum()  # (bs, s)

        prob_sum_local = (old_prob.masked_fill(~token_mask, 0.0)).sum()  # (bs, s)
        prob_cnt_local = token_mask.sum()  # (bs, s)

        eos_mask = (ability_idx == ab_idx)  # (B,)
        eos_sum_local = eos_adv.masked_fill(~eos_mask, 0.0).sum()  # (B,)
        eos_cnt_local = eos_mask.sum()  # (B,)

        # 全局统计
        ent_sum = allreduce_sum(ent_sum_local.clone())
        ent_cnt = allreduce_sum(ent_cnt_local.clone())
        prob_sum = allreduce_sum(prob_sum_local.clone())
        prob_cnt = allreduce_sum(prob_cnt_local.clone())
        eos_sum = allreduce_sum(eos_sum_local.clone())
        eos_cnt = allreduce_sum(eos_cnt_local.clone())

        # global中存在有效数据时才进行统计
        ent_mean = float((ent_sum / ent_cnt.clamp(min=1)).item()) if ent_cnt.item() > 0 else 0.0
        prob_mean = float((prob_sum / prob_cnt.clamp(min=1)).item()) if prob_cnt.item() > 0 else 0.0
        eos_mean = float((eos_sum / eos_cnt.clamp(min=1)).item()) if eos_cnt.item() > 0 else 0.0

        if use_critic:
            return_diff_local = (returns - values).masked_fill(~token_mask, 0.0).sum()  # (bs, s)
            return_local = (returns.masked_fill(~token_mask, 0.0)).sum()
            return_diff_var = allreduce_var(return_diff_local.clone())
            return_var = allreduce_var(return_local.clone())

            if ent_cnt.item() > 0:
                vf_explained_var = (1.0 - return_diff_var / (return_var + 1e-5)).detach().item()
            else:
                vf_explained_var = 0.0
            metrics.update({
                f'ability_infos/ability/vf_explained_var_{ab_name}': vf_explained_var,
            })

        metrics.update({
            f'ability_infos/ability/entropy_{ab_name}': ent_mean,
            f'ability_infos/ability/adv_eos_{ab_name}': eos_mean,
            f'ability_infos/ability/prob_mean_{ab_name}': prob_mean,
        })

    return DataProto.from_dict({'dummy': torch.ones(size=(1,))}, meta_info={'metrics': metrics})
