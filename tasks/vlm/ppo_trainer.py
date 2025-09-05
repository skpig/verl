import ray
import wandb
from codetiming import Timer
from alpha_seed.trainer.ppo import RayPPOTrainer
from alpha_seed.workers.xperf_rollout.profiler.visualizer import visualize_standalone_usage
from alpha_seed.utils.observility.pretty_print import pprint
from alpha_seed.utils.functional import save_simple_train_data_to_hdfs
from mono_rl.utils.dataset.dist_data_util import load_image_data_dist, init_or_get_dist_data_manager, release_object


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
                    if self.use_critic:
                        batch, critic_future = self._compute_values(batch, critic_future, input_batch, metrics)

                    batch = self._compute_adv(batch, metrics, use_async_gen)

                    if self.global_step == 1:
                        print('Debugging', batch.batch)

                    input_batch = batch
                    if self.enable_actor_critic_spatial_mux:
                        input_batch = input_batch.repeat(2, interleave=False)

                    # update actor
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_step and self.global_step % self.config.trainer.actor_update_freq == 0:
                        actor_future = self.actor_rollout_wg.update_actor(input_batch)

                    # remove old_experts after policy update
                    if "old_experts" in batch.batch:
                        batch.batch.pop("old_experts")

                    # update critic
                    if self.use_critic:
                        critic_future = self.critic_wg.update_critic(input_batch)

                    self._update_actor(actor_future, batch, metrics)

                    self._update_critic(batch, critic_future, metrics)

                    # update ref ema
                    with Timer(name='update_ref_ema', logger=None) as timer:
                        self.ref_policy_wg.update_ref_ema()
                    metrics['timing/update_ref_ema'] = timer.last

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_step % self.config.trainer.test_freq == 0:
                        with Timer(name='testing', logger=None) as timer:
                            self.validation_manager.validate(is_async=self.use_standalone_validator,
                                                             global_step=self.global_step)
                        metrics['timing/testing'] = timer.last

                    with Timer(name='save_output_batch', logger=None) as timer:
                        save_simple_train_data_to_hdfs(batch, self.tokenizer, self.global_step,
                                                       self.config.trainer.default_hdfs_dir,
                                                       self.config.data.max_prompt_length)
                    metrics['timing/save_output_batch'] = timer.last

                    self.compute_metrics(batch, metrics)

                    self._save_checkpoint(metrics)

                    # collect sandbox client remaining results
                    if self.config.trainer.use_remote_sandbox:
                        remote_client = ray.get_actor('remote_client')
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