"""
PPO actor with Megatron backend
"""

from megatron.schedules import get_forward_backward_func
from megatron.core import tensor_parallel
from megatron.core import parallel_state as mpu

from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from alpha_seed import core_algos
import torch.distributed
from tensordict import TensorDict

from verl.utils.py_functional import append_to_dict
from verl import DataProto

from verl.trainer.ppo.actor import BasePPOActor

from flash_attn.bert_padding import pad_input
from functools import partial

from typing import Dict
import itertools


class MegatronPPOActor(BasePPOActor):

    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config)
        # for compatibility with FSDP
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.module = self.actor_module
        self.optimizer = self.actor_optimizer

    def _optimizer_step(self, is_last_mini_batch):
        """
        We manually try grad sync on last mini_batch
        """
        from megatron import get_args, get_timers
        from janus.utils import JanusLaterOperationManager

        metrics = {}
        optimizers = self.optimizer

        assert len(optimizers) == 1
        optimizer = optimizers[0]

        args = get_args()
        timers = get_timers()
        # metrics[f'actor/lr'] = optimizer[0].param_groups[0]['lr']

        # Reduce gradients.
        optimizer.reduce_model_grads(args, timers)

        # Update parameters.
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        JanusLaterOperationManager.wait()
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
        timers('optimizer').stop()

        self.megatron_update_successful = update_successful

        assert self.megatron_update_successful

        if self.megatron_update_successful:
            # Gather params.
            optimizer.gather_model_params(args, timers)
            self.megatron_grad_norm = grad_norm
            self.megatron_num_zeros_in_grad = num_zeros_in_grad
            self.megatron_skipped_iter = 0

            # do param sync manually for weight update in last minibatch
            if args.overlap_dp_param_comm and is_last_mini_batch:
                dist_opt = optimizer
                for model_id, model in enumerate(dist_opt.models):
                    optimizer.try_param_sync(model, model_id, False)
                    optimizer.try_param_sync(model, model_id, True)
                if args.early_prefetch_dp_allgather:
                    for model_id, model in enumerate(dist_opt.models):
                        optimizer.try_param_sync(model, model_id, False)
        else:
            self.megatron_skipped_iter = 1
        if args.clip_grad > 0:
            metrics[f'actor/grad_norm'] = grad_norm

        return metrics

    def _optimizer_zero_grad(self):
        from megatron import get_args
        optimizers = self.optimizer

        assert len(optimizers) == 1
        args = get_args()
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
            training_models = self.module
            if not args.overlap_dp_param_comm:
                for partition in training_models:
                    partition.zero_grad_buffer()
        optimizers[0].zero_grad()

    def _forward_backward_batch(self, batches: list[TensorDict], response_length, forward_only=False):
        from megatron import get_args
        from flash_attn.bert_padding import unpad_input
        from verl.utils.megatron.sequence_parallel import pad_to_sequence_parallel
        from verl.utils.model import compute_position_id_with_mask
        from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits

        num_microbatches = len(batches)

        args = get_args()
        assert not args.scale_loss_in_gradient
        # TODO: select pipeline strategy here. Force to use Any1F1B to support any num_micro_batches
        forward_backward_func = get_forward_backward_func(pipeline_strategy='Any1F1B')

        use_rollout_log_probs = self.config.get("use_rollout_log_probs", False)

        def loss_func(output, micro_batch):
            if forward_only:
                return 1.0, output

            log_prob = output['log_probs']
            assert output['entropy'] is None

            # compute logprobs and entropy here. We only compute entropy when forward_only=True
            attention_mask = micro_batch['attention_mask']
            response_mask = attention_mask[:, -response_length:]

            if use_rollout_log_probs:
                # use ewma if use_rollout_log_probs: importance sampling by rollout_logp)rob, clip by old_log_prob
                use_ewma_loss = True
                old_log_prob = micro_batch['rollout_log_probs']
                ref_log_prob = micro_batch['old_log_probs']
            else:
                use_ewma_loss = self.config.use_ewma_loss
                old_log_prob = micro_batch['old_log_probs']
                ref_log_prob = micro_batch.get('ref_log_prob', None)

            old_log_prob = micro_batch['old_log_probs']
            advantages = micro_batch['advantages']
            ref_log_prob = micro_batch.get('ref_log_prob', None)
            upgo_advantages = micro_batch['upgo_advantages']
            overlong_mask = micro_batch.get('overlong_mask', None)

            clip_ratio = self.config.clip_ratio
            clip_ratio_low = clip_ratio
            clip_ratio_high = clip_ratio
            if self.config.clip_ratio_low:
                clip_ratio_low = self.config.clip_ratio_low
            if self.config.clip_ratio_high:
                clip_ratio_high = self.config.clip_ratio_high
            clip_ratio2 = self.config.clip_ratio2
            scale_pg_by_kl = self.config.scale_pg_by_kl
            scale_pg_by_local_kl = self.config.scale_pg_by_local_kl
            entropy_coeff = self.config.entropy_coeff
            upgo_loss_weight = self.config.upgo_loss_weight
            kl_loss_weight = self.config.kl_loss_weight
            lm_loss_weight = self.config.lm_loss_weight
            kl_penalty_type = self.config.kl_penalty
            loss_average_method = self.config.loss_average_method

            total_loss, pg_loss, upgo_loss, pg_clipfrac, pg_clipfrac_hi, pg_clipfrac_lo, pg_clipfrac2, ppo_kl, ppo_kl_sum = core_algos.compute_policy_loss(
                old_log_prob=old_log_prob,
                ref_log_prob=ref_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                upgo_advantages=upgo_advantages,
                eos_mask=response_mask,
                cliprange_low=clip_ratio_low,
                cliprange_high=clip_ratio_high,
                cliprange2=clip_ratio2,
                scale_pg_by_kl=scale_pg_by_kl,
                scale_pg_by_local_kl=scale_pg_by_local_kl,
                upgo_loss_weight=upgo_loss_weight,
                use_ewma_loss=use_ewma_loss,
                kl_penalty_type=kl_penalty_type,
                overlong_mask=overlong_mask,
                loss_average_method=loss_average_method)

            # if self.config.early_stop_by_kl != 0 and ppo_kl > self.config.early_stop_by_kl and batch_idx > 0:
            #     minibatch_early_stop = True
            #     break

            if kl_loss_weight > 0.0:
                kl_loss = core_algos.compute_kl_loss(log_prob, ref_log_prob, response_mask, kl_penalty_type)
            else:
                kl_loss = torch.zeros((), device=pg_loss.device)

            if lm_loss_weight > 0.0:
                eos_ids = micro_batch['eos_ids']
                raw_scores = micro_batch['token_level_scores']
                lm_loss = core_algos.compute_lm_loss(log_prob, raw_scores, eos_ids)
            else:
                lm_loss = torch.zeros((), device=pg_loss.device)

            policy_loss = total_loss - kl_loss_weight * kl_loss + lm_loss_weight * lm_loss

            # correctly scale policy_loss
            loss = policy_loss * (len(micro_batch) / self.config.ppo_mini_batch_size)
            #(zhangchi.usc1992) we do this because in megatron pp schedule, the loss will be divided by num_microbatches
            loss = loss * num_microbatches

            # return loss and stats
            stats = {
                'actor/pg_loss': pg_loss.detach().item(),
                'actor/upgo_loss': upgo_loss.detach().item(),
                'actor/kl_loss': kl_loss.detach().item(),
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                'actor/pg_clipfrac_hi': pg_clipfrac_hi.detach().item(),
                'actor/pg_clipfrac_lo': pg_clipfrac_lo.detach().item(),
                'actor/pg_clipfrac2': pg_clipfrac2.detach().item(),
                'actor/ppo_kl': ppo_kl.detach().item(),
                'actor/ppo_kl_sum': ppo_kl_sum.detach().item(),
                'actor/tokens_per_micro_batch_update': attention_mask.sum().detach().item(),
                # 'actor/seqlen': seqlen,
            }
            return loss, stats

        def forward_step(batch_iter, model):
            micro_batch = next(batch_iter)
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            input_ids = micro_batch['input_ids'].to(torch.int64)
            attention_mask = micro_batch['attention_mask'].to(torch.int64)

            position_ids = compute_position_id_with_mask(attention_mask)

            batch_size, sequence_length = input_ids.shape

            # remove padding here
            input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(input_ids.unsqueeze(-1),
                                                                                    attention_mask=attention_mask)
            # TODO(zhangchi.usc1992): optimize this
            position_ids_rmpad = unpad_input(position_ids.unsqueeze(-1),
                                             attention_mask=attention_mask)[0]  # (total_nnz, 1)

            total_s = input_ids_rmpad.shape[0]

            # pad to sequence parallel size
            input_ids_rmpad_padded = pad_to_sequence_parallel(input_ids_rmpad)  # (total_nnz + pad_size, 1)
            input_ids_rmpad_padded = input_ids_rmpad_padded.transpose(0, 1)  # (1, total_nnz + pad_size)
            position_ids_rmpad = position_ids_rmpad.transpose(0, 1)

            # form a batch and feed into the model
            forward_batch = {
                'input_ids': input_ids_rmpad_padded,
                'cu_seqlens': cu_seqlens,
                'max_s': max_seqlen_in_batch,
                'total_s': total_s,
                'position_ids': position_ids_rmpad,
                'host_seqlens': cu_seqlens.cpu(),
                'padded_seq_len':
                    input_ids_rmpad_padded.shape[-1]  # how should we pass this?
            }

            output = model(batch=forward_batch)

            if mpu.is_pipeline_last_stage():
                labels = torch.roll(input_ids_rmpad_padded, shifts=-1, dims=1).squeeze(dim=0)  # (total_nnz + pad_size,)
                logits = output['logits'].squeeze(dim=1)  # (total_nnz_padded, vocab_size // tp)

                # TODO(zhangchi.usc1992) optimize accurate entropy computation
                # if forward_only:
                #     entropy = vocab_parallel_entropy(logits)  # (total_nnz + pad_size,)
                #     entropy = entropy[:total_s]
                #     entropy = pad_input(entropy.unsqueeze(-1), indices, batch_size, sequence_length).squeeze(-1)
                #     entropy = entropy[:, -response_length - 1:-1]
                # else:
                #     entropy = None

                # vocab_parallel logprobs and vocab_parallel entropy
                # Note(zhangchi.usc1992) very important. This function will modify logits inplace
                log_prob = vocab_parallel_log_probs_from_logits(logits=logits, labels=labels)  # (total_nnz + pad_size,)
                log_prob = log_prob[:total_s]

                # pad log_prob into full
                log_prob = pad_input(log_prob.unsqueeze(-1), indices, batch_size,
                                     sequence_length).squeeze(-1)  # (batch_size, sequence_length)
                log_prob = log_prob[:, -response_length - 1:-1]

                # TODO(zhangchi.usc1992) switch to using accurate entropy computation
                if forward_only:
                    entropy = -log_prob
                else:
                    entropy = None

                output = {'log_probs': log_prob, 'entropy': entropy}

                return output, partial(loss_func, micro_batch=micro_batch)
            else:
                hidden_states = output['hidden_states']
                return hidden_states, partial(loss_func, micro_batch=micro_batch)

        from verl.utils.megatron.pipeline_parallel import make_batch_generator, compute_transformers_input_shapes
        batch_generator = make_batch_generator(batches, vpp_size=len(self.module))

        input_shapes = compute_transformers_input_shapes(
            batches,
            meta_info={
                'sequence_parallel': True,
                'hidden_size':
                    args.hidden_size  # bad! we assume this is universal
            })

        assert args.use_distributed_optimizer

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.module,
            dtype=args.params_dtype,
            tensor_shape=(1, 1, 1),  # useless if input_shapes is passed
            input_shapes=input_shapes,
            input_shapes_unpad=None,  # set to None for now as there is no pp
            sequence_parallel=True,
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
            num_microbatches=num_microbatches,
            grad_scaler=None if forward_only else self.optimizer[0].scale_loss,
            grad_sync_func=self.optimizer[0].try_grad_sync
            if args.overlap_dp_grad_comm and args.use_distributed_optimizer else None,
            param_sync_func=self.optimizer[0].try_param_sync
            if args.overlap_dp_param_comm and args.use_distributed_optimizer else None,
            timers=None,
            deallocate_pipeline_outputs=args.deallocate_pipeline_outputs,
            forward_only=forward_only)

        return losses_reduced

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """data should contain input_ids and attention_mask"""

        # TODO: optimize this
        data = data.to(torch.cuda.current_device())

        data.batch = data.batch.contiguous()
        select_keys = ['responses', 'input_ids', 'attention_mask']
        batch = data.select(batch_keys=select_keys).batch
        input_ids = batch['input_ids']
        batch_size = input_ids.size(0)
        response = batch['responses']
        response_length = response.size(1)

        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
        assert use_dynamic_bsz
        max_token_len = data.meta_info['max_token_len']

        # perform dynamic bsz
        micro_batches, num_micro_batches, indices = rearrange_micro_batches(batch=batch,
                                                                            max_token_len=max_token_len,
                                                                            dp_group=mpu.get_data_parallel_group())

        with torch.no_grad():
            output = self._forward_backward_batch(micro_batches, response_length=response_length, forward_only=True)
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                log_probs = torch.cat([o['log_probs'] for o in output], dim=0)  # (bs, seq_size)
                log_probs = log_probs.to(torch.float32)

                entropy = torch.cat([o['entropy'] for o in output], dim=0)  # (bs, seq_size)
                entropy = entropy.to(torch.float32)
            else:
                log_probs = torch.empty(size=(batch_size, response_length),
                                        dtype=torch.float32,
                                        device=input_ids.device)

                entropy = torch.empty_like(log_probs)

            # broadcast across pp ranks
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                # now every model
                torch.distributed.broadcast(tensor=log_probs,
                                            src=mpu.get_pipeline_model_parallel_last_rank(),
                                            group=mpu.get_pipeline_model_parallel_group(),
                                            async_op=False)

                torch.distributed.broadcast(tensor=entropy,
                                            src=mpu.get_pipeline_model_parallel_last_rank(),
                                            group=mpu.get_pipeline_model_parallel_group(),
                                            async_op=False)

        # reorder log_probs and entropy
        indices = list(itertools.chain.from_iterable(indices))
        assert len(indices) == entropy.size(0), f"{len(indices)} vs. {entropy.size()} vs. {entropy.size()}"
        revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
        log_probs = log_probs[revert_indices]
        entropy = entropy[revert_indices]

        return entropy, log_probs

    def update_policy(self, data: DataProto) -> Dict:
        """
        We have to make sure that data is identical in tp/pp region
        """

        # TODO: optimize this
        data = data.to(torch.cuda.current_device())

        select_keys = ['responses', 'input_ids', 'attention_mask', 'old_log_probs', 'advantages', 'upgo_advantages']
        if 'ref_log_prob' in data.batch.keys():
            select_keys.append('ref_log_prob')
        if 'rollout_log_probs' in data.batch.keys():
            select_keys.append('rollout_log_probs')
        if 'overlong_mask' in data.batch.keys():
            select_keys.append('overlong_mask')
        if 'eos_ids' in data.batch.keys():
            select_keys.append('eos_ids')
        if 'token_level_scores' in data.batch.keys():
            select_keys.append('token_level_scores')
        batch = data.select(batch_keys=select_keys).batch

        response = batch['responses']
        response_length = response.size(1)

        dataloader = batch.split(self.config.ppo_mini_batch_size)

        num_mini_batches = len(dataloader)

        # TODO(zhangchi.usc1992): fix metrics. enable dp_overlap
        metrics = {}
        for batch_idx, mini_batch in enumerate(dataloader):
            self._optimizer_zero_grad()

            micro_batches, _, _ = rearrange_micro_batches(batch=mini_batch,
                                                          max_token_len=self.config.ppo_max_token_len,
                                                          dp_group=mpu.get_data_parallel_group())

            metric_micro_batch = self._forward_backward_batch(micro_batches,
                                                              response_length=response_length,
                                                              forward_only=False)
            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.
            optimizer_metrics = self._optimizer_step(is_last_mini_batch=batch_idx == num_mini_batches - 1)
            append_to_dict(metrics, optimizer_metrics)

        return metrics
