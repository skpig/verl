"""
PPO actor with Megatron backend
"""

from megatron.schedules import get_forward_backward_func
from megatron.core import tensor_parallel
from megatron.core import parallel_state as mpu

from verl.utils.megatron.tensor_parallel import vocab_parallel_compute_entropy_loss, vocab_parallel_log_probs_from_logits
from alpha_seed import core_algos
import torch.distributed
from tensordict import TensorDict

from verl.utils.py_functional import append_to_dict
from verl import DataProto

from verl.trainer.ppo.actor import BasePPOActor

from flash_attn.bert_padding import pad_input
from functools import partial

from typing import Dict


class MegatronPPOActor(BasePPOActor):

    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config)
        # config should contain PPO logics
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

    def _optimizer_step(self):
        from megatron import get_args, get_timers
        from janus.utils import JanusLaterOperationManager

        metrics = {}
        optimizers = self.actor_optimizer

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
        if self.megatron_update_successful:
            # Gather params.
            optimizer.gather_model_params(args, timers)
            self.megatron_grad_norm = grad_norm
            self.megatron_num_zeros_in_grad = num_zeros_in_grad
            self.megatron_skipped_iter = 0

            # do param sync manually for weight update in last minibatch

            if args.overlap_dp_param_comm and self.is_last_mini_batch:
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
        optimizers = self.actor_optimizer

        assert len(optimizers) == 1
        args = get_args()
        if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
            training_models = self.actor_module
            if not args.overlap_dp_param_comm:
                for partition in training_models:
                    partition.zero_grad_buffer()
        optimizers[0].zero_grad()

    def _lr_scheduler_step(self) -> None:
        schedulers = self.actor_optimizer_scheduler
        if self.megatron_update_successful:
            assert len(schedulers) == 1
            scheduler = schedulers[0]
            scheduler.step(increment=1)

    def _forward_backward_batch(self, data: TensorDict, forward_only=False):
        from megatron import get_args
        from flash_attn.bert_padding import unpad_input
        from verl.utils.megatron.sequence_parallel import pad_to_sequence_parallel
        from verl.utils.model import compute_position_id_with_mask

        from verl.utils.torch_functional import logprobs_from_logits, entropy_from_logits

        args = get_args()
        # TODO: select pipeline strategy here
        forward_backward_func = get_forward_backward_func(pipeline_strategy=None)

        # TODO: add sequence balancing and dynamic bsz here. For now, we simply split the data according to micro_bsz
        batch = data

        batches = batch.split(self.config.ppo_micro_batch_size)
        response_length = data['responses'].size(-1)

        def loss_func(output, data, meta_info):
            # compute logprobs and entropy here. We only compute entropy when forward_only=True
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            responses = data['input_ids'][:, -response_length:]

            # compute policy loss
            logits = output
            # TODO(zhangchi.usc1992): optimize this
            logits = logits[:, -response_length - 1:-1]
            # log_probs = vocab_parallel_log_probs_from_logits(logits, responses)
            log_prob = logprobs_from_logits(logits, responses)

            if forward_only:
                entropy = entropy_from_logits(logits)
                return 1.0, {'log_probs': log_prob, 'entropy': entropy}

            old_log_prob = data['old_log_probs']
            advantages = data['advantages']
            clip_ratio = meta_info['clip_ratio']
            ref_log_prob = data.get('ref_log_prob', None)
            # entropy_coeff = meta_info['entropy_coeff']
            upgo_advantages = data['upgo_advantages']
            overlong_mask = data.get('overlong_mask', None)

            clip_ratio = self.config.clip_ratio
            clip_ratio2 = self.config.clip_ratio2
            scale_pg_by_kl = self.config.scale_pg_by_kl
            scale_pg_by_local_kl = self.config.scale_pg_by_local_kl
            entropy_coeff = self.config.entropy_coeff
            upgo_loss_weight = self.config.upgo_loss_weight
            kl_loss_weight = self.config.kl_loss_weight
            lm_loss_weight = self.config.lm_loss_weight
            kl_penalty_type = self.config.kl_penalty

            total_loss, pg_loss, upgo_loss, pg_clipfrac, pg_clipfrac2, ppo_kl, ppo_kl_sum = core_algos.compute_policy_loss(
                old_log_prob=old_log_prob,
                ref_log_prob=ref_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                upgo_advantages=upgo_advantages,
                eos_mask=response_mask,
                cliprange=clip_ratio,
                cliprange2=clip_ratio2,
                scale_pg_by_kl=scale_pg_by_kl,
                scale_pg_by_local_kl=scale_pg_by_local_kl,
                upgo_loss_weight=upgo_loss_weight,
                use_ewma_loss=self.config.use_ewma_loss,
                kl_penalty_type=kl_penalty_type,
                overlong_mask=overlong_mask)

            # if self.config.early_stop_by_kl != 0 and ppo_kl > self.config.early_stop_by_kl and batch_idx > 0:
            #     minibatch_early_stop = True
            #     break

            if kl_loss_weight > 0.0:
                kl_loss = core_algos.compute_kl_loss(log_prob, ref_log_prob, response_mask, kl_penalty_type)
            else:
                kl_loss = torch.zeros((), device=pg_loss.device)

            if lm_loss_weight > 0.0:
                eos_ids = data['eos_ids']
                raw_scores = data['token_level_scores']
                lm_loss = core_algos.compute_lm_loss(log_prob, raw_scores, eos_ids)
            else:
                lm_loss = torch.zeros((), device=pg_loss.device)

            policy_loss = total_loss - kl_loss_weight * kl_loss + lm_loss_weight * lm_loss

            # return loss and stats
            stats = {
                'actor/pg_loss': pg_loss.detach().item(),
                'actor/upgo_loss': upgo_loss.detach().item(),
                'actor/kl_loss': kl_loss.detach().item(),
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                'actor/pg_clipfrac2': pg_clipfrac2.detach().item(),
                'actor/ppo_kl': ppo_kl.detach().item(),
                'actor/ppo_kl_sum': ppo_kl_sum.detach().item(),
                'actor/tokens_per_micro_batch_update': attention_mask.sum().detach().item(),
                # 'actor/seqlen': seqlen,
            }
            return policy_loss, stats

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
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
            input_ids_rmpad_padded = input_ids_rmpad_padded.transpose(0, 1)
            position_ids_rmpad = position_ids_rmpad.transpose(0, 1)

            # form a batch and feed into the model
            forward_batch = {
                'input_ids': input_ids_rmpad_padded,
                'cu_seqlens': cu_seqlens,
                'max_s': max_seqlen_in_batch,
                'total_s': total_s,
                'position_ids': position_ids_rmpad
            }

            output = model(batch=forward_batch)

            logits = output['logits']
            logits = tensor_parallel.gather_from_tensor_model_parallel_region(
                logits)  # (total_nnz_padded, 1, vocab_size)

            # from IPython import embed
            # if dist.get_rank() == 0:
            #     embed()
            # dist.barrier()

            # all gather from sequence parallel region. This makes replicate on each tp rank
            logits = logits[:total_s]  # (total_nnz_padded)

            logits = torch.squeeze(logits, dim=1)  # remove the artificial batch dimension
            # add removed padding back
            logits = pad_input(logits, indices, batch_size,
                               seqlen=sequence_length)  # (batch_size, sequence_length, vocab_size)

            # TODO(zhangchi.usc1992)
            # currently, we allgather from sequence parallel region of logits and remove padding from tp here
            # in fact, we can first perform reduction and then directly outputs logprobs and

            if forward_only:
                meta_info = None
            else:
                meta_info = {'clip_ratio': self.config.clip_ratio, 'entropy_coeff': self.config.entropy_coeff}
            return logits, partial(loss_func, data=batch, meta_info=meta_info)

        from verl.utils.megatron.pipeline_parallel import make_batch_generator, compute_transformers_input_shapes
        batch_generator = make_batch_generator(batches, vpp_size=len(self.actor_module))

        num_microbatches = len(batches)

        input_shapes = compute_transformers_input_shapes(
            batches,
            meta_info={
                'sequence_parallel': True,
                'hidden_size':
                    args.hidden_size  # bad! we assume this is universal
            })

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            dtype=args.params_dtype,
            tensor_shape=(args.seq_length, num_microbatches, args.hidden_size),
            input_shapes=input_shapes,
            input_shapes_unpad=None,  # set to None for now as there is no pp
            sequence_parallel=True,
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
            num_microbatches=num_microbatches,
            grad_scaler=None if forward_only else self.actor_optimizer[0].scale_loss,
            grad_sync_func=self.actor_optimizer[0].try_grad_sync
            if args.overlap_dp_grad_comm and args.use_distributed_optimizer else None,
            param_sync_func=self.actor_optimizer[0].try_param_sync
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
        with torch.no_grad():
            output = self._forward_backward_batch(batch, forward_only=True)
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
        if 'overlong_mask' in data.batch.keys():
            select_keys.append('overlong_mask')
        if 'eos_ids' in data.batch.keys():
            select_keys.append('eos_ids')
        if 'token_level_scores' in data.batch.keys():
            select_keys.append('token_level_scores')
        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        # TODO(zhangchi.usc1992): fix metrics. enable dp_overlap
        metrics = {}
        for batch_idx, mini_batch in enumerate(dataloader):
            self._optimizer_zero_grad()
            metric_micro_batch = self._forward_backward_batch(mini_batch, forward_only=False)
            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.
            optimizer_metrics = self._optimizer_step()
            append_to_dict(metrics, optimizer_metrics)

        return metrics
