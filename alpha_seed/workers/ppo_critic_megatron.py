"""
PPO critic with Megatron backend
"""

from megatron.schedules import get_forward_backward_func
from megatron.core import tensor_parallel
from megatron.core import parallel_state as mpu

from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from alpha_seed import core_algos
import torch.distributed
from tensordict import TensorDict

from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl import DataProto

from verl.trainer.ppo.critic import BasePPOCritic
from alpha_seed import core_algos

from flash_attn.bert_padding import pad_input
from functools import partial

from typing import Dict
import itertools


class MegatronPPOCritic(BasePPOCritic):

    def __init__(self, config, module, optimizer=None):
        super().__init__(config)
        # config should contain PPO logics
        self.module = module
        self.optimizer = optimizer

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
            metrics[f'critic/grad_norm'] = grad_norm

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

        def loss_func(output, micro_batch):
            if forward_only:
                return 1.0, output

            vpreds = output['values']

            responses = micro_batch['responses']
            attention_mask = micro_batch['attention_mask']
            values = micro_batch['values']
            returns = micro_batch['returns']
            overlong_mask = micro_batch.get('overlong_mask', None)

            response_length = responses.size(1)

            eos_mask = attention_mask[:, -response_length - 1:-1]

            # assert not torch.any(torch.isnan(vpreds)).item()

            cliprange_value_low = self.config.cliprange_value
            cliprange_value_high = self.config.cliprange_value
            if self.config.cliprange_value_low:
                cliprange_value_low = self.config.cliprange_value_low
            if self.config.cliprange_value_high:
                cliprange_value_high = self.config.cliprange_value_high
            vf_loss, vf_clipfrac, seq_vf = core_algos.compute_value_loss(
                vpreds=vpreds,
                values=values,
                returns=returns,
                eos_mask=eos_mask,
                cliprange_value_low=cliprange_value_low,
                cliprange_value_high=cliprange_value_high,
                overlong_mask=overlong_mask,
                loss_average_method=self.config.loss_average_method)

            # correctly scale policy_loss
            loss = vf_loss * (len(micro_batch) / self.config.ppo_mini_batch_size)
            #(zhangchi.usc1992) we do this because in megatron pp schedule, the loss will be divided by num_microbatches
            loss = loss * num_microbatches

            # return loss and stats
            stats = {
                'critic/vf_loss': vf_loss.detach().item(),
                'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
                'critic/tokens_per_micro_batch_update': attention_mask.sum().detach().item(),
                # 'critic/seqlen': seqlen,
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
                values = output['logits'].squeeze(dim=1)  # (total_nnz_padded,)
                values = values[:total_s]

                # pad log_prob into full
                values = pad_input(values.unsqueeze(-1), indices, batch_size,
                                   sequence_length).squeeze(-1)  # (batch_size, sequence_length)
                values = values[:, -response_length - 1:-1]

                output = {'values': values}

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

    def compute_values(self, data: DataProto) -> torch.Tensor:
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
                values = torch.cat([o['values'] for o in output], dim=0)  # (bs, seq_size)
                values = values.to(torch.float32)
            else:
                values = torch.empty(size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device)

            # broadcast across pp ranks
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                # now every model
                torch.distributed.broadcast(tensor=values,
                                            src=mpu.get_pipeline_model_parallel_last_rank(),
                                            group=mpu.get_pipeline_model_parallel_group(),
                                            async_op=False)

        # reorder log_probs and entropy
        indices = list(itertools.chain.from_iterable(indices))
        assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()} vs. {values.size()}"
        revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
        values = values[revert_indices]
        return values

    def update_critic(self, data: DataProto) -> Dict:
        """
        We have to make sure that data is identical in tp/pp region
        """

        # TODO: optimize this
        data = data.to(torch.cuda.current_device())

        select_keys = ['input_ids', 'responses', 'attention_mask', 'values', 'returns']
        if 'overlong_mask' in data.batch.keys():
            select_keys.append('overlong_mask')
        batch = data.select(batch_keys=select_keys).batch

        response = batch['responses']
        response_length = response.size(1)

        dataloader = batch.split(self.config.ppo_mini_batch_size)

        num_mini_batches = len(dataloader)

        # TODO(zhangchi.usc1992): fix metrics. enable dp_overlap
        seq_level_vf_lst = []
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

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # note that metrics is only available on last pp rank. We have to broadcast to every pp rank
            object_list = [None] * mpu.get_pipeline_model_parallel_world_size()
            object_list[-1] = metrics
            torch.distributed.broadcast_object_list(object_list=object_list,
                                                    src=mpu.get_pipeline_model_parallel_last_rank(),
                                                    group=mpu.get_pipeline_model_parallel_group())
            metrics = object_list[-1]  # take from last pp
        return None, metrics
