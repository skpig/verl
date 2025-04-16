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

from .ppo_actor import make_mini_step_dataloader, default_loss_fn


class MegatronPPOActor(BasePPOActor):

    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config)
        # for compatibility with FSDP
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.module = self.actor_module
        self.optimizer = self.actor_optimizer

        self.loss_fn = default_loss_fn

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

    def _preprocess_micro_batches(self, micro_batches: list[TensorDict]):
        """
        Preprocess a list of micro_batches into (forward_batch, input_shapes), 
        where forward_batch is a dictionary that is directly passed into the model,
        and input_shapes is a list of torch.Size that used for pp communication shapes
        """
        from flash_attn.bert_padding import unpad_input, index_first_axis, rearrange
        from verl.utils.megatron.sequence_parallel import pad_to_sequence_parallel
        from verl.utils.model import compute_position_id_with_mask
        from megatron import get_args

        args = get_args()
        cp_size = mpu.get_context_parallel_world_size()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        micro_batches_with_inputs = []
        input_shapes = []

        for micro_batch in micro_batches:

            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            input_ids = micro_batch['input_ids'].to(torch.int64)
            attention_mask = micro_batch['attention_mask'].to(torch.int64)
            response_length = micro_batch['responses'].size(-1)

            position_ids = compute_position_id_with_mask(attention_mask)
            # remove padding here
            input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(input_ids.unsqueeze(-1),
                                                                                    attention_mask=attention_mask)
            # TODO(zhangchi.usc1992): optimize this
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                  indices)  # (total_nnz, 1)

            total_s = input_ids_rmpad.shape[0]

            if cp_size > 1:
                # CP manager
                from mariana.data.text.transforms.text_microbatch_transform import ContextParallelProcessor

                cp_processor = ContextParallelProcessor(
                    pad_idx=1,  # TODO(zhangchi.usc1992) harcode for seed tokenizer for now
                    tp_size=mpu.get_tensor_model_parallel_world_size(),
                    cp_size=mpu.get_context_parallel_world_size(),
                    cp_rank=mpu.get_context_parallel_rank(),
                    hidden_size=-1)

                cp_forward_batch = {
                    'input_ids': input_ids_rmpad.squeeze(-1),
                    'seq_lens': cu_seqlens,
                    'cu_seqlens': cu_seqlens,
                    'loss_mask': torch.ones_like(input_ids_rmpad).squeeze(-1)
                }

                forward_batch = cp_processor(data=cp_forward_batch)
                # cp_manager'tensor stay in cpu originally, a liitle bit cheated; move them to cuda
                forward_batch['cu_seqlens_splited'] = forward_batch['cp_manager'].cu_seqlens_splited.cuda()
                # for non-bigOps
                # note that cp_manager.cu_seqlens_splited_rmpad is a list of tensor
                for i in range(len(forward_batch['cp_manager'].cu_seqlens_splited_rmpad)):
                    forward_batch['cp_manager'].cu_seqlens_splited_rmpad[i] = forward_batch[
                        'cp_manager'].cu_seqlens_splited_rmpad[i].cuda()
                # for bigOps
                # note that cp_manager.cu_seqlens_splited is a tensor
                forward_batch['cp_manager'].cu_seqlens_splited = forward_batch['cp_manager'].cu_seqlens_splited.cuda()
                forward_batch['cp_manager'].cu_seqlens = forward_batch['cp_manager'].cu_seqlens.cuda()
                forward_batch['cp_manager'].cu_seqlens_rmpad = forward_batch['cp_manager'].cu_seqlens_rmpad.cuda()
                forward_batch['input_ids'] = forward_batch['input_ids'].unsqueeze(0)
                forward_batch['position_ids'] = forward_batch['position_ids'].cuda()

                input_shape = torch.Size([forward_batch['input_ids'].shape[-1] // tp_size, 1, args.hidden_size])

            else:
                # pad to sequence parallel size
                input_ids_rmpad_padded = pad_to_sequence_parallel(input_ids_rmpad)  # (total_nnz + pad_size, 1)
                input_ids_rmpad_padded = input_ids_rmpad_padded.transpose(0, 1)  # (1, total_nnz + pad_size)
                position_ids_rmpad_padded = pad_to_sequence_parallel(position_ids_rmpad)  # (total_nnz + pad_size, 1)
                position_ids_rmpad_padded = position_ids_rmpad_padded.transpose(0, 1)  # (1, total_nnz + pad_size)

                # form a batch and feed into the model
                forward_batch = {
                    'input_ids': input_ids_rmpad_padded,
                    'cu_seqlens': cu_seqlens,
                    'max_s': max_seqlen_in_batch,
                    'total_s': total_s,
                    'position_ids': position_ids_rmpad_padded.squeeze(0),
                    'host_seqlens': cu_seqlens.cpu(),
                }

                input_shape = torch.Size([input_ids_rmpad_padded.shape[-1] // tp_size, 1, args.hidden_size])

            input_shapes.append(input_shape)

            forward_batch['indices'] = indices
            micro_batches_with_inputs.append({'forward_batch': forward_batch, 'micro_batch': micro_batch})

        return micro_batches_with_inputs, input_shapes

    def _forward_backward_batch(self, batches: list[TensorDict], forward_only=False):
        from megatron import get_args
        from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
        from dist_attn import context_parallel

        num_microbatches = len(batches)

        pp_size = mpu.get_pipeline_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        cp_group = mpu.get_context_parallel_group()

        assert num_microbatches >= pp_size, \
            f'num_microbatches must be greater or equal to pp size. Got {num_microbatches=}, {pp_size=}'

        args = get_args()
        assert not args.scale_loss_in_gradient
        # TODO: select pipeline strategy here. Force to use Any1F1B to support any num_micro_batches
        forward_backward_func = get_forward_backward_func(pipeline_strategy='Any1F1B')

        def loss_func(output, micro_batch):
            if forward_only:
                return 1.0, output

            log_prob = output['log_probs']
            assert output['entropy'] is None

            seqlen = output['seqlen']
            policy_loss, stats = self.loss_fn(self.config, micro_data=micro_batch, full_entropy=None, log_prob=log_prob)
            stats['actor/seqlen'] = seqlen

            # correctly scale policy_loss
            loss = policy_loss * (len(micro_batch) / self.config.ppo_mini_batch_size)
            #(zhangchi.usc1992) we do this because in megatron pp schedule, the loss will be divided by num_microbatches
            loss = loss * num_microbatches
            return loss, stats

        def forward_step(batch_iter, model):
            micro_batch_with_inputs = next(batch_iter)
            micro_batch = micro_batch_with_inputs['micro_batch']
            forward_batch = micro_batch_with_inputs['forward_batch']

            input_ids = micro_batch['input_ids']
            responses = micro_batch['responses']
            indices = forward_batch['indices']
            total_s = forward_batch['total_s']
            response_length = responses.size(1)
            batch_size, sequence_length = input_ids.shape

            if cp_size > 1:
                # note that this is splitted in cp
                labels = forward_batch['shift_labels']
                cu_seqlens_splited = forward_batch.get("cu_seqlens_splited", None)
                max_s = forward_batch.get("max_s", None)
            else:
                input_ids_rmpad_padded = forward_batch['input_ids']
                labels = torch.roll(input_ids_rmpad_padded, shifts=-1, dims=1).squeeze(dim=0)  # (total_nnz + pad_size,)

            output = model(batch=forward_batch)

            if mpu.is_pipeline_last_stage():

                logits = output['logits'].squeeze(dim=1)  # (total_nnz_padded // cp, vocab_size // tp)
                logits = logits.float()

                # TODO(zhangchi.usc1992) switch to using accurate entropy computation
                # because -log_prob is not unbias estimator of entropy when there is off-policy
                if forward_only:
                    # Note (zhangchi.usc1992) that we have to compute entropy before log_prob as later will alter logits
                    entropy = vocab_parallel_entropy(logits)  # (total_nnz + pad_size // cp,)

                    if cp_size > 1:
                        # context parallel all gather its inputs inside the group
                        # hybrid data parallel leavesw the variables distributed

                        # cross_entropy doesn't all-gather 'loss' among CP group
                        # all-gather 'loss' among CP group; and re-arange to disable zig-zag string mode
                        entropy = context_parallel.get_context_parallel_output(entropy, cp_group, 0, cu_seqlens_splited,
                                                                               max_s, True, True)

                        # remove padding
                        entropy = context_parallel.rmpad_context_parallel_output(
                            x=entropy,
                            seqlens_in_batch=cu_seqlens_splited,
                            cp_group=mpu.get_context_parallel_group(),
                            extra_data=forward_batch.get('cp_manager'))
                    else:
                        entropy = entropy[:total_s]
                    entropy = pad_input(entropy.unsqueeze(-1), indices, batch_size, sequence_length).squeeze(-1)
                    entropy = entropy[:, -response_length - 1:-1]
                else:
                    entropy = None

                # vocab_parallel logprobs and vocab_parallel entropy
                # Note(zhangchi.usc1992) very important. This function will modify logits inplace
                log_prob = vocab_parallel_log_probs_from_logits(logits=logits,
                                                                labels=labels)  # (total_nnz + pad_size // cp,)

                if cp_size > 1:
                    loss_mask = forward_batch['loss_mask']
                    log_prob = context_parallel.get_context_parallel_output(log_prob, cp_group, 0, cu_seqlens_splited,
                                                                            max_s, True, True)

                    loss_mask = context_parallel.get_context_parallel_output(loss_mask, cp_group, 0, cu_seqlens_splited,
                                                                             max_s, True, True)

                    # remove padding
                    log_prob = torch.masked_select(log_prob, loss_mask.bool())
                else:
                    log_prob = log_prob[:total_s]

                # pad log_prob into full
                log_prob = pad_input(log_prob.unsqueeze(-1), indices, batch_size,
                                     sequence_length).squeeze(-1)  # (batch_size, sequence_length)
                log_prob = log_prob[:, -response_length - 1:-1]

                output = {'log_probs': log_prob, 'entropy': entropy, 'seqlen': total_s}

                return output, partial(loss_func, micro_batch=micro_batch)
            else:
                hidden_states = output['hidden_states']
                return hidden_states, partial(loss_func, micro_batch=micro_batch)

        from verl.utils.megatron.pipeline_parallel import make_batch_generator

        micro_batches_with_inputs, input_shapes = self._preprocess_micro_batches(batches)
        batch_generator = make_batch_generator(micro_batches_with_inputs, vpp_size=len(self.module))

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
        micro_batches, num_micro_batches, indices = rearrange_micro_batches(
            batch=batch,
            max_token_len=max_token_len,
            dp_group=mpu.get_data_parallel_group(),
            min_num_micro_batch=mpu.get_pipeline_model_parallel_world_size())

        with torch.no_grad():
            output = self._forward_backward_batch(micro_batches, forward_only=True)
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

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def update_policy(self, data: DataProto) -> Dict:
        """
        We have to make sure that data is identical in tp/pp region
        """

        # TODO: optimize this
        data = data.to(torch.cuda.current_device())
        dataloader = make_mini_step_dataloader(data,
                                               ppo_mini_batch_size=self.config.ppo_mini_batch_size,
                                               return_dataproto=False)

        num_mini_batches = len(dataloader)

        # TODO(zhangchi.usc1992): fix metrics. enable dp_overlap
        metrics = {}
        for batch_idx, mini_batch in enumerate(dataloader):
            self._optimizer_zero_grad()

            micro_batches, _, _ = rearrange_micro_batches(
                batch=mini_batch,
                max_token_len=self.config.ppo_max_token_len,
                dp_group=mpu.get_data_parallel_group(),
                min_num_micro_batch=mpu.get_pipeline_model_parallel_world_size())

            metric_micro_batch = self._forward_backward_batch(micro_batches, forward_only=False)
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
        return metrics
