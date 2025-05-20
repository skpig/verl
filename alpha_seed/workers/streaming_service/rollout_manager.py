from typing import *
import asyncio
import copy
import ray
import torch
import numpy as np
import pandas as pd
import uuid
import time
import threading
from contextlib import suppress, contextmanager, nullcontext
from codetiming import Timer
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from transformers import AutoTokenizer
from hdfs_io import hexists, makedirs, hcopy
from alpha_seed.trainer.tensorcore_collect import tensorcore_collection
from alpha_seed.utils.observility.pretty_print import pprint
from alpha_seed.utils.functional import print_dataproto_size
from alpha_seed.workers.streaming_service.streaming_utils import record_xperf_metrics
from alpha_seed.workers.agents import select_handler_fn
from alpha_seed.workers.agents import TaskContext
from alpha_seed.workers.streaming_service.streaming_utils import pad, process_output


class SaveDataProtoFunc(Protocol):

    def __call__(self, data: DataProto, prefix: str = ""):
        ...


def _setup_standalone_comm(hybrid_wg, standalone_wg, role: str, port: int):
    hybrid_master_address = hybrid_wg.get_master_addr()
    standalone_master_address = standalone_wg.get_master_addr()

    hybrid_wg.setup_standalone_worker_comm(hybrid_master_address, standalone_master_address, str(port), role)
    standalone_wg.setup_standalone_worker_comm(hybrid_master_address, standalone_master_address, str(port), role)


def _update_standalone_weights(hybrid_wg, standalone_wg, standalone_role: str):
    actor_fut = hybrid_wg.update_standalone_worker(standalone_role)
    standalone_fut = standalone_wg.update_standalone_worker(standalone_role)
    ray.get(actor_fut)
    ray.get(standalone_fut)
    hybrid_wg.release_param_and_cache()


@contextmanager
def server_update_weights_ctx(server_wg):
    toggled = False
    try:
        server_wg.stop_server_before_weights_update()
        toggled = True
        yield
    finally:
        if toggled:
            server_wg.restart_server_after_weights_update()


@contextmanager
def hybrid_enable_server_ctx(hybrid_wg):
    toggled = False
    try:
        hybrid_wg.toggle_inference_server_state(sleep=False)
        toggled = True
        yield
    finally:
        if toggled:
            hybrid_wg.toggle_inference_server_state(sleep=True)


class RolloutManager:

    def __init__(
        self,
        config: OmegaConf,
        logger: Tracking,
        tokenizer: AutoTokenizer,
        port_bias: int = 8000,
    ):
        self.config = config
        self.logger = logger
        self.tokenizer = tokenizer

        self._initialized = False

        self._use_server = self.config.actor_rollout_ref.rollout.mode == "server"
        self._server_args = self.config.rollout_server

        # batch for last step's input batch for standalone
        # initialized with [] to avoid len(None) error
        self.standalone_batch: DataProto = []
        # batch for unfinished generating
        self.pending_batch: List[DataProto] = []

        self.standalone_gen_batch_output_resume: DataProto = None
        self.standalone_batch_resume: DataProto = None

        self.rollout_pool_warmup_step = self.config.actor_rollout_ref.rollout.rollout_pool.get("warmup_step", 0)

        self._hybrid_port = port_bias
        self._train_standalone_port = port_bias + 1
        self._val_standalone_port = port_bias + 2
        self._train_comm_port = port_bias + 3
        self._val_comm_port = port_bias + 4

        self._hybrid_wg_lock = threading.Lock()

    def _init_servers(self):
        if not self._use_server:
            return

        def start_background_loop(loop):
            asyncio.set_event_loop(loop)
            with suppress(asyncio.CancelledError):
                loop.run_forever()

        self.listen_loop = asyncio.new_event_loop()
        self.loop = asyncio.new_event_loop()
        self._listen_thread = threading.Thread(target=start_background_loop,
                                               args=(self.listen_loop,),
                                               daemon=True,
                                               name="server_listen")
        self._listen_thread.start()
        self._client_thread = threading.Thread(target=start_background_loop,
                                               args=(self.loop,),
                                               daemon=True,
                                               name="client")
        self._client_thread.start()

        self._server_futs = []
        self._server_start_events = []
        self._server_stop_events = []

        self._start_server(self.hybrid_wg, port=self._hybrid_port)

        if self.train_standalone_wg is not None:
            self._start_server(self.train_standalone_wg, port=self._train_standalone_port)
        if self.val_standalone_wg is not None:
            self._start_server(self.val_standalone_wg, port=self._val_standalone_port)

        async def wait_events(events):
            for event in events:
                await event.wait()

        asyncio.run_coroutine_threadsafe(wait_events(self._server_start_events), self.listen_loop).result()
        print("[rollout manager] servers started.")

    def stop_servers(self):
        if not self._use_server:
            return

        async def set_stop_events():
            for event in self._server_stop_events:
                event.set()

        asyncio.run_coroutine_threadsafe(set_stop_events(), self.listen_loop).result()
        for fut in self._server_futs:
            fut.result()
        print("[rollout manager] servers stopped.")

    def _init_standalone_comms(self):
        if self.train_standalone_wg is not None:
            _setup_standalone_comm(self.hybrid_wg,
                                   standalone_wg=self.train_standalone_wg,
                                   role='standalone_rollout',
                                   port=self._train_comm_port)
        if self.val_standalone_wg is not None:
            _setup_standalone_comm(self.hybrid_wg,
                                   standalone_wg=self.val_standalone_wg,
                                   role='standalone_validator',
                                   port=self._val_comm_port)

    def _init_eos_callback(self):
        remote_reward_style = []
        if self.config.trainer.use_remote_sandbox:
            remote_reward_style.append('code-sandbox')
        if self.config.trainer.use_remote_verifier:
            remote_reward_style.append('verifier_service')
        # add more reward style here that are going to be pipelined inside generation

        # set the eos_callback_fn of actor_rollout
        from alpha_seed.workers.xperf_rollout.component.query import Query

        def sandbox_callback_fn(query: Query):
            input_ids = query.input_ids + query.new_token_ids
            req_id = query.meta_info['uid']
            reward_model = query.meta_info['reward_model']
            reward_style = reward_model['style']
            ground_truth = reward_model['ground_truth']

            # note that the uid of padding dataproto should be None
            if reward_style in remote_reward_style and req_id is not None:
                # get the sandbox ray handler
                handler = ray.get_actor('remote_client')
                # this is non-blocking
                handler.add_requests.remote(req_id=req_id,
                                            input_ids=input_ids,
                                            ground_truth=ground_truth,
                                            reward_style=reward_style)

        if self.config.trainer.use_remote_sandbox:
            self.hybrid_wg.set_eos_callback_fn(sandbox_callback_fn)
            if self.train_standalone_wg is not None:
                self.train_standalone_wg.set_eos_callback_fn(sandbox_callback_fn)
            if self.val_standalone_wg is not None:
                self.val_standalone_wg.set_eos_callback_fn(sandbox_callback_fn)

    def initialize(self, hybrid_wg, rollout_pool=None, train_standalone_wg=None, val_standalone_wg=None):
        assert not self._initialized

        assert hybrid_wg is not None, "hybrid_wg must be provided"
        self.hybrid_wg = hybrid_wg
        self.rollout_pool = rollout_pool
        self.train_standalone_wg = train_standalone_wg
        self.val_standalone_wg = val_standalone_wg

        self._init_servers()
        self._init_standalone_comms()
        self._init_eos_callback()
        self._initialized = True

    def resume(self, remote_global_step_folder: str, load_dataproto_fn: Callable):
        # async resume
        # we only resume standalone buffer when the number of warmup steps is zero. This is reasonable because
        # 1. warmup and load old standalone is conflict. 2. when the prompt/response length changes, we can't load
        if self.rollout_pool_warmup_step == 0 and hexists(
                f"{remote_global_step_folder}/standalone_gen_batch_output.batch.pt"):
            pprint("resume standalone rollout.")
            self.standalone_gen_batch_output_resume = load_dataproto_fn(path=remote_global_step_folder,
                                                                        prefix="standalone_gen_batch_output")
            self.standalone_batch_resume = load_dataproto_fn(path=remote_global_step_folder, prefix="standalone_batch")

    def train_generate(
        self,
        batch: DataProto,
        step: int,
        save_dataproto_fn: SaveDataProtoFunc,
        is_warmup_step: bool,
        metrics: Dict = None,
    ) -> DataProto:
        """
        :param batch: current training input batch
        :param step: current training step
        :param save_dataproto_fn: function for saving dataproto in hdfs for resuming
        :param is_warmup_step: if True, wait all running batch to finish
        :param metrics: metrics dict
        :return: batch to be train after generation
        """
        step_start = time.time()
        assert self._initialized

        assert (prompt_len := batch.batch['input_ids'].shape[1]
               ) == self.config.data.max_prompt_length, f"{prompt_len} != {self.config.data.max_prompt_length}"
        assert batch.batch['input_ids'].shape == batch.batch['attention_mask'].shape

        gen_batch, batch = self._prepare_gen_batch(batch, is_train=True)
        metrics = {} if metrics is None else metrics
        complete_ratio = (1.0 if is_warmup_step else self.config.actor_rollout_ref.rollout.get("complete_ratio", 1.0))
        gen_batch.meta_info.update({"complete_ratio": complete_ratio})

        if self._use_server:
            assert complete_ratio in (0.0, 1.0), "complete_ratio must be 1.0 or 0.0 for server mode"
            # hybrid server mode
            gen_batch.union(batch)
            ready_batch, self.pending_batch = self._train_server_gen(gen_batch,
                                                                     step=step,
                                                                     metrics=metrics,
                                                                     pending_batch=copy.copy(self.pending_batch),
                                                                     is_standalone=(complete_ratio == 0.0),
                                                                     is_warmup_step=is_warmup_step)
        else:
            # batch mode, hybrid + (optional) standalone
            ready_batch, self.standalone_batch, self.pending_batch = (self._train_batch_gen(
                batch,
                gen_batch,
                step=step,
                metrics=metrics,
                save_dataproto_fn=save_dataproto_fn,
                standalone_batch=copy.copy(self.standalone_batch),
                pending_batch=copy.copy(self.pending_batch),
            ))

        finished_num = len(ready_batch)
        incomplete_num = len(gen_batch) + len(self.pending_batch) - finished_num
        pprint(
            f"train step #{step} gen complete {finished_num}, incomplete {incomplete_num}, pending {len(self.pending_batch)}"
        )
        ray.get(self.rollout_pool.fill_rollout_pool.remote(ready_batch))
        if is_warmup_step:
            print(f"warmup gen step #{step}, elapsed: {time.time() - step_start}")
            return None

        num_bon = self.config.actor_rollout_ref.rollout.get("num_bon", 1)
        # get the training batch
        return_batch_size = (self.config.data.train_batch_size *
                             self.config.trainer.league_training_config.buffer_size * num_bon)
        train_batch = ray.get(self.rollout_pool.get_train_batch.remote(return_batch_size))
        batch = DataProto.concat(train_batch)

        if self._use_server:
            # maintain keys not handled in server mode
            batch.batch["prompts"] = batch.batch["input_ids"][:, :self.config.data.max_prompt_length]
            batch.batch["responses"] = batch.batch["input_ids"][:, self.config.data.max_prompt_length:]
            batch.pop(batch_keys=['is_finished'])

        batch.meta_info["generation_kwargs"] = self.config.actor_rollout_ref.rollout.train_generate_kwargs
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        if 'pixel_values' in batch.non_tensor_batch:
            batch.meta_info['global_img_token_num'] = [
                t.shape[0] if t is not None else 0 for t in batch.non_tensor_batch['pixel_values']
            ]

        if self.config.algorithm.force_append_eos:
            batch.batch["input_ids"][:, -1] = self.tokenizer.eos_token_id
            batch.batch["responses"][:, -1] = self.tokenizer.eos_token_id

        metrics["rollout/training_batch"] = len(batch)
        pprint(f"training batches {len(batch)}.")
        print(f"gen step #{step}, elapsed: {time.time() - step_start}")
        return batch

    def val_generate(self, batch: DataProto, step: int = 0, is_async: bool = False, metrics: Dict = None) -> DataProto:
        """
        :param batch: current training input batch
        :param step: current training step
        :return: batch to be train after generation, metrics
        """
        assert self._initialized
        gen_batch, batch = self._prepare_gen_batch(batch, is_train=False)
        metrics = {} if metrics is None else metrics

        if self._use_server:
            gen_batch.union(batch)
            gen_out_batch = self._val_server_gen(gen_batch, step=step, metrics=metrics, is_standalone=is_async)
        else:
            gen_out_batch = self._val_batch_gen(gen_batch, step=step, metrics=metrics, is_standalone=is_async)

        same_keys = batch.non_tensor_batch.keys() & gen_out_batch.non_tensor_batch.keys()
        batch.pop(non_tensor_batch_keys=list(same_keys))
        batch.union(gen_out_batch)
        return batch

    def _train_batch_gen(
        self,
        batch: DataProto,
        gen_batch: DataProto,
        step: int,
        metrics: Dict,
        standalone_batch: DataProto,
        pending_batch: List[DataProto],
        save_dataproto_fn: SaveDataProtoFunc,
    ):
        ready_batch = []

        with Timer(name="gen", logger=None) as timer:
            with tensorcore_collection():
                gen_batch_output = self.hybrid_wg.generate_sequences(gen_batch)
                # TODO: The following two lines should be memory view. However it's not. Let's remove it by removing all its dependency
                gen_batch_output.batch["prompts"] = gen_batch_output.batch["input_ids"][:, :self.config.data.
                                                                                        max_prompt_length]
                gen_batch_output.batch["responses"] = gen_batch_output.batch["input_ids"][:, self.config.data.
                                                                                          max_prompt_length:]
        metrics["timing/gen"] = timer.last
        metrics["rollout/hybrid_input_batch"] = len(gen_batch)
        metrics["memory/gen_max_allocated"] = gen_batch_output.meta_info["memory/gen_max_allocated"]
        metrics["memory/gen_max_reserved"] = gen_batch_output.meta_info["memory/gen_max_reserved"]
        metrics["timing/weight_binding"] = gen_batch_output.meta_info["timing/weight_binding"]

        # for debugging purpose only. we manually set all the attention_mask to 1 to
        # test the training performance under maximum workload.
        if self.config.trainer.set_fake_attention_mask:
            with Timer(name="fake_mask", logger=None) as timer:
                from verl.utils.model import create_random_mask

                total_length = (self.config.data.max_prompt_length + self.config.data.max_response_length)

                min_ratio_of_valid_token = self.config.trainer.fake_seqlen_ratio
                max_ratio_of_valid_token = self.config.trainer.fake_seqlen_ratio

                assert (self.config.trainer.fake_seqlen_ratio > (self.config.data.max_prompt_length + 1) / total_length)

                max_ratio_of_left_padding = 0
                attention_mask = create_random_mask(
                    gen_batch_output.batch["input_ids"],
                    max_ratio_of_valid_token=max_ratio_of_valid_token,
                    max_ratio_of_left_padding=max_ratio_of_left_padding,
                    min_ratio_of_valid_token=min_ratio_of_valid_token,
                )

                gen_batch_output.batch["attention_mask"] = attention_mask

                # force actor and critic stop updating weights because the data is fake
                self.config.actor_rollout_ref.actor.optim.lr = 0
                self.config.critic.optim.lr = 0

            metrics["timing/fake_mask"] = timer.last
            pprint(f"set fake attention mask")

        # only report metrics from one generation replica
        record_xperf_metrics(gen_batch_output, metrics, self.logger, step, prefix="hybrid")

        # stop hybrid rollout
        finished_num, ready_batch, pending_batch = process_output(
            batch,
            gen_batch_output,
            self.tokenizer,
            ready_batch,
            pending_batch,
            self.config,
        )
        pprint(f"stop hybrid rollout, completed_batch {finished_num}, incompleted_batch {len(batch) - finished_num} " +
               f"ready_queue {len(ready_batch)}, pending_queue {len(pending_batch)}.")
        metrics["rollout/hybrid_completed_batch"] = finished_num
        metrics["rollout/hybrid_incompleted_batch"] = len(batch) - finished_num

        # stop standalone rollout to update model
        if self.standalone_batch_resume is not None:
            standalone_batch = self.standalone_batch_resume
            self.standalone_batch_resume = None

        with Timer(name="async_gen", logger=None) as timer:
            finished_num = 0
            if len(standalone_batch) > 0:
                if not self.standalone_gen_batch_output_resume:
                    gen_batch_output = (self.train_standalone_wg.generate_sequences_get())
                else:
                    gen_batch_output = self.standalone_gen_batch_output_resume
                    self.standalone_gen_batch_output_resume = None
                if (self.config.trainer.save_freq > 0 and (step - 1) % self.config.trainer.save_freq == 0 and
                        step != 1):
                    print(f"step {step}, saving... standalone_gen_batch")
                    # save standalone_batch and gen_batch_output
                    save_dataproto_fn(
                        gen_batch_output,
                        prefix="standalone_gen_batch_output",
                    )
                    save_dataproto_fn(standalone_batch, prefix="standalone_batch")
                # only report metrics from one generation replica
                record_xperf_metrics(
                    gen_batch_output,
                    metrics,
                    self.logger,
                    step,
                    prefix="standalone",
                )
                finished_num, ready_batch, pending_batch = process_output(
                    standalone_batch,
                    gen_batch_output,
                    self.tokenizer,
                    ready_batch,
                    pending_batch,
                    self.config,
                    standalone=True,
                )
                pprint(
                    f"stop standalone rollout, completed_batch {finished_num}, incompleted_batch {len(standalone_batch) - finished_num}"
                    + f"ready_queue {len(ready_batch)}, pending_queue {len(pending_batch)}.")
            metrics["rollout/standalone_completed_batch"] = finished_num
            metrics["rollout/standalone_incompleted_batch"] = (len(standalone_batch) - finished_num)
        metrics["timing/async_gen"] = timer.last

        # update standalone rollout weights
        with Timer(name="update_standalone", logger=None) as timer:
            with self._hybrid_wg_lock:
                if self.train_standalone_wg is not None:
                    _update_standalone_weights(self.hybrid_wg, self.train_standalone_wg, "standalone_rollout")
        metrics["timing/update_standalone"] = timer.last

        # standalone generate (off policy)
        standalone_batch = []
        max_standalone_len = 0
        # sort by staleness, put items with larger off_policy_steps at the end of the list so they can be popped early
        pending_batch = sorted(pending_batch, key=lambda item: item.batch["off_policy_steps"].max().item())
        while (self.train_standalone_wg is not None and len(pending_batch) >= self.train_standalone_wg.world_size):
            for _ in range(self.train_standalone_wg.world_size):
                standalone_batch.append(pending_batch.pop())
                max_standalone_len = max(
                    max_standalone_len,
                    standalone_batch[-1].batch["attention_mask"].sum(-1),
                )
        for i in range(len(standalone_batch)):
            standalone_batch[i] = pad(standalone_batch[i], max_standalone_len, self.tokenizer)
        if len(standalone_batch) > 0:

            def make_interleave(batch: list, mp_size: int):
                assert len(batch) % mp_size == 0
                indices = sum(
                    [list(range(start, len(batch), mp_size)) for start in range(mp_size)],
                    [],
                )
                return [batch[i] for i in indices]

            # interleave standalone batch so that the data is still in FIFO order after dp compute dispatch
            standalone_batch = make_interleave(standalone_batch, self.train_standalone_wg.world_size)

            standalone_batch = DataProto.concat(standalone_batch)
            standalone_gen_batch = standalone_batch.pop(batch_keys=gen_batch.batch.keys())

            standalone_gen_batch.non_tensor_batch = standalone_batch.non_tensor_batch
            standalone_gen_batch.meta_info["generation_kwargs"] = (
                self.config.actor_rollout_ref.rollout.train_generate_kwargs)
            self.train_standalone_wg.generate_sequences_put(standalone_gen_batch)
            pprint(f"start standalone rollout, input batches {len(standalone_gen_batch)}.")
        metrics["rollout/standalone_input_batch"] = len(standalone_batch)
        return ready_batch, standalone_batch, pending_batch

    def _train_server_gen(
        self,
        gen_batch: DataProto,
        step: int,
        metrics: Dict,
        pending_batch: List[DataProto],
        is_standalone: bool,
        is_warmup_step: bool,
    ) -> Tuple[List[DataProto], List[DataProto]]:
        """streaming gen with server, only for train"""
        if is_standalone:
            with Timer(name="update_rollout_server", logger=None) as timer:
                with server_update_weights_ctx(self.train_standalone_wg):
                    with self._hybrid_wg_lock:
                        _update_standalone_weights(self.hybrid_wg, self.train_standalone_wg, "standalone_rollout")
            print(f"[INFO] {step} generate streaming[update weights and restart] {timer.last}")
            metrics["timing/update_rollout_server"] = timer.last
            server_port = self._train_standalone_port
        else:
            server_port = self._hybrid_port

        ready_batch = []
        handler_fn = select_handler_fn(self.config.rollout_server.handler)
        context = TaskContext(
            config=self.config,
            tokenizer=self.tokenizer,
            global_step=step,
            server_port=server_port,
        )

        async def submit_and_wait():
            # submit the training batch to the rollout server
            start = time.time()
            running_batch = []
            for item in gen_batch.chunk(len(gen_batch)):
                task = asyncio.create_task(handler_fn(item, context))
                running_batch.append(task)
            print(f"[INFO] {step} train generate server[submit], batch size: {len(gen_batch)}, {time.time() - start}")
            start = time.time()

            if is_warmup_step or (not is_standalone):
                # for warmup, wait all ready
                await asyncio.gather(*running_batch)

            print(
                f"[INFO] {step} train generate server[as_completed], batch size: {len(gen_batch)}, {time.time() - start}"
            )
            start = time.time()

            done, pending = await asyncio.wait(running_batch + pending_batch,
                                               timeout=0,
                                               return_when=asyncio.ALL_COMPLETED)
            return done, pending

        with nullcontext() if is_standalone else self._hybrid_wg_lock:
            with nullcontext() if is_standalone else hybrid_enable_server_ctx(self.hybrid_wg):
                done, pending = asyncio.run_coroutine_threadsafe(submit_and_wait(), self.loop).result()

        pending = list(pending)
        results = []
        for task in done:
            if task.exception():
                raise task.exception()
            else:
                results.append(task.result())

        ready_batch = results

        dummy_batch = DataProto(meta_info={"xperf_metrics": self._merge_xperf_metrics(ready_batch)})
        record_xperf_metrics(dummy_batch,
                             metrics,
                             self.logger,
                             step,
                             prefix="standalone" if is_standalone else "hybrid")
        return ready_batch, pending

    def _val_batch_gen(self, gen_batch: DataProto, step: int, metrics: Dict, is_standalone: bool) -> DataProto:
        if is_standalone:
            with self._hybrid_wg_lock:
                _update_standalone_weights(self.hybrid_wg, self.val_standalone_wg, "standalone_validator")
            validator_wg = self.val_standalone_wg
        else:
            validator_wg = self.hybrid_wg
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, validator_wg.world_size)
        # mark the paddig data uid to None
        for i in range(pad_size):
            gen_batch_padded.non_tensor_batch['uid'][-1 - i] = None
        with Timer(name="gen", logger=None) as timer:
            gen_out_batch_padded = validator_wg.generate_sequences(gen_batch_padded)
            gen_out_batch_padded.batch["prompts"] = gen_out_batch_padded.batch["input_ids"][:, :self.config.data.
                                                                                            max_prompt_length]
            gen_out_batch_padded.batch["responses"] = gen_out_batch_padded.batch["input_ids"][:, self.config.data.
                                                                                              max_prompt_length:]
        metrics['timing/gen'] = timer.last
        gen_out_batch = unpad_dataproto(gen_out_batch_padded, pad_size)
        record_xperf_metrics(gen_out_batch,
                             metrics,
                             self.logger,
                             step,
                             prefix="standalone" if is_standalone else "hybrid")
        return gen_out_batch

    def _val_server_gen(self, gen_batch: DataProto, step: int, metrics: Dict, is_standalone: bool) -> DataProto:
        if is_standalone:
            with Timer(name="update_rollout_server", logger=None) as timer:
                with server_update_weights_ctx(self.val_standalone_wg):
                    with self._hybrid_wg_lock:
                        _update_standalone_weights(self.hybrid_wg, self.val_standalone_wg, "standalone_validator")
            print(f"[INFO] {step} val generate server[update weights and restart] {timer.last}")
            metrics["timing/update_rollout_server"] = timer.last
            server_port = self._val_standalone_port
        else:
            server_port = self._hybrid_port

        ready_batch = []
        handler_fn = select_handler_fn(self.config.rollout_server.handler)
        context = TaskContext(
            config=self.config,
            tokenizer=self.tokenizer,
            global_step=step,
            server_port=server_port,
        )

        async def _submit_and_wait():
            # submit the training batch to the rollout server
            start = time.time()
            running_batch = []
            for item in gen_batch.chunk(len(gen_batch)):
                task = asyncio.create_task(handler_fn(item, context))
                running_batch.append(task)
            print(f"[INFO] {step} val generate streaming[submit], batch size: {len(gen_batch)}, {time.time() - start}")
            start = time.time()

            ready_batch = await asyncio.gather(*running_batch, return_exceptions=True)
            print(f"[INFO] {step} val gen server[as_completed], batch size: {len(gen_batch)}")
            return ready_batch

        with nullcontext() if is_standalone else self._hybrid_wg_lock:
            with nullcontext() if is_standalone else hybrid_enable_server_ctx(self.hybrid_wg):
                ready_batch = asyncio.run_coroutine_threadsafe(_submit_and_wait(), self.loop).result()

        for res in ready_batch:
            if isinstance(res, Exception):
                raise res

        gen_out = DataProto.concat(ready_batch)
        gen_out.meta_info['xperf_metrics'] = self._merge_xperf_metrics(ready_batch)
        record_xperf_metrics(gen_out, metrics, self.logger, step, prefix="standalone" if is_standalone else "hybrid")
        return gen_out

    def _merge_xperf_metrics(self, batch_list: List[DataProto]) -> Dict:
        """Merge per query xperf_metrics"""
        merged_metrics = dict()
        for item in batch_list:
            if "xperf_metrics" not in item.meta_info:
                continue
            query_metrics = item.meta_info['xperf_metrics']
            for key, val in query_metrics.items():
                if key not in merged_metrics:
                    merged_metrics[key] = val
                if type(val) != type(merged_metrics[key]):
                    continue
                merged_metrics[key] += val
        return merged_metrics

    def _prepare_gen_batch(self, batch: DataProto, is_train: bool):

        def _get_response_tensor(dtype, pad_val=-1):
            return torch.zeros(
                batch.batch["input_ids"].shape[0],
                self.config.data.max_response_length,
                dtype=dtype,
                device=batch.batch["input_ids"].device,
            ).fill_(pad_val)

        gen_batch_required_keys = ["input_ids", "attention_mask"]
        for key in [
                "rollout_log_probs",
                "probs_gt_threshold_num",
                "probs_lt_threshold_sum",
                "off_policy_steps",
        ]:
            if key not in batch:
                batch.batch[key] = _get_response_tensor(dtype=torch.bfloat16)
            gen_batch_required_keys.append(key)

        if is_train and self.config.algorithm.use_model_output_mask:
            if (key := "model_output_mask") not in batch:
                batch.batch[key] = _get_response_tensor(dtype=torch.int8, pad_val=0)
            gen_batch_required_keys.append(key)

        gen_batch = batch.pop(batch_keys=gen_batch_required_keys)
        gen_batch.non_tensor_batch = batch.non_tensor_batch
        sample_kwargs = (self.config.actor_rollout_ref.rollout.train_generate_kwargs
                         if is_train else self.config.actor_rollout_ref.rollout.val_generate_kwargs)
        sample_kwargs_dict = OmegaConf.to_container(sample_kwargs, resolve=True)
        gen_batch.meta_info.update({"generation_kwargs": sample_kwargs_dict})
        if not is_train:
            gen_batch.meta_info.update({
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'validate': True,
                'complete_ratio': 1.0,
            })

        return gen_batch, batch

    def _start_server(self, rollout_wg, port: int):
        server_started = asyncio.Event()
        server_stop = asyncio.Event()

        async def listen():
            from alpha_seed.workers.streaming_service.streaming_rollout_server import AsyncXPerfGPTRolloutServer
            server = AsyncXPerfGPTRolloutServer(self.config, self.tokenizer, port=port)
            async with server as rollout:
                rollout.attach_actors(rollout_wg)
                server_started.set()
                await server_stop.wait()
                print(f"[INFO] server stopped on port={port}")
            return server

        server_fut = asyncio.run_coroutine_threadsafe(listen(), self.listen_loop)
        self._server_futs.append(server_fut)
        self._server_start_events.append(server_started)
        self._server_stop_events.append(server_stop)
