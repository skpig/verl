import itertools
import random
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
from alpha_seed.workers.actors.rollout_pool import RolloutPool
from contextlib import suppress, contextmanager, nullcontext
from codetiming import Timer
from omegaconf import OmegaConf, DictConfig
from ray import ObjectRef

from alpha_seed.utils.server_client import is_local_ray_instance
from alpha_seed.workers.streaming_service.auto_scaling import HorizontalAutoScaling, ScalePolicyConfig
from alpha_seed.workers.streaming_service.rollout_proxy import FixedReplicatedRayWorkerGroupAdapter, \
    RolloutWorkerGroupProxy
from alpha_seed.workers.streaming_service.streaming_rollout import RemoteAsyncXPerfGPTRollout
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, RayResourcePool
from verl.single_controller.ray.replicated_worker_group import ReplicatedRayWorkerGroup, ScalingRayWorkerGroup
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


def _setup_standalone_comm(hybrid_wg, standalone_wg, role: str):
    hybrid_master_address = hybrid_wg.get_master_addr()
    hybrid_master_port = hybrid_wg.get_master_free_port()
    standalone_master_address = standalone_wg.get_master_addr()

    master_fut = hybrid_wg.setup_standalone_worker_comm(hybrid_master_address, standalone_master_address,
                                                        str(hybrid_master_port), role)
    slave_fut = standalone_wg.setup_standalone_worker_comm(hybrid_master_address, standalone_master_address,
                                                           str(hybrid_master_port), role)
    # 这里同步，避免刚找出来的可用端口还没来得及建联就被占用了
    ray.get(master_fut)
    ray.get(slave_fut)


def _setup_standalone_comm_ucx(source_addresses_fut, standalone_wg, role: str):
    all_actor_addresses = ray.get(source_addresses_fut)
    print(f"all ucx source addresses: {all_actor_addresses}")
    source_address_iter = itertools.cycle(all_actor_addresses)
    addresses = [next(source_address_iter) for _ in range(standalone_wg.world_size)]
    standalone_wg.setup_as_client(role, addresses, all_actor_addresses)


def _update_standalone_weights(hybrid_wg, server_wg, server_role: str):
    # update the rollout server, do weights binding
    actor_fut = hybrid_wg.update_standalone_worker(server_role)
    standalone_fut = server_wg.update_standalone_worker(server_role)
    # note that we should wait for the weight sync to be completed to avoid standalone fail and driver continues
    ray.get(standalone_fut)
    server_wg.update_standalone_worker_end()
    ray.get(actor_fut)
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
        config: DictConfig,
        logger: Tracking,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.logger = logger
        self.tokenizer = tokenizer

        self._initialized = False

        self._use_server = self.config.actor_rollout_ref.rollout.mode == "server"
        self._rollout_elastic_enabled = self.config.streaming_rollout.elastic.enable
        self._server_args = self.config.rollout_server
        self.rollout_server_started = threading.Event()

        # batch for last step's input batch for standalone
        # initialized with [] to avoid len(None) error
        self.standalone_batch: DataProto = []
        # batch for unfinished generating
        self.pending_batch: List[DataProto] = []

        self.standalone_gen_batch_output_resume: DataProto = None
        self.standalone_batch_resume: DataProto = None

        self.rollout_pool_warmup_step = self.config.actor_rollout_ref.rollout.rollout_pool.get("warmup_step", 0)

        self.weights_communicator = self.config.actor_rollout_ref.rollout.weights_communicator
        self._source_addresses_fut = None

        # worker groups
        self.hybrid_wg = None
        self.rollout_pool = None
        self.train_standalone_wg = None
        self.val_standalone_wg = None
        self.hybrid_wg_proxy = None
        self.train_standalone_wg_proxy = None
        self.hybrid_val_wg_proxy = None
        self.val_wg_proxy = None

        # xperf openai servers
        self.hybrid_rollout_server = None
        self.standalone_rollout_server = None
        self.hybrid_validation_rollout_server = None
        self.validation_rollout_server = None

        self._hybrid_wg_lock = threading.Lock()

    def _init_servers(self):
        # server mode 下 start 各种 server
        if not self._use_server:
            return

        def start_background_loop(loop):
            asyncio.set_event_loop(loop)
            with suppress(asyncio.CancelledError):
                loop.run_forever()

        self.loop = asyncio.new_event_loop()
        self._client_thread = threading.Thread(target=start_background_loop,
                                               args=(self.loop,),
                                               daemon=True,
                                               name='gen-client-event-loop')
        self._client_thread.start()

        # start the openai server
        def start_server_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            import aiomonitor
            with aiomonitor.Monitor(loop, termui_port=11000, console_enabled=False):
                loop.run_until_complete(self._start_server())

        threading.Thread(target=start_server_thread, daemon=True, name='rollout-server-event-loop').start()

    def stop_servers(self):
        if not self._use_server:
            return

        if self.hybrid_wg_proxy is not None:
            self.hybrid_wg_proxy.stop()
        if self.train_standalone_wg_proxy is not None:
            self.train_standalone_wg_proxy.stop()
        if self.hybrid_val_wg_proxy is not None:
            self.hybrid_val_wg_proxy.stop()
        if self.val_wg_proxy is not None:
            self.val_wg_proxy.stop()

    def _init_standalone_comms(self):
        # 初始化参数更新的方式，其中elastic rollout必须只能用ucx
        # 其他的既可以nccl也可以ucx
        if self.weights_communicator == 'ucx':
            # setup actor as server to serve weights update request
            self._source_addresses_fut = self.hybrid_wg.setup_as_server()

            # setup standalone worker as client
            if self.train_standalone_wg is not None and not self._rollout_elastic_enabled:
                # elastic rollout由每个实例scale up后setup，这里跳过
                _setup_standalone_comm_ucx(self._source_addresses_fut, self.train_standalone_wg, "standalone_rollout")
            if self.val_standalone_wg is not None:
                _setup_standalone_comm_ucx(self._source_addresses_fut, self.val_standalone_wg, "standalone_validator")
        else:
            if self.train_standalone_wg is not None:
                _setup_standalone_comm(self.hybrid_wg,
                                       standalone_wg=self.train_standalone_wg,
                                       role='standalone_rollout')
            if self.val_standalone_wg is not None:
                _setup_standalone_comm(self.hybrid_wg,
                                       standalone_wg=self.val_standalone_wg,
                                       role='standalone_validator')

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

        if len(remote_reward_style) > 0:
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
            self.rollout_server_started.wait()
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
        RolloutPool.dynamic_call(self.rollout_pool, "fill_rollout_pool", ready_batch)

        if is_warmup_step:
            print(f"warmup gen step #{step}, elapsed: {time.time() - step_start}")
            return None
        num_bon = self.config.actor_rollout_ref.rollout.get("num_bon", 1)
        # get the training batch
        return_batch_size = (self.config.data.train_batch_size *
                             self.config.trainer.league_training_config.buffer_size * num_bon)
        train_batch = RolloutPool.dynamic_call(self.rollout_pool, "get_train_batch", return_batch_size)
        batch = DataProto.concat(train_batch)

        if self._use_server:
            # maintain keys not handled in server mode
            batch.batch["prompts"] = batch.batch["input_ids"][:, :self.config.data.max_prompt_length]
            batch.batch["responses"] = batch.batch["input_ids"][:, self.config.data.max_prompt_length:]
            batch.pop(batch_keys=['is_finished'])

        batch.meta_info["generation_kwargs"] = OmegaConf.to_container(
            self.config.actor_rollout_ref.rollout.train_generate_kwargs, resolve=True)
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
            standalone_gen_batch.meta_info["generation_kwargs"] = OmegaConf.to_container(
                self.config.actor_rollout_ref.rollout.train_generate_kwargs, resolve=True)
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

        ready_batch = []
        server_host = self.standalone_rollout_server.host if is_standalone else self.hybrid_rollout_server.host
        server_port = self.standalone_rollout_server.port if is_standalone else self.hybrid_rollout_server.port
        handler_fn = select_handler_fn(self.config.rollout_server.handler)
        context = TaskContext(
            config=self.config,
            tokenizer=self.tokenizer,
            global_step=step,
            server_host=server_host,
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
        self.rollout_server_started.wait()
        if is_standalone:
            with Timer(name="update_rollout_server", logger=None) as timer:
                with server_update_weights_ctx(self.val_standalone_wg):
                    with self._hybrid_wg_lock:
                        _update_standalone_weights(self.hybrid_wg, self.val_standalone_wg, "standalone_validator")
            print(f"[INFO] {step} val generate server[update weights and restart] {timer.last}")
            metrics["timing/update_rollout_server"] = timer.last

        ready_batch = []
        handler_fn = select_handler_fn(self.config.rollout_server.handler)
        server = self.validation_rollout_server if is_standalone else self.hybrid_validation_rollout_server
        context = TaskContext(
            config=self.config,
            tokenizer=self.tokenizer,
            global_step=step,
            server_host=server.host,
            server_port=server.port,
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
        # pack fields into extra_data (for tool calling...)
        if (key := "extra_data") not in gen_batch.non_tensor_batch:
            gen_batch.non_tensor_batch[key] = np.array([{} for _ in range(len(gen_batch))])
        if (key := 'agent_env') in gen_batch.non_tensor_batch:
            for i in range(len(gen_batch)):
                agent_env = gen_batch.non_tensor_batch[key][i]
                if isinstance(agent_env, np.ndarray):
                    agent_env = agent_env.tolist()
                gen_batch.non_tensor_batch['extra_data'][i].update({'agent_env': agent_env})

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

    async def _start_server(self):

        async def listen(request_manager_name):
            from alpha_seed.workers.streaming_service.streaming_rollout_server import AsyncXPerfGPTRolloutServer
            server = AsyncXPerfGPTRolloutServer(self.config, self.tokenizer, request_manager_name)
            await server.start_server()
            return server

        gen_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        poll_interval = self.config.streaming_rollout.proxy.poll_internal_seconds

        # train
        # create replicated worker group and rollout proxy
        self.hybrid_wg_proxy = RolloutWorkerGroupProxy(
            FixedReplicatedRayWorkerGroupAdapter(self.hybrid_wg, gen_tp_size, 'actor_rollout_ref'), [],
            'hybrid_rollout', poll_interval)
        self.hybrid_rollout_server = await listen('hybrid_rollout')

        if self.train_standalone_wg is not None:
            self.train_standalone_wg_proxy = RolloutWorkerGroupProxy(
                FixedReplicatedRayWorkerGroupAdapter(self.train_standalone_wg, gen_tp_size, 'standalone_rollout'), [],
                'standalone_rollout', poll_interval)
            self.standalone_rollout_server = await listen('standalone_rollout')

        if self._rollout_elastic_enabled:
            assert self.weights_communicator == 'ucx', 'weights_communicator must be "ucx" when using elastic rollout'
            assert self.train_standalone_wg is None, 'should not initialize train standalone when using elastic rollout'
            # 替换掉train_standalone_wg，接口一致
            self.train_standalone_wg = self._init_elastic_rollout()
            self.standalone_rollout_server = await listen('standalone_rollout')

        # validation on hybrid engine
        self.hybrid_val_wg_proxy = RolloutWorkerGroupProxy(
            FixedReplicatedRayWorkerGroupAdapter(self.hybrid_wg, gen_tp_size, 'actor_rollout_ref'), [],
            'hybrid_validation', poll_interval)
        self.hybrid_validation_rollout_server = await listen('hybrid_validation')

        # standalone validation
        if self.val_standalone_wg is not None:
            self.val_wg_proxy = RolloutWorkerGroupProxy(
                FixedReplicatedRayWorkerGroupAdapter(self.val_standalone_wg, gen_tp_size, 'standalone_validator'), [],
                'validation', poll_interval)
            self.validation_rollout_server = await listen('validation')

        self.rollout_server_started.set()
        print("[rollout manager] servers started.")

        # 不能让这个event loop结束，因为每个oai server里面还有一个自己的server_task
        await asyncio.Future()

    def _init_elastic_rollout(self):
        poll_interval = self.config.streaming_rollout.proxy.poll_internal_seconds
        # 每个rollout_worker用1个gpu，每个gpu对应1个rank
        res_shape = [self.config.streaming_rollout.n_gpus_per_node] * self.config.streaming_rollout.nnodes
        tp_size = sum(res_shape)

        # 依赖actor的address作为ucx endpoint
        hybrid_rollout_addrs = ray.get(self._source_addresses_fut)
        print(f"all ucx source addresses: {hybrid_rollout_addrs}")

        # 按照rollout的dp world进行切分，一定是正好切够的
        assert len(hybrid_rollout_addrs) % tp_size == 0, \
            f"hybrid rollout world size({len(hybrid_rollout_addrs)}) should be divisible by dp_world_size({tp_size})"
        # 按TP维度切片，将相同tp rank的放一起
        # shape: (tp_size, dp_size)
        # [[TP0, ...] [TP1, ...] [TP2, ...] [TP3, ...]]
        hybrid_dp_size = len(hybrid_rollout_addrs) // tp_size
        hybrid_rollout_addresses_tp_groups = [
            [hybrid_rollout_addrs[i * tp_size + j] for i in range(hybrid_dp_size)] for j in range(tp_size)
        ]

        # streaming+elastic的standalone rollout初始化
        # rollout worker 初始化方式定义
        rollout_cls = RayClassWithInitArgs(cls=RemoteAsyncXPerfGPTRollout,
                                           config=self.config.actor_rollout_ref,
                                           role="rollout_server")

        def model_init(wg: Union[RayWorkerGroup, RemoteAsyncXPerfGPTRollout]) -> List[ObjectRef]:
            # 需要确保actor初始化好才能setup rollout作为client去获取参数
            # note(lixiang): 1 实际上这里引用了外层的actor_rollout_init_fut不是太好，但因为构造回调只能定义在这里，所以先这么写
            # note(lixiang): 2
            #  setup_rollout 的执行顺序为，init_model然后setup_as_client
            #  合理的顺序为init_model，然后wait actor_rollout_init_fut，最后setup_as_client，
            #  这里位了简化，暂时不拆开setup_rollout
            ray.get(wg.init_model())

            # 从dp group里随机选一组地址以负载平衡，更大规模的负载平衡再换别的分配的方式
            random_dp_rank = random.randint(0, len(hybrid_rollout_addresses_tp_groups[0]) - 1)
            dp_groups = [group[random_dp_rank] for group in hybrid_rollout_addresses_tp_groups]
            return wg.setup_as_client('standalone_rollout_server', dp_groups, hybrid_rollout_addrs)

        # 用代理类表示这个wg，里面会兼容ppo这里用到的方法
        # 每个rollout_worker用1个gpu
        res_shape = [self.config.streaming_rollout.n_gpus_per_node] * self.config.streaming_rollout.nnodes
        stable_pool_name = self.config.streaming_rollout.elastic.stable_pool_name
        elastic_pool_name = self.config.streaming_rollout.elastic.elastic_pool_name
        stable_pool_res = [stable_pool_name]
        elastic_pool_res = [elastic_pool_name]
        if is_local_ray_instance():
            # local ray的debug trial因为没有那些role的定义，所以这里不额外指定调度
            stable_pool_res = []
            elastic_pool_res = []

        stable_res_pool = RayResourcePool(
            process_on_nodes=res_shape,
            use_gpu=True,
            max_colocate_count=1,
            additional_resources=stable_pool_res,  # 用于表示调度到指定资源池
            name_prefix=f'standalone_rollout_stable_')
        elastic_res_pool = RayResourcePool(
            process_on_nodes=res_shape,
            use_gpu=True,
            max_colocate_count=1,
            additional_resources=elastic_pool_res,  # 用于表示调度到指定资源池
            name_prefix=f'standalone_rollout_elastic_')
        # 稳定池跑最小副本数
        min_guaranteed_replicas = ReplicatedRayWorkerGroup(rollout_cls, stable_res_pool, model_init)
        # 弹性池跑伸缩副本
        best_effort_replicas = ReplicatedRayWorkerGroup(rollout_cls, elastic_res_pool, model_init)
        # 两个副本组合并一起组成伸缩组
        replicas = ScalingRayWorkerGroup(min_guaranteed_replicas, best_effort_replicas)
        # 封装给worker group的接口代理
        rollout_proxy = RolloutWorkerGroupProxy(replicas, hybrid_rollout_addrs, 'standalone_rollout', poll_interval)

        # 拉起最小副本数
        model_init_futs = min_guaranteed_replicas.scale_up(self.config.streaming_rollout.elastic.min_replicas)
        # 等actor创建好可以接收请求，不是等init_model完成
        min_guaranteed_replicas.wait_for_alive(self.config.streaming_rollout.elastic.min_replicas)

        # initialize rollout horizontal auto scaling control handle
        elastic_pool_name = self.config.streaming_rollout.elastic.elastic_pool_name
        policy = ScalePolicyConfig(
            scale_up_threshold=self.config.streaming_rollout.elastic.scale_up_threshold,
            scale_down_threshold=self.config.streaming_rollout.elastic.scale_down_threshold,
            scale_up_wait=self.config.streaming_rollout.elastic.scale_up_wait,
            scale_down_wait=self.config.streaming_rollout.elastic.scale_down_wait,
            min_replicas=self.config.streaming_rollout.elastic.min_replicas,
            max_replicas=self.config.streaming_rollout.elastic.max_replicas,
        )
        self.standalone_rollout_ha = HorizontalAutoScaling(replicas,
                                                           elastic_pool_name,
                                                           policy,
                                                           metric_source=rollout_proxy)
        ray.get(model_init_futs)
        return rollout_proxy
