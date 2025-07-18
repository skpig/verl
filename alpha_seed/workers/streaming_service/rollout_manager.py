import itertools
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

from alpha_seed.utils.debug.aiomonitor import get_aiomonitor_cls
from alpha_seed.workers.actors.rollout_pool import RolloutPool
from contextlib import suppress, contextmanager, nullcontext
from codetiming import Timer
from omegaconf import OmegaConf, DictConfig
from ray import ObjectRef

from alpha_seed.workers.agents.executor import RayActorExecutor, ExecutorBase, LocalExecutor
from alpha_seed.workers.streaming_service.elastic_rollout_manager import ElasticRolloutManager
from alpha_seed.workers.streaming_service.rollout_proxy import FixedReplicatedRayWorkerGroupAdapter, \
    RolloutWorkerGroupProxy, BalancedRolloutWorkerGroupProxy, CombinedRayWorkerGroupAdapter
from alpha_seed.workers.streaming_service.streaming_rollout import RemoteAsyncXPerfGPTRollout
from mono_rl import DataProto
from mono_rl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, RayResourcePool
from mono_rl.single_controller.ray.replicated_worker_group import ReplicatedRayWorkerGroup, ScalingRayWorkerGroup
from verl.utils.tracking import Tracking
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from transformers import AutoTokenizer
from hdfs_io import hexists, makedirs, hcopy
from alpha_seed.utils.observility.pretty_print import pprint
from alpha_seed.utils.functional import print_dataproto_size
from alpha_seed.workers.streaming_service.streaming_utils import record_xperf_metrics
from alpha_seed.workers.agents.handlers import select_handler_fn
from alpha_seed.workers.agents.handlers import TaskContext
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


def _setup_standalone_comm_ucx(all_actor_addresses, standalone_wg, role: str):
    print(f"all ucx source addresses: {all_actor_addresses}")
    source_address_iter = itertools.cycle(all_actor_addresses)
    addresses = [next(source_address_iter) for _ in range(standalone_wg.world_size)]
    standalone_wg.setup_as_client(role, addresses, all_actor_addresses)
    # 及时纯stable standalone也setup as relay是为了在weights同步过程中等传输完了再返回，如果不是relay则直接返回，在后台自动传完
    standalone_wg.setup_as_relay()


def _update_standalone_weights(hybrid_wg,
                               standalone_wg,
                               standalone_role: str,
                               threadsafe_nccl_comm: threading.Event = None):
    actor_fut = hybrid_wg.update_standalone_worker(standalone_role)
    standalone_fut = standalone_wg.update_standalone_worker(standalone_role)
    # note that we should wait for the weight sync to be completed to avoid standalone fail and driver continues
    ray.get(standalone_fut)
    standalone_wg.update_standalone_worker_end()
    ray.get(actor_fut)
    # In the scenario of async val with multi-thread multi-stream nccl, set event to notify driver that weights have been updated, otherwise it might encounter the deadlock.
    # For the async gen in training, it is safe. Only the main thread is used.
    if threadsafe_nccl_comm is not None:
        threadsafe_nccl_comm.set()
    hybrid_wg.release_param_and_cache()


@contextmanager
def server_update_weights_ctx(server_wg):
    server_wg.stop_server_before_weights_update()
    yield
    server_wg.restart_server_after_weights_update()


@contextmanager
def hybrid_enable_server_ctx(hybrid_wg):
    hybrid_wg.toggle_inference_server_state(sleep=False)
    yield
    hybrid_wg.toggle_inference_server_state(sleep=True)


class RolloutManager:

    def __init__(
        self,
        config: DictConfig,
        logger: Tracking,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.config_dict = OmegaConf.to_container(self.config, resolve=True)
        self.logger = logger
        self.tokenizer = tokenizer
        self.train_client_executor: Optional[ExecutorBase] = None
        self.val_client_executor: Optional[ExecutorBase] = None

        self._initialized = False

        self._use_server = self.config.actor_rollout_ref.rollout.mode == "server"
        self._rollout_elastic_enabled = self.config.streaming_rollout.elastic.enable
        self._server_args = self.config.rollout_server
        self.rollout_server_started = threading.Event()
        self.threadsafe_nccl_comm = threading.Event()
        # batch for last step's input batch for standalone
        # initialized with [] to avoid len(None) error
        self.standalone_batch: DataProto = []
        # batch for unfinished generating
        self.pending_batch: List[asyncio.Task] = []

        self.standalone_gen_batch_output_resume: DataProto = None
        self.standalone_batch_resume: DataProto = None

        self.rollout_pool_warmup_step = self.config.actor_rollout_ref.rollout.rollout_pool.get("warmup_step", 0)

        self.weights_communicator = self.config.actor_rollout_ref.rollout.weights_communicator
        self.elastic_rollout_mgr = ElasticRolloutManager(self.config)

        # worker groups
        self.hybrid_wg = None
        self.rollout_pool = None
        self.train_standalone_wg = None
        self.val_standalone_wg = None
        self.train_rollout_proxy = None
        self.val_rollout_proxy = None
        self.train_replicas: CombinedRayWorkerGroupAdapter = None
        self.val_replicas: CombinedRayWorkerGroupAdapter = None

        # xperf openai servers
        self.train_rollout_server = None
        self.val_rollout_server = None
        self._hybrid_wg_lock = threading.Lock()

        # off_policy_step counter
        self._task_id_counter = 0
        self._task_id_to_task_and_step: Dict[int, Tuple[asyncio.Task, int]] = {}

    def _init_servers(self):
        # server mode 下 start 各种 server
        if not self._use_server:
            return

        def start_background_loop(loop):
            asyncio.set_event_loop(loop)
            use_aiomonitor = self.config.misc.aiomonitor.enable
            Monitor = get_aiomonitor_cls(use_aiomonitor)
            with suppress(asyncio.CancelledError):
                with Monitor(loop, termui_port=11001, console_enabled=False):
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
            use_aiomonitor = self.config.misc.aiomonitor.enable
            Monitor = get_aiomonitor_cls(use_aiomonitor)
            with Monitor(loop, termui_port=11000, console_enabled=False):
                loop.run_until_complete(self._start_server())

        threading.Thread(target=start_server_thread, daemon=True, name='rollout-server-event-loop').start()

    def stop_servers(self):
        if not self._use_server:
            return

        if self.train_rollout_proxy is not None:
            self.train_rollout_proxy.stop()
        if self.val_rollout_proxy is not None:
            self.val_rollout_proxy.stop()

    def _init_standalone_comms(self):
        # 初始化参数更新的方式，其中elastic rollout必须只能用ucx
        # 其他的既可以nccl也可以ucx
        if self.weights_communicator == 'ucx':
            # setup actor as server to serve weights update request
            self._source_addresses_fut = self.hybrid_wg.setup_as_server()
            self.elastic_rollout_mgr.set_hybrid_rollout_address(self._source_addresses_fut)

            # setup standalone worker as client
            if self.train_standalone_wg is not None and not self._rollout_elastic_enabled:
                # elastic rollout由每个实例scale up后setup，这里跳过
                _setup_standalone_comm_ucx(ray.get(self._source_addresses_fut), self.train_standalone_wg,
                                           "standalone_rollout")
            if self.val_standalone_wg is not None:
                _setup_standalone_comm_ucx(ray.get(self._source_addresses_fut), self.val_standalone_wg,
                                           "standalone_validator")
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
        if self.config.trainer.use_remote_sandbox:
            remote_reward_style.append('aider')
        if self.config.trainer.use_remote_verifier:
            remote_reward_style.append('verifier_service')
            remote_reward_style.append('gaokao_verifier_service')
        if self.config.trainer.use_remote_swe_sandbox:
            remote_reward_style.append('swe_repair_verifier')
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

    def _init_client_executor(self):
        if not self._use_server:
            return
        max_workers = self.config.rollout_server.agent.max_workers
        worker_max_concurrency = self.config.rollout_server.agent.worker_max_concurrency
        executor_cls = self.config.rollout_server.agent.executor_class
        self.rollout_server_started.wait()
        ExecutorCls = None
        if executor_cls == "LocalExecutor":
            ExecutorCls = LocalExecutor
        elif executor_cls == "RayActorExecutor":
            ExecutorCls = RayActorExecutor
        else:
            raise ValueError(f"Unsupported executor class: {executor_cls}")
        self.train_client_executor = ExecutorCls("train", self.config, self.tokenizer, self.train_rollout_server.host,
                                                 self.train_rollout_server.port)
        self.val_client_executor = ExecutorCls("val", self.config, self.tokenizer, self.val_rollout_server.host,
                                               self.val_rollout_server.port)

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
        self._init_client_executor()
        self._initialized = True

    def wait_nccl_comm_threadsafe(self):
        if self.val_standalone_wg is not None:
            # wait for standalone validator weights updated before proceeding
            self.threadsafe_nccl_comm.wait()
            self.threadsafe_nccl_comm.clear()

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
        :param is_warmup_step: if True, batch is used to warmup rollout pool
        :param metrics: metrics dict
        :return: batch to be train after generation
        """
        step_start = time.time()
        assert self._initialized

        assert (prompt_len := batch.batch['input_ids'].shape[1]
               ) == self.config.data.max_prompt_length, f"{prompt_len} != {self.config.data.max_prompt_length}"
        assert batch.batch['input_ids'].shape == batch.batch['attention_mask'].shape

        gen_batch, batch = self._prepare_gen_batch(batch, step, is_train=True)
        metrics = {} if metrics is None else metrics
        complete_ratio = self.config.actor_rollout_ref.rollout.get("complete_ratio", 1.0)
        if is_warmup_step:
            assert complete_ratio == 0.0, 'warmup_step should only be used for fully async, you should set rollout_pool.warmup_step=0 with complete_ratio>0!'
        gen_batch.meta_info.update({"complete_ratio": complete_ratio})

        if self._use_server:
            self.rollout_server_started.wait()
            self.train_rollout_proxy.step(step)
            # hybrid server mode
            gen_batch.union(batch)
            ready_batch, self.pending_batch = self._train_server_gen(gen_batch,
                                                                     step=step,
                                                                     metrics=metrics,
                                                                     pending_batch=copy.copy(self.pending_batch),
                                                                     is_warmup_step=is_warmup_step,
                                                                     complete_ratio=complete_ratio)
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
        if len(ready_batch) > 0:
            RolloutPool.dynamic_call(self.rollout_pool, "fill_rollout_pool", ready_batch)

        if is_warmup_step:
            print(f"warmup gen step #{step}, elapsed: {time.time() - step_start}")
            return None
        # get the training batch
        train_batch = RolloutPool.dynamic_call(self.rollout_pool, "get_train_batch")
        batch = DataProto.concat(train_batch)

        if (key := 'model_output_mask') in batch.batch:
            tensor = batch.batch[key]
            tensor[tensor < 0] = 0

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
        elif 'img_token_num' in batch.non_tensor_batch:
            batch.meta_info['global_img_token_num'] = list(batch.non_tensor_batch['img_token_num'])

        if self.config.algorithm.force_append_eos:
            batch.batch["input_ids"][:, -1] = self.tokenizer.eos_token_id
            batch.batch["responses"][:, -1] = self.tokenizer.eos_token_id

        metrics["rollout/training_batch"] = len(batch)

        # collect metrics from proxy on server mode
        if self._use_server:
            proxy_metrics = self.train_rollout_proxy.get_step_metrics()
            metrics.update(proxy_metrics)

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
        gen_batch, batch = self._prepare_gen_batch(batch, step, is_train=False)
        metrics = {} if metrics is None else metrics

        if self._use_server:
            gen_batch.union(batch)
            gen_out_batch = self._val_server_gen(gen_batch, step=step, metrics=metrics, is_standalone=is_async)
        else:
            gen_out_batch = self._val_batch_gen(gen_batch, step=step, metrics=metrics, is_standalone=is_async)

        same_keys = batch.non_tensor_batch.keys() & gen_out_batch.non_tensor_batch.keys()
        batch.pop(non_tensor_batch_keys=list(same_keys))
        batch.union(gen_out_batch)

        if self._use_server:
            # maintain keys not handled in server mode
            batch.batch["prompts"] = batch.batch["input_ids"][:, :self.config.data.max_prompt_length]
            batch.batch["responses"] = batch.batch["input_ids"][:, self.config.data.max_prompt_length:]
            batch.pop(batch_keys=['is_finished'])

        return batch

    async def _wait_max_off_policy_steps(self, step: int, metrics: Dict):
        max_off_policy_steps = self.config.actor_rollout_ref.rollout.get('max_off_policy_steps', None)
        if max_off_policy_steps is None:
            return
        should_wait = []
        to_pop_ids = []
        for task_id, (task, submit_step) in self._task_id_to_task_and_step.items():
            if (step - submit_step) >= max_off_policy_steps:
                should_wait.append(task)
                to_pop_ids.append(task_id)

        for task_id in to_pop_ids:
            del self._task_id_to_task_and_step[task_id]

        if len(should_wait) > 0:
            with Timer(name="gen", logger=None) as timer:
                await asyncio.wait(should_wait, return_when=asyncio.ALL_COMPLETED)

            elapsed = timer.last
            metrics['timing/wait_max_off_policy'] = elapsed
            print(
                f"[INFO]: step #{step} waited for {len(should_wait)} tasks reaching {max_off_policy_steps=}, {elapsed=:.3f}s"
            )

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

    def _train_server_gen(self, gen_batch: DataProto, step: int, metrics: Dict, pending_batch: List[DataProto],
                          is_warmup_step: bool, complete_ratio: float) -> Tuple[List[DataProto], List[DataProto]]:
        """streaming gen with server, only for train"""
        if self.train_standalone_wg is not None:
            with Timer(name="update_rollout_server", logger=None) as timer:
                self.update_standalone_server_weights(is_train=True)
            print(f"[INFO] {step} generate streaming[update weights and restart] {timer.last}")
            metrics["timing/update_rollout_server"] = timer.last

        global_handler = select_handler_fn(self.config.rollout_server.handler,
                                           external_lib=self.config.rollout_server.external_lib)
        context = TaskContext(
            config=self.config,
            global_step=step,
            server_host=self.train_rollout_server.host,
            server_port=self.train_rollout_server.port,
            is_train=True,
        )

        async def submit_and_wait():
            # submit the training batch to the rollout server
            start = time.time()
            running_batch = []

            for item in gen_batch.chunk(len(gen_batch)):
                handler = None
                if 'agent_handler' in item.non_tensor_batch and not pd.isna(item.non_tensor_batch['agent_handler'][0]):
                    handler = select_handler_fn(item.non_tensor_batch['agent_handler'][0],
                                                external_lib=self.config.rollout_server.external_lib)

                task_id = self._task_id_counter
                self._task_id_counter += 1
                task = asyncio.create_task(self.train_client_executor.submit(handler or global_handler, item, context))
                self._task_id_to_task_and_step[task_id] = (task, step)
                running_batch.append(task)
            await self._wait_max_off_policy_steps(step=step, metrics=metrics)

            print(f"[INFO] {step} train generate server[submit], batch size: {len(gen_batch)}, {time.time() - start}")
            start = time.time()

            if complete_ratio > 0:
                # for warmup, if fully_async, waiting for pool is handled in train_generate,
                # for non-fully_async, wait all ready and fill the pool here
                if is_warmup_step:
                    await asyncio.gather(*running_batch)

                print(
                    f"[INFO] {step} train generate server[as_completed], batch size: {len(gen_batch)}, {time.time() - start}"
                )
                start = time.time()

                finished = 0
                for future in asyncio.as_completed(running_batch):
                    # 获取已完成结果
                    await future
                    finished += 1

                    # 计算完成比例
                    if finished / len(running_batch) >= complete_ratio:
                        print(f"达到{complete_ratio*100}%完成率，提前退出")
                        break
                done, pending = await asyncio.wait(running_batch + pending_batch,
                                                   timeout=0,
                                                   return_when=asyncio.ALL_COMPLETED)
                print(f"[INFO] {len(done)=}, {len(pending)=}")
            else:
                all_batch = running_batch + pending_batch
                if is_warmup_step:
                    done, pending = [], all_batch
                else:
                    done, pending = await asyncio.wait(all_batch, timeout=0, return_when=asyncio.ALL_COMPLETED)
                    while len(done) < len(gen_batch) and len(pending) > 0:
                        new_done, pending = await asyncio.wait(list(pending), return_when=asyncio.FIRST_COMPLETED)
                        done.update(new_done)

                    done, pending = list(done), list(pending)
                    # For fully async, each step only fill the pool with the same batch size of gen_batch,
                    # If more than len(gen_batch) items are done, append them to pending and they can be used in next steps
                    prev_done_cnt = len(done)
                    prev_pending_cnt = len(pending)
                    if len(done) > len(gen_batch):
                        pending.extend(done[len(gen_batch):])
                        done = done[:len(gen_batch)]
                    print(
                        f"[INFO] fully async #{step} real_done:{prev_done_cnt} -> done:{len(done)}, real_pending:{prev_pending_cnt} -> {len(pending)}"
                    )

            done, pending = list(done), list(pending)
            return done, pending

        with self.enable_hybrid_server_gen_ctx(is_train=True):
            # print(f"[INFO] {step} generate streaming[update weights and restart] {timer.last}")
            # metrics["timing/update_rollout_server"] = timer.last
            done, pending = asyncio.run_coroutine_threadsafe(submit_and_wait(), self.loop).result()

        pending = list(pending)
        results = []
        for task in done:
            if task.exception():
                raise task.exception()
            else:
                task_result = task.result()
                if isinstance(task_result, DataProto):
                    results.append(task_result)
                elif isinstance(task_result, list):
                    results.extend(task_result)
                else:
                    raise ValueError(
                        f"AgentLoop only support DataProto or list[DataProto] at this moment, got {type(task_result)}")

        ready_batch = results
        finished_num = len(ready_batch)

        dummy_batch = DataProto(meta_info={"xperf_metrics": self._merge_xperf_metrics(ready_batch)})
        record_xperf_metrics(dummy_batch,
                             metrics,
                             self.logger,
                             step,
                             prefix="standalone" if complete_ratio == 0.0 else "hybrid")
        metrics["rollout/standalone_completed_batch"] = finished_num
        metrics["rollout/standalone_incompleted_batch"] = len(pending_batch) + len(gen_batch) - finished_num
        return ready_batch, pending

    def _val_batch_gen(self, gen_batch: DataProto, step: int, metrics: Dict, is_standalone: bool) -> DataProto:
        if is_standalone:
            with self._hybrid_wg_lock:
                _update_standalone_weights(self.hybrid_wg, self.val_standalone_wg, "standalone_validator",
                                           self.threadsafe_nccl_comm)
            validator_wg = self.val_standalone_wg
        else:
            self.threadsafe_nccl_comm.set()  # NOTE: let wait_nccl_comm_threadsafe at step 0 pass
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
        if self.val_standalone_wg is not None:
            with Timer(name="update_rollout_server", logger=None) as timer:
                self.update_standalone_server_weights(is_train=False)
            print(f"[INFO] {step} val generate server[update weights and restart] {timer.last}")
            metrics["timing/update_rollout_server"] = timer.last

        global_handler = select_handler_fn(self.config.rollout_server.handler,
                                           external_lib=self.config.rollout_server.external_lib)
        context = TaskContext(
            config=self.config,
            global_step=step,
            server_host=self.val_rollout_server.host,
            server_port=self.val_rollout_server.port,
            is_train=False,
        )

        async def _submit_and_wait():
            # submit the training batch to the rollout server
            start = time.time()
            running_batch = []
            for item in gen_batch.chunk(len(gen_batch)):
                task = asyncio.create_task(self.val_client_executor.submit(global_handler, item, context))
                running_batch.append(task)
            print(f"[INFO] {step} val generate streaming[submit], batch size: {len(gen_batch)}, {time.time() - start}")
            start = time.time()

            ready_batch = await asyncio.gather(*running_batch, return_exceptions=True)
            ready_batch = [task for task in ready_batch if task is not None]
            print(f"[INFO] {step} val gen server[as_completed], batch size: {len(gen_batch)}")
            return ready_batch

        with nullcontext() if is_standalone else self.enable_hybrid_server_gen_ctx(is_train=False):
            ready_batch = asyncio.run_coroutine_threadsafe(_submit_and_wait(), self.loop).result()

        # flatten ready_batch
        results = []
        for res in ready_batch:
            if isinstance(res, Exception):
                raise res
            if isinstance(res, DataProto):
                results.append(res)
            elif isinstance(res, list):
                results.extend(res)
            else:
                raise ValueError(
                    f"AgentLoop only support DataProto or list[DataProto] at this moment, got({type(results)})")
        ready_batch = results

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
                    merged_metrics[key] = copy.deepcopy(val)
                if type(val) != type(merged_metrics[key]):
                    continue
                merged_metrics[key] += val
        return merged_metrics

    def _prepare_gen_batch(self, batch: DataProto, step, is_train: bool):

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
                "off_policy_steps",
        ]:
            if key not in batch:
                batch.batch[key] = _get_response_tensor(dtype=torch.bfloat16)
            gen_batch_required_keys.append(key)

        if is_train and self.config.algorithm.use_model_output_mask:
            if (key := "model_output_mask") not in batch:
                batch.batch[key] = _get_response_tensor(dtype=torch.int8, pad_val=-1)
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
        for i in range(len(gen_batch)):
            gen_batch.non_tensor_batch['extra_data'][i].update({'config': self.config_dict})
        # More efficient for server-client interaction
        if (key := "prompt") not in gen_batch.non_tensor_batch and self._use_server:
            input_ids_list = gen_batch.batch["input_ids"].tolist()
            decoded = []
            pad_token_id = copy.deepcopy(self.tokenizer.pad_token_id)
            for ids in input_ids_list:
                # Remove only padding tokens (usually tokenizer.pad_token_id)
                filtered_ids = [id for id in ids if id != pad_token_id]
                text = self.tokenizer.decode(filtered_ids, skip_special_tokens=False)
                decoded.append(text)
            gen_batch.non_tensor_batch[key] = np.array(decoded, dtype=object)
        sample_kwargs = (self.config.actor_rollout_ref.rollout.train_generate_kwargs
                         if is_train else self.config.actor_rollout_ref.rollout.val_generate_kwargs)
        sample_kwargs_dict = OmegaConf.to_container(sample_kwargs, resolve=True)
        gen_batch.meta_info.update({
            "step": step,
            "generation_kwargs": sample_kwargs_dict,
        })
        if not is_train:
            gen_batch.meta_info.update({
                'eos_token_id': self.tokenizer.eos_token_id,  # noqa
                'pad_token_id': self.tokenizer.pad_token_id,  # noqa
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
        rollout_proxy_config = self.config.streaming_rollout.proxy
        lb_mode = rollout_proxy_config.lb_mode
        if lb_mode == "even-distribution":
            ProxyClass = RolloutWorkerGroupProxy
        elif lb_mode == "dynamic-balancing":
            ProxyClass = BalancedRolloutWorkerGroupProxy
        else:
            raise ValueError(f"config.streaming_rollout.proxy.lb_mode does not support {lb_mode=}, "
                             f"please choose from ['even-distribution', 'dynamic-balancing']")

        # train
        # create replicated worker group and rollout proxy
        hybrid_replica = FixedReplicatedRayWorkerGroupAdapter(self.hybrid_wg, gen_tp_size, 'actor_rollout_ref')

        if self._rollout_elastic_enabled:
            assert self.weights_communicator == 'ucx', 'weights_communicator must be "ucx" when using elastic rollout'
            assert self.train_standalone_wg is None, 'should not initialize train standalone when using elastic rollout'
            self.train_rollout_proxy, self.train_standalone_wg, self.train_replicas = self.elastic_rollout_mgr.init_elastic_rollout(
                hybrid_replica=hybrid_replica)
        else:
            train_intermittent_replicas = {'hybrid': hybrid_replica}
            train_persistent_replicas = {}
            if self.train_standalone_wg is not None:
                train_persistent_replicas['standalone'] = FixedReplicatedRayWorkerGroupAdapter(
                    self.train_standalone_wg, gen_tp_size, 'standalone_rollout')
            self.train_replicas = CombinedRayWorkerGroupAdapter(train_intermittent_replicas, train_persistent_replicas)
            self.train_rollout_proxy = ProxyClass(self.train_replicas, [], 'train_rollout', rollout_proxy_config)

        # Turn off hybrid for gen by default (i.e. train mode initially)
        self.train_replicas.set_replica_ready_state(name='hybrid', ready=False)
        self.train_rollout_server = await listen('train_rollout')

        # validation on hybrid engine
        val_intermittent_replicas = {
            'hybrid': FixedReplicatedRayWorkerGroupAdapter(self.hybrid_wg, gen_tp_size, 'actor_rollout_ref')
        }
        val_persistent_replicas = {}

        # standalone validation
        if self.val_standalone_wg is not None:
            val_persistent_replicas['standalone'] = FixedReplicatedRayWorkerGroupAdapter(
                self.val_standalone_wg, gen_tp_size, 'standalone_validator')

        self.val_replicas = CombinedRayWorkerGroupAdapter(val_intermittent_replicas, val_persistent_replicas)
        # Turn off hybrid for gen by default (i.e. train mode initially)
        self.val_replicas.set_replica_ready_state(name='hybrid', ready=False)

        self.val_rollout_proxy = ProxyClass(self.val_replicas, [], 'val_rollout', rollout_proxy_config)
        self.val_rollout_server = await listen('val_rollout')

        self.rollout_server_started.set()
        print("[rollout manager] servers started.")

        # 不能让这个event loop结束，因为每个oai server里面还有一个自己的server_task
        await asyncio.Future()

    @contextmanager
    def enable_hybrid_server_gen_ctx(self, is_train: bool):
        """Set hybrid server to be ready for gen"""
        replicas = self.train_replicas if is_train else self.val_replicas
        replicas.set_replica_ready_state(name='hybrid', ready=True)
        with self._hybrid_wg_lock:
            with hybrid_enable_server_ctx(self.hybrid_wg):
                yield
        replicas.set_replica_ready_state(name='hybrid', ready=False)
        self.hybrid_wg.release_running_queries()

    def update_standalone_server_weights(self, is_train: bool):
        standalone_wg = self.train_standalone_wg if is_train else self.val_standalone_wg
        standalone_role = "standalone_rollout" if is_train else "standalone_validator"

        with server_update_weights_ctx(standalone_wg):
            with self._hybrid_wg_lock:
                _update_standalone_weights(self.hybrid_wg, standalone_wg, standalone_role,
                                           self.threadsafe_nccl_comm if not is_train else None)
