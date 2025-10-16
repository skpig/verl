import os
import itertools
from typing import *
import asyncio
import copy
import ray
import torch
import numpy as np
import pandas as pd
import time
import threading
import json
import random
import os
import logging

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from alpha_seed.utils.debug.aiomonitor import get_aiomonitor_cls
from alpha_seed.utils.profile.timeline import Tracer
from alpha_seed.workers.actors.rollout_pool import RolloutPool
from contextlib import suppress, contextmanager, nullcontext
from codetiming import Timer
from omegaconf import OmegaConf, DictConfig
try:
    import aiofiles
except ImportError:
    aiofiles = None

from alpha_seed.workers.agents.executor import RayActorExecutor, ExecutorBase, LocalExecutor
from alpha_seed.workers.agents.metrics_collector import init_agent_metrics_collector
from alpha_seed.workers.agents.trajectory import init_agent_trajectory_collector
from alpha_seed.workers.streaming_service.elastic_rollout_manager import ElasticRolloutManager
from alpha_seed.workers.streaming_service.rollout_proxy import FixedReplicatedRayWorkerGroupAdapter, \
    RolloutWorkerGroupProxy, BalancedRolloutWorkerGroupProxy, CacheAwareBalancedRolloutWorkerGroupProxy, \
    CombinedRayWorkerGroupAdapter
from alpha_seed.workers.streaming_service.streaming_rollout import RemoteAsyncXPerfGPTRollout
from alpha_seed.workers.xperf_rollout.utils.base_weights_communicator import WeightsRankInfo
from mono_rl import DataProto
from mono_rl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, RayResourcePool
from mono_rl.single_controller.ray.replicated_worker_group import ReplicatedRayWorkerGroup, ScalingRayWorkerGroup
from verl.utils.tracking import Tracking
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from transformers import AutoTokenizer, AutoProcessor
from hdfs_io import hexists, makedirs, hcopy
from alpha_seed.utils.observility.pretty_print import pprint
from alpha_seed.utils.functional import print_dataproto_size
from alpha_seed.workers.streaming_service.streaming_utils import record_xperf_metrics, metrics_for_recommend_standalone_usage
from alpha_seed.workers.agents.handlers import select_handler_fn
from alpha_seed.workers.agents.handlers import TaskContext
from alpha_seed.workers.streaming_service.streaming_utils import pad, process_output, create_response_tensor
from alpha_seed.utils.reward_score import NON_AGENT_PLACE_HOLDER_SCORE
from alpha_seed.utils.reward_score.utils import Verifier

logger = logging.getLogger(__name__)


class SaveDataProtoFunc(Protocol):

    def __call__(self, data: DataProto, prefix: str = ""):
        ...


@contextmanager
def try_lock(lock_to_acquire: threading.Lock):
    got_the_lock = lock_to_acquire.acquire(blocking=False)

    try:
        yield got_the_lock
    finally:
        if got_the_lock:
            lock_to_acquire.release()


def _setup_standalone_comm(hybrid_wg, standalone_wg, role: str):
    hybrid_master_address = hybrid_wg.get_master_addr()
    hybrid_master_port = hybrid_wg.get_master_free_port()
    standalone_master_address = standalone_wg.get_master_addr()
    hybrid_master_address_ref = ray.put(hybrid_master_address)
    standalone_master_address_ref = ray.put(standalone_master_address)
    master_fut = hybrid_wg.setup_standalone_worker_comm(hybrid_master_address_ref, standalone_master_address_ref,
                                                        str(hybrid_master_port), role)
    slave_fut = standalone_wg.setup_standalone_worker_comm(hybrid_master_address_ref, standalone_master_address_ref,
                                                           str(hybrid_master_port), role)
    # 这里同步，避免刚找出来的可用端口还没来得及建联就被占用了
    ray.get(master_fut)
    ray.get(slave_fut)


def _setup_standalone_comm_ucx(all_actor_info: List[WeightsRankInfo], standalone_wg, role: str):
    print(f"all ucx source addresses: {all_actor_info}")
    tp_size = all_actor_info[-1].tp_rank + 1  # 按rank排序，最后一个rank一定是最大的tp_rank
    source_by_dp = [all_actor_info[i:i + tp_size] for i in range(0, len(all_actor_info), tp_size)]

    standalone_wg.setup_as_client(role, source_by_dp, all_actor_info)
    # 即使纯stable standalone也setup as relay是为了在weights同步过程中等传输完了再返回，如果不是relay则直接返回，在后台自动传完
    standalone_wg.setup_as_relay()


def _update_standalone_weights(hybrid_wg,
                               standalone_wg,
                               standalone_role: str,
                               threadsafe_nccl_comm: threading.Event = None,
                               need_hybrid_weights_update: bool = True,
                               offload_hybrid_mem: bool = True):
    actor_fut = hybrid_wg.update_standalone_worker(standalone_role, need_hybrid_weights_update, offload_hybrid_mem)
    standalone_fut = standalone_wg.update_standalone_worker(standalone_role)
    # note that we should wait for the weight sync to be completed to avoid standalone fail and driver continues
    ray.get(standalone_fut)
    standalone_wg.update_standalone_worker_end()
    ray.get(actor_fut)
    # In the scenario of async val with multi-thread multi-stream nccl, set event to notify driver that weights have been updated, otherwise it might encounter the deadlock.
    # For the async gen in training, it is safe. Only the main thread is used.
    if threadsafe_nccl_comm is not None:
        threadsafe_nccl_comm.set()


@contextmanager
def server_update_weights_ctx(server_wg, role: str):
    server_wg.stop_server_before_weights_update()
    print(f"[INFO] {role=} stopped server before weights update")
    yield
    print(f"[INFO] {role=} restart server after weights update")
    server_wg.restart_server_after_weights_update()


@contextmanager
def hybrid_enable_server_ctx(hybrid_wg, need_hybrid_weights_update: bool = True):
    hybrid_wg.toggle_inference_server_state(sleep=False, need_hybrid_weights_update=need_hybrid_weights_update)
    yield
    hybrid_wg.toggle_inference_server_state(sleep=True)


class RolloutManager:

    def __init__(self, config: DictConfig, logger: Tracking, tokenizer: AutoTokenizer, processor: AutoProcessor):
        self.config = config
        self.config_dict = OmegaConf.to_container(self.config, resolve=True)
        self.logger = logger
        self.tokenizer = tokenizer
        self.processor = processor

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

        # agent
        stable_pool_names = self.config.elastic.resource_pools.stable_pool_names
        stable_pool_name = stable_pool_names[0] if stable_pool_names else ''
        self.agent_metrics_collector = init_agent_metrics_collector(self.config, stable_pool_name)
        self.agent_trajectory_collector = init_agent_trajectory_collector(self.config, stable_pool_name)
        self.train_client_executor: Optional[ExecutorBase] = None
        self.val_client_executor: Optional[ExecutorBase] = None

        # xperf openai servers
        self.train_rollout_server = None
        self.val_rollout_server = None
        self._hybrid_wg_lock = threading.Lock()
        self._val_standalone_wg_lock = threading.Lock()  # protect replica status and wg weights
        self._save_executor = ThreadPoolExecutor(max_workers=2)
        self._save_task_pool = []

        # off_policy_step counter
        self._task_id_counter = 0
        self._task_id_to_task_and_step: Dict[int, Tuple[asyncio.Task, int]] = {}

        # sync param
        self._sync_param_once = False

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

        self.val_client_executor.stop()
        self.train_client_executor.stop()

    def _init_standalone_comms(self):
        # 初始化参数更新的方式，其中elastic rollout必须只能用ucx
        # 其他的既可以nccl也可以ucx
        if self.weights_communicator == 'ucx':
            # setup actor as server to serve weights update request
            self._source_info = self.hybrid_wg.setup_as_server()
            self.elastic_rollout_mgr.set_hybrid_rollout_source_info(self._source_info)

            # setup standalone worker as client
            if self.train_standalone_wg is not None and not self._rollout_elastic_enabled:
                # elastic rollout由每个实例scale up后setup，这里跳过
                _setup_standalone_comm_ucx(self._source_info, self.train_standalone_wg, "standalone_rollout")
            if self.val_standalone_wg is not None:
                _setup_standalone_comm_ucx(self._source_info, self.val_standalone_wg, "standalone_validator")
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
        # set the eos_callback_fn of actor_rollout
        from alpha_seed.workers.xperf_rollout.component.query import Query
        raw_config = self.config

        def sandbox_callback_fn(query: Query):
            reward_model = query.meta_info.get('reward_model')
            has_remote_verifier = True
            if reward_model is not None and 'style' in reward_model and reward_model['style'] != 'remote_service':
                verifier = Verifier.get_verifier(reward_model['style'], raw_config)
                if verifier is None or not verifier.is_remote():
                    has_remote_verifier = False
            else:
                has_remote_verifier = False
            req_id = query.meta_info['uid']
            if has_remote_verifier:
                input_ids = query.input_ids + query.new_token_ids
                reward_style = reward_model['style']
                ground_truth = reward_model['ground_truth']

                if verifier.is_remote() and req_id is not None:
                    # this is non-blocking
                    verifier.add_requests(req_id=req_id,
                                          input_ids=input_ids,
                                          ground_truth=ground_truth,
                                          reward_style=reward_style)

            # remote rm verifier
            rm_required_type = reward_model.get('rm_required_type', None)
            if rm_required_type is not None:
                verifier = Verifier.get_verifier(f'{rm_required_type}_service', raw_config)
                input_ids = query.input_ids + query.new_token_ids
                reward_model = query.meta_info['reward_model']
                ground_truth = reward_model['ground_truth']

                params = dict(
                    input_ids=input_ids,
                    ground_truth=ground_truth,
                    reward_style=f'{rm_required_type}_service',
                    call_rm_service=True,
                    reward_model=reward_model,
                )
                verifier.add_requests(req_id=req_id, **params)

        if self.config.reward_model.enable_eos_callback:
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
        self.train_client_executor = ExecutorCls("train", self.config, self.tokenizer, self.processor,
                                                 self.train_rollout_server.host, self.train_rollout_server.port,
                                                 "train_rollout", self.loop)
        self.val_client_executor = ExecutorCls("val", self.config, self.tokenizer, self.processor,
                                               self.val_rollout_server.host, self.val_rollout_server.port,
                                               "val_rollout", self.loop)

    def initialize(self, hybrid_wg, rollout_pool=None, train_standalone_wg=None, val_standalone_wg=None):
        assert not self._initialized

        assert hybrid_wg is not None, "hybrid_wg must be provided"
        self.hybrid_wg = hybrid_wg
        self.rollout_pool = rollout_pool
        self.train_standalone_wg = train_standalone_wg
        self.val_standalone_wg = val_standalone_wg

        self._init_standalone_comms()
        self._init_servers()
        self._init_eos_callback()
        self._init_client_executor()
        self._initialized = True

    async def wait_nccl_comm_threadsafe(self):
        if self.val_standalone_wg is not None:
            # wait for standalone validator weights updated before proceeding
            await asyncio.to_thread(self.threadsafe_nccl_comm.wait)
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

    def _normalize_done_tasks(self, done_tasks: List[asyncio.Task]) -> List[DataProto]:
        """Parse done tasks to ready batch"""
        ready_batch = []
        for task in done_tasks:
            if task.exception():
                raise task.exception()
            else:
                task_result = task.result()
                if isinstance(task_result, DataProto):
                    ready_batch.append(task_result)
                elif isinstance(task_result, list):
                    ready_batch.extend(task_result)
                else:
                    raise ValueError(
                        f"AgentLoop only support DataProto or list[DataProto] at this moment, got {type(task_result)}")
        return ready_batch

    def train_generate_fill(
        self,
        batch: DataProto,
        step: int,
        save_dataproto_fn: SaveDataProtoFunc,
        is_warmup_step: bool,
        metrics: Dict = None,
    ) -> None:
        """
        :param batch: current training input batch
        :param step: current training step
        :param save_dataproto_fn: function for saving dataproto in hdfs for resuming
        :param is_warmup_step: if True, batch is used to warmup rollout pool
        :param metrics: metrics dict
        :return: None
        """
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
            self.train_client_executor.set_global_step(step)
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
            fill_rollout_metrics = RolloutPool.dynamic_call(self.rollout_pool, "fill_rollout_pool", ready_batch, step)
            metrics.update(fill_rollout_metrics)

    def train_generate_fetch(
        self,
        step: int,
        is_warmup_step: bool,
        metrics: Dict = None,
    ) -> DataProto:
        """
        :param is_warmup_step: if True, wait all running batch to finish
        :param metrics: metrics dict
        :param step: current train step
        :return: batch to be train after generation
        """
        assert self._initialized

        metrics = {} if metrics is None else metrics

        if is_warmup_step:
            return None

        # get the training batch
        train_batch, get_batch_metrics = RolloutPool.dynamic_call(self.rollout_pool, "get_train_batch")
        metrics.update(get_batch_metrics)
        if len(train_batch) == 0:
            return None  # no batch ready, wait for next step
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
            agent_metrics = ray.get(self.agent_metrics_collector.get_current_metrics.remote(step))
            metrics.update(proxy_metrics)
            metrics.update(agent_metrics)

        if 'step' in batch.non_tensor_batch:
            min_gen_start_step = batch.non_tensor_batch['step'].min()
            mean_gen_start_step = batch.non_tensor_batch['step'].mean()
            metrics["rollout/max_off_policy_steps"] = step - min_gen_start_step
            metrics["rollout/mean_off_policy_steps"] = step - mean_gen_start_step

        pprint(f"training batches {len(batch)}.")
        return batch

    def update_swalm_rollout_agent_metrics(self, metrics: Dict, rollout_agent_tmp_metrics: List = None):
        if rollout_agent_tmp_metrics:
            finish_reason_dict = {
                "finish": 0,
                "finish_with_early_stop_learn": 0,
                "finish_with_prompt_truncated_learn": 0,
                "finish_with_response_truncated_learn": 0,
                "finish_with_max_turn_learn": 0,
                "stop_wtih_early_stop_drop": 0,
                "stop_wtih_prompt_truncated_drop": 0,
                "stop_wtih_response_truncated_drop": 0,
                "stop_wtih_max_turn_drop": 0,
                "stop_with_no_valid_response": 0,
                "stop_with_offpolicy_drop": 0,
                "stop_with_error_stop": 0,
            }
            all_iterations = []
            all_success_iterations = []
            for agent_metrics in rollout_agent_tmp_metrics:
                iterations = agent_metrics.get("all_turns_sum", 0)
                all_iterations.append(iterations)
                finish_reason = agent_metrics.get("finish_reason", "")
                if finish_reason:
                    assert finish_reason in finish_reason_dict, f"Unsupported finish_reason: {finish_reason} in {list(finish_reason_dict.keys())}"
                    finish_reason_dict[finish_reason] += 1
                    if "finish" in finish_reason:
                        all_success_iterations.append(iterations)
            metrics["rollout/agent/all_swalm_task"] = len(all_iterations)
            metrics["rollout/agent/all_iterations"] = sum(all_iterations) / max(1, len(all_iterations))
            metrics["rollout/agent/all_success_iterations"] = sum(all_success_iterations) / max(
                1, len(all_success_iterations))
            for finish_reason in finish_reason_dict:
                metrics[f"rollout/agent/{finish_reason}_ratio"] = finish_reason_dict[finish_reason] / max(
                    1, len(all_iterations))

    def train_generate(
        self,
        batch: DataProto,
        step: int,
        save_dataproto_fn: SaveDataProtoFunc,
        is_warmup_step: bool,
    ) -> Tuple[DataProto, dict]:
        """
        :param batch: current training input batch
        :param step: current training step
        :param save_dataproto_fn: function for saving dataproto in hdfs for resuming
        :param is_warmup_step: if True, wait all running batch to finish
        :param metrics: metrics dict
        :return: batch to be train after generation
        """
        metrics = {}
        if self.config.actor_rollout_ref.rollout.get("complete_ratio", 1.0) == 0.0:
            assert self.rollout_pool_warmup_step >= 1, "fully async must have warmup_step>0"

        step_start = time.time()
        self.train_generate_fill(batch, step, save_dataproto_fn, is_warmup_step, metrics)
        rollout_agent_tmp_metrics = metrics.pop("rollout/agent/tmp_agent_metrics", None)
        self.update_swalm_rollout_agent_metrics(metrics, rollout_agent_tmp_metrics)
        batch = self.train_generate_fetch(step, is_warmup_step, metrics)
        print(f"[INFO] gen step #{step}, elapsed: {time.time() - step_start}")
        return batch, metrics

    def train_generate_queued(self, train_batch_iter: Iterator[DataProto], step: int) -> Tuple[DataProto, dict]:
        """Train generation in queued style
        :param train_batch_iter: generator to get gen input batch
        :param step: current training step
        :return: batch to be trained after generation
        """
        metrics = {}

        queued_rollout_config = self.config.trainer.queued_rollout_config
        assert self._use_server, "train_generate_queued only valid for server mode"
        complete_ratio = self.config.actor_rollout_ref.rollout.get("complete_ratio", 1.0)
        assert complete_ratio == 0.0, "train_generate_queued only valid for complete_ratio=0.0"
        assert not self.config.actor_rollout_ref.rollout.rollout_pool.clear_rollout_pool, "train_generate_queued incompatible with clear_rollout_pool=True"

        def check_condition():
            wait_condition = queued_rollout_config.wait_condition
            if wait_condition == 'pool_ready_count':
                return_batch_size = self.config.data.train_batch_size * self.config.trainer.league_training_config.buffer_size * self.config.actor_rollout_ref.rollout.get(
                    "num_bon", 1)
                ready_count = RolloutPool.dynamic_call(self.rollout_pool, "get_ready_pool_size")
                return ready_count >= return_batch_size
            else:
                raise ValueError(f'Invalid wait_condition={wait_condition}')

        async def wait_for_pending():
            assert isinstance(self.pending_batch,
                              list), f"expect pending_batch to be list, got {type(self.pending_batch)}"
            done, pending = await asyncio.wait(self.pending_batch, return_when=asyncio.FIRST_COMPLETED)
            self.pending_batch = list(pending)
            return self._normalize_done_tasks(done)

        max_concurrency = queued_rollout_config.concurrency
        max_buffer_size = queued_rollout_config.max_buffer_size

        dataloader_time = 0
        rollout_agent_tmp_metrics = []
        if self.train_standalone_wg is not None:
            with Timer(name="update_rollout_server_queued", logger=None) as timer:
                xperf_metrics = self.update_standalone_server_weights(is_train=True, is_main_thread=True)
                dummy_batch = DataProto(meta_info={"xperf_metrics": xperf_metrics})
                record_xperf_metrics(dummy_batch, metrics, self.logger, step, prefix="standalone")
            print(f"[INFO] {step} train generate queued[update weights and restart] {timer.last}")
            metrics["timing/update_rollout_server_queued"] = timer.last

        if self.val_standalone_wg is not None:
            with try_lock(self._val_standalone_wg_lock) as locked:
                if locked and not self.val_replicas.is_replica_ready('standalone'):
                    self.train_replicas.set_replica_ready_state('val', ready=True)
                    with Timer(name="update_val_rollout_server_queued", logger=None) as timer:
                        _ = self.update_standalone_server_weights(is_train=False, is_main_thread=True)
                    print(f"[INFO] {step} train generate queued[update val weights and restart] {timer.last}")
                    metrics["timing/update_val_rollout_server_queued"] = timer.last

        with self.suppress_train_update_standalone():
            xperf_metrics = {}
            with self.enable_hybrid_server_gen_ctx(is_train=True, xperf_metrics=xperf_metrics):
                while True:
                    if len(self.pending_batch) >= max_concurrency:
                        # at maximum concurrency, check condition to start training
                        if check_condition():
                            break
                        ready_batch = asyncio.run_coroutine_threadsafe(wait_for_pending(), self.loop).result()
                        if self.config.data.get("enable_swalm_agent", False):
                            for batch in ready_batch:
                                agent_metrics = batch.meta_info.get("agent_metrics", {})
                                if agent_metrics:
                                    rollout_agent_tmp_metrics.append(agent_metrics)
                        RolloutPool.dynamic_call(self.rollout_pool, "fill_rollout_pool", ready_batch, step)
                    else:
                        # if ready count in the pool is greater than max_buffer_size, stop adding new gen batch
                        if max_buffer_size > 0:
                            ready_count = RolloutPool.dynamic_call(self.rollout_pool, "get_ready_pool_size")
                            if ready_count > max_buffer_size:
                                assert check_condition(
                                ), "ready_count > max_buffer_size but wait_condition not satisfied, please increase max_buffer_size"
                                break

                        # push new data to rollout server
                        with Timer(name='dataloader', logger=None) as timer:
                            batch: DataProto = next(train_batch_iter)
                        dataloader_time += timer.last

                        self.train_generate_fill(batch,
                                                 step=step,
                                                 save_dataproto_fn=None,
                                                 is_warmup_step=True,
                                                 metrics=metrics)
                        rollout_agent_tmp_metric = metrics.pop("rollout/agent/tmp_agent_metrics", None)
                        if rollout_agent_tmp_metric:
                            rollout_agent_tmp_metrics.extend(rollout_agent_tmp_metric)

            dummy_batch = DataProto(meta_info={"xperf_metrics": xperf_metrics})
            record_xperf_metrics(dummy_batch, metrics, self.logger, step, prefix="hybrid")
        metrics['timing/dataloader'] = dataloader_time
        self.update_swalm_rollout_agent_metrics(metrics, rollout_agent_tmp_metrics)
        batch: DataProto = self.train_generate_fetch(step, is_warmup_step=False, metrics=metrics)
        return batch, metrics

    async def val_generate_async(self,
                                 batch: DataProto,
                                 step: int = 0,
                                 is_async: bool = False) -> Tuple[DataProto, dict]:
        return await asyncio.to_thread(self.val_generate, batch, step, is_async)

    def val_generate(self, batch: DataProto, step: int = 0, is_async: bool = False) -> Tuple[DataProto, dict]:
        """
        :param batch: current training input batch
        :param step: current training step
        :return: batch to be train after generation, metrics
        """
        assert self._initialized
        gen_batch, batch = self._prepare_gen_batch(batch, step, is_train=False)
        metrics = {}

        if self._use_server:
            gen_batch.union(batch)
            self.rollout_server_started.wait()
            self.val_client_executor.set_global_step(step)
            with self.acquire_val_standalone_for_val_ctx(metrics):
                if self.config.rollout_server.evals.enable:
                    batch = self._val_server_gen_with_ckpt(gen_batch,
                                                           step=step,
                                                           metrics=metrics,
                                                           is_standalone=is_async)
                else:
                    batch = self._val_server_gen(gen_batch, step=step, metrics=metrics, is_standalone=is_async)
        else:
            gen_out_batch = self._val_batch_gen(gen_batch, step=step, metrics=metrics, is_standalone=is_async)
            same_keys = batch.non_tensor_batch.keys() & gen_out_batch.non_tensor_batch.keys()
            batch.pop(non_tensor_batch_keys=list(same_keys))
            batch.union(gen_out_batch)

        if self._use_server and not self.config.rollout_server.evals.enable:
            # maintain keys not handled in server mode
            batch.batch["prompts"] = batch.batch["input_ids"][:, :self.config.data.max_prompt_length]
            batch.batch["responses"] = batch.batch["input_ids"][:, self.config.data.max_prompt_length:]
            batch.pop(batch_keys=['is_finished'])

        return batch, metrics

    async def _wait_max_off_policy_steps(self, step: int, metrics: Dict):
        max_off_policy_steps = self.config.actor_rollout_ref.rollout.get('max_off_policy_steps', None)
        if max_off_policy_steps is None:
            self._task_id_to_task_and_step = {}
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

    def _merge_xperf_metrics(self, batch_list: List[DataProto], merged_metrics: Dict) -> Dict:
        """Merge per query xperf_metrics"""
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
            if len(self._save_task_pool) > 0:
                wait(self._save_task_pool, return_when=ALL_COMPLETED)
                for fut in self._save_task_pool:
                    fut.result()  # raise error
                self._save_task_pool.clear()
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
                    save_future1 = self._save_executor.submit(save_dataproto_fn,
                                                              data=gen_batch_output,
                                                              prefix="standalone_gen_batch_output")
                    save_future2 = self._save_executor.submit(save_dataproto_fn,
                                                              data=standalone_batch,
                                                              prefix="standalone_batch")
                    self._save_task_pool.extend([save_future1, save_future2])
                if not gen_batch_output.meta_info.get('xperf_metrics', None):
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
            with self.acquire_hybrid_wg(is_train=True):
                if self.train_standalone_wg is not None:
                    # TODO: redundant hybrid update
                    _update_standalone_weights(self.hybrid_wg, self.train_standalone_wg, "standalone_rollout", None,
                                               True, True)
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

        # hybrid train -> hybrid rollout weights update
        # hybrid rollout -> standalone rollout weights update
        if self.train_standalone_wg is not None and self.should_train_update_standalone():
            with Timer(name="update_rollout_server", logger=None) as timer:
                xperf_metrics = self.update_standalone_server_weights(is_train=True, is_main_thread=True)
            print(f"[INFO] {step} generate streaming[update weights and restart] {timer.last}")
            metrics["timing/update_rollout_server"] = timer.last

            if xperf_metrics is not None:
                dummy_batch = DataProto(meta_info={"xperf_metrics": xperf_metrics})
                record_xperf_metrics(dummy_batch, metrics, self.logger, step, prefix="standalone")

        if self.val_standalone_wg is not None and self.should_train_update_standalone():
            with try_lock(self._val_standalone_wg_lock) as locked:
                if locked and not self.val_replicas.is_replica_ready('standalone'):
                    self.train_replicas.set_replica_ready_state('val', ready=True)
                    # Update val standalone wg if val replica is ready for train rollout
                    with Timer(name="update_val_rollout_server", logger=None) as timer:
                        _ = self.update_standalone_server_weights(is_train=False, is_main_thread=True)
                    print(f"[INFO] {step} generate streaming[update val weights and restart] {timer.last}")
                    metrics["timing/update_val_rollout_server"] = timer.last

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
                if 'agent_handler' in item.non_tensor_batch and not pd.isna(
                        item.non_tensor_batch['agent_handler'][0]) and item.non_tensor_batch['agent_handler'][0].strip(
                        ):
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

        xperf_metrics: dict = {}
        with self.enable_hybrid_server_gen_ctx(is_train=True,
                                               xperf_metrics=xperf_metrics,
                                               need_hybrid_weights_update=self.train_standalone_wg is None):
            done, pending = asyncio.run_coroutine_threadsafe(submit_and_wait(), self.loop).result()
        pending = list(pending)
        ready_batch = self._normalize_done_tasks(done)
        finished_num = len(ready_batch)

        if self.config.actor_rollout_ref.rollout.recommend_standalone_usage.enable:
            metrics_for_recommend_standalone_usage(xperf_metrics)

        dummy_batch = DataProto(meta_info={"xperf_metrics": self._merge_xperf_metrics(ready_batch, xperf_metrics)})
        record_xperf_metrics(dummy_batch, metrics, self.logger, step, prefix="hybrid")
        metrics["rollout/standalone_completed_batch"] = finished_num
        metrics["rollout/standalone_incompleted_batch"] = len(pending_batch) + len(gen_batch) - finished_num

        if self.config.data.get("enable_swalm_agent", False):
            all_agent_metrics = []
            for batch in ready_batch:
                agent_metrics = batch.meta_info.get("agent_metrics", {})
                if agent_metrics:
                    all_agent_metrics.append(agent_metrics)
            if all_agent_metrics:
                metrics.update({"rollout/agent/tmp_agent_metrics": all_agent_metrics})

        return ready_batch, pending

    def _val_batch_gen(self, gen_batch: DataProto, step: int, metrics: Dict, is_standalone: bool) -> DataProto:
        if is_standalone:
            with self.acquire_hybrid_wg(is_train=False):
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
        """For val during training"""
        if self.val_standalone_wg is not None:
            with Timer(name="update_rollout_server", logger=None) as timer:
                self.update_standalone_server_weights(is_train=False, is_main_thread=False)
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
                handler = None
                if 'agent_handler' in item.non_tensor_batch and not pd.isna(
                        item.non_tensor_batch['agent_handler'][0]) and item.non_tensor_batch['agent_handler'][0].strip(
                        ):
                    handler = select_handler_fn(item.non_tensor_batch['agent_handler'][0],
                                                external_lib=self.config.rollout_server.external_lib)
                task = asyncio.create_task(self.val_client_executor.submit(handler or global_handler, item, context))
                running_batch.append(task)
            print(f"[INFO] {step} val generate streaming[submit], batch size: {len(gen_batch)}, {time.time() - start}")
            start = time.time()

            ready_batch = await asyncio.gather(*running_batch, return_exceptions=True)
            ready_batch = [task for task in ready_batch if task is not None]
            print(f"[INFO] {step} val gen server[as_completed], batch size: {len(gen_batch)}")
            return ready_batch

        xperf_metrics: dict = {}
        with nullcontext() if is_standalone else self.enable_hybrid_server_gen_ctx(is_train=False,
                                                                                   xperf_metrics=xperf_metrics):
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

        if self.config.data.get("enable_swalm_agent", False):
            success_ready_batch = []
            for res in ready_batch:
                if ('swalm_agent_score' in res.batch):
                    from alpha_seed.workers.agents.handlers.swalm.swalm_handler import SWALM_ENV_FAIL_SCORE
                    if (res.batch['swalm_agent_score'][0].item() == SWALM_ENV_FAIL_SCORE):
                        continue
                    else:
                        success_ready_batch.append(res)
                else:
                    res.batch['swalm_agent_score'] = torch.tensor(NON_AGENT_PLACE_HOLDER_SCORE).repeat(len(res))
                    success_ready_batch.append(res)
            failed_nums = len(ready_batch) - len(success_ready_batch)
            fake_success_ready_batch = random.choices(success_ready_batch, k=failed_nums)
            for res in fake_success_ready_batch:
                res.batch['swalm_agent_score'] = torch.tensor(-1.).repeat(len(res))
                res.non_tensor_batch["extra_info"][0]['all_turns_sum'] = -99  # -99 as the env failure flag
                if os.getenv("ENABLE_SWALM_LOG", False):
                    res.non_tensor_batch["extra_info"][0]['agent_traj_url'] = ""
                success_ready_batch.append(res)
            ready_batch = success_ready_batch
        else:
            ready_batch = results

        gen_out = DataProto.concat(ready_batch)
        # only use DP[0] for metrics presentation
        gen_out.meta_info['xperf_metrics'] = self._merge_xperf_metrics(ready_batch, xperf_metrics)
        record_xperf_metrics(gen_out, metrics, self.logger, step, prefix="standalone" if is_standalone else "hybrid")
        return gen_out

    def _get_filter_key(self, item):
        if isinstance(item, dict):
            val_epoch_id = item['val_epoch_id']
            index_id = item['index_id']
            bon_id = item['bon_id']
        else:
            val_epoch_id = item.meta_info['epoch_id']
            index_id = item.non_tensor_batch['index'][0]
            bon_id = item.non_tensor_batch['bon_id'][0]
        return f"{val_epoch_id}_{index_id}_{bon_id}"

    async def _write_results_to_file(self, ready_batch):
        assert self.config.trainer.default_hdfs_dir.startswith("/mnt/hdfs/"), \
            f"default_hdfs_dir must start with /mnt/hdfs/, got {self.config.trainer.default_hdfs_dir}"
        save_file = os.path.join(self.config.trainer.default_hdfs_dir, f"val_results.jsonl")
        logger.info(f"[INFO] Writing {len(ready_batch)} items to {save_file}")
        index_file = os.path.join(self.config.trainer.default_hdfs_dir, f"val_results_index.json")
        if aiofiles is None:
            raise RuntimeError("please pip install aiofiles")
        async with aiofiles.open(save_file, "a") as f:
            for item in ready_batch:
                await f.write(json.dumps(item) + "\n")
        indices = []
        for item in ready_batch:
            indices.append(self._get_filter_key(item))
        async with aiofiles.open(index_file, "a") as f:
            await f.write(json.dumps(indices) + "\n")

    async def _filter_history_data(self, gen_batch):
        assert self.config.trainer.default_hdfs_dir.startswith("/mnt/hdfs/"), \
            f"default_hdfs_dir must start with /mnt/hdfs/, got {self.config.trainer.default_hdfs_dir}"
        save_file = os.path.join(self.config.trainer.default_hdfs_dir, f"val_results_index.json")
        indices = []
        if not os.path.exists(save_file):
            return gen_batch
        if aiofiles is None:
            raise RuntimeError("please pip install aiofiles")
        async with aiofiles.open(save_file, "r") as f:
            lines = await f.read()
            lines = lines.strip().split('\n')
            for line in lines:
                indices.extend(json.loads(line))
        gen_batch = list(filter(lambda x: self._get_filter_key(x) not in indices, gen_batch))
        return gen_batch

    def _val_server_gen_with_ckpt(self, gen_batch: DataProto, step: int, metrics: Dict,
                                  is_standalone: bool) -> DataProto:
        """rollout server gen with ckpt, for val_only, sync param once. 用户可以自定义handler, handler返回的结果会保存到fused hdfs路径中,
        保存时间间隔为config.rollout_server.evals.ckpt_interval_seconds"""
        if self.val_standalone_wg is not None and not self._sync_param_once:
            with Timer(name="update_rollout_server", logger=None) as timer:
                self.update_standalone_server_weights(is_train=False, is_main_thread=False)
            print(f"[INFO] {step} val generate server[update weights and restart] {timer.last}")
            metrics["timing/update_rollout_server"] = timer.last
            self._sync_param_once = True

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
            val_gen_batch = gen_batch.chunk(len(gen_batch))
            val_gen_batch = await self._filter_history_data(val_gen_batch)
            if not val_gen_batch:
                return

            last_completed_num = len(gen_batch) - len(val_gen_batch)

            for item in val_gen_batch:
                handler = None
                if 'agent_handler' in item.non_tensor_batch and not pd.isna(
                        item.non_tensor_batch['agent_handler'][0]) and item.non_tensor_batch['agent_handler'][0].strip(
                        ):
                    handler = select_handler_fn(item.non_tensor_batch['agent_handler'][0],
                                                external_lib=self.config.rollout_server.external_lib)
                task = asyncio.create_task(self.val_client_executor.submit(handler or global_handler, item, context))
                running_batch.append(task)
            logger.info(
                f"[INFO] {step} val generate streaming[submit], batch size: {len(val_gen_batch)}, {time.time() - start}"
            )
            start = time.time()

            start_time = time.time()
            write_task = None
            ready_batch_buffer = []
            completed_num = 0
            total_len = len(val_gen_batch)
            log_interval = total_len / 100
            log_threshold = log_interval

            for fut in asyncio.as_completed(running_batch):
                result = await fut
                if result is not None:
                    ready_batch_buffer.append(result)
                    completed_num += 1
                if completed_num >= log_threshold:
                    time_elapse = time.time() - start
                    log_threshold += log_interval
                    logger.info(f"[INFO] {step} val generate progress: {completed_num}/{total_len}, "
                                f"time elapse: {time_elapse:.2f}s, throughput: {completed_num / time_elapse:.2f}")
                if time.time() - start_time > self.config.rollout_server.evals.ckpt_interval_seconds:
                    start_time = time.time()
                    if write_task is not None:
                        # wait last write task to finish
                        await write_task
                    # async write
                    write_task = asyncio.create_task(self._write_results_to_file(ready_batch_buffer))
                    ready_batch_buffer = []

            if write_task is not None:
                await write_task
            if len(ready_batch_buffer) > 0:
                await self._write_results_to_file(ready_batch_buffer)

            logger.info(f"[INFO] {step} val gen server[as_completed], batch size: {len(val_gen_batch)}")

        xperf_metrics: dict = {}
        with nullcontext() if is_standalone else self.enable_hybrid_server_gen_ctx(is_train=False,
                                                                                   xperf_metrics=xperf_metrics):
            asyncio.run_coroutine_threadsafe(_submit_and_wait(), self.loop).result()

    def _prepare_gen_batch(self, batch: DataProto, step, is_train: bool):
        gen_batch_required_keys = ["input_ids", "attention_mask"]
        for key in [
                "rollout_behavior_log_probs",
                "off_policy_steps",
        ]:
            if key not in batch:
                batch.batch[key] = create_response_tensor(name=key,
                                                          bs=batch.batch["input_ids"].shape[0],
                                                          length=self.config.data.max_response_length,
                                                          device=batch.batch["input_ids"].device)
            gen_batch_required_keys.append(key)

        if self.config.critic.get("use_decouple_critic", False):
            for key in [
                    "input_ids_critic",
                    "attention_mask_critic",
            ]:
                # if key not in batch:
                #     batch.batch[key] = _get_response_tensor(dtype=torch.int64)
                gen_batch_required_keys.append(key)
        if is_train and self.config.algorithm.use_model_output_mask:
            if (key := "model_output_mask") not in batch:
                batch.batch[key] = create_response_tensor(name=key,
                                                          bs=batch.batch["input_ids"].shape[0],
                                                          length=self.config.data.max_response_length,
                                                          device=batch.batch['input_ids'].device)
            gen_batch_required_keys.append(key)

        for key in ["ability_idx", "no_thinking_required"]:
            if key in batch.batch:
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
        if (key := "initial_files") in gen_batch.non_tensor_batch:
            #Here some CI tasks need initial files to upload to sandbox
            for i in range(len(gen_batch)):
                agent_env_initial_files = gen_batch.non_tensor_batch['initial_files'][i]
                if not pd.isna(agent_env_initial_files):
                    gen_batch.non_tensor_batch['extra_data'][i].update(
                        {'agent_env_initial_files': json.loads(agent_env_initial_files)['initial_files']})
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
        gen_batch.non_tensor_batch['step'] = np.array([step] * len(gen_batch), dtype=object)
        sample_kwargs = (self.config.actor_rollout_ref.rollout.train_generate_kwargs
                         if is_train else self.config.actor_rollout_ref.rollout.val_generate_kwargs)
        sample_kwargs_dict = OmegaConf.to_container(sample_kwargs, resolve=True)
        gen_batch.meta_info.update({
            "step": step,
            "generation_kwargs": sample_kwargs_dict,
            "return_selected_experts": is_train,
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
        rollout_config = self.config.streaming_rollout
        lb_mode = rollout_config.proxy.lb_mode
        if lb_mode == "even-distribution":
            ProxyClass = RolloutWorkerGroupProxy
        elif lb_mode == "dynamic-balancing":
            ProxyClass = BalancedRolloutWorkerGroupProxy
        elif lb_mode == "cache-aware-balancing":
            ProxyClass = CacheAwareBalancedRolloutWorkerGroupProxy
        else:
            raise ValueError(f"config.streaming_rollout.proxy.lb_mode does not support {lb_mode=}, "
                             f"please choose from ['even-distribution', 'dynamic-balancing', 'cache-aware-balancing']")

        hybrid_replica = FixedReplicatedRayWorkerGroupAdapter(self.hybrid_wg, gen_tp_size, 'actor_rollout_ref')

        # validation on hybrid engine
        val_intermittent_replicas = {
            'hybrid': hybrid_replica,
        }
        # standalone validation
        val_standalone_replica = None
        if self.val_standalone_wg is not None:
            val_standalone_replica = FixedReplicatedRayWorkerGroupAdapter(self.val_standalone_wg, gen_tp_size,
                                                                          'standalone_validator')
            val_intermittent_replicas['standalone'] = val_standalone_replica

        val_persistent_replicas = {}
        self.val_replicas = CombinedRayWorkerGroupAdapter(val_intermittent_replicas, val_persistent_replicas)
        # Turn off hybrid for gen by default (i.e. train mode initially)
        self.val_replicas.set_replica_ready_state(name='hybrid', ready=False)
        if 'standalone' in self.val_replicas.replicas:
            self.val_replicas.set_replica_ready_state(name='standalone', ready=False)

        self.val_rollout_proxy = ProxyClass(self.val_replicas, [], 'val_rollout', rollout_config)
        self.val_rollout_server = await listen('val_rollout')

        # train
        # create replicated worker group and rollout proxy

        if self._rollout_elastic_enabled:
            assert self.weights_communicator == 'ucx', 'weights_communicator must be "ucx" when using elastic rollout'
            assert self.train_standalone_wg is None, 'should not initialize train standalone when using elastic rollout'
            self.train_rollout_proxy, self.train_standalone_wg, self.train_replicas = self.elastic_rollout_mgr.init_elastic_rollout(
                hybrid_replica=hybrid_replica, val_standalone_replica=val_standalone_replica)
        else:
            train_intermittent_replicas = {'hybrid': hybrid_replica}

            # Add val standalone replica to train replicas by default
            if val_standalone_replica is not None:
                train_intermittent_replicas['val'] = val_standalone_replica

            train_persistent_replicas = {}
            if self.train_standalone_wg is not None:
                train_persistent_replicas['standalone'] = FixedReplicatedRayWorkerGroupAdapter(
                    self.train_standalone_wg, gen_tp_size, 'standalone_rollout')
            self.train_replicas = CombinedRayWorkerGroupAdapter(train_intermittent_replicas, train_persistent_replicas)
            self.train_rollout_proxy = ProxyClass(self.train_replicas, [], 'train_rollout', rollout_config)

        # Turn off hybrid for gen by default (i.e. train mode initially)
        self.train_replicas.set_replica_ready_state(name='hybrid', ready=False)
        self.train_rollout_server = await listen('train_rollout')

        self.rollout_server_started.set()
        print("[rollout manager] servers started.")

        # 不能让这个event loop结束，因为每个oai server里面还有一个自己的server_task
        await asyncio.Future()

    @contextmanager
    def enable_hybrid_server_gen_ctx(self,
                                     is_train: bool,
                                     xperf_metrics: dict = None,
                                     need_hybrid_weights_update: bool = True):
        """Set hybrid server to be ready for gen"""
        flag_key = f'_hybrid_server_enabled__is_train_{is_train}'
        if getattr(self, flag_key, False):
            # No effect if already enabled
            yield
            return
        setattr(self, flag_key, True)

        has_standalone = self.train_standalone_wg is not None
        replicas = self.train_replicas if is_train else self.val_replicas
        replicas.set_replica_ready_state(name='hybrid', ready=True)
        with self.acquire_hybrid_wg(is_train):
            with hybrid_enable_server_ctx(self.hybrid_wg, need_hybrid_weights_update):
                yield
                if has_standalone:
                    replicas.set_replica_ready_state(name='hybrid', ready=False)
        ret_xperf_metrics = self.get_metrics()
        self.hybrid_wg.empty_engine_cache(only_clear_metrics=self.config.trainer.queued_rollout_config.enable)
        if xperf_metrics is not None:
            xperf_metrics.update(ret_xperf_metrics)
        setattr(self, flag_key, False)

    @contextmanager
    def acquire_val_standalone_for_val_ctx(self, metrics: Dict):
        """Set standalone replica for val gen, disable it for train gen"""
        if self.val_standalone_wg is None:
            yield
            return

        with self._val_standalone_wg_lock:
            with Timer(name='train_to_val') as timer:
                self.train_replicas.set_replica_ready_state('val', ready=False)
                self.val_standalone_wg.empty_engine_cache()
                self.val_replicas.set_replica_ready_state('standalone', ready=True)
            metrics['timing/val_standalone_to_val'] = timer.last
            print(f"[INFO] acquired val_standalone for validation")
            yield
            print(f"[INFO] release val_standalone to enable training")
            self.val_replicas.set_replica_ready_state('standalone', ready=False)
            # NOTE: Do not set `self.train_replicas.set_replica_ready_state('val', ready=True)`,
            # because the weights is not guaranteed to be updated, let train thread to decide
            # whether to use val standalone replica

    @contextmanager
    def acquire_hybrid_wg(self, is_train: bool):
        role = 'train' if is_train else 'val'
        with self._hybrid_wg_lock:
            print(f'[INFO] {role} acquired hybrid_wg')
            yield
            print(f'[INFO] {role} released hybrid_wg')

    @contextmanager
    def suppress_train_update_standalone(self):
        flag_key = f"suppress_update_standalone"
        setattr(self, flag_key, True)
        yield
        setattr(self, flag_key, False)

    def get_metrics(self):
        """After exiting ctx, collect metrics"""
        return self.hybrid_wg.get_metrics()

    def should_train_update_standalone(self):
        flag_key = f"suppress_update_standalone"
        return not getattr(self, flag_key, False)

    def get_standalone_metrics(self, standalone_wg):
        if not self._rollout_elastic_enabled and standalone_wg is not None:
            metrics = standalone_wg.return_metrics()
            standalone_wg.empty_engine_cache()
            return metrics
        return {}

    def update_standalone_server_weights(self, is_train: bool, is_main_thread: bool) -> Optional[dict]:
        flag_key = f"suppress_update_standalone__is_train_{is_train}"
        if getattr(self, flag_key, False):
            # suppressed update, do nothing
            return None
        standalone_wg = self.train_standalone_wg if is_train else self.val_standalone_wg
        standalone_role = "standalone_rollout" if is_train else "standalone_validator"

        if standalone_wg is None:
            return None

        with server_update_weights_ctx(standalone_wg, role=standalone_role):
            with self.acquire_hybrid_wg(is_train):
                _update_standalone_weights(self.hybrid_wg, standalone_wg, standalone_role,
                                           self.threadsafe_nccl_comm if not is_main_thread else None, True, False)
                standalone_metric = self.get_standalone_metrics(standalone_wg)
        return standalone_metric

    def dump_trace_spans(self, after_ts: float = 0.):
        return Tracer.merge_all(after_ts=after_ts)
