import random
import time
from typing import Union, List, Optional

import ray
from ray import ObjectRef

from alpha_seed.utils.server_client import is_local_ray_instance
from alpha_seed.workers.streaming_service.auto_scaling import ScalePolicyConfig, HorizontalAutoScaling
from alpha_seed.workers.streaming_service.rollout_proxy import BalancedRolloutWorkerGroupProxy, \
    CombinedRayWorkerGroupAdapter, StandaloneRolloutWGAdapter, CacheAwareBalancedRolloutWorkerGroupProxy
from alpha_seed.workers.streaming_service.streaming_rollout import ElasticAsyncXPerfGPTRollout
from alpha_seed.workers.xperf_rollout.utils.base_weights_communicator import WeightsRankInfo
from mono_rl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, RayResourcePool
from mono_rl.single_controller.ray.replicated_worker_group import ReplicatedRayWorkerGroup, ScalingRayWorkerGroup


class ElasticRolloutManager:

    def __init__(self, config):
        self.config = config
        self.standalone_rollout_ha: Optional[HorizontalAutoScaling] = None
        self._hybrid_rollout_source_info: List[WeightsRankInfo] = []
        self._rollout_relay_addresses: List[List[WeightsRankInfo]] = []  # [[tp0, tp1], [tp0, tp1], ...]

    def set_hybrid_rollout_source_info(self, hybrid_rollout_address):
        self._hybrid_rollout_source_info = hybrid_rollout_address

    def init_elastic_rollout(self, hybrid_replica: ReplicatedRayWorkerGroup):
        rollout_config = self.config.streaming_rollout
        # 每个rollout_worker用1个gpu，每个gpu对应1个rank
        res_shape = [self.config.streaming_rollout.n_gpus_per_node] * self.config.streaming_rollout.nnodes
        tp_size = sum(res_shape)

        # 依赖actor的address作为ucx endpoint
        hybrid_rollout_rank_info: List[WeightsRankInfo] = self._hybrid_rollout_source_info
        print(f"all ucx source addresses: {hybrid_rollout_rank_info}")

        # 按照rollout的dp world进行切分，一定是正好切够的
        assert len(hybrid_rollout_rank_info) % tp_size == 0, \
            f"hybrid rollout world size({len(hybrid_rollout_rank_info)}) should be divisible by dp_world_size({tp_size})"
        # shape: (dp_size, tp_size)
        # [[TP0, TP1, ...] [TP0, TP1, ...] ...]
        hybrid_rollout_info_tp_groups = [
            hybrid_rollout_rank_info[i:i + tp_size] for i in range(0, len(hybrid_rollout_rank_info), tp_size)
        ]

        # streaming+elastic的standalone rollout初始化
        # rollout worker 初始化方式定义
        rollout_cls = RayClassWithInitArgs(cls=ElasticAsyncXPerfGPTRollout,
                                           config=self.config.actor_rollout_ref,
                                           role="rollout_server",
                                           hybrid_rollout_info=hybrid_rollout_rank_info)

        def initial_stable_model_setup_comm(wg: Union[RayWorkerGroup, ElasticAsyncXPerfGPTRollout]) -> List[ObjectRef]:
            # 需要确保actor初始化好才能setup rollout作为client去获取参数
            # 连上hybrid rollout 或者 stable standalone rollout获取参数
            # 返回relay address，即此rollout自己as server的address
            return wg.init_and_setup(hybrid_rollout_info_tp_groups, setup_relay=True)

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
        else:
            assert stable_pool_name is not None and stable_pool_name != '', \
                'you have enabled the elastic rollout, but streaming_rollout.elastic.stable_pool_name is not set'

        # 稳定池资源和副本配置
        stable_res_pool = RayResourcePool(
            process_on_nodes=res_shape,
            use_gpu=True,
            max_colocate_count=1,
            additional_resources=stable_pool_res,  # 用于表示调度到指定资源池
            name_prefix=f'sr_stable_')
        min_guaranteed_replicas = ReplicatedRayWorkerGroup(rollout_cls, stable_res_pool,
                                                           initial_stable_model_setup_comm)
        # 拉起最小副本数
        num_guaranteed = self.config.streaming_rollout.elastic.min_replicas
        min_replicas_init_fut = min_guaranteed_replicas.scale_up(num_guaranteed)

        # 等待stable standalone rollout启动完成
        # [tp0, tp1, tp0, tp1, ...]
        relay_addrs: List[WeightsRankInfo] = ray.get(min_replicas_init_fut)
        # [[tp0, tp1], [tp0, tp1], ...]
        self._rollout_relay_addresses = [relay_addrs[i:i + tp_size] for i in range(0, len(relay_addrs), tp_size)]
        print(f'relay ready, address tp groups: {self._rollout_relay_addresses}')

        def elastic_model_setup_comm(wg: Union[RayWorkerGroup, ElasticAsyncXPerfGPTRollout]) -> List[ObjectRef]:
            # 直接拉参数，拉完立刻ready可接受请求
            return wg.init_and_setup(self._rollout_relay_addresses,
                                     setup_relay=False,
                                     intermediately_update_weights=True)

        # 弹性池跑伸缩副本
        elastic_res_pool = RayResourcePool(
            process_on_nodes=res_shape,
            use_gpu=True,
            max_colocate_count=1,
            additional_resources=elastic_pool_res,  # 用于表示调度到指定资源池
            name_prefix=f'sr_elastic_')
        best_effort_replicas = ReplicatedRayWorkerGroup(rollout_cls, elastic_res_pool, elastic_model_setup_comm)
        # 两个副本组合并一起组成伸缩组
        elastic_replicas = ScalingRayWorkerGroup(min_guaranteed_replicas, best_effort_replicas)
        # 组合 hybrid replica
        replicas = CombinedRayWorkerGroupAdapter(
            intermittent={'hybrid': hybrid_replica},
            persistent={'elastic': elastic_replicas},
        )
        lb_mode = rollout_config.proxy.lb_mode
        if lb_mode == "dynamic-balancing":
            ProxyClass = BalancedRolloutWorkerGroupProxy
        elif lb_mode == "cache-aware-balancing":
            ProxyClass = CacheAwareBalancedRolloutWorkerGroupProxy
        else:
            raise ValueError(f"config.streaming_rollout.proxy.lb_mode does not support {lb_mode=} in elastic rollout, "
                             f"please choose from ['dynamic-balancing', 'cache-aware-balancing']")

        # 封装给worker group的接口代理
        rollout_proxy = ProxyClass(replicas, hybrid_rollout_rank_info, 'train_rollout', rollout_config)

        # initialize rollout horizontal auto scaling control handle
        elastic_pool_name = self.config.streaming_rollout.elastic.elastic_pool_name
        policy = ScalePolicyConfig(
            metrics_sampling_seconds=self.config.streaming_rollout.elastic.metrics_sampling_seconds,
            scale_up_threshold=self.config.streaming_rollout.elastic.scale_up_threshold,
            scale_down_threshold=self.config.streaming_rollout.elastic.scale_down_threshold,
            scale_up_wait=self.config.streaming_rollout.elastic.scale_up_wait,
            scale_down_wait=self.config.streaming_rollout.elastic.scale_down_wait,
            min_replicas=self.config.streaming_rollout.elastic.min_replicas,
            max_replicas=self.config.streaming_rollout.elastic.max_replicas,
        )
        self.standalone_rollout_ha = HorizontalAutoScaling(elastic_replicas,
                                                           elastic_pool_name,
                                                           policy,
                                                           metric_source=rollout_proxy)
        self.standalone_rollout_wg = StandaloneRolloutWGAdapter(elastic_replicas)

        # 必须等min_replicas部分变成initialized状态才可以返回
        # 因为完成scale_up调用并不会立即变成initialized状态，由liveness probe线程将其设置为initialized，
        # 这期间可能大约1-2s滞后，可能会在接下来第一次gen时错过update_standalone_weights，导致guaranteed部分没有weights，
        # 这里等一下，保证guaranteed部分是一定能参数weights update的
        while len(min_guaranteed_replicas.initialized_worker_group_ids) < num_guaranteed:
            time.sleep(0.5)

        # 返回的3个对象
        #  rollout_proxy: 负载均衡query
        #  standalone_rollout_wg: 控制standalone rollout更新weights
        #  replicas: 控制hybrid active/inactive
        return rollout_proxy, self.standalone_rollout_wg, replicas
