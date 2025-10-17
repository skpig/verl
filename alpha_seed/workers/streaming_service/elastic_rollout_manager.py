import random
import time
import traceback
from typing import Union, List, Optional, Any

import ray
from ray import ObjectRef
from ray.exceptions import ActorDiedError

from alpha_seed.utils.server_client import is_local_ray_instance, KVStore, recreate_actor
from alpha_seed.workers.streaming_service.auto_scaling import ScalePolicyConfig, HorizontalAutoScaling
from alpha_seed.workers.streaming_service.rollout_proxy import BalancedRolloutWorkerGroupProxy, \
    CombinedRayWorkerGroupAdapter, StandaloneRolloutWGAdapter, CacheAwareBalancedRolloutWorkerGroupProxy
from alpha_seed.workers.streaming_service.streaming_rollout import ElasticAsyncXPerfGPTRollout
from alpha_seed.workers.xperf_rollout.utils.base_weights_communicator import WeightsRankInfo
from mono_rl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, RayResourcePool
from mono_rl.single_controller.ray.replicated_worker_group import ReplicatedRayWorkerGroup, ScalingRayWorkerGroup
import threading


class ElasticRolloutManager:

    def __init__(self, rm_name: str, streaming_config, actor_rollout_ref_config):
        self.rm_name = rm_name  # 用于区分不同的rollout，是train rollout还是val rollout之类
        self.streaming_config = streaming_config
        self.actor_rollout_ref_config = actor_rollout_ref_config
        self.standalone_rollout_ha: Optional[HorizontalAutoScaling] = None
        self._hybrid_rollout_source_info: List[WeightsRankInfo] = []
        self._rollout_relay_addresses: List[List[WeightsRankInfo]] = []  # [[tp0, tp1], [tp0, tp1], ...]

        self.guaranteed_replicas: ReplicatedRayWorkerGroup = None  # noqa: set soon
        self.tp_size = -1
        try:
            self.kv_store_actor = ray.get_actor(KVStore.name)
        except:
            # 在ci中kv_store_actor是没有在main中被提前创建的，因此在这里创建
            print("In ci test case, create kv_store_actor")
            self.kv_store_actor = recreate_actor(KVStore, name=KVStore.name)

        # 开启一个后台线程，每隔10s执行update_relay_addr_list
        self.update_relay_addr_thread = threading.Thread(target=self._update_relay_addr_loop, daemon=True)
        self.stable_immediately_update_weights = False  # 在完成RobustRolloutManager初始化后，为True，保证stable_rollout的权重及时更新

    def set_hybrid_rollout_source_info(self, hybrid_rollout_address):
        self._hybrid_rollout_source_info = hybrid_rollout_address

    def trans_relay_addrs_format(self, relay_addrs: List[WeightsRankInfo]):
        return self._chunk_by_wg(relay_addrs, self.tp_size)

    def _update_relay_addr(self):
        current_relay_addr_dict = dict()
        # 只需要维护init的worker足够了，因为就算不ready，也会在weight_communicator中检测出来，避免elastic rollout拉到错误的权重
        init_wgs = self.guaranteed_replicas.get_initialized_worker_groups()
        for wg_id, wg in init_wgs.items():
            wg: ElasticAsyncXPerfGPTRollout
            relay_info = ray.get(wg.get_relay_info())
            current_relay_addr_dict[wg_id] = relay_info

        flat_relay_addrs = [item for sublist in current_relay_addr_dict.values() for item in sublist]
        self._rollout_relay_addresses = self.trans_relay_addrs_format(flat_relay_addrs)
        # manager更新地址
        ray.get(
            self.kv_store_actor.set_key_val.remote(f'{self.rm_name}.rollout_relay_addresses',
                                                   self._rollout_relay_addresses))

    def _update_relay_addr_loop(self):
        # 获取新增的和移除的relay_addr，并更新self._rollout_relay_addresses
        # 间隔10s执行一次
        while True:
            time.sleep(10)
            self._update_relay_addr()

    def init_elastic_rollout(self,
                             hybrid_replica: ReplicatedRayWorkerGroup,
                             borrowed_standalone_replica: Optional[ReplicatedRayWorkerGroup] = None):
        rollout_config = self.streaming_config
        # 每个rollout_worker用1个gpu，每个gpu对应1个rank
        res_shape = [rollout_config.n_gpus_per_node] * rollout_config.nnodes
        tp_size = sum(res_shape)
        self.tp_size = tp_size

        # 依赖actor的address作为ucx endpoint
        hybrid_rollout_rank_info: List[WeightsRankInfo] = self._hybrid_rollout_source_info
        print(f"all ucx source addresses: {hybrid_rollout_rank_info}")

        # 按照rollout的dp world进行切分，一定是正好切够的
        assert len(hybrid_rollout_rank_info) % tp_size == 0, \
            f"hybrid rollout world size({len(hybrid_rollout_rank_info)}) should be divisible by dp_world_size({tp_size})"
        # shape: (dp_size, tp_size)
        # [[TP0, TP1, ...] [TP0, TP1, ...] ...]
        hybrid_rollout_info_tp_groups = self.trans_relay_addrs_format(hybrid_rollout_rank_info)

        # streaming+elastic的standalone rollout初始化
        # rollout worker 初始化方式定义
        rollout_cls = RayClassWithInitArgs(cls=ElasticAsyncXPerfGPTRollout,
                                           config=self.actor_rollout_ref_config,
                                           role=self.rm_name,
                                           hybrid_rollout_info=hybrid_rollout_rank_info)

        def initial_stable_model_setup_comm(wg: Union[RayWorkerGroup, ElasticAsyncXPerfGPTRollout]) -> List[ObjectRef]:
            # 需要确保actor初始化好才能setup rollout作为client去获取参数
            # 连上hybrid rollout 或者 stable standalone rollout获取参数
            # 返回relay address，即此rollout自己as server的address
            # 如果是重新拉起的，第一次找其它stable更新权重，后面找hybrid
            return wg.init_and_setup(hybrid_rollout_info_tp_groups,
                                     setup_relay=True,
                                     immediately_update_weights=self.stable_immediately_update_weights)

        # 每个rollout_worker用1个gpu
        res_shape = [rollout_config.n_gpus_per_node] * rollout_config.nnodes
        stable_pool_name = rollout_config.elastic.stable_pool_name
        elastic_pool_name = rollout_config.elastic.elastic_pool_name
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
            name_prefix=f'{self.rm_name}_st_')
        min_guaranteed_replicas = ReplicatedRayWorkerGroup(rollout_cls, stable_res_pool,
                                                           initial_stable_model_setup_comm)
        self.guaranteed_replicas = min_guaranteed_replicas
        # 拉起最小副本数
        num_guaranteed = rollout_config.elastic.min_replicas
        min_replicas_init_fut = min_guaranteed_replicas.scale_up(num_guaranteed)

        # 等待stable standalone rollout启动完成
        # [tp0, tp1, tp0, tp1, ...] -> [[tp0, tp1], [tp0, tp1], ...]
        min_replicas_init_fut_by_wg = self._chunk_by_wg(min_replicas_init_fut, self.tp_size)
        self._rollout_relay_addresses = self._wait_wgs_refs(min_replicas_init_fut_by_wg)
        print(f'relay ready, address tp groups: {self._rollout_relay_addresses}')
        assert len(self._rollout_relay_addresses) > 0, \
            "None of the guaranteed replica has successfully initialized. please check the log of RolloutManager"

        ray.get(
            self.kv_store_actor.set_key_val.remote(f'{self.rm_name}.rollout_relay_addresses',
                                                   self._rollout_relay_addresses))

        def elastic_model_setup_comm(wg: Union[RayWorkerGroup, ElasticAsyncXPerfGPTRollout]) -> List[ObjectRef]:
            # 直接拉参数，拉完立刻ready可接受请求
            return wg.init_and_setup(self._rollout_relay_addresses, setup_relay=False, immediately_update_weights=True)

        # 弹性池跑伸缩副本
        elastic_res_pool = RayResourcePool(
            process_on_nodes=res_shape,
            use_gpu=True,
            max_colocate_count=1,
            additional_resources=elastic_pool_res,  # 用于表示调度到指定资源池
            name_prefix=f'{self.rm_name}_el_')
        best_effort_replicas = ReplicatedRayWorkerGroup(rollout_cls, elastic_res_pool, elastic_model_setup_comm)
        # 两个副本组合并一起组成伸缩组
        elastic_replicas = ScalingRayWorkerGroup(min_guaranteed_replicas, best_effort_replicas)
        # 组合 hybrid replica
        intermittent_replicas = {'hybrid': hybrid_replica}
        if borrowed_standalone_replica is not None:
            intermittent_replicas['borrow'] = borrowed_standalone_replica

        replicas = CombinedRayWorkerGroupAdapter(
            intermittent=intermittent_replicas,
            persistent={'elastic': elastic_replicas},
        )
        lb_mode = rollout_config.proxy.lb_mode
        if lb_mode == "dynamic-balancing":
            ProxyClass = BalancedRolloutWorkerGroupProxy
        elif lb_mode == "cache-aware-balancing":
            ProxyClass = CacheAwareBalancedRolloutWorkerGroupProxy
        else:
            raise ValueError(f"config.streaming_*.proxy.lb_mode does not support {lb_mode=} in elastic rollout, "
                             f"please choose from ['dynamic-balancing', 'cache-aware-balancing']")

        # 封装给worker group的接口代理
        rollout_proxy = ProxyClass(replicas, hybrid_rollout_rank_info, self.rm_name, rollout_config)

        # 必须等min_replicas部分变成initialized状态才可以返回
        # 因为完成scale_up调用并不会立即变成initialized状态，由liveness probe线程将其设置为initialized，
        # 这期间可能大约1-2s滞后，可能会在接下来第一次gen时错过update_standalone_weights，导致guaranteed部分没有weights，
        # 这里等一下，保证guaranteed部分是一定能参数weights update的
        # 按第一批的实际可用rollout relay数量来等
        while len(min_guaranteed_replicas.initialized_worker_group_ids) < len(self._rollout_relay_addresses):
            time.sleep(0.5)

        # 状态切换前，先更新一次relay_addr的信息
        self._update_relay_addr()
        self.update_relay_addr_thread.start()
        self.stable_immediately_update_weights = True

        # 启动auto scaling
        # stable rollout的autoscaling策略, 少了立刻拉起而且不看metrics，所以与metrics相关的scale阈值设为-1
        min_guaranteed_policy = ScalePolicyConfig(
            metrics_sampling_seconds=rollout_config.elastic.metrics_sampling_seconds,
            scale_up_threshold=-1,
            scale_down_threshold=-1,
            scale_up_wait=rollout_config.elastic.scale_up_wait,
            scale_down_wait=rollout_config.elastic.scale_down_wait,
            min_replicas=rollout_config.elastic.min_replicas,
            max_replicas=rollout_config.elastic.min_replicas,
        )
        # elastic rollout的autoscaling策略
        best_effort_policy = ScalePolicyConfig(
            metrics_sampling_seconds=rollout_config.elastic.metrics_sampling_seconds,
            scale_up_threshold=rollout_config.elastic.scale_up_threshold,
            scale_down_threshold=rollout_config.elastic.scale_down_threshold,
            scale_up_wait=rollout_config.elastic.scale_up_wait,
            scale_down_wait=rollout_config.elastic.scale_down_wait,
            min_replicas=0,  # best_effort是弹性伸缩的，所以最小值是0
            max_replicas=rollout_config.elastic.max_replicas - rollout_config.elastic.min_replicas,
        )
        elastic_pool_name = rollout_config.elastic.elastic_pool_name
        self.stable_rollout_ha = HorizontalAutoScaling(min_guaranteed_replicas,
                                                       stable_pool_name,
                                                       min_guaranteed_policy,
                                                       metric_source=rollout_proxy)
        self.elastic_rollout_ha = HorizontalAutoScaling(best_effort_replicas,
                                                        elastic_pool_name,
                                                        best_effort_policy,
                                                        metric_source=rollout_proxy)

        # 返回的3个对象
        #  rollout_proxy: 负载均衡query
        #  standalone_rollout_wg: 控制standalone rollout更新weights
        #  replicas: 控制hybrid active/inactive
        standalone_rollout_wg = StandaloneRolloutWGAdapter(elastic_replicas)
        return rollout_proxy, standalone_rollout_wg, replicas

    def _wait_wgs_refs(self, refs: List[List[ObjectRef]]) -> List[List[Any]]:
        # 按照wg分组的ref获取结果，忽略出现ActorDiedError的wg
        # 按输入的顺序返回
        ret = []
        for wg_ref in refs:
            try:
                result = ray.get(wg_ref)
                ret.append(result)
            except ActorDiedError as e:
                print(f"actor died during initialization. {e.actor_id=}. print trace back only")
                traceback.print_exc()
        return ret

    def _chunk_by_wg(self, lst: List[Any], size: int) -> List[List[Any]]:
        # 把 [dp0tp0, dp0tp1, ..., dpNtp0, dpNtp1] 按各个dp切开，注意输入的list的排序
        # -> [[dp0tp*...], ..., [dpNtp*]]
        return [lst[i:i + size] for i in range(0, len(lst), size)]
