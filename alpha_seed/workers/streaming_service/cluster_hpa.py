import math
import os
import threading
import time
import traceback
import warnings
from dataclasses import dataclass

import ray
import requests
from omegaconf import DictConfig


@dataclass
class NodeSpec:
    accelerator_type: str
    accelerator_count: int


class ArnoldAPIError(Exception):
    pass


class ArnoldTrialError(Exception):
    pass


class ArnoldTrialResourceManager:
    """
    管理arnold trial伸缩，详见文档 https://bytedance.larkoffice.com/docx/KEwqd6XcFojnpGx272zcjHMXnPe
    """

    def __init__(self, worker_pool_name: str, config: DictConfig):
        """
        :param worker_pool_name: 可伸缩的arnold role的名称
        :param config:
            name 与merlin trial对应的role name
            hard_min 缩容时可缩到的最小值
            hard_max 扩容时可缩到的物理最大值，一般是一个不太会达到的很大的值，暂时写一个数字避免扩爆
            buffer_size 保证有多少个空闲的buffer node
        """
        self.worker_pool_name = worker_pool_name
        self.config = config
        self.num_buffer_nodes = config.buffer_size
        self._jwt_token_cache = None
        self._jwt_token_timestamp = None
        self._jwt_token_expire_seconds = 180
        self._previous_pending_ticket = None

        # 每次扩缩容就扩这个unit
        self._resource_scale_unit: NodeSpec = self._get_resource_scale_unit()

        threading.Thread(target=self._buffer_reservation_loop, daemon=True,
                         name=f'buffer-reserve/{worker_pool_name}').start()

    def reserve(self, num_new_nodes: int, timeout_seconds: float = 300):
        check_interval = 5
        retry_interval = 20

        # 等待上一轮工单完成（如果timeout无法完成会抛出异常）
        if self._previous_pending_ticket is not None:
            timeout_at = time.time() + timeout_seconds
            while True:
                time.sleep(check_interval)
                finished = self._check_scaling_ticket_status(self._previous_pending_ticket)
                if finished:
                    self._previous_pending_ticket = None
                    break
                if time.time() > timeout_at:
                    raise ArnoldTrialError("unable to fulfill previous scaling ticket, "
                                           "please check the detail status on arnold/merlin portal")

        # 调用接口扩容（重试timeout则退出）
        timeout_at = time.time() + timeout_seconds
        while time.time() < timeout_at:
            try:
                self._previous_pending_ticket = self._reserve_node(num_new_nodes)
            except ArnoldAPIError:
                traceback.print_exc()
                time.sleep(retry_interval)
            else:
                break

        # 等待这次扩容工单完成（timeout则退出）
        if self._previous_pending_ticket is not None:
            timeout_at = time.time() + timeout_seconds
            while time.time() < timeout_at:
                time.sleep(check_interval)
                finished = self._check_scaling_ticket_status(self._previous_pending_ticket)
                if finished:
                    self._previous_pending_ticket = None
                    break

    def _buffer_reservation_loop(self):
        # 确保ray集群的buffer资源
        # 每次
        check_interval = 5
        initial_delay = 30
        time.sleep(initial_delay)
        while True:
            time.sleep(check_interval)

            # note(lixiang): 暂时根据ray能看到的资源数进行扩容，无法获取准确的分池的资源量
            total_available = ray.available_resources()
            num_gpus_available = total_available.get('GPU', 0)
            desired_buffer_resource = self._resource_scale_unit.accelerator_count * self.num_buffer_nodes
            if num_gpus_available <= desired_buffer_resource:
                num_nodes_to_scale_up = math.ceil(
                    (desired_buffer_resource - num_gpus_available) / self._resource_scale_unit.accelerator_count)
                print(f"desired buffer resource={desired_buffer_resource}, current available gpu={num_gpus_available}, "
                      f"will scale up {num_nodes_to_scale_up} nodes")
                # block until scaling finished
                try:
                    self.reserve(num_nodes_to_scale_up)
                except Exception:
                    # 暂时先只打印错误，好像也没办法做什么，不要让这个线程挂掉
                    print("-" * 60, "  WARNING ONLY  ", "-" * 60)
                    traceback.print_exc()
                    print("-" * 130)

    def _get_resource_scale_unit(self) -> NodeSpec:
        # 看一次扩容的节点gpu数量和型号
        represent_ray_node_info = None
        for n in ray.nodes():
            if n['Resources'].get(self.worker_pool_name, 0) > 0:
                represent_ray_node_info = n
                break
        if represent_ray_node_info is None:
            warnings.warn("cannot find ray node with role name {self.worker_pool_name}, "
                          "possibly this resource pool is not ready. will use the node info in config")
            return NodeSpec(self.config.gpu_type, self.config.n_gpus_per_node)

        accelerator_type = None
        accelerator_count = 0
        for res_name, val in represent_ray_node_info['Resources'].items():
            if res_name.startswith('accelerator_type:'):
                accelerator_type = res_name[len('accelerator_type:'):]
                accelerator_count = val
        if accelerator_count == 0 or accelerator_type is None:
            raise RuntimeError(f"cannot find available accelerator in this node({represent_ray_node_info})")
        return NodeSpec(accelerator_type, accelerator_count)

    def _reserve_node(self, num_new_nodes: int):
        jwt_token = self._get_jwt_token()

        # note(lixiang)
        # 先通过ray的api获取在集群里的node数量，之后看情况要不要换成更准确的arnold API
        # 处以100是因为raylet里面会根据arnold role，每个node注册100个占位符资源
        num_current_nodes = int(ray.cluster_resources().get(self.worker_pool_name, 0)) // 100
        desired_nodes = num_current_nodes + num_new_nodes

        # 2. Scale up/down workers
        arnold_url = "https://arnold.bytedance.net/api/v3/trial_role_scale/"
        resp = requests.post(
            arnold_url,
            headers={"X-Jwt-Token": jwt_token},
            json={
                "trial_id": os.environ["ARNOLD_TRIAL_ID"],
                "scale_roles": {
                    self.worker_pool_name: desired_nodes
                }
            },
        )
        if resp.status_code >= 300:
            logid = resp.headers.get('x-tt-logid')
            raise ArnoldAPIError(f"arnold api(POST {arnold_url}) does not respond with status code 2xx, "
                                 f"got({resp.status_code}), {logid=}, body: {resp.text}")
        resp_body = resp.json()
        status = resp_body.get('status')
        if status == 'failed':
            message = resp_body.get('msg')
            logid = resp.headers.get('x-tt-logid')
            raise ArnoldAPIError(f"arnold api(POST {arnold_url}) failed with {message=}. {logid=}, body: {resp_body}")

        # response example
        # resp: {
        # 'id': 165,
        # 'create_time': '2025-04-09T05:22:59.750214Z',
        # 'update_time': '2025-04-09T05:23:00.436749Z',
        # 'scale_roles': {'worker': {'num': 1, 'robust_run_index': 0}},
        # 'status': 'failed',
        # 'msg': 'submit to metis failed',
        # 'submitter': {'id': 54142, 'username': 'x.lixiang', 'is_superuser': False, 'is_staff': False}}
        ticket_id = resp_body["id"]
        print(f'successfully call arnold api to scale up to {desired_nodes} nodes. {ticket_id=}')
        return ticket_id

    def _check_scaling_ticket_status(self, ticket_id) -> bool:
        jwt_token = self._get_jwt_token()
        arnold_url = f"https://arnold.bytedance.net/api/v3/trial_role_scale/{ticket_id}/"
        resp = requests.get(
            arnold_url,
            headers={"X-Jwt-Token": jwt_token},
        )
        # resp: {
        # 'id': 165,
        # 'create_time':
        # '2025-04-09T05:22:59.750214Z',
        # 'update_time': '2025-04-09T05:23:00.436749Z',
        # 'scale_roles': {'worker': {'num': 1, 'robust_run_index': 0}},
        # 'status': 'failed',
        # 'msg': 'submit to metis failed',
        # 'submitter': {'id': 54142, 'username': 'x.lixiang', 'is_superuser': False, 'is_staff': False}}
        if resp.status_code >= 300:
            logid = resp.headers.get('x-tt-logid')
            raise ArnoldAPIError(f"arnold api({arnold_url}) does not respond with status code 2xx, "
                                 f"got({resp.status_code}), {logid=}, body: {resp.text}")
        resp_body = resp.json()
        status = resp_body.get('status')
        if status == 'failed':
            # 是发你了
            logid = resp.headers.get('x-tt-logid')
            message = resp_body.get('msg')
            raise ArnoldAPIError(f"arnold api({arnold_url}) failed with {message=}. {logid=}, body: {resp_body}")
        elif status == 'success':
            return True
        else:
            # waiting/scaling
            # 其余状态全都按未完成，需要等待处理
            return False

    def _get_jwt_token(self):
        if self._jwt_token_cache is not None:
            now = time.time()
            expired = now > self._jwt_token_timestamp + self._jwt_token_expire_seconds
            if not expired:
                return self._jwt_token_cache
            else:
                self._jwt_token_cache = None
                self._jwt_token_timestamp = None

        with open("/etc/tce_dynamic/identity.token") as f:
            zti_token = f.read()

        byte_region = ""
        if os.environ.get("BYTE_REGION", "") not in ("CN", ""):  # 如果没有BYTE_REGION，默认当作cn
            byte_region = "-i18n"
        resp = requests.get(
            f"https://cloud{byte_region}.bytedance.net/auth/api/v1/jwt",
            headers={"X-ZTI-Token": zti_token},
        )

        self._jwt_token_cache = resp.headers["x-jwt-token"]
        self._jwt_token_timestamp = time.time()
        return self._jwt_token_cache
