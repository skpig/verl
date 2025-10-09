from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class WeightsRankInfo:
    rank: int
    dp_rank: int
    tp_rank: int
    ip: str  # oob address (eth0 or bond0 or so on)
    worker_name: str
    ucx_address: str  # ip:port 格式的ucx server的地址
    oob_address: str  # ip:port 格式的oob控制地址(http或tcp)
    pid: int  # rank process id


class WeightsCommunicator:

    def setup_as_client(self, role, source_endpoint_info: List[WeightsRankInfo], init_recv_buffer: bool = True):
        pass

    def setup_as_server(self, ifname=None) -> Tuple[str, str]:
        """
        setup the server client/server mode
        return the main traffic address and oob control traffic address in format ip:port
        """
        pass

    def setup_standalone_worker_comm(self, hybrid_master_address, standalone_master_address, port, role):
        """
        for nccl only
        """
        pass

    def update_standalone_worker(self, role):
        """
        standalone worker starts to read weights from designated source
        """
        raise NotImplementedError()

    def update_standalone_worker_wait(self):
        """
        wait for update_standalone_worker complete, in case of update_standalone_worker()
        returns early for reading data from relay
        """
        return

    def update_standalone_worker_end(self, addresses: List[WeightsRankInfo]):
        """
        broadcast finishing signal to all hybrid rollout instance to tell them to exit weights serving mode.
        (due to be unable to interrupt the hybrid rollout rpc when serving is up, introduce this function)
        """
        pass

    def wait_for_setup_completed(self):
        """
        block until communicator has setup as client completed
        """
        raise NotImplementedError()

    def on_will_start_update(self):
        """
        在即将调用update_standalone_worker之前会调用，可用于处理一些状态变更，例如设置一些event不再接受新的client请求
        """
        pass

    def on_updated(self):
        """
        在调用完所有rank的update_standalone_worker之后会调用，可用于释放一些锁或者状态
        """
        pass

    @property
    def is_relay(self):
        return False

    @property
    def has_setup(self):
        raise NotImplementedError()
