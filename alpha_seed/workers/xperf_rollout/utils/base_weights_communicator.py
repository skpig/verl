from typing import List


class WeightsCommunicator:

    def setup_as_client(self, role, source_address: str):
        pass

    def setup_as_server(self, ifname=None) -> str:
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

    def update_standalone_worker_end(self, addresses: List[str]):
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
