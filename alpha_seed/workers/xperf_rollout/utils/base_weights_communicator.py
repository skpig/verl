from typing import List


class WeightsCommunicator:

    def setup_as_client(self, role, source_address: str):
        pass

    def setup_as_server(self, ifname=None):
        pass

    def setup_standalone_worker_comm(self, hybrid_master_address, standalone_master_address, port, role):
        pass

    def update_standalone_worker(self, role):
        raise NotImplementedError()

    def update_standalone_worker_end(self, addresses: List[str]):
        pass

    def wait_for_setup_completed(self):
        raise NotImplementedError()
