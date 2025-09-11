import os
import ray
import json
import torch
import socket
import logging
import argparse
import subprocess
import xperf_gpt
from transformers import AutoTokenizer, AutoConfig
from alpha_seed.workers.xperf_rollout.utils import get_xperf_gpt_config
from alpha_seed.utils.ckpt.hdfs import download_minimal_required_files
from xperf_gpt.inference.session import InferenceSession
from hdfs_io.hdfs_io import hcopy, hmkdir


class XEngine():

    def __init__(self, rank, world_size, master_addr, master_port, use_ep, vocab_tp, dump_config):
        self.rank = rank
        self.world_size = world_size
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        self.use_ep = use_ep
        self.vocab_tp = vocab_tp
        self.dump_config = dump_config

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        address = address.lstrip("[")
        address = address.rstrip("]")
        return address

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

    def setup(self):
        os.environ['MASTER_ADDR'] = str(self._master_addr)
        os.environ['MASTER_PORT'] = str(self._master_port)
        os.environ['LOCAL_RANK'] = str(self.rank)
        os.environ['RANK'] = str(self.rank)
        os.environ['LOCAL_WORLD_SIZE'] = str(self.world_size)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['XPERF_SESSION_SET_TORCH_DEVICE'] = "0"
        os.environ["XPERF_MOE_HACK_EXPERT"] = "1"
        os.environ["USE_SESSION_CACHE"] = "0"
        os.environ["XGPT_TUNER_ENABLE"] = "1"
        os.environ["XPERF_TUNER_ONLINE_VERSION"] = "2.0.0+xgpt"
        if self.dump_config:
            os.environ["XPERF_TUNER_INHOUSE_ENABLE"] = "1"
            os.environ["XPERF_TUNER_CONFIG_DUMP_ENABLE"] = "1"
            os.environ["XPERF_TUNER_CONFIG_DUMP_PATH"] = "/opt/tiger/config_dump"
        else:
            os.environ["XPERF_TUNER_CONFIG_DUMP_ENABLE"] = "0"

        is_multihost = int(os.getenv("WORLD_SIZE", "1")) > 8
        if is_multihost:
            os.environ["NCCL_SOCKET_IFNAME"] = os.getenv("NCCL_SOCKET_IFNAME", "eth0")
            os.environ["NCCL_IB_HCA"] = os.getenv("NCCL_IB_HCA", "^=mlx5_0")
            os.environ["NCCL_NVLS_ENABLE"] = "0"
            os.environ["NCCL_IB_GID_INDEX"] = "3"
            os.environ["NCCL_IB_DISABLE"] = "0"
            os.environ["NCCL_IB_TIMEOUT"] = "25"
            os.environ["NCCL_IB_RETRY_CNT"] = "7"
            os.environ["NCCL_MULTI_HOST"] = "1"

        torch.cuda.set_device(0)
        xperf_gpt.load_xperf_gpt()

        max_batch_size = 1
        num_slots = 1024
        max_length = 2048
        max_new_tokens = 2048
        self.engine = InferenceSession(num_slots=num_slots,
                                       max_batch_size=max_batch_size,
                                       max_length=max_length,
                                       use_vllm=True,
                                       slot_block_size=256,
                                       vocab_tp=self.vocab_tp)
        generate_kwargs = dict(max_new_tokens=max_new_tokens,
                               do_sample=False,
                               top_k=1,
                               top_p=0.7,
                               temperature=1.0,
                               context_only=False)
        self.engine.init_inference_engine("/opt/tiger/xperf_config.json",
                                          generate_kwargs,
                                          multi_host_tp=is_multihost,
                                          enable_metrics=True,
                                          use_ep=self.use_ep,
                                          rank0_split=False,
                                          mock_weights=True)

    def generate(self, prompt):
        self.engine.execute(prompt)
        return self.engine.get_metrics()["dp_rank_0"]["decode_per_token_latency"]


@ray.remote
class XServer():

    def __init__(self, tp_size, use_ep, vocab_tp, dump_config):
        super().__init__()
        self.tp_size = tp_size
        self.world_size = tp_size
        self.use_ep = use_ep
        self.vocab_tp = vocab_tp
        self.dump_config = dump_config

        self.workers = []
        self.setup_actors()

    def generate(self):
        remote_list = []
        for rank in range(self.world_size):
            remote_list.append(self.workers[rank].generate.remote(["小炒肉怎么做才好吃"]))
        results = ray.get(remote_list[0])
        return results

    def setup_actors(self):
        WorkerActor = ray.remote(num_cpus=1, num_gpus=1)(XEngine)
        master_actor = WorkerActor.remote(0, self.world_size, None, None, self.use_ep, self.vocab_tp, self.dump_config)
        self.workers.append(master_actor)
        master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
        logging.info("[setup_actors] workerActor initiating {}".format(WorkerActor))

        for rank in range(1, self.world_size):
            worker = WorkerActor.remote(rank, self.world_size, master_addr, master_port, self.use_ep, self.vocab_tp,
                                        self.dump_config)
            self.workers.append(worker)
        logging.info("[setup_actors] init workerActor {}".format(len(self.workers)))

        remote_list = []
        for worker in self.workers:
            remote_list.append(worker.setup.remote())
        ray.get(remote_list)


@ray.remote(num_gpus=8)
class Tuner():

    def __init__(self, save_path):
        self.save_path = save_path

    def tune_and_upload(self):

        def system_with_output(command):
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return result.returncode, result.stdout

        status, output = system_with_output(
            "XPERF_TUNER_CONFIG_DUMP_PATH=/opt/tiger/config_dump python3 -m xperf_tuner.inhouse_tuning")
        print("[TUNING] status {}, output {}".format(status, output))

        status, output = system_with_output(
            "xperf_tuner_helper.config_upload --path /opt/tiger/config_dump --version 2.0.0")
        print("[UPLOAD TUNING CONFIG TO SERVER] status {}, output {}".format(status, output))

        hmkdir(self.save_path)
        hcopy(f"/opt/tiger/config_dump", self.save_path)
        print("[UPLOAD TUNING CONFIG TO HDFS] {}".format(self.save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--quant_mode", type=str, choices=["NO_QUANT", "WFP8", "W8A8", "W4A8"], default="NO_QUANT")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--use_ep", type=bool, default=True)
    parser.add_argument("--vocab_tp", type=bool, default=True)
    args = parser.parse_args()

    local_path = download_minimal_required_files(args.model_path, False, 0, 1)
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=False)
    model_hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=False)
    model_xperf_config = get_xperf_gpt_config(model_config=model_hf_config, tokenizer=tokenizer)
    model_xperf_config["quant_mode"] = args.quant_mode
    with open(file="/opt/tiger/xperf_config.json", mode='w') as f:
        json.dump(model_xperf_config, f)
    logging.info("[XPERF CONFIG] {}".format(model_xperf_config))

    server = XServer.remote(tp_size=args.tp_size, use_ep=args.use_ep, vocab_tp=args.vocab_tp, dump_config=True)
    logging.info("[DECODE LATENCY BEFORE TUNING] {}".format(ray.get(server.generate.remote())))
    ray.kill(server)

    tuner = Tuner.remote(args.save_path)
    ray.get(tuner.tune_and_upload.remote())
    ray.kill(tuner)

    server = XServer.remote(tp_size=args.tp_size, use_ep=args.use_ep, vocab_tp=args.vocab_tp, dump_config=False)
    logging.info("[DECODE LATENCY AFTER TUNING] {}".format(ray.get(server.generate.remote())))
    ray.kill(server)


if __name__ == "__main__":
    main()
