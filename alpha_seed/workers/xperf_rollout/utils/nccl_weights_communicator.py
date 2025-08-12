import os
import threading

import torch
from unittest.mock import patch

from alpha_seed.workers.xperf_rollout.utils.vit_inferencer import TorchVitInferencer
from alpha_seed.workers.xperf_rollout.utils.base_weights_communicator import WeightsCommunicator
from verl.utils.debug import log_gpu_memory_usage


class NCCLWeightsCommunicator(WeightsCommunicator):

    def __init__(self, inference_engine, standalone=False, device_mesh=None, enable_aiomonitor=False):
        self.inference_engine = inference_engine
        self.standalone = standalone
        self.device_mesh = device_mesh
        self.enable_aiomonitor = enable_aiomonitor
        self._setup_completed = threading.Event()

    def wait_for_setup_completed(self):
        self._setup_completed.wait()

    @property
    def has_setup(self):
        return self._setup_completed.is_set()

    def setup_standalone_worker_comm(self, hybrid_master_address, standalone_master_address, port, role):
        assert role in ["standalone_rollout", "standalone_validator", "standalone_rollout_server"]
        assert (hybrid_master_address is not None)
        assert (standalone_master_address is not None)

        master_address = hybrid_master_address[0].meta_info["hybrid_master_addr"]
        hybrid_world_size = len(hybrid_master_address)
        standalone_world_size = len(standalone_master_address)
        rank = torch.distributed.get_rank() + (0 if not self.standalone else hybrid_world_size)
        world_size = hybrid_world_size + standalone_world_size
        comm_info = {
            "hybrid_world_size": hybrid_world_size,
            "standalone_world_size": standalone_world_size,
            "rank": rank,
            "world_size": world_size,
            "master_address": master_address,
            "master_port": port
        }
        print(comm_info)

        with patch.dict(
                os.environ,
            {
                'RANK': str(comm_info["rank"]),
                'WORLD_SIZE': str(comm_info["world_size"]),
                'LOCAL_RANK': str(comm_info["rank"] % 8),
                'LOCAL_WORLD_SIZE': str(min(8, comm_info["world_size"])),
                'MASTER_ADDR': str(comm_info["master_address"]),
                'MASTER_PORT': str(comm_info["master_port"]),  # find a free port
                # need to disable custom ar for global connection
                'XPERF_CUSTOM_ALL_REDUCE': "0",
            }):
            comm_info["nccl_layer"] = torch.classes.XGPT.NCCLPrimitive()
            comm_info["nccl_layer"].init(f"standalone_{int(port)}", comm_info["world_size"], comm_info["rank"], "tcp",
                                         0)
        setattr(self, f"{role}_comm_info", comm_info)
        self._setup_completed.set()

    def update_standalone_worker(self, role):

        def _update_xperf_vit_model(comm_fn, comm_rank):

            def comm_and_assign(module, param_names=None):
                if param_names is None:
                    param_names = ["weight", "bias"]
                for param_name in param_names:
                    param = getattr(module, param_name).cuda()
                    comm_fn(param, comm_rank)
                    setattr(module, param_name, param)

            # TODO: if freezed, we don't need to update vit model
            vit_engine = self.inference_engine.vit_engine

            # for torch vit
            if isinstance(vit_engine, TorchVitInferencer):
                vit_engine.update_standalone_weights(comm_fn, comm_rank)
                return

            if hasattr(vit_engine.visual_encoder.module, "layers_weight"):
                for layer, layer_weight in enumerate(vit_engine.visual_encoder.module.layers_weight):
                    for i, weight in enumerate(layer_weight):
                        if isinstance(weight, torch.Tensor):
                            weight = weight.cuda()
                            comm_fn(weight, comm_rank)
                            vit_engine.visual_encoder.module.layers_weight[layer][i].data = weight.data
                comm_and_assign(vit_engine.visual_encoder.module.patch_embed.proj)
            else:
                for layer in vit_engine.visual_encoder.module.custom_decoder.layers:
                    comm_and_assign(layer.norm1)
                    comm_and_assign(layer.norm2)
                    comm_and_assign(layer.mlp.fc1)
                    comm_and_assign(layer.mlp.fc2)
                    comm_and_assign(layer.attn.proj)
                    comm_and_assign(layer.attn.qkv, ['weight'])
                    comm_and_assign(layer.attn, param_names=['q_bias', 'v_bias'])
                comm_and_assign(vit_engine.visual_encoder.module.custom_decoder.patch_embed.proj)
            comm_and_assign(vit_engine.ln_vision)
            comm_and_assign(vit_engine.seed_proj[0])
            comm_and_assign(vit_engine.seed_proj[2])

        def _update_xperf_model(comm_fn, comm_rank):
            layernorm_weight = self.inference_engine.engine.module.layernorm_weight.cuda()
            lm_head_weight = self.inference_engine.engine.module.lm_head_weight.cuda()
            wte_weight = self.inference_engine.engine.module.wte_weight.cuda()
            comm_fn(layernorm_weight, comm_rank)
            comm_fn(lm_head_weight, comm_rank)
            comm_fn(wte_weight, comm_rank)
            self.inference_engine.engine.module.layernorm_weight = layernorm_weight
            self.inference_engine.engine.module.lm_head_weight = lm_head_weight
            self.inference_engine.engine.module.wte_weight = wte_weight
            layers_weight = self.inference_engine.engine.module.layers_weight
            for layer, layer_weight in enumerate(layers_weight):
                for i, weight in enumerate(layer_weight):
                    if isinstance(weight, torch.Tensor):
                        origin_dtype = weight.dtype
                        if origin_dtype == torch.float8_e4m3fn or origin_dtype == torch.uint8:
                            # use int8 to communicate
                            weight = weight.view(torch.int8)
                            if (self.inference_engine.engine.module.quant_mode == "WFP8"):
                                origin_dtype = torch.float8_e4m3fn
                        weight = weight.cuda()
                        comm_fn(weight, comm_rank)
                        self.inference_engine.engine.module.layers_weight[layer][i].data = weight.view(
                            origin_dtype).data
            if hasattr(self.inference_engine, 'vit_engine'):
                _update_xperf_vit_model(comm_fn, comm_rank)
            self.inference_engine.current_steps = 0

        comm_info = getattr(self, f"{role}_comm_info")
        hybrid_world_size = comm_info["hybrid_world_size"]
        standalone_world_size = comm_info["standalone_world_size"]
        rank = comm_info["rank"]
        world_size = comm_info["world_size"]
        nccl_layer = comm_info["nccl_layer"]

        if self.standalone:
            from_rank = rank % hybrid_world_size
            _update_xperf_model(nccl_layer.recv, from_rank)
        else:
            for i in range((world_size - 1) // hybrid_world_size):
                to_rank = rank + hybrid_world_size + i * hybrid_world_size
                if to_rank < world_size:
                    _update_xperf_model(nccl_layer.send, to_rank)
        log_gpu_memory_usage(f'After {role} update')
