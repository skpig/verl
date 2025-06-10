"""
Generate runtime parallel configuration for specific models and hardware.

Note if you run it standalone, you should not involve a pure-cpu head

Example:

```sh
# Recommended recipe format: gpu_ngpus_model_size_seqlen.yaml, e.g.,
`h800_128_m8_20b_18k.yaml`

python3 -m alpha_seed.tuner.auto_tuner \
    --model hdfs://haruna/home/byte_data_seed/lf_lq/user/zhiqi.0/models/p6dense-72B-Instruct \
    --max-seqlen 22528 \
    --gpu-type H800 \
    --nnodes 32 \
    --ngpus-per-node 8 \
    --export test.yaml \
    --tp-size 1

```

* Run with generated recipe

```sh
python3 main_ppo.py \
    recipe=hdfs://xxx.yaml \
    ...
```
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from verl.utils.fs import copy_local_path_from_hdfs
from argparse import Namespace
import yaml
import numpy as np
import copy
import warnings
import torch
import os

from transformers import AutoConfig, AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from alpha_seed.workers.fsdp.initialize import create_mesh, meta_device_init
from alpha_seed.workers.hybrid_engine.fsdp_gather import ulysses_pad_and_slice_inputs
from dist_attn.ulysses.parallel_states import set_ulysses_sequence_parallel_group
from dist_attn.ulysses.ops import gather_outputs
from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch, get_parallel_plan
from alpha_seed.workers.fsdp import fully_shard
from alpha_seed.workers.fsdp.offload import offload_fsdp_optimizer, load_fsdp_optimizer
from tests.hybrid_engine.test_parallel import init_random_data
from alpha_seed.workers.fsdp.offload import activation_offload
import hdfs_io
import verl.utils.torch_functional as verl_F
import torch.distributed as dist
import matplotlib.pyplot as plt

import ray
from mono_rl.single_controller import Worker
from mono_rl.single_controller import register, Dispatch
from mono_rl.single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

GpuMemorySpec = {
    "H100": 80.0,
    "H800": 80.0,
    "H20": 95.0,
    "L20": 45.0,
    "A100-SXM4-80GB": 80.0,
}
HaveNVLink = {
    "H100": True,
    "H800": True,
    "H20": True,
    "L20": False,
    "A100-SXM4-80GB": True,
}
GpusPerNode = {
    "H100": 8,
    "H800": 8,
    "H20": 8,
    "L20": 16,
    "A100-SXM4-80GB": 8,
}


def m8_extra_memory(config, ntokens) -> float:
    # k, v mirror
    head_dim = config.hidden_size / config.num_attention_heads
    kv_heads = config.num_key_value_heads
    kv_mirror_act = kv_heads * head_dim * ntokens * 2 / (1024**3) * 2
    nlayers = len(config.kv_mirror_layers)
    return kv_mirror_act * nlayers


ExtraMemory = {
    "seed_m8": m8_extra_memory,
}


@dataclass
class ParallelConfig:
    max_token_len: int
    fsdp_size: int
    sp_size: int
    tp_size: int
    act_offload: bool


@dataclass
class Constraints:
    tp_size: Optional[int]


@dataclass
class Env:
    gpu_type: str
    nnodes: int
    ngpus_per_node: int
    ngpus: int
    memory: float
    nvlink: bool


def print0(*msg):
    if dist.get_rank() == 0:
        print(*msg)


class AutoTuner:

    def __init__(self, model_path: str, max_seqlen: int, env: Env):

        self.env = env
        self.model_path = copy_local_path_from_hdfs(model_path)
        self.config = AutoConfig.from_pretrained(self.model_path)
        self.max_seqlen: int = max_seqlen
        self.total_params = None  # will be set in init_model_info
        self.root_params = None  # will be set in init_model_info
        self.default_filename = None  # will be set in init_model_info
        self.init_model_info()
        self.seed = 42

    def init_model_info(self):
        # init original model
        with meta_device_init(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoModelForCausalLM.from_config(config=self.config,
                                                     torch_dtype=torch.float32,
                                                     attn_implementation="flash_attention_2")
            self.total_params = sum(p.numel() for p in model.parameters())
            self.root_params = sum(p.numel() for p in model.parameters(recurse=False))
            nparams = self.total_params / (1e9)
            nparams_str = f"{int(nparams)}B" if nparams > 1 else f"{int(nparams*1000)}M"
            print0(f"total parameters of the model: {self.total_params / (1024 ** 3):.2f} B")
            self.default_filename = f"{self.env.gpu_type}.{self.env.ngpus}.{self.config.model_type}.{nparams_str}.seq{self.max_seqlen//1024}k.yaml"

    def init_model_and_optimizer(self, parallel_config: ParallelConfig, num_layers: int = 10):
        meshes = create_mesh(parallel_config.fsdp_size, parallel_config.tp_size, parallel_config.sp_size)
        fsdp_mesh, tp_mesh = meshes[:2]

        setattr(self.config, '_moe_implementation', 'fused')
        setattr(self.config, 'embd_pdrop', 0.0)
        setattr(self.config, 'attention_dropout', 0.0)
        setattr(self.config, 'resid_pdrop', 0.0)

        # init a layer-shrinked model for evaluation
        with meta_device_init(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = copy.deepcopy(self.config)

            # shrink layers
            if config.num_hidden_layers > num_layers:
                setattr(config, 'num_hidden_layers', num_layers)
                if config.model_type == "seed_m8":
                    mirror_layers = int(num_layers * 0.2)
                    setattr(config, "kv_mirror_imitated_layers", list(range(0, mirror_layers)))
                    setattr(config, "kv_mirror_layers", list(range(num_layers - mirror_layers, num_layers)))
                    setattr(config, "pre_post_layernorm_layers", list(range(0, num_layers)))

            apply_monkey_patch(config)
            model = AutoModelForCausalLM.from_config(config=config,
                                                     torch_dtype=torch.float32,
                                                     attn_implementation="flash_attention_2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, _ = fully_shard(
                model=model,
                block_cls=model._no_split_modules[0],
                fsdp_mesh=fsdp_mesh,
                tp_plan=get_parallel_plan(config, tp_mesh),
                tp_mesh=tp_mesh,
                recompute=True,
                act_offload=True,
                param_offload=False,
                weights=self.model_path,
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        print0(f"after init model: {torch.cuda.memory_allocated() / (1024**3):.2f} GB | "
               f"max allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        torch.cuda.reset_peak_memory_stats()
        print0(f"after reset peak memory states: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        return model, optimizer, meshes

    def train_one_step(self, model: FSDP, optimizer, meshes, max_token, accum_steps: int = -1):

        fsdp_mesh, tp_mesh, sp_mesh, gather_mesh = meshes

        torch.manual_seed(self.seed)
        # self.seed += 1
        input_ids, input_ids_rolled, masks, position_ids, seqs = init_random_data(self.max_seqlen, max_token)
        unpad_size = input_ids.size(1)
        if sp_mesh.size() > 1:
            set_ulysses_sequence_parallel_group(sp_mesh.get_group())
            input_ids, position_ids, _ = ulysses_pad_and_slice_inputs(input_ids, position_ids, sp_mesh.size())
            input_ids_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rolled.unsqueeze(0), None, sp_mesh.size())
            input_ids_rolled = input_ids_rolled.squeeze(0)

        for _ in range(max(2, accum_steps)):
            # forward
            output = model(input_ids=input_ids,
                           position_ids=position_ids,
                           use_cache=False,
                           labels=input_ids_rolled,
                           temperature=1.0,
                           fuse_lm_head_ce_loss=True)
            log_probs = output.loss
            if sp_mesh.size() > 1:
                log_probs = gather_outputs(log_probs, gather_dim=0, padding_dim=0, unpad_dim_size=unpad_size)
            loss = verl_F.masked_mean(log_probs, masks)
            # backward
            loss.backward()

        model.clip_grad_norm_(max_norm=1.0).item()
        # zero out grad to get avoid of token dispatch randomness
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        if accum_steps > 0:
            load_fsdp_optimizer(optimizer, torch.cuda.current_device())
            optimizer.step()
            optimizer.zero_grad()
            offload_fsdp_optimizer(optimizer)
        for module in FSDP.fsdp_modules(model):
            module._flat_param.grad = None

        print0(f"seqlen: {input_ids.size(1)}, max allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return input_ids.size(1)

    def gen_memory_est_fn(
        self,
        token_stop: int = 256 * 1024,
        token_step_per_gpu: int = 4096,
        constraints: Constraints = None,
    ):
        """
        Generate estimate memory function
        """
        model: FSDP
        token_stop = max(self.max_seqlen * 2, token_stop)
        config = self.empirical_config(constraints)
        model, optimizer, meshes = self.init_model_and_optimizer(config)
        fsdp_mesh, tp_mesh, sp_mesh, gather_mesh = meshes
        fsdp_size = fsdp_mesh.size()
        tp_size = tp_mesh.size()
        sp_size = sp_mesh.size()

        # warmup for 2 steps and create optimizer states
        self.train_one_step(model, optimizer, meshes, max_token=self.max_seqlen)
        self.train_one_step(model, optimizer, meshes, max_token=self.max_seqlen)
        torch.cuda.empty_cache()

        # get static memory for running model
        nparams = sum(p.numel() for p in model.parameters())
        memory_at_update = (nparams * 16) / (1024**3)
        print0(f"state memory at update: {memory_at_update:.2f} GB")
        num_layers = len(FSDP.fsdp_modules(model)) - 1  # -1 remove root module
        print0(f"detected fsdp wrapped layers: {num_layers}")
        # root module params doesn't use tp
        num_param_per_layer = (nparams - self.root_params / fsdp_size) / num_layers * fsdp_size
        memory_at_fbw = (nparams * 8 + num_param_per_layer * 10) / (1024**3)
        print0(f"layer params: {num_param_per_layer / 1e9:.2f}B | memory: {num_param_per_layer * 4 / (1024**3):.2f} GB")
        print0(f"state memory at fw/bw: {memory_at_fbw:.2f} GB")

        maxtokens = []
        allocated_memories = []
        reserved_memories = []
        memory_retries = []

        try:
            # up to 256k
            for maxtoken in range(self.max_seqlen, token_stop, token_step_per_gpu * sp_size):
                self.train_one_step(model, optimizer, meshes, maxtoken)
                max_allocated = torch.tensor(torch.cuda.max_memory_allocated() / (1024**3),
                                             device=torch.cuda.current_device())
                dist.all_reduce(max_allocated, op=dist.ReduceOp.MAX)
                real_max_allocated = max_allocated.item()
                max_allocated = max_allocated.item() - memory_at_fbw
                max_reserved = torch.tensor(torch.cuda.max_memory_reserved() / (1024**3),
                                            device=torch.cuda.current_device())
                dist.all_reduce(max_reserved, op=dist.ReduceOp.MAX)
                max_reserved = max_reserved.item() - memory_at_fbw
                maxtokens.append(maxtoken)
                allocated_memories.append(max_allocated)
                reserved_memories.append(max_reserved)
                memory_retries.append(memory_retries)
                alloc_retries = torch.cuda.memory_stats()["num_alloc_retries"]
                print0(f"adding data: maxtokens: {maxtoken // 1024}K | "
                       f"allocated: {max_allocated:.2f} GB | reserved: {max_reserved:.2f} GB | "
                       f"mretry: {alloc_retries}")
                if real_max_allocated >= self.env.memory:
                    break

        except Exception as e:
            print(f"Got error in profiling: {e}, stopping...")
            torch.cuda.empty_cache()

        # deg=1 denotes for linear estimation
        assert len(maxtokens) > 1, f"detected profiled data number < 2"
        slope, intercept = self.predict(maxtokens, allocated_memories, "predict.png")
        # y_fit = slope * x + intercept
        assert slope > 0, f"expected act memory increases when number of token increases, but got {slope=}"
        print0(f"act_memory = {slope} * tokens + {intercept}")

        tp_size, sp_size = tp_mesh.size(), sp_mesh.size()
        assert self.env.ngpus % (tp_size * sp_size) == 0
        fsdp_size = self.env.ngpus // tp_size

        def estimate_memory(max_tokens: int):
            memory_at_update = (self.total_params * 16 / fsdp_size / tp_size) / (1024**3)
            # note: num_param_per_layer already counts the affect of tp
            static_memory = (self.total_params * 8 / fsdp_size / tp_size + num_param_per_layer * 10) / (1024**3)
            memory_at_fbw = static_memory + slope * max_tokens + intercept
            extra_memory_fn = ExtraMemory.get(self.config.model_type, None)
            extra_memory = extra_memory_fn(self.config, max_tokens / sp_size) if extra_memory_fn is not None else 0
            # print0(f"static: {static_memory}, fbw: {memory_at_fbw}, extra_memory: {extra_memory} GB")
            peak_memory = max(memory_at_update, memory_at_fbw + extra_memory)
            return peak_memory

        return estimate_memory

    def empirical_config(self, constraints: Constraints = None):
        have_tp_implementation = self.config.model_type in (
            "seed_m8",
            "deepseek_v3",
            "seed_p6dense",
        )
        # get num heads
        num_heads = getattr(self.config, "num_attention_heads", None)
        # TODO(zhiqi.0): remove this constraints after the codebase supports
        # tp / sp on num heads for qwen
        if self.config.model_type == "seed_p6dense":
            num_heads = getattr(self.config, "num_key_value_heads", None)
        if num_heads is None:
            warnings.warn(f"Cannot get num_attention_heads in {self.config.model_type}, assume to be 1")
            num_heads = 1
        num_heads = max(num_heads, 1)
        # determine tp size
        if (not constraints) or (constraints.tp_size is None):
            tp_size = 1
            # we only enable tp for models > 60B
            if have_tp_implementation and (self.total_params / 1e9) > 60:
                tp_size = 2 if self.env.nvlink else 4
            # shrink tp size to be divisible to num_heads
            while num_heads % tp_size != 0:
                tp_size = max(tp_size // 2, 1)
        else:
            tp_size = constraints.tp_size
        # determine sp size
        ngpus_per_node = min(self.env.ngpus_per_node, dist.get_world_size())
        assert ngpus_per_node % tp_size == 0, f"{ngpus_per_node=}, {tp_size=}"
        sp_size = ngpus_per_node // tp_size
        while num_heads % sp_size != 0:
            sp_size = max(sp_size // 2, 1)
        return ParallelConfig(
            max_token_len=self.max_seqlen,
            fsdp_size=-1,
            sp_size=sp_size,
            tp_size=tp_size,
            act_offload=True,
        )

    def search(
        self,
        constraints: Constraints = None,
        step: int = 512,
        export_path: str = None,
        plot_file: str = None,
    ) -> ParallelConfig:

        # check if search file exists
        if export_path:
            if export_path.startswith("hdfs://") and hdfs_io.exists(export_path):
                print(f"{export_path} exists, search stopped.")
                return export_path
            elif os.path.exists(export_path):
                print(f"{export_path} exists, search stopped.")
                return export_path

        memory_fn = self.gen_memory_est_fn(constraints=constraints)
        curr_config = self.empirical_config(constraints)
        init_memory = memory_fn(curr_config.max_token_len)
        assert init_memory < self.env.memory, f"No solution found. (Init memory: {init_memory:.2f} GB)"
        tokens = []
        memories = []
        while True:
            config = copy.copy(curr_config)
            config.max_token_len = config.max_token_len + step
            memory = memory_fn(config.max_token_len)
            print0(f"{config}: memory: {memory:.2f} GB")
            tokens.append(config.max_token_len)
            memories.append(memory)
            if memory > self.env.memory:
                break
            curr_config = config
        if plot_file and dist.get_rank() == 0:
            plt.plot(tokens, memories, 'o-', label='real')
            plt.xlabel('Token Number')
            plt.ylabel('Memory Usage (GB)')
            plt.savefig(plot_file)
            plt.close()

        if export_path:
            if dist.get_rank() == 0:
                print(f"Find solution: {config}")
                self.export_recipe(curr_config, export_path)
        dist.barrier()
        return export_path

    def predict(self, tokens: List[int], memories: List[int], plot_filepath: str = None) -> Tuple[float, float]:
        """
        Predict the memory usage given token number, and plot to the filepath if `plot_filepath` is provided.
        
        Returns:
            slope: float
            interception: float
        """
        # we use last 10 data points for prediction to get rid of
        # slow-increasement of memory due to large gradients
        tokens = np.array(tokens[-10:])
        memories = np.array(memories[-10:])
        slope, intercept = np.polyfit(tokens, memories, deg=1)
        if plot_filepath and dist.get_rank() == 0:
            plt.plot(tokens, memories, 'o-', label='real')
            fit_line = slope * tokens + intercept
            plt.plot(tokens, fit_line, 'r-', label='predict')
            plt.xlabel('Token Number')
            plt.ylabel('Memory Usage')
            plt.legend()
            plt.savefig(plot_filepath)
            plt.close()
        return slope, intercept

    def export_recipe(self, config: ParallelConfig, filepath: str):
        assert filepath.endswith(".yaml"), f"{filepath} must ends with .yaml"
        with open('tasks_scripts/recipes/template.yaml', "r") as f:
            template = yaml.safe_load(f)
        template["actor_rollout_ref"]["actor"]["ppo_max_token_len"] = config.max_token_len
        template["actor_rollout_ref"]["actor"]["fsdp_size"] = config.fsdp_size
        template["actor_rollout_ref"]["actor"]["tp_size"] = config.tp_size
        template["actor_rollout_ref"]["actor"]["ulysses_sequence_parallel_size"] = config.sp_size
        template["actor_rollout_ref"]["actor"]["act_offload"] = config.act_offload
        # if config.act_offload:
        #     template["actor_rollout_ref"]["actor"]["gc_freq"] = "micro"

        template["actor_rollout_ref"]["ref"]["max_token_len"] = config.max_token_len
        template["actor_rollout_ref"]["ref"]["fsdp_size"] = config.fsdp_size
        template["actor_rollout_ref"]["ref"]["tp_size"] = config.tp_size
        template["actor_rollout_ref"]["ref"]["ulysses_sequence_parallel_size"] = config.sp_size

        template["critic"]["ppo_max_token_len"] = config.max_token_len
        template["critic"]["fsdp_size"] = config.fsdp_size
        template["critic"]["tp_size"] = config.tp_size
        template["critic"]["ulysses_sequence_parallel_size"] = config.sp_size
        template["critic"]["act_offload"] = config.act_offload
        # if config.act_offload:
        #     template["critic"]["gc_freq"] = "micro"

        # calculate rollout tp size
        tp_size = self.env.ngpus_per_node
        num_heads = getattr(self.config, "num_attention_heads", None)
        # TODO(zhiqi.0): remove this constraints after the codebase supports
        # tp / sp on num_attention_heads for qwen
        if self.config.model_type == "seed_p6dense":
            num_heads = getattr(self.config, "num_key_value_heads", None)
        if num_heads is None:
            num_heads = 1
        while tp_size > 1 and num_heads % tp_size != 0:
            tp_size = tp_size // 2
        tp_size = max(tp_size, 1)
        template["actor_rollout_ref"].setdefault("rollout", {})["tensor_model_parallel_size"] = tp_size
        template["actor_rollout_ref"]["rollout"]["max_token_len"] = self.max_seqlen

        local_filepath = filepath
        if filepath.startswith("hdfs://"):
            local_filepath = os.path.join('/tmp/', os.path.basename(filepath))
        with open(local_filepath, "w") as f:
            yaml.safe_dump(template, f)
        print(f"recipe is dumped at local path {local_filepath}")
        if filepath.startswith("hdfs://"):
            folder = os.path.dirname(filepath)
            hdfs_io.makedirs(folder, exist_ok=True)
            hdfs_io.hput(local_filepath, folder)
            print(f"recipe is dumped at hdfs path: {filepath}")
        return template


@ray.remote
class RayAutoTuner(Worker):

    def __init__(self, model_path: str, max_seqlen: int, env: Env = None, mem_margin: float = 0.15):
        super().__init__()

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

        ngpus = dist.get_world_size()
        if not env:
            # get from runtime
            gpu_type = torch.cuda.get_device_name().split()[-1]
            env = Env(
                gpu_type,
                min(ngpus // GpusPerNode.get(gpu_type, 8), 1),
                min(GpusPerNode.get(gpu_type, 8), ngpus),
                ngpus,
                torch.cuda.get_device_properties(0).total_memory / (1024**3) * (1 - mem_margin),
                HaveNVLink.get(gpu_type, True),
            )

        self.tuner = AutoTuner(model_path=model_path, max_seqlen=max_seqlen, env=env)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def search(self, constraints: Constraints = None, export_path: str = None, export_dir: str = None):
        if export_path is None:
            if export_dir:
                export_path = os.path.join(export_dir, self.tuner.default_filename)
            else:
                export_path = self.tuner.default_filename
        return self.tuner.search(
            constraints=constraints,
            export_path=export_path,
            plot_file="search.png",
        )


@ray.remote
def auto_tune_task(config, ngpus_per_node: int, nnodes: int, save_dir: str = None):

    # standalone run
    if isinstance(config, Namespace):
        env = Env(
            args.gpu_type,
            args.nnodes,
            args.ngpus_per_node,
            args.nnodes * args.ngpus_per_node,
            GpuMemorySpec[args.gpu_type] * (1.0 - args.mem_margin),
            HaveNVLink[args.gpu_type],
        )
        constraints = Constraints(tp_size=args.tp_size)
        model_path = args.model
        max_seqlen = args.max_seqlen
        export_path = args.export
    # alphaseed rl job run
    else:
        env = None  # infer by runtime
        constraints = None
        model_path = config.actor_rollout_ref.model.path
        max_seqlen = config.data.max_prompt_length + config.data.max_response_length
        export_path = None  # will be constructed at runtime

    print(f"creating resource group for tuner: {ngpus_per_node}x{nnodes}")
    resource_pool = RayResourcePool([ngpus_per_node] * nnodes, use_gpu=True)
    class_with_args = RayClassWithInitArgs(
        RayAutoTuner,
        model_path,
        max_seqlen,
        env,
    )
    tuner = RayWorkerGroup(resource_pool, class_with_args, name_prefix="autotuner")
    filepath = tuner.search(constraints, export_path, save_dir)
    # release resources
    for pg in resource_pool.get_placement_groups():
        ray.util.remove_placement_group(pg)
    return filepath


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-seqlen", type=int, default=16384, help="max sequence length of a sample")
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--ngpus-per-node", type=int)
    parser.add_argument("--gpu-type", type=str, choices=["H20", "H800", "H100", "L20"])
    parser.add_argument("--mem-margin", type=float, default=0.15, help="ratio of reserved memory in GB of total memory")
    parser.add_argument("--export", type=str, default='./auto.yaml', help="can be local or hdfs path")
    parser.add_argument("--tp-size", type=int, default=None, help="constraints of tensor parallelism size")
    args = parser.parse_args()
    print(args)

    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise NotImplementedError("Ray cluster with head is not supported. "
                                  "Please create trial without ray configuration")

    # init ray
    if not ray.is_initialized():
        runtime_env = {
            'env_vars': {
                # redis server for triton
                'TRITON_CACHE_MANAGER': 'triton.runtime.cache:RemoteCacheManager',
                'TRITON_REMOTE_CACHE_BACKEND': 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend',
                # variables
                'NCCL_DEBUG': '0',
                'BPEX_NO_WARN_ON_UNTUNED_CASE': '1',
                'TORCH_NCCL_AVOID_RECORD_STREAMS': '1'
            }
        }
        ray.init(runtime_env=runtime_env)
    # run the task
    ray.get(auto_tune_task.remote(args, ngpus, 1))
