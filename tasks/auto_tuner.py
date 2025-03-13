"""
Generate runtime parallel configuration for specific models and hardware

Example:

```sh
# Recommanded recipe format: gpu_ngpus_model_size_seqlen.yaml, e.g.,
`h800_128_m8_20b_18k.yaml`


# Generate using torchrun
torchrun --nproc_per_node=$ARNOLD_WORKER_GPU --nnodes=$ARNOLD_WORKER_NUM --node_rank=$ARNOLD_ID \
    --master_addr=$ARNOLD_WORKER_0_HOST --master_port=12321 \
    tasks/auto_tuner.py \
    --model hdfs://haruna/home/byte_data_seed/ssd_hldy/user/gracexu/exp/qwen2.5_32b_instruct_mariana/qwen2.5_32b_ins_v7.1_refge3_sp_fix-chatml_250217/rl_init/1230a1 \
    --max-seqlen 22528 \
    --nnodes 32 --ngpus-per-node 8 --gpu-type H800 \
    --recipe-out hdfs://haruna/home/byte_data_seed/lf_lq/user/zhiqi.0/rlhf/recipes/H800_256_qwen_32b_22k.yaml \
    2>&1 | tee log.txt

cat ./auto.yaml
```

* Run with generated recipe
```sh
python3 main_ppo.py \
    recipe=./auto.yaml \
    ...
```
"""

from typing import List, Tuple
from dataclasses import dataclass
from verl.utils.fs import copy_local_path_from_hdfs
import argparse
import yaml
import numpy as np
import copy
import warnings
import torch
import os

from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision
from transformers import AutoConfig, AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from alpha_seed.workers.actors.initialize import create_mesh, parallel_load_safetensors, parallel_init_fsdp_fn, meta_device_init
from alpha_seed.workers.hybrid_engine.fsdp_gather import ulysses_pad_and_slice_inputs
from dist_attn.ulysses.parallel_states import set_ulysses_sequence_parallel_group
from dist_attn.ulysses.ops import gather_outputs
from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch
from alpha_seed.workers.actors.checkpoint.extensions import register_dtensor_save_hook
from alpha_seed.models.transformers.parallel import apply_parallel_plan
from alpha_seed.models.transformers.ops import clip_grad_norm_
from alpha_seed.workers.actors.offload import offload_fsdp_optimizer, load_fsdp_optimizer
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from tests.hybrid_engine.utils import print_each_rank
from tests.hybrid_engine.test_parallel import init_random_data
from alpha_seed.workers.actors import activation_offload
import hdfs_io
import verl.utils.torch_functional as verl_F
import torch.distributed as dist
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import os

os.environ['TRITON_CACHE_MANAGER'] = 'triton.runtime.cache:RemoteCacheManager'
os.environ['TRITON_REMOTE_CACHE_BACKEND'] = 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend'
os.environ['BPEX_NO_WARN_ON_UNTUNED_CASE'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--max-seqlen", type=int, default=16384, help="max sequence length of a sample")
parser.add_argument("--nnodes", type=int)
parser.add_argument("--ngpus-per-node", type=int)
parser.add_argument("--gpu-type", type=str, choices=["H20", "H800", "H100", "L20"])
parser.add_argument("--mem-margin", type=float, default=0.15, help="ratio of reserved memory in GB of total memory")
parser.add_argument("--recipe-output", type=str, default='./auto.yaml', help="can be local or hdfs path")
args = parser.parse_args()
print(args)

GpuMemorySpec = {
    "H100": 80.0,
    "H800": 80.0,
    "H20": 95.0,
    "L20": 45.0,
}
HaveNVLink = {
    "H100": True,
    "H800": True,
    "H20": True,
    "L20": False,
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
        self.total_params = None  # will be set in init_model_and_optimizer
        self.root_params = None  # will be set in init_model_and_optimizer
        self.max_seqlen: int = max_seqlen
        self.seed = 42
        # runtime config
        self.act_offload_ctx = None  # set in profiling

    def init_model_and_optimizer(self, parallel_config: ParallelConfig, num_layers: int = 10):
        meshes = create_mesh(parallel_config.fsdp_size, parallel_config.tp_size, parallel_config.sp_size)
        fsdp_mesh, tp_mesh = meshes[:2]

        setattr(self.config, '_moe_implementation', 'fused')
        setattr(self.config, 'embd_pdrop', 0.0)
        setattr(self.config, 'attention_dropout', 0.0)
        setattr(self.config, 'resid_pdrop', 0.0)

        # init original model
        with meta_device_init(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoModelForCausalLM.from_config(config=self.config,
                                                     torch_dtype=torch.float32,
                                                     attn_implementation="flash_attention_2")
            self.total_params = sum(p.numel() for p in model.parameters())
            self.root_params = sum(p.numel() for p in model.parameters(recurse=False))
            print_each_rank(f"total parameters of the model: {self.total_params / (1024 ** 3):.2f} B")

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
            # recompute and act offload
            torch.utils.checkpoint.CheckpointFunction = activation_offload.CheckpointFunction
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
            # parallelize model
            shard_plan = apply_parallel_plan(model, config, tp_mesh)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)
        auto_wrap_policy = get_fsdp_wrap_policy(module=model)
        shards = parallel_load_safetensors(self.model_path)
        init_fn = parallel_init_fsdp_fn(model, shards)
        strategy = ShardingStrategy.HYBRID_SHARD if fsdp_mesh.ndim > 1 and fsdp_mesh.size(
        ) > 1 else ShardingStrategy.FULL_SHARD

        model = FSDP(model,
                     use_orig_params=True,
                     param_init_fn=init_fn,
                     auto_wrap_policy=auto_wrap_policy,
                     sharding_strategy=strategy,
                     mixed_precision=mixed_precision,
                     cpu_offload=None,
                     forward_prefetch=True,
                     sync_module_states=False,
                     device_id=torch.cuda.current_device(),
                     device_mesh=fsdp_mesh)
        shards.clear()
        register_dtensor_save_hook(model, shard_plan)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        print_each_rank(f"after init model: {torch.cuda.memory_allocated() / (1024**3):.2f} GB | "
                        f"max allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        torch.cuda.reset_peak_memory_stats()
        print_each_rank(f"after reset peak memory states: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        return model, optimizer, meshes

    def train_one_step(self, model, optimizer, meshes, max_token, accum_steps: int = -1):

        fsdp_mesh, tp_mesh, sp_mesh, gather_mesh = meshes
        self.act_offload_ctx = activation_offload.get_offload_context(True, model)

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
            with self.act_offload_ctx:
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

        clip_grad_norm_(model, max_norm=1.0).item()
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
    ):
        """
        Generate estimate memory function
        """
        model: FSDP
        token_stop = max(self.max_seqlen * 2, token_stop)
        config = self.empirical_config()
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

    def empirical_config(self):
        have_tp_implementation = self.config.model_type in (
            "seed_m8",
            "deepseek_v3",
            "seed_p6dense",
        )
        tp_size = 1
        if have_tp_implementation:
            tp_size = 2 if self.env.nvlink else 4
        ngpus_per_node = min(self.env.ngpus_per_node, dist.get_world_size())
        assert ngpus_per_node % tp_size == 0
        sp_size = ngpus_per_node // tp_size
        return ParallelConfig(
            max_token_len=self.max_seqlen,
            fsdp_size=-1,
            sp_size=sp_size,
            tp_size=tp_size,
            act_offload=True,
        )

    def search(self, step: int = 512, plot_file: str = None) -> ParallelConfig:

        memory_fn = self.gen_memory_est_fn()
        curr_config = self.empirical_config()
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
        return curr_config

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
        while tp_size > 1 and self.config.num_attention_heads % tp_size != 0:
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


if __name__ == '__main__':

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    filepath: str = args.recipe_output
    if filepath.startswith("hdfs://"):
        if hdfs_io.exists(filepath):
            print(f"{filepath} exists, search stopped.")
            dist.destroy_process_group()
            exit(0)

    env = Env(
        args.gpu_type,
        args.nnodes,
        args.ngpus_per_node,
        args.nnodes * args.ngpus_per_node,
        GpuMemorySpec[args.gpu_type] * (1.0 - args.mem_margin),
        HaveNVLink[args.gpu_type],
    )
    tuner = AutoTuner(args.model, args.max_seqlen, env)

    config = tuner.search(plot_file="search.png")
    if dist.get_rank() == 0:
        print(f"Find solution: {config}")
        recipe = tuner.export_recipe(config, filepath)

    dist.destroy_process_group()
