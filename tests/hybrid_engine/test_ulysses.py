"""
Run with

PYTHONPATH=.:$PYTHONPATH torchrun --nproc_per_node=4 tests/hybrid_engine/test_ulysses.py
"""

import os

os.environ['TRITON_CACHE_MANAGER'] = 'triton.runtime.cache:RemoteCacheManager'
os.environ['TRITON_REMOTE_CACHE_BACKEND'] = 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend'
os.environ['BPEX_NO_WARN_ON_UNTUNED_CASE'] = '1'
os.environ['NCCL_DEBUG'] = 'warn'

# make deterministic
os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

torch.use_deterministic_algorithms(True)

import warnings

import seed_models
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig, AutoModelForCausalLM
from alpha_seed.workers.actors.initialize import parallel_load_safetensors, parallel_init_fsdp_fn, meta_device_init, create_init_fn
from alpha_seed.workers.hybrid_engine.fsdp_ulysses import ulysses_pad_and_slice_inputs
from dist_attn.ulysses.parallel_states import set_ulysses_sequence_parallel_group
from dist_attn.ulysses.ops import gather_outputs

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from tests.hybrid_engine.utils import prepare_data, print_each_rank, ref_loss_fn

p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
p6dense_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf'
p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'
m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/25B_MoE_SFT29_32k_bsz6_lr2e5_tp4_hf'


def get_model(model_path, init_empty=False, num_layers: int = 8, recompute=False):

    world_size = dist.get_world_size()
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    model_path = copy_local_path_from_hdfs(model_path)

    with meta_device_init(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = AutoConfig.from_pretrained(model_path)
        setattr(config, '_moe_implementation', 'fused')
        setattr(config, 'num_hidden_layers', num_layers)
        setattr(config, 'embd_pdrop', 0.0)
        setattr(config, 'attention_dropout', 0.0)
        setattr(config, 'resid_pdrop', 0.0)
        # for P7
        if hasattr(config, 'sliding_window') and config.sliding_window is not None:
            # only keep the first num_layers layers
            config.sliding_window = config.sliding_window[:num_layers]
        # if dist.get_rank() == 0:
        #     print(config)
        model = AutoModelForCausalLM.from_config(config=config,
                                                 torch_dtype=torch.float32,
                                                 attn_implementation="flash_attention_2")
        # enable recompute
        if recompute:
            model.gradient_checkpointing_enable()

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    auto_wrap_policy = get_fsdp_wrap_policy(module=model)

    if init_empty:
        init_fn = create_init_fn(model)
    else:
        shards = parallel_load_safetensors(model_path)
        init_fn = parallel_init_fsdp_fn(model, shards)

    actor_module_fsdp = FSDP(model,
                             use_orig_params=False,
                             param_init_fn=init_fn,
                             auto_wrap_policy=auto_wrap_policy,
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             mixed_precision=mixed_precision,
                             sync_module_states=False,
                             device_id=torch.cuda.current_device(),
                             device_mesh=device_mesh)

    return actor_module_fsdp


@torch.no_grad()
def gather_inputs(input_ids, input_ids_rolled, masks, position_ids):
    input_ids = gather_outputs(input_ids, gather_dim=1)
    input_ids_rolled = gather_outputs(input_ids_rolled, gather_dim=0)
    position_ids = gather_outputs(position_ids, gather_dim=1)
    masks = gather_outputs(masks, gather_dim=0)
    return input_ids, input_ids_rolled, masks, position_ids


def _run_one_step(fsdp_model, use_sp: bool = False):
    input_ids, input_ids_rolled, masks, position_ids = prepare_data()

    if use_sp:
        set_ulysses_sequence_parallel_group(dist.group.WORLD)
        input_ids, input_ids_rolled, masks, position_ids = gather_inputs(input_ids, input_ids_rolled, masks,
                                                                         position_ids)
        unpad_size = input_ids.size(1)
        input_ids, position_ids, _ = ulysses_pad_and_slice_inputs(input_ids, position_ids, dist.get_world_size())

    torch.manual_seed(42)
    output = fsdp_model(input_ids=input_ids, position_ids=position_ids, use_cache=False)

    if use_sp:
        output.logits = gather_outputs(output.logits, gather_dim=1, padding_dim=1, unpad_dim_size=unpad_size)

    loss, probs = ref_loss_fn(output, input_ids_rolled, masks)
    loss.backward()
    return loss


def test_ulysses(model: str):
    model_path = {"p6": p6_path, "p6dense": p6dense_path, "p7": p7_path, "m8": m8_path}[model]

    # run baseline
    fsdp_model = get_model(model_path)
    ref_loss = _run_one_step(fsdp_model, use_sp=False)
    print_each_rank(f"baseline loss: {ref_loss}")

    # run with ulysses
    if model == "p6":
        from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch_to_p6
        apply_monkey_patch_to_p6()
    elif model == "p6dense":
        from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch_to_p6_dense
        apply_monkey_patch_to_p6_dense()
    elif model == "p7":
        from alpha_seed.models.transformers.monkey_patch import apply_monkey_patch_to_p7
        apply_monkey_patch_to_p7()
    else:
        raise NotImplementedError(f"Unknown model type:{model}")
    fsdp_model = get_model(model_path)
    sp_loss = _run_one_step(fsdp_model, use_sp=True)
    print_each_rank(f"ulysses loss: {sp_loss}")

    torch.testing.assert_close(ref_loss, sp_loss)
    assert torch.allclose(ref_loss, sp_loss, atol=0, rtol=0)
    print(f"test passed")


if __name__ == '__main__':

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

    test_ulysses(model='p6')

    dist.barrier()
    dist.destroy_process_group()
