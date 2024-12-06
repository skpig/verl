"""
PYTHONPATH=.:$PYTHONPATH torchrun --nproc_per_node=8 tests/hybrid_engine/test_save_load.py
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TRITON_CACHE_MANAGER'] = 'triton.runtime.cache:RemoteCacheManager'
os.environ['TRITON_REMOTE_CACHE_BACKEND'] = 'alpha_seed.utils.redis.triton_redis:BytedRedisRemoteCacheBackend'
os.environ['BPEX_NO_WARN_ON_UNTUNED_CASE'] = '1'

import seed_models  # noqa
import warnings

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision

from torch.distributed.device_mesh import init_device_mesh
from alpha_seed.workers.actors.initialize import parallel_load_safetensors, parallel_init_fsdp_fn, meta_device_init, create_init_fn
from alpha_seed.workers.actors.checkpoint.extensions import register_dtensor_save_hook

from verl.utils.fs import copy_local_path_from_hdfs
from tests.hybrid_engine.utils import prepare_data, print_each_rank, ref_loss_fn

p6_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf'
p6dense_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf'
p7_path = 'hdfs://haruna/home/byte_data_seed/ssd_lq/public/seed_models/Seed-2B5-P7_32k_sft29_32gpu'
m8_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/25B_MoE_SFT29_32k_bsz6_lr2e5_tp4_hf'


def get_model(init_empty=False, num_layers: int = 8, recompute=False):

    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    model_path = copy_local_path_from_hdfs(p6_path)

    with meta_device_init(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = AutoConfig.from_pretrained(model_path)
        setattr(config, '_moe_implementation', 'fused')
        setattr(config, 'num_hidden_layers', num_layers)
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
    # bind extensions so that we can flexibly shard DTensor to avoid padding
    register_dtensor_save_hook(actor_module_fsdp)
    return actor_module_fsdp


def train_one_step(fsdp_model):
    input_ids, input_ids_rolled, masks, position_ids = prepare_data()
    torch.manual_seed(42)
    output = fsdp_model(input_ids=input_ids, position_ids=position_ids)
    loss, probs = ref_loss_fn(output, input_ids_rolled, masks)
    loss.backward()
    return loss


def test_model_save_load():

    test_ckpt_folder = "tests/checkpoint"
    filepath = os.path.join(test_ckpt_folder, f"test_ckpt.rank{dist.get_rank()}.pt")
    os.makedirs(test_ckpt_folder, exist_ok=True)

    # test save
    fsdp_model = get_model(init_empty=False)
    save_loss = train_one_step(fsdp_model)
    print_each_rank(f"loss of saved model: {save_loss}")
    print_each_rank("saving state dict...")
    with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
        state_dict = fsdp_model.state_dict()
        torch.save(state_dict, filepath)

    # test load
    fsdp_model = get_model(init_empty=True)
    init_loss = train_one_step(fsdp_model)
    print_each_rank(f"loss before load: {init_loss}")
    print_each_rank(f"loading state dict...")
    with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
        device = torch.cuda.current_device()
        torch.load(filepath, map_location=f"cuda:{device}", weights_only=False)
        fsdp_model.load_state_dict(state_dict)

    load_loss = train_one_step(fsdp_model)
    print_each_rank(f"loss after load: {load_loss}")

    assert not torch.allclose(init_loss, load_loss)
    # bitwise correct test
    assert torch.allclose(save_loss, load_loss, rtol=0, atol=0)


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    torch.cuda.set_device(dist.get_rank())
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

    test_model_save_load()
    dist.destroy_process_group()
