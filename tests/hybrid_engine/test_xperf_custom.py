import os
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from alpha_seed.utils.server_client import KVStore
from mono_rl.single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
from omegaconf import OmegaConf
import ray
import ipdb

os.environ["XPERF_DUMP_NAN"] = "0"

default_conf = OmegaConf.load("/opt/tiger/alpha-seed/tasks/config/ppo_trainer.yaml")
default_actor_rollout_ref_config = default_conf.actor_rollout_ref
config = OmegaConf.create({
    "model": {
        "path": "hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf"
    },
    "rollout": {
        "name": "xperf_gpt",
        "prompt_length": 1024,
        "response_length": 1024,
        "tensor_model_parallel_size": 2,
        "enable_paged_attention": False,
        "use_vllm": False,
        "num_slots": 256,
        "slot_block_size": 1024,
        "gpu_memory_utilization": 0.6,
        "quant_mode": "NO_QUANT",
        "xperf_custom": {
            "enable": True,
            "backbone": "m8",
        },
    },
})

config = OmegaConf.merge(default_actor_rollout_ref_config, config)

ray.init(namespace="alphaseed")
resource_pool = RayResourcePool(process_on_nodes=[2], use_gpu=True)
ray_cls_with_init = RayClassWithInitArgs(cls=AsyncActorRolloutRefWorker, config=config, role="rollout")
wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
ray.get(wg.init_model())
kv_store = ray.remote(KVStore).options(name=KVStore.name, lifetime="detached").remote()
kv_store.set_key_val.remote("worker_names", wg.worker_names)

import torch
from mono_rl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from transformers import AutoTokenizer

prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

chat = [{"role": "user", "content": prompt}]

model_path = copy_local_path_from_hdfs(config.model.path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"

sentences = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

input_data = tokenizer(sentences, return_tensors="pt").to("cuda")

input_ids = input_data["input_ids"]
attention_mask = input_data["attention_mask"]
get_dummy_tensor = lambda: torch.zeros(input_ids.shape[0], config.rollout.response_length, dtype=torch.bfloat16).fill_(
    -1)

data = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "off_policy_steps": get_dummy_tensor(),
    "rollout_log_probs": get_dummy_tensor(),
}

data = DataProto.from_dict(data, meta_info={"generation_kwargs": config.rollout.train_generate_kwargs})

data = data.repeat(2)
output = wg.generate_sequences(data)
# output = wg.generate_sequences(data)

output_ids = output.batch["input_ids"]

text_out = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

for i in range(len(text_out)):
    print(text_out[i].replace(tokenizer.pad_token, ""))

ipdb.set_trace()
