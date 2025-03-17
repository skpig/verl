import sys

paths = ["/opt/tiger/alpha_seed", "/opt/tiger/seed_models", "/opt/tiger/verl"]
sys.path.extend(paths)
import ray
import torch
import wandb
from alpha_seed.workers.actors.async_actor_ref_worker import AsyncActorRolloutRefWorker
from alpha_seed import core_algos
from alpha_seed.utils.reward_score import math_v1
from verl import DataProto
from verl.utils.py_functional import append_to_dict
import verl.utils.torch_functional as verl_F
from alpha_seed.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torch.utils.data import DataLoader
from single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from single_controller.ray.base import create_colocated_worker_cls


def init_client():
    # please start server first with:     server_client.role="server"
    runtime_env = {'env_vars': {"PYTHONPATH": ":".join(paths)}}
    # attach to the server
    ray.init(namespace="alphaseed", runtime_env=runtime_env, address="auto")
    # get actors from server
    kv_store = ray.get_actor(name="kv_store")
    worker_names = ray.get(kv_store.get_by_key.remote("hybrid_pool"))
    actor_rollout_cls = RayClassWithInitArgs(cls=AsyncActorRolloutRefWorker, config=None, role="actor_rollout_ref")
    worker_dict_cls = create_colocated_worker_cls(class_dict={"actor_rollout_ref": actor_rollout_cls})
    wg_dict = RayWorkerGroup(ray_cls_with_init=worker_dict_cls, worker_names=worker_names)
    spawn_wg = wg_dict.spawn(prefix_set=["actor_rollout_ref"])
    wg = spawn_wg["actor_rollout_ref"]
    return wg, kv_store


# todo: fixme
def set_default_values(batch, max_response_length):
    missing_keys = ["rollout_log_probs", "probs_gt_threshold_num", "probs_lt_threshold_sum", "off_policy_steps"]
    for key in missing_keys:
        if key not in batch:
            batch.batch[key] = torch.zeros(batch.batch['input_ids'].shape[0],
                                           max_response_length,
                                           dtype=torch.bfloat16,
                                           device=batch.batch['input_ids'].device).fill_(-1)
    return batch


TRAIN_FILE = "hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/qiying/projects/alphaseed/datasets/opensource/deepscaler_train_40k.parquet"
batch_size = 128
mini_steps = 8
num_bon = 8
wandb.init(project="alphaseed_example", config={"batch_size": batch_size, "num_bon": num_bon, "mini_steps": mini_steps})

wg, kv_store = init_client()
remote_config = ray.get(kv_store.get_by_key.remote("config"))
tokenizer = ray.get(kv_store.get_by_key.remote("tokenizer"))
tokenizer.padding_side = "left"
max_prompt_length = 2048
train_dataset = RLHFDataset(parquet_files=TRAIN_FILE,
                            tokenizer=tokenizer,
                            prompt_key="prompt",
                            answer_key="answer",
                            max_prompt_length=max_prompt_length)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=None,
                              drop_last=True,
                              collate_fn=collate_fn)
generation_kwargs = {
    'do_sample': True,
    'top_k': 0,
    'top_p': 1.,
    'temperature': 1.,
}


def pg_loss_fn(config, micro_data, full_entropy, log_prob):
    old_log_prob = micro_data['old_log_probs']
    advantages = micro_data['advantages']
    clip_ratio = config.clip_ratio

    responses = micro_data['responses']
    response_length = responses.size(1)
    attention_mask = micro_data['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    ratio = torch.exp(log_prob - old_log_prob)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_losses_clip = torch.maximum(pg_losses1, pg_losses2)
    pg_loss = verl_F.masked_mean(pg_losses_clip, response_mask)

    metrics = {"pg_loss": pg_loss.detach().item()}
    metrics["actor/ppo_kl_sum"] = 0  # todo: remove it
    return pg_loss, metrics


# wg.set_actor_loss_fn(pg_loss_fn)

step = 1
for batch_dict in train_dataloader:
    batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info={'generation_kwargs': generation_kwargs})
    batch = set_default_values(batch, remote_config.data.max_response_length)
    prompts = tokenizer.batch_decode(batch.batch["input_ids"], skip_special_tokens=True)

    print(f"step: {step}, gen")
    batch = batch.repeat(num_bon)
    gen_out = wg.generate_sequences(batch)

    print(f"step: {step}, logprob")
    gen_out.batch["prompts"] = gen_out.batch["input_ids"][:, :max_prompt_length]
    gen_out.batch["responses"] = gen_out.batch["input_ids"][:, max_prompt_length:]
    gen_out.meta_info = {'generation_kwargs': generation_kwargs}
    gen_out = wg.old_log_probs(gen_out)

    print(f"step: {step}, reward")
    sequences = tokenizer.batch_decode(gen_out.batch["input_ids"], skip_special_tokens=True)
    response_ids = gen_out.batch['input_ids'][:, max_prompt_length:]
    reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
    scores = []
    seqlens = []
    for i in range(batch_size * num_bon):
        # compute reward
        ground_truth = batch[i].non_tensor_batch['reward_model']['ground_truth']
        score = math_v1.compute_score(sequences[i], ground_truth)
        scores.append(score)
        prompt_length = batch[i].batch['input_ids'].shape[-1]
        valid_response_length = gen_out[i].batch['attention_mask'][prompt_length:].sum().item()
        seqlens.append(valid_response_length)
        # give score to the eos token
        reward_tensor[i, valid_response_length - 1] = score

    avg_score = sum(scores) / len(scores)
    avg_seqlen = sum(seqlens) / len(seqlens)
    print(f"step: {step}, score: {avg_score}, seqlen: {avg_seqlen}")
    metrics = {
        "train/score": avg_score,
        "train/response_length": avg_seqlen,
    }

    print(f"step: {step}, advantage")
    response_length = gen_out.batch['responses'].size(1)
    response_mask = gen_out.batch['attention_mask'][:, -response_length:]
    index = batch.non_tensor_batch['index']
    advantages, returns, adv_metrics = core_algos.compute_grpo_advantage_return(token_level_scores=reward_tensor,
                                                                                eos_mask=response_mask,
                                                                                index=index,
                                                                                num_bon=num_bon)
    gen_out.batch['advantages'] = advantages
    gen_out.batch['returns'] = returns
    gen_out.batch['upgo_advantages'] = torch.zeros_like(advantages)
    gen_out.meta_info['global_token_num'] = torch.sum(gen_out.batch['attention_mask'], dim=-1).tolist()

    print(f"step: {step}, train actor")
    mini_batches = gen_out.chunk(mini_steps)

    lr_scheduler_step = False
    for batch_idx, mini_batch in enumerate(mini_batches):
        if batch_idx == (mini_steps - 1):
            lr_scheduler_step = True
        mini_batch.meta_info["lr_scheduler_step"] = lr_scheduler_step
        actor_output = wg.train_actor(mini_batch)
        print(actor_output)
    # actor_output = wg.update_actor(gen_out)
    # print(actor_output)

    wandb.log(metrics, step=step)
    step += 1
