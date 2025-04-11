import torch
from openai import OpenAI
from verl import DataProto
import os
''' example input: 
DataProtoItem(batch=TensorDict(
    fields={
        answer_attention_mask: Tensor(shape=torch.Size([1, 256]), device=cpu, dtype=torch.int8, is_shared=False),
        answer_input_ids: Tensor(shape=torch.Size([1, 256]), device=cpu, dtype=torch.int32, is_shared=False),
        attention_mask: Tensor(shape=torch.Size([1, 256]), device=cpu, dtype=torch.int8, is_shared=False),
        input_ids: Tensor(shape=torch.Size([1, 256]), device=cpu, dtype=torch.int32, is_shared=False)},
    batch_size=torch.Size([1]),
    device=None,
    is_shared=False), non_tensor_batch= ... '''


def _internal_call(item, config):
    # openAI client call rollout
    client = OpenAI(api_key="useless-api-key", base_url=f"http://0.0.0.0:8001")
    item.batch = item.batch.reshape(-1)

    input_ids = item.batch['input_ids']
    attention_mask = item.batch['attention_mask']
    valid_input_len = torch.sum(attention_mask)
    prompt_ids = input_ids[0, -valid_input_len:].tolist()
    data = {"prompt": prompt_ids}

    completion = client.chat.completions.create(
        model="rollout",
        messages=data,
        extra_body={
            "top_p": config.train_generate_kwargs['top_p'],
            "top_k": config.train_generate_kwargs['top_k'],
            "max_tokens": config.train_generate_kwargs['max_new_tokens'],
            "max_length": config.prompt_length + config.response_length
        },
    )
    return completion


def process_single_batch(item, context):
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout

    completion = _internal_call(item, config)

    from alpha_seed.workers.agents import DataPack, pack_to_dataproto
    data_pack = DataPack.create_from_completion(completion.choices[0].message)
    out = pack_to_dataproto(item, tokenizer, data_pack, config)  # dataproto

    # response_outputs = [completion.choices[0].message.raw_output_ids]
    # get reward and decide whether to end episod
    return out


def reward_fn(data_item, completion, context):
    from alpha_seed.utils.reward_score import math_v2
    compute_score_fn = math_v2.compute_score
    ground_truth = data_item.non_tensor_batch['reward_model'][0]['ground_truth']
    data_item.batch['prompts'] = data_item.batch['input_ids'][:, :context.config.data.max_prompt_length]
    data_item.batch['responses'] = data_item.batch['input_ids'][:, context.config.data.max_prompt_length:]
    response_outputs = [completion.choices[0].message.raw_output_ids]
    solution_str = context.tokenizer.decode(response_outputs[0], skip_special_tokens=True)
    score_fn_inputs = {
        "batch_info": data_item.batch,
        "tokenizer": context.tokenizer,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "config": context.config,
        'data_uid': None,
        "solution_len": None,
        "solution_ids": None,
        'rm_name': None,
        'pause_tokens_index': None
    }
    final_reward = compute_score_fn(**score_fn_inputs)
    return final_reward


def process_single_batch_v2(item, context):
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    final_reward = 0
    round_idx = 0
    while (final_reward <= 0 and round_idx < 3):
        completion = _internal_call(item, config)
        from alpha_seed.workers.agents import DataPack, pack_to_dataproto
        data_pack = DataPack.create_from_completion(completion.choices[0].message)
        out = pack_to_dataproto(item, tokenizer, data_pack, config)  # dataproto
        final_reward = reward_fn(out, completion, context)
        round_idx += 1
    return out
