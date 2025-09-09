import asyncio


async def post_process_eval_result(item, out, executor, val_reward_fn):
    loop = asyncio.get_running_loop()
    reward_tensor, prompt_str, solution_str = await loop.run_in_executor(executor, val_reward_fn, out, 0, False, True,
                                                                         True)
    reward_score = reward_tensor.sum(-1)[0].item()
    val_epoch_id = item.meta_info['epoch_id']
    bon_id = item.non_tensor_batch['bon_id'][0] if 'bon_id' in item.non_tensor_batch else 0
    result = {
        'prompt_id': item.non_tensor_batch['prompt_id'][0],
        'index_id': item.non_tensor_batch['index'][0],
        'val_epoch_id': val_epoch_id,
        'bon_id': bon_id,
        'reward': reward_score,
        'prompt': prompt_str,
        'response': solution_str,
    }
    return result
