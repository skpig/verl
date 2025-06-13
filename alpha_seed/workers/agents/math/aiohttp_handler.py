import torch
from openai import AsyncOpenAI

from alpha_seed.workers.streaming_service.streaming_utils import is_ipv6
from mono_rl import DataProto
import asyncio
import os
import aiohttp
import copy
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


async def chat_completions(content, meta_info, config, host, port: int):
    if is_ipv6(host):
        host = f'[{host}]'
    try:
        timeout = aiohttp.ClientTimeout(total=9600)
        session = aiohttp.ClientSession(timeout=timeout)
        generation_kwargs = meta_info['generation_kwargs']
        async with session.post(url=f"http://{host}:{port}/chat/completions",
                                headers={"Authorization": "Bearer token-abc123"},
                                json={
                                    "model": "rollout",
                                    "messages": content,
                                    "top_p": generation_kwargs['top_p'],
                                    "top_k": generation_kwargs['top_k'],
                                    "temperature": generation_kwargs['temperature'],
                                    "max_tokens": generation_kwargs['max_new_tokens'],
                                    "max_length": config.prompt_length + config.response_length,
                                    "meta_info": meta_info,
                                },
                                timeout=timeout) as resp:
            ret = await resp.json()
            assert resp.status == 200, f"chat_completions failed msg: {ret}"
            return ret
    except Exception as e:
        raise (e)
    finally:
        await session.close()


async def _internal_call(item, config, host, port: int):
    completion = None
    try:
        item.batch = item.batch.reshape(-1)
        input_ids = item.batch['input_ids']
        attention_mask = item.batch['attention_mask']
        valid_input_len = torch.sum(attention_mask)
        prompt_ids = input_ids[0, -valid_input_len:].tolist()
        data = {"prompt": prompt_ids}
        meta_info = copy.copy(item.meta_info)
        # required for eos callback
        meta_info['uid'] = item.non_tensor_batch['uid'][0]
        meta_info['reward_model'] = item.non_tensor_batch['reward_model'][0]
        meta_info['server_meta'] = {
            'host': host,
            'port': port,
        }
        # required for tool calling
        if (key := 'extra_data') in item.non_tensor_batch:
            meta_info[key] = item.non_tensor_batch[key][0]

        completion = await chat_completions(data, meta_info, config, host, port)
    except asyncio.CancelledError:
        # Handle task cancellation (e.g., cleanup)
        print("Request was cancelled!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise  # Re-raise to propagate the cancellation
    except Exception as e:
        print(f"Error occurred!!!!!!!!!!!!!!!!!!", e)
        raise  # Re-raise the exception to propagate it further
    return completion


async def process_single_batch(item, context):
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    host = context.server_host
    port = context.server_port

    completion = await _internal_call(item, config, host, port)

    from alpha_seed.workers.agents import DataPack, pack_to_dataproto
    data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
    out = pack_to_dataproto(item, tokenizer, data_pack, config)  # dataproto
    return out


if __name__ == '__main__':
    os.environ["no_proxy"] = ""
    data = {"prompt": "1+1="}
    import aiohttp

    async def chat_completions(content):
        try:
            session = aiohttp.ClientSession()
            async with session.post(
                    url="http://0.0.0.0:8001/chat/completions",
                    headers={"Authorization": "Bearer token-abc123"},
                    json={
                        "model": "rollout",
                        "messages": content,
                        "top_p": 1.0,
                        "top_k": 1,
                        "max_tokens": 16,
                        "max_length": 1024
                    },
            ) as resp:
                return await resp.json()
        except Exception as e:
            print(e)
            return e
        finally:
            await session.close()

    out = asyncio.run(chat_completions(data))
    print(out)
    print(out['choices'][0]['message'])
