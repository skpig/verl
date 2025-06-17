from alpha_seed.workers.agents.handlers import register_handler
from alpha_seed.workers.streaming_service.streaming_utils import internal_call
import asyncio
import os
import aiohttp
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


@register_handler("math/aiohttp")
async def process_single_batch(item, context):
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    host = context.server_host
    port = context.server_port

    completion = await internal_call(item, config, host, port)

    from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto
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
