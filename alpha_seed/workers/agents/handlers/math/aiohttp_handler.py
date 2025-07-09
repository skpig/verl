from transformers.utils import PaddingStrategy

from alpha_seed.workers.agents.handlers.base import AsyncAgent, ThreadedAgent
from alpha_seed.workers.agents.handlers import register_handler, TaskContext
from alpha_seed.workers.streaming_service.streaming_utils import internal_call
import torch
import asyncio
import os

from mono_rl import DataProto
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
async def process_single_batch(item: DataProto, context: TaskContext, **kwargs):
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    host = context.server_host
    port = context.server_port
    if 'image_grid_hw' in item.non_tensor_batch:
        prompt = ''
    else:
        prompt = item.non_tensor_batch['prompt'][0]
    completion = await internal_call(item, config, host, port, prompt=prompt)
    from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto
    data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
    out = pack_to_dataproto(item, tokenizer, data_pack, config)  # dataproto
    return out


@register_handler("general/single_turn")
class SingleTurn(AsyncAgent):

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        os.environ["no_proxy"] = ""
        tokenizer = self.tokenizer
        config = context.config
        rollout_config = context.config.actor_rollout_ref.rollout
        if 'image_grid_hw' in item.non_tensor_batch:
            # vlm mode里input_ids已经提前处理好，所以这里不用prompt
            prompt = ''
        else:
            # tokenize and left pad
            prompt = item.non_tensor_batch['prompt'][0]
            prompt_data = await tokenizer.batch_encode_plus_async([prompt],
                                                                  padding=PaddingStrategy.MAX_LENGTH,
                                                                  padding_side='left',
                                                                  add_special_tokens=False,
                                                                  max_length=config.data.max_prompt_length)
            item.batch['input_ids'] = torch.tensor(prompt_data.input_ids, dtype=torch.int32)
            item.batch['attention_mask'] = torch.tensor(prompt_data.attention_mask, dtype=torch.int8)
        # 因为已经提前tokenize好，不传prompt
        completion = await self.llm.complete(item, rollout_config)
        from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto
        data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
        out = pack_to_dataproto(item, tokenizer, data_pack, rollout_config)  # dataproto
        return out


@register_handler("general/single_turn/sync")
class SingleTurnSync(ThreadedAgent):

    def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        pass


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
