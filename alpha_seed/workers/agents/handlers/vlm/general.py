from transformers.utils import PaddingStrategy

from alpha_seed.workers.agents.handlers.base import AsyncAgent, ThreadedAgent
from alpha_seed.workers.agents.handlers import register_handler, TaskContext
from alpha_seed.workers.streaming_service.streaming_utils import internal_call
from alpha_seed.utils.reward_score.utils import Verifier
import torch
import asyncio
import os
import ray
import time

from mono_rl import DataProto


@register_handler("vlm/single_turn")
class SingleTurn(AsyncAgent):

    def _reward_fn(self, item: DataProto, out: DataProto):
        reward_model = out.non_tensor_batch['reward_model'][0]
        reward_style = reward_model['style']
        req_id = item.non_tensor_batch['uid'][0]
        input_ids = out.batch['input_ids'][0]
        ground_truth = reward_model['ground_truth']
        verifier = None
        if reward_model is not None and 'style' in reward_model and reward_model['style'] != 'remote_service':
            # 非remote rm verifier
            verifier = Verifier.get_verifier(reward_model['style'], self.config, self.tokenizer.tokenizer)
        if 'no_thinking_required' in item.batch:
            no_thinking_required = item.batch["no_thinking_required"][0]
        else:
            no_thinking_required = False

        if verifier is not None and req_id is not None:
            # this is non-blocking
            verifier.add_requests(req_id=req_id,
                                  input_ids=input_ids,
                                  ground_truth=ground_truth,
                                  reward_style=reward_style,
                                  no_thinking_required=no_thinking_required)
        # grm verifier
        remote_rm_type = reward_model.get('rm_required_type', None)
        if remote_rm_type is not None:
            reward_style = f'{remote_rm_type}_service'
            verifier = Verifier.get_verifier(reward_style, self.config, self.tokenizer.tokenizer)
            response_ids = input_ids[self.config.data.max_prompt_length:]
            response_length = out.batch['attention_mask'][0][self.config.data.max_prompt_length:].sum()
            response_ids = response_ids[:response_length]
            params = dict(input_ids=input_ids,
                          ground_truth=ground_truth,
                          reward_style=reward_style,
                          response_ids=response_ids,
                          call_rm_service=True,
                          reward_model=reward_model,
                          call_fn_time=time.time())
            verifier.add_requests(req_id=req_id, **params)

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        os.environ["no_proxy"] = ""
        tokenizer = self.tokenizer
        config = context.config
        rollout_config = context.config.actor_rollout_ref.rollout
        reward_model = item.non_tensor_batch.pop('reward_model')
        if 'image_data_ref' in item.non_tensor_batch:
            # vlm mode里input_ids已经提前处理好，所以这里不用prompt
            prompt = ''
        else:
            # tokenize and left pad
            prompt = item.non_tensor_batch['prompt'][0]
            prompt_data = await tokenizer.batch_encode_plus_async([prompt],
                                                                  padding=PaddingStrategy.MAX_LENGTH,
                                                                  padding_side='left',
                                                                  truncation=True,
                                                                  add_special_tokens=False,
                                                                  max_length=config.data.max_prompt_length)
            item.batch['input_ids'] = torch.tensor(prompt_data.input_ids, dtype=torch.int32)
            item.batch['attention_mask'] = torch.tensor(prompt_data.attention_mask, dtype=torch.int8)
        # 因为已经提前tokenize好，不传prompt
        completion = await self.llm.complete(item, rollout_config)
        from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto
        data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
        out = pack_to_dataproto(item, tokenizer, data_pack, rollout_config)  # dataproto
        out.non_tensor_batch['reward_model'] = reward_model
        self._reward_fn(item, out)
        return out
