"""
Implement custom functions for math expression task
"""
import asyncio
import logging

from transformers import PreTrainedTokenizer
from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.agents.handlers import register_handler, TaskContext
from alpha_seed.workers.agents.handlers.base import AsyncAgent
from alpha_seed.workers.agents.llm import AsyncLLMInterface
from alpha_seed.workers.agents.handlers.vlm.parser import VisualCotParser
from alpha_seed.workers.agents.envs.visual_cot import create_from_env_str
from alpha_seed.utils.functional import import_from_string
from mono_rl import DataProto

logger = logging.getLogger(__name__)


@register_handler("agent/tool/vlm_evals")
class VLMEvalsAgent(AsyncAgent):

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        super().__init__(tokenizer, llm, **kwargs)
        processor = kwargs['processor']
        self.visual_cot = create_from_env_str('visual_cot@',
                                              tokenizer=tokenizer,
                                              image_processor=processor.image_processor)
        self.processor = processor
        self.tools = {"visual_cot": self.visual_cot}
        self.tool_parser = VisualCotParser(tokenizer)
        config = kwargs['config']
        reward_manager_cls = import_from_string(config.tasks.reward_manager)
        self.val_reward_fn = reward_manager_cls(tokenizer=tokenizer,
                                                config=config,
                                                logger=None,
                                                rm_name="val",
                                                single_batch=True)

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        tokenizer = self.tokenizer
        rollout_config = context.config.actor_rollout_ref.rollout
        completion = await self.llm.complete(item, rollout_config)
        from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto
        data_pack = DataPack.create_from_completion_dict(completion['choices'][0]['message'])
        out = pack_to_dataproto(item, tokenizer, data_pack, rollout_config)
        loop = asyncio.get_running_loop()
        reward_tensor, prompt_str, solution_str = await loop.run_in_executor(self.executor, self.val_reward_fn, out, 0,
                                                                             False, True, True)
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
