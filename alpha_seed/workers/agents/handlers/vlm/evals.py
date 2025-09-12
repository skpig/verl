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
from alpha_seed.workers.agents.handlers.vlm import post_process_eval_result
from mono_rl import DataProto

logger = logging.getLogger(__name__)


@register_handler("agent/tool/vlm_evals")
class VLMEvalsAgent(AsyncAgent):

    def __init__(self, tokenizer: AsyncTokenizer | PreTrainedTokenizer, llm: AsyncLLMInterface, **kwargs):
        super().__init__(tokenizer, llm, **kwargs)
        config = kwargs['config']
        processor = kwargs['processor']
        self.visual_cot = create_from_env_str('visual_cot@',
                                              tokenizer=tokenizer,
                                              image_processor=processor.image_processor,
                                              config=config)
        self.processor = processor
        self.tools = {"visual_cot": self.visual_cot}
        self.tool_parser = VisualCotParser(tokenizer)
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
        return await post_process_eval_result(item, out, self.executor, self.val_reward_fn)
