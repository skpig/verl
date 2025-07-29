import os
import re

import torch
from openai import AsyncOpenAI

from verl import DataProto
from typing import Optional, Tuple, List, Dict, Union
import asyncio
import aiohttp
import copy

from alpha_seed.workers.agents.handlers import TaskContext
from alpha_seed.utils.tokenizer.async_tokenizer import AsyncTokenizer
from alpha_seed.workers.streaming_service.streaming_utils import DataPack, pack_to_dataproto, is_ipv6
from alpha_seed.workers.agents.handlers import register_handler
from alpha_seed.workers.agents.handlers.base import AsyncAgent
import random
from verl.utils import torch_functional

try:
    from groot.action import AsyncEnv
    from groot.action import AsyncJupyterEnv, AsyncJupyterServerEnv
    from groot.action.search.toutiao_search import AsyncToutiaoSearch
    from groot.action.search.web_search import AsyncWebSearch
    from groot.action.search.global_search import AsyncGlobalSearch
    from groot.action.search.browser_reader import AsyncBrowserReader
    from groot.action.search.page_reader import AsyncPageReader
    from groot.action.search.global_search_apihub import AsyncGlobalSearchApiHub
    from groot.action.search.browser_reader_apihub import AsyncBrowserReaderApiHub, BrowserReaderApiHub
    from groot.action.call_llm import AsyncCallLLM
    from groot.state import BaseState
    from .agent import AsyncSimpleAgent
    from alpha_seed.workers.agents.handlers.tool_use.verifier.custom_prompt import get_custom_sp
    from alpha_seed.workers.agents.handlers.tool_use.verifier.judger import judge_answer, get_last_ans
    from alpha_seed.workers.agents.handlers.tool_use.utils import get_llm, extract_conversation, encode_conversation, decode_conversation, LLMInterface, tokenize_and_postprocess_data, print_conversation
    from groot.llm import GPTAPI, AsyncGPTAPI, AsyncByteLLM
except:
    pass
from transformers import PreTrainedTokenizer
from bytedance import servicediscovery
import time


async def process_single_batch(
        item: DataProto,
        context: TaskContext,
        async_tokenizer: Optional[Union[AsyncTokenizer, PreTrainedTokenizer]] = None) -> DataProto:
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer if async_tokenizer is None else async_tokenizer
    config = context.config.actor_rollout_ref.rollout
    host = context.server_host
    port = context.server_port

    item.batch = item.batch.reshape(-1)
    meta_info = copy.copy(item.meta_info)
    meta_info['uid'] = item.non_tensor_batch['uid'][0]
    meta_info['reward_model'] = item.non_tensor_batch['reward_model'][0]
    # config.plugin.fc_type = item.non_tensor_batch['ability'][0].split("@")[1]

    # TODO: 【亟需检查】很神奇，会什么用doubao的decoder + qwen的数据，但是这边解码出来是对的？
    # print_str = f"[DEBUG] 检查decode是否正常：{context.tokenizer.decode(item.batch['input_ids'][0])}"
    # print(print_str)

    conversation, decode_str = extract_conversation(item, context.tokenizer)
    if len(conversation) == 1:
        assert conversation[0]['role'] == 'user'
        system_prompt = ""
        user_prompt = conversation[0]['content']
    elif len(conversation) == 2:
        assert conversation[0]['role'] == 'system'
        assert conversation[1]['role'] == 'user'
        system_prompt = conversation[0]['content']
        user_prompt = conversation[1]['content']
    else:
        raise ValueError(f"Invalid conversation: {conversation}\n{decode_str}")

    lang = getattr(config.plugin, 'lang', 'zh')
    if "lang" in item.non_tensor_batch:
        lang = item.non_tensor_batch["lang"][0]

    current_date = time.strftime("%Y-%m-%d", time.localtime())
    llm = LLMInterface(tokenizer=context.tokenizer, config=config, host=host, port=port, meta_info=meta_info)

    if config.plugin.fc_type == "jupyter":
        search_engine_obj = AsyncToutiaoSearch(search_psm=config.plugin.search_psm,
                                               max_retry=getattr(config.plugin, 'search_max_retry', 1),
                                               control_qps=getattr(config.plugin, 'control_qps', False))
        external_llm = get_llm(config.plugin.external_llm_psm,
                               api_llm_cls=GPTAPI,
                               api_model_type=getattr(config.plugin, 'external_llm_type', 'Qwen2.5-32B-Instruct'))
        page_reader = BrowserReaderApiHub(control_qps=False)
        env_class = AsyncJupyterServerEnv
    elif config.plugin.fc_type == "function":
        # if config.plugin.search_psm == "web_search":
        search_engine = getattr(config.plugin, "search_engine", "toutiao")
        if search_engine == "web_search":
            search_engine_obj = AsyncWebSearch(
                use_cache=getattr(config.plugin, "use_cache", False),
                use_global=getattr(config.plugin, 'use_global_search', False),
                control_qps=getattr(config.plugin, 'control_qps', False),
                max_retry=getattr(config.plugin, 'search_max_retry', 2),
                use_bing_ratio=getattr(config.plugin, 'use_bing_ratio', 0.5),
                retry_interval=1.0,
                qps_timeout=getattr(config.plugin, 'qps_timeout', 300),
            )
        elif search_engine == "global_search":
            search_engine_obj = AsyncGlobalSearch(search_psm=config.plugin.search_psm,
                                                  max_retry=getattr(config.plugin, 'search_max_retry', 1),
                                                  control_qps=getattr(config.plugin, 'control_qps', False))
        elif search_engine == "global_search_hub":
            search_engine_obj = AsyncGlobalSearchApiHub(max_retry=getattr(config.plugin, 'search_max_retry', 1),
                                                        control_qps=getattr(config.plugin, 'control_qps', False))
        else:
            search_engine_obj = AsyncToutiaoSearch(search_psm=config.plugin.search_psm,
                                                   max_retry=getattr(config.plugin, 'search_max_retry', 1),
                                                   control_qps=getattr(config.plugin, 'control_qps', False))
        reader_type = getattr(config.plugin, 'link_reader_type', 'page_reader')
        if reader_type == 'browser_reader':
            page_reader = AsyncBrowserReader(control_qps=getattr(config.plugin, 'control_qps', False))
        elif reader_type == 'browser_reader_hub':
            page_reader = AsyncBrowserReaderApiHub(control_qps=getattr(config.plugin, 'control_qps', False))
        elif reader_type == 'page_reader':
            external_llm = get_llm(config.plugin.external_llm_psm,
                                   api_llm_cls=AsyncGPTAPI,
                                   api_model_type=getattr(config.plugin, 'external_llm_type', 'Qwen2.5-32B-Instruct'))
            page_reader = AsyncPageReader(llm=external_llm, lang=lang)
        else:
            raise ValueError(f"Invalid reader_type: {reader_type}")
        env_class = AsyncEnv
    else:
        raise ValueError(f"Invalid fc_type: {config.plugin.fc_type}")
    mapping_dict = {
        f"{search_engine_obj.name}.search": "search",
        f"{page_reader.name}.open": "open",
    }
    tool2action = {
        "open_url": page_reader,
        "search": search_engine_obj,
    }

    tool_list = getattr(config.plugin, "tool_list", "search,open_url").split(",")
    action_list = [tool2action[tool] for tool in tool_list]

    # add call llm to non function case for larger action space
    # if config.plugin.fc_type != 'function':
    #     action_list.append(call_llm)
    #     mapping_dict[f"{call_llm.name}.call_llm"] = "call_llm"

    if env_class == AsyncJupyterServerEnv:
        if config.plugin.server_type == 'companion':
            trial_id = os.getenv('ARNOLD_TRIAL_ID')
            servers = servicediscovery.lookup(f"data.aml.arnold_env_manager_{trial_id}", address_family='v6')
        else:
            trial_id = config.plugin.jupyter_server_id
            servers = servicediscovery.lookup(trial_id, address_family='v6')
        env = env_class(action_list, mapping_dict, servers[0]['Host'], servers[0]['Port'])
        await env.ainit()
    else:
        env = env_class(action_list, mapping_dict)

    # select max step
    if config.plugin.fc_type == 'jupyter':
        from groot.agent.prompts import REACT_SYSTEM_PROMPT_JUPYTER as system_prompt_dict
        from groot.agent.protocol import ReWorkflowProtocol as base_protocol
    elif config.plugin.fc_type == 'function':
        from groot.agent.prompts import REACT_SYSTEM_PROMPT as system_prompt_dict
        from groot.agent.protocol import DoubaoProtocol as base_protocol
    else:
        raise ValueError(f"Invalid fc_type: {config.plugin.fc_type}")

    if "max_turn" in item.meta_info:
        max_turn = item.meta_info["max_turn"]
    else:
        if context.is_train:
            max_turn = config.plugin.max_turn
        else:
            if "val_max_turn" in config.plugin:
                max_turn = config.plugin.val_max_turn
            else:
                max_turn = config.plugin.max_turn

    custom_sp = getattr(config.plugin, 'custom_sp', None)
    _system_prompt_dict = get_custom_sp(custom_sp) if custom_sp is not None else system_prompt_dict
    agent = AsyncSimpleAgent(
        llm=llm,
        env=env,
        max_turn=max_turn,
        verbose=False,
        system_prompt=_system_prompt_dict,
        protocol=base_protocol(),
        tool_format='func',
        lang=lang,
        current_date=current_date,
        max_repeat_action=getattr(config.plugin, 'max_repeat_action', -1),
    )
    if system_prompt and custom_sp is None:
        state, action_history = await agent.run(conversation, think_end_str='</think>')
        recompute_input_ids = False
    else:
        new_system_prompt = agent.system_prompt
        conversation = [
            {
                "role": "system",
                "content": new_system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ]
        state, action_history = await agent.run(conversation, think_end_str='</think>')
        recompute_input_ids = True

    try:
        print("releasing env")
        await env.release_env()
    except:
        pass

    if recompute_input_ids:
        messages = []
        for turn in state.history:
            if turn['role'] in ['assistant', 'tool']:
                break
            messages.append(turn)
        assert len(messages) <= 2, messages
        prompt_w_chat_template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids, attention_mask = await tokenize_and_postprocess_data(prompt=prompt_w_chat_template,
                                                                        tokenizer=tokenizer,
                                                                        max_length=item.batch['input_ids'].size(-1),
                                                                        pad_token_id=tokenizer.pad_token_id,
                                                                        left_pad=True,
                                                                        truncation='error')
        item.batch['input_ids'] = input_ids
        item.batch['attention_mask'] = attention_mask

    if random.random() < 0.01:
        print_conversation(state.history)

    env_failures = env.failures
    action_stats = action_history.action_stats
    stats = dict(env_failures)
    stats.update(action_stats)

    score = None
    if config.plugin.remote_verify_config.get("async", False):
        acc, reward = await judge_answer(
            plugin_config=config.plugin,
            item=item,
            state=state,
            tokenizer=tokenizer,
            is_training=context.is_train,
        )
        score = reward

    return await llm.pack_out_dataproto(item, state, stats, score)


@register_handler("agent/tool_use/search")
class ReActAgent(AsyncAgent):

    async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
        return await process_single_batch(item, context, self.tokenizer)
