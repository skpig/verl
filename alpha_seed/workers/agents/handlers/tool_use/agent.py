import collections

try:
    from groot.state import BaseState
    from groot.action import Env
    from groot.agent import BaseAgent, AsyncAgentMixin
    from groot.utils import get_curr_date
    from groot.llm import BaseLLM
    from groot.agent.protocol import DoubaoProtocol, BaseProtocol
    from groot.agent.prompts import REACT_SYSTEM_PROMPT
    from groot.utils import print_with_verbose
except:
    BaseState = object
    BaseAgent = object
    AsyncAgentMixin = list
    BaseProtocol = object
    DoubaoProtocol = object
    BaseLLM = object
    Env = object
    ActionHistory = object
    REACT_SYSTEM_PROMPT = None
from termcolor import colored
from typing import Optional, Union, List, Dict, Any, Tuple
import json


class ActionHistory:

    def __init__(self):
        self.action_cnt = collections.Counter()
        self.action_stats = collections.Counter()

    def add(self, action: Dict[str, Any]):
        action_str = json.dumps(action, sort_keys=True)
        self.action_cnt[action_str] += 1
        self.action_stats[f"num_{action['name']}"] += 1

    def query(self, action: Dict[str, Any]):
        action_str = json.dumps(action, sort_keys=True)
        return self.action_cnt[action_str]


class SimpleAgent(BaseAgent):

    def __init__(self,
                 llm: BaseLLM,
                 env: Env = Env(),
                 max_turn: int = 30,
                 current_date: str = None,
                 system_prompt: dict | str = REACT_SYSTEM_PROMPT,
                 fewshot_prompt: Optional[str] = None,
                 protocol: Optional[BaseProtocol] = DoubaoProtocol(),
                 verbose: bool = False,
                 tool_format: str = 'func',
                 llm_kvcache: bool = False,
                 lang: str = 'zh',
                 force_system_prompt: bool = False,
                 max_env_response: Optional[int] = None,
                 max_repeat_action: int = -1,
                 **kwargs):
        """
        Initialize the ReAct agent with a language model, environment, and protocol.

        Args:
            llm (BaseLLM): The language model to interact with.
            env (Env): The environment for actions.
            max_turn (int): Maximum number of turns for the agent to interact.
            system_prompt (str): Initial system prompt.
            fewshot_prompt (Optional[str]): Few-shot example prompt to be added to system prompt.
            protocol (Optional[BaseProtocol]): Protocol for parsing responses and actions.
            verbose (bool): Whether to print verbose output.
            tool_format (str): The format type of the returned actions. 
                               'openai' for OpenAI functioncall format,
                               'func' for function docstring format. 
                               Defaults to 'openai'.
            llm_kvcache (bool): Whether to use kvcache for the language model.
        """
        if isinstance(system_prompt, dict):
            system_prompt = system_prompt[lang]
        self.protocol = protocol
        self.max_turn = max_turn
        self.verbose = verbose
        self.current_date = current_date
        self.llm_kvcache = llm_kvcache
        if llm_kvcache:
            self.llm_kvcache_id = None
        self.max_env_response = max_env_response
        self.max_repeat_action = max_repeat_action

        # Format the system prompt using environment actions and protocol tokens
        api_list = env.get_actions_info(format_type=tool_format)
        if tool_format == 'func':
            api_list = '\n'.join(api_list)
        else:
            api_list = json.dumps(api_list, ensure_ascii=False, indent=2)

        if not force_system_prompt:
            # 原本的格式匹配模式
            if system_prompt:
                system_prompt = system_prompt.format(
                    api_list=api_list,
                    timestamp=get_curr_date() if current_date is None else current_date,
                    **protocol.special_tokens,
                )

            # Add few-shot prompt examples to system prompt, if available
            if fewshot_prompt:
                system_prompt += fewshot_prompt.format(**self.protocol.special_tokens)
        else:
            # 强制直接使用system_prompt
            pass

        # Print the system prompt for debugging
        print_with_verbose(colored(f"System: {system_prompt}", 'cyan'), verbose=self.verbose)

        # Initialize the base agent
        super().__init__(llm, env, system_prompt=system_prompt, **kwargs)

    def run(self, inputs: Optional[str] = None, think_end_str: str = "") -> str:
        """
        Run the ReAct agent in an interactive loop.

        Args:
            inputs (Optional[str]): The user input to start the conversation.
            think_end_str (Optional[str]): The think end string used by LLM to stop internal reasoning.
        Returns:
            str: Final response after completing the interaction.
        """
        state = BaseState()
        # Add user input to conversation state
        state.add(role='user', content=inputs)
        print_with_verbose(colored(f"User: {inputs}", 'light_magenta'), verbose=self.verbose)

        def react_chat(state):
            if self.llm_kvcache:
                if self.llm_kvcache_id is None:
                    llm_response, self.llm_kvcache_id = self.llm_chat(state, use_kvcache=True)
                else:
                    llm_response = self.llm_chat(state, kvcache_id=self.llm_kvcache_id)
            else:
                llm_response = self.llm_chat(state)
            if llm_response is None:
                # 超过最大token了
                return None
            print_with_verbose(colored(f"Assistant: {llm_response}", 'blue'), verbose=self.verbose)
            # Parse the LLM response to determine if an action is required
            truncated_response = llm_response
            prefix = ""
            if think_end_str:
                if think_end_str in truncated_response:
                    prefix, truncated_response = truncated_response.split(think_end_str, 1)
                    prefix = prefix + think_end_str
            message, action = self.protocol.parse(truncated_response)
            message = prefix + message
            # Add LLM response to conversation state
            state.add(role='assistant', content=llm_response)
            # If no action is required, assume the conversation has reached the terminal state
            return message, action, state

        exceed_token_limit = False

        for turn in range(self.max_turn):
            # Get response from the language model
            resp = react_chat(state)
            if resp is None:
                exceed_token_limit = True
                break
            message, action, state = resp

            # If no action is required, assume the conversation has reached the terminal state
            if action is None:
                if self.system_prompt:
                    # Add system prompt to the beginning of the conversation history
                    state.history.insert(0, dict(role='system', content=self.system_prompt))
                return state

            # Execute action in the environment and capture the response
            try:
                env_response = []
                for each_action in action:
                    each_response = self.env(**each_action)
                    env_response.append(json.dumps(each_response, ensure_ascii=False, indent=2))
                env_response = '\n\n\n'.join(env_response)
            except Exception as e:
                env_response = f"Error during environment interaction: {str(e)}"
            if self.max_env_response is not None and len(env_response) > self.max_env_response:
                env_response = env_response[:self.max_env_response]
            print_with_verbose(colored(f"Environment: {env_response}", 'green'), verbose=self.verbose)

            # Add environment response to conversation state
            state.add(role='tool', content=env_response)

        # If max turns reached without termination, return this message
        # TODO:
        # 1. Possibly force direct summary if no terminal state is reached
        # 2. #open page too much
        # 3. #query too much

        # If max turns reached without termination, add one final assistant message
        if not exceed_token_limit:
            resp = react_chat(state)
            if resp is not None:
                message, action, state = resp
        # if self.system_prompt:
        #     # Add system prompt to the beginning of the conversation history
        #     state.history.insert(0, dict(role='system', content=self.system_prompt))
        return state


class AsyncSimpleAgent(AsyncAgentMixin, SimpleAgent):

    async def run(self,
                  inputs: Union[str, Dict[str, str], List[Dict]] = None,
                  think_end_str: str = "") -> Tuple[BaseState, ActionHistory]:
        """
        Run the AsyncReAct agent in an interactive loop.
        Args:
            inputs (Optional[str]): The user input to start the conversation.
            think_end_str: (Optional[str]) The think end string used by LLM for reasoning content
        Returns:
            str: Final response after completing the interaction.
        """
        state = BaseState()
        # Add user input to conversation state
        if isinstance(inputs, str):
            state.add(role='user', content=inputs)
        elif isinstance(inputs, dict):
            assert inputs['role'] == 'user'
            state.add(inputs['content'])
        elif isinstance(inputs, list):
            state.batch_add(inputs)
        else:
            raise ValueError(inputs)
        print_with_verbose(colored(f"User: {inputs}", 'light_magenta'), verbose=self.verbose)

        action_history = ActionHistory()

        # @async_retry_until_success(max_try=3)
        async def react_chat(state):
            if self.llm_kvcache:
                if self.llm_kvcache_id is None:
                    resp = await self.llm_chat(state, use_kvcache=True)
                    llm_response, self.llm_kvcache_id = resp
                else:
                    llm_response = await self.llm_chat(state, kvcache_id=self.llm_kvcache_id)
            else:
                llm_response = await self.llm_chat(state)
            if llm_response is None:
                # 超过最大token了 或者 超时
                return None
            print_with_verbose(colored(f"Assistant: {llm_response}", 'blue'), verbose=self.verbose)
            # Parse the LLM response to determine if an action is required
            truncated_response = llm_response
            prefix = ""
            if think_end_str:
                if think_end_str in truncated_response:
                    prefix, truncated_response = truncated_response.split(think_end_str, 1)
                    prefix = prefix + think_end_str
            message, action = self.protocol.parse(truncated_response)
            message = prefix + message
            # Add LLM response to conversation state
            state.add(role='assistant', content=llm_response)
            # If no action is required, assume the conversation has reached the terminal state
            return message, action, state

        exceed_token_limit = False
        reach_action_max_repeat = False
        for turn in range(self.max_turn):
            # Get response from the language model
            resp = await react_chat(state)
            if resp is None:
                exceed_token_limit = True
                break
            message, action, state = resp
            # message, action, state = await react_chat(state)

            # If no action is required, assume the conversation has reached the terminal state
            if action is None:
                return state, action_history

            if self.max_repeat_action > 0:
                for each_action in action:
                    action_repeat_num = action_history.query(each_action)
                    if action_repeat_num >= self.max_repeat_action:
                        reach_action_max_repeat = True
                        break
                    action_history.add(each_action)
                if reach_action_max_repeat:
                    print(f"Reach max repeated action: {each_action}. Stop interaction.")
                    break

            # Execute action in the environment and capture the response
            try:
                env_response = []
                for each_action in action:
                    each_response = await self.env(**each_action)
                    env_response.append(json.dumps(each_response, ensure_ascii=False, indent=2))
                env_response = '\n\n\n'.join(env_response)
            except Exception as e:
                env_response = f"Error during environment interaction: {str(e)}"
            print_with_verbose(colored(f"Environment: {env_response}", 'green'), verbose=self.verbose)
            # Add environment response to conversation state
            state.add(role='tool', content=env_response)
        # If max turns reached without termination, return this message
        # return state  # Possibly force direct summary if no terminal state is reached

        # If max turns reached without termination, and it is not stopped due to repeated action, add one final assistant message
        if not exceed_token_limit and not reach_action_max_repeat:
            resp = await react_chat(state)
            if resp is not None:
                message, action, state = resp
        # message, action, state = await react_chat(state)
        return state, action_history


if __name__ == "__main__":
    from groot.llm import ByteLLM
    from groot.action import Env
    from groot.action.search.toutiao_search import ToutiaoSearch
    from groot.action.search.link_reader import LinkReader
    # from groot.action.search.duck_search import DuckDuckGoSearch
    from groot.action.search.bing_search import BingSearch

    llm = ByteLLM(mode='xperf', psm_cfg=dict(psm='agent_20b_qrl'))
    env = Env([ToutiaoSearch(), LinkReader()])
    current_date = "2024-03-15"  # 使用标准格式的日期
    agent = SimpleAgent(llm, env, verbose=True, current_date=current_date)

    test_query = "特斯拉最近三个月股价最高和最低分别是多少？"
    states = []
    state = agent.run(test_query)
    states.append(state)
    print("-" * 100)

    # env = Env([DuckDuckGoSearch(), LinkReader()])
    # agent = SimpleAgent(llm, env, verbose=True, current_date=current_date)
    # state = agent.run(test_query)
    # states.append(state)
    # print("-"*100)

    # env = Env([BingSearch(), LinkReader()])
    # agent = SimpleAgent(llm, env, verbose=True, current_date=current_date)
    # state = agent.run(test_query)
    # states.append(state)
    # print("-"*100)

    # print("##############不同搜索工具的对比##############")
    # print(f"Query: {test_query}")
    # print(f"Current Date: {current_date}")
    # search_tools = ["头条搜索", "DuckDuckGo搜索", "Bing搜索"]
    # for state, tool in zip(states, search_tools):
    #     print(f"搜索工具: {tool}")
    #     print(f"交互轮次: {len(state.history)}")
    #     print(state.history[-1]["content"])
    #     print("-"*100)
