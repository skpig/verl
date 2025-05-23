# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations

import json
import os
import os.path as osp
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from mcts import MCTS

# from .agents.mcts import MCTS
# from .agents.tree import BaseTree
# from .constants import ERROR_COLOR, TIMEOUT_SECONDS
# from .llms.llm_engine import llm_engine, rm_engine
# from .llms.llms import llm_generate, rm_generate


class Solver:
    # model_config = ConfigDict(arbitrary_types_allowed=True)
    # config: Any
    # stop: List[str] = None
    # llm: Optional[Callable[..., List[str]]] = None
    # llm_engine: Optional[LLM] = None
    # rollout_sampling_params: Optional[SamplingParams] = None
    # max_agent_steps: int = 1

    def __init__(self, **kwargs) -> None:
        self.max_agent_steps = kwargs.get("max_agent_steps", 16)
        self.rollout_sampling_params = SamplingParams(
            n=kwargs.get("expand_num", 4), 
            max_tokens=2048, 
            temperature=1.0
        )
        self.llm_engine = LLM(model="/data/pretrain/Qwen/Qwen2.5-3B-Instruct", 
            tensor_parallel_size=4,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True)


    # @staticmethod
    # def processor(agent, output) -> BaseTree:
    #     agent.generate_next_step(output)
    #     return agent


    # @staticmethod
    # def selector(agent, output) -> BaseTree:
    #     agent.select_next_step(output)
    #     return agent


    def generate_preprocess(self, agents: List[MCTS]):
        prompts = []
        prompts_span = [0]
        valid_agents = [] # agents need to be generated
        invalid_agents = [] # FIXME: ?

        for agent in agents:
            # aggregate prompt for all agents
            if agent.should_generate_next():
                # if agent.has_expanded(): # FIXME: what is has_expanded()
                #     expanded_agents.append(agent)
                # else:
                agent_prompts = agent.create_prompt() # can be multiple prompts? all prompts of `current_nodes`
                # rewards.extend(agent.get_rewards()) # can be multiple rewards? all rewards of `current_nodes`
                prompts.extend(agent_prompts)
                prompts_span.append(prompts_span[-1] + len(agent_prompts)) # since each agent can have multiple prompts, we need to record the span of each agent
                valid_agents.append(agent)
            else:
                invalid_agents.append(agent)
        return prompts, prompts_span, valid_agents, invalid_agents


    def generate_postprocess(
        self, 
        outputs: List[List[RequestOutput]], 
        valid_agents: List[MCTS],
    ) -> List[MCTS]:
        for id, (agent, output) in enumerate(zip(valid_agents, outputs)):
            assert agent is not None, "agent is None"
            agent.generate_next_step(output)
            agent.draw_tree(id)
            # post_agents.append(agent)
        #with ProcessPool(max_workers=min(len(valid_agents), os.cpu_count())) as pool:
        # iterator = (self.__class__.processor(agent, output) for agent, output in zip(valid_agents, outputs))
        
        # progress_bar = tqdm(total=len(valid_agents), desc="generate_postprocess")  
        # while True:
        #     try:
        #         result = next(iterator) # A MCTS object
        #         post_agents.append(result)
        #     except StopIteration:
        #         break
        #     except Exception as error:
        #         print(f"{error}\n", ERROR_COLOR)
        #         post_agents.append(None)
        #     progress_bar.update(1) 
        # progress_bar.close() 
            
        # update agents, reset post agents if encountered error
        # updated_agents = [
        #     post_agent if post_agent is not None else valid_agent
        #     for post_agent, valid_agent in zip(post_agents, valid_agents)
        # ]
        # return updated_agents
    

    # def value_preprocess(self, agents: List[BaseTree]) -> Tuple[List[str], List[int]]:
    #     prompts = []
    #     prompts_span = [0]
    #     for agent in agents:
    #         agent_prompts = agent.create_prompt(is_value_only=True) # use candidate nodes to create prompt, FIXME: why?
    #         prompts.extend(agent_prompts)
    #         prompts_span.append(prompts_span[-1] + len(agent_prompts))
    #     return prompts, prompts_span
    
    
    def value_postprocess(
        self, 
        outputs, 
        valid_agents,
    ) -> List[MCTS]:
        for agent, output in zip(valid_agents, outputs):
            if agent is not None:
                agent.select_next_step(output)
        return valid_agents
    

    def save_intermediate_metric(self, path: str, agents: List[MCTS], rollout) -> None:
        if self.config.is_sampling: return
        states = [s.intermediate_metric for s in agents]
        statics = []
        for i in range(rollout + 1):
            pass1, passn = 0, 0
            for idx, state in enumerate(states):
                max_value = -100
                max_value_result = False
                pass1_ans = False
                for idx, rollout_index in enumerate(state["rollout_indexs"]):
                    if rollout_index <= i:
                        if state["value_estimate"][idx] > max_value:
                            max_value = state["value_estimate"][idx]
                            max_value_result = state["judgements"][idx]
                        if state["judgements"][idx]:
                            pass1_ans = True
                if max_value_result:
                    pass1 += 1
                if pass1_ans:
                    passn += 1
            statics.append({
                "rollout": i,
                "pass1": pass1,
                "passn": passn,
                "len": len(states),
            })
        with open(path, "w", encoding='utf-8') as f:
            json.dump([statics,states], f, ensure_ascii=False, indent=4)

    
    def save_intermediate_rollouts(self, saved_jsonl_file, cur_data, agents, rollout_idx):
        if self.config.save_intermediate_rollouts and saved_jsonl_file and self.config.mode == "mcts":
            saved_json_dir = osp.dirname(saved_jsonl_file)
            saved_jsonl_file_name = osp.basename(saved_jsonl_file)
            saved_json_path = osp.join(saved_json_dir, f"rollout")
            if not os.path.exists(saved_json_path):
                os.mkdir(saved_json_path)
            self.save_intermediate_metric(osp.join(saved_json_path, f"intermediate_metric_{saved_jsonl_file_name}"), agents, rollout_idx)
            outs = self.output(agents)
            with open(osp.join(saved_json_path, f"rollout{rollout_idx:02}" + saved_jsonl_file_name), "a+", encoding='utf-8') as writer:
                for d in cur_data:
                    question = d["question"]
                    d["rstar"] = outs[question]
                    writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                    writer.flush()
    
    def output(self, agents: List[MCTS]):
        jsonlines = {}
        for i, agent in enumerate(agents):         
            jsonlines[agent.question] = agent.return_states()
        
        return jsonlines
    
    def calculate_reward(self, outputs_lst: List[List[RequestOutput]]) -> List[float]:
        """
        Calculate the reward for each agent. Assign each output in RequestOutput directly
        """
        # TODO: multiprocessing
        # TODO: use reward_fn to calculate reward
        for output_object in outputs_lst:
            for output in output_object.outputs:
                # score = reward_fn(output)
                score = random.choice([0, 1])
                output.score = score
        return outputs_lst
    
    def dummy_generate(self, prompts: List[str]):
        import random

        from vllm.outputs import CompletionOutput, RequestOutput
        outputs = []
        for index, prompt in enumerate(prompts):
            # 生成虚拟的 CompletionOutput 对象
            completion_output = CompletionOutput(
                index=index,
                text=''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=20)),
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason='length',
                stop_reason=None,
                lora_request=None
            )
            # 生成虚拟的 RequestOutput 对象
            request_output = RequestOutput(
                request_id='dummy_id',
                prompt=prompt,
                prompt_token_ids=None,
                prompt_logprobs=None,
                outputs=[completion_output],
                finished=True,
                metrics=None,
                lora_request=None,
                encoder_prompt=None,
                encoder_prompt_token_ids=None,
                num_cached_tokens=None
            )
            outputs.append(request_output)
        return outputs
    

    def solve(self, agents: List[MCTS]):
        
        for rollout_idx in tqdm(range(self.max_agent_steps), desc="Rollout Processing"):
            # 1. select one node from all to generate
            # 2. conduct one step expansion on 

            # for step in range(self.config.max_depth):
            print("-----------------Current Rollout: ", rollout_idx, "-----------------")

            """ selection """
            for agent in agents:
                agent.select_next_step() # initialize `current_nodes`


            """ expansion """
            # 1. prepare prompts for expansion
            prompts, prompts_span, valid_agents, invalid_agents = self.generate_preprocess(agents) # aggregate prompts from different agents for vllm generation
            
            # if len(valid_agents) < 10: # to increase utility
            #     print("Early stop due to insufficient agents.")
            #     break
            
            # 2. step expansion & simulation
            outputs = self.llm_engine.generate(prompts, self.rollout_sampling_params)
            # outputs = self.dummy_generate(prompts)
            
            # for output, reward in zip(outputs, valid_rewards): # attach reward to prevent repeat rewarding
            #     output.value_estimate = reward # 相当于从parent处继承了reward
            
            # 3. reward
            outputs = self.calculate_reward(outputs)

            reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])] # since each agent can have multiple prompts, we need to reconstruct the outputs for corresponding agents
            
            # 4. expand & simulate & backpropagation
            self.generate_postprocess(reconstructed_outputs, valid_agents)

            # step evaluation for PRM
            # if self.need_value_func:
                # prompts, prompts_span = self.value_preprocess(valid_agents) # aggregate prompts from different agents for reward model generation
                # outputs = self.reward_model(prompts=prompts)
                # reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
            # reconstructed_outputs = [None] * (len(prompts_span) - 1)
            
            # # select the next `current_nodes`
            # valid_agents = self.value_postprocess(reconstructed_outputs, valid_agents)
            # expanded_agents = self.value_postprocess([None] * len(expanded_agents), expanded_agents) # for expanded agents, just do selection step
            
            # keep all agents
            agents = valid_agents + invalid_agents

            # # Save agents internal rollouts
            # self.save_intermediate_rollouts(saved_jsonl_file, cur_data, agents, rollout_idx)
            
        return self.output(agents)
    
def format_question_to_prompt(question):
    system_prompt = """
When tackling complex reasoning tasks, you should first thinks about the reasoning process in the mind and then provides the answer. 

You should strictly follow the format below:

## Reasoning step 1:
Your reasoning process here

## Reasoning step 2:
Your reasoning process here

## Reasoning step 3:
Your reasoning process here
...
## Reasoning step n:
Your reasoning process here

## Answer:
Your answer here
"""
    user_prompt = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "## Reasoning step 1:\n"}
    ]

    
if __name__ == "__main__":
    solver = Solver()
    questions = [
        """Consider the terms of an arithmetic sequence: $-\\frac{{1}}{{3}}, y+2, 4y, \\ldots$. Solve for $y$.""",
        """	
Suppose that $g(x) = 5x - 3$. What is $g^{{-1}}(g^{{-1}}(14))$?""",
    ]

    tokenizer = AutoTokenizer.from_pretrained("/data/pretrain/Qwen/Qwen2.5-0.5B-Instruct")

    agents = [
        MCTS(
            query_ids=tokenizer.apply_chat_template(format_question_to_prompt(query), continue_final_message=True),
            split_sequence=tokenizer.encode("## Reasoning step", add_special_tokens=False),
            max_depth=4,
            tokenizer=tokenizer,
        )
        for query in questions
    ]

    outputs = solver.solve(agents)


