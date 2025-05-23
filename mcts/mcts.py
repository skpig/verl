# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO

from __future__ import annotations

import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from mcts_node import MCTSNode
from pydantic import field_validator
from vllm.outputs import CompletionOutput, RequestOutput

# from rstar_deepthink.constants import (CODE_END, NO_VALID_CHILD,
#                                        TOO_MANY_CODE_ERRORS, TOO_MANY_STEPS)
# from rstar_deepthink.nodes import MCTSNode
# # from rstar_deepthink.agents.utils import math_equiv as is_equiv
# from rstar_deepthink.nodes.base_node import BaseNode

def build_kmp_table(pattern):
    """构建部分匹配表（前缀函数）"""
    n = len(pattern)
    table = [0] * n
    j = 0  # length of previous longest prefix suffix

    for i in range(1, n):
        while j > 0 and pattern[i] != pattern[j]:
            j = table[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            table[i] = j
    return table

def kmp_search(text, pattern):
    """KMP算法在list上搜索 pattern 出现在 text 中的位置"""
    if not pattern:
        return list(range(len(text) + 1))

    table = build_kmp_table(pattern)
    result = []

    j = 0  # index for pattern
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = table[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            result.append(i - j + 1 + len(pattern))
            j = table[j - 1]

    return result


# MCTS 类继承自 Beam Search，用于多步决策问题中的树搜索策略
class MCTS:
    # 中间记录指标，包括问题、答案、评估值等
    intermediate_metric: Dict = {
        "question": "",
        "gt": "", 
        "answers": [],
        "judgements": [],
        "value_estimate": [],
        "rollout_indexs": [],
    }

    def __init__(self, query_ids, split_sequence, max_depth=4, tokenizer=None, c_puct=2, **kwargs) -> None:
        self.root = MCTSNode(prefix_ids=[],
                            resp_ids=query_ids,
                            is_expand=True,
                            is_terminal=False,
                            resp_logprob=0,
                            parent=None,
                            tag="0")
        self.split_sequence = split_sequence
        self.max_depth = max_depth
        self.tokenizer = tokenizer
        self.c_puct = c_puct


         
        # self.prompt_wrap = trivial_prompt_wrap
        # self.obs_wrap = obs_wrap
        # self.step_unwrap = step_result_unwrap
        return


    # # MCTS-related: 从 search_node 或 root 进行选择过程
    # def selection(self, start_node) -> Optional[Type[MCTSNode]]:
    #     """
    #     Possible returns:
    #         1. if start_node has children (not expanded yet), return the first child
    #     """
    #     # if from_root:
    #     #     start_node = self.root
    #     # else:
    #     #     start_node = self.search_node
    #     node = start_node
    #     if node is None: return None
    #     if node.has_children() or node.is_terminal:
    #         next_node = self._select_child(node)  # 根据 PUCT 策略选择最优子节点
    #         if next_node is None:
    #             node.is_terminal = True  # 所有子节点都是终止节点，标记当前节点为终止
    #         node = next_node
    #     return None if (node is None or node.is_terminal) else node

    # 根据 PUCT 值选择子节点
    def _select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue
            puct_value = child.puct(c_puct=self.c_puct)  # 计算当前节点的 puct 值
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        # return best_childs[0] if best_childs else None  # 返回唯一最佳子节点, 尽管可能有多个相同的 puct 值的子节点
        return random.choice(best_childs) if best_childs else None  # 随机选择一个最佳子节点

    # 基于生成结果，展开当前节点，一次性生成多个子节点
    # TODO: 每一个output都是一个完整的rollout, create_child需要调用多次
    def expand_and_simulate_node(self, output_object: RequestOutput, node: Type[MCTSNode]) -> None:
        for idx, output in enumerate(output_object.outputs):
            # if not output.stop_reason: output.stop_reason = ""
            # step_result, parser_result = self.step_unwrap(output.text + output.stop_reason) # concat back the step_end_token (i.e. stop_reason)
            # self.create_child(step_result, parser_result, node, idx)
            self.recursive_create_child(
                node=node,
                step_completion=output.token_ids,
                step_prefix_ids=output_object.prompt_token_ids,
                step_logprobs=output.logprobs,
                rollout_score=output.score,
                # all_resp_ids=output.token_ids,
                # all_logprobs=output.all
            )
        
        # activate all children (some of them might be created but not-activated in previous run)
        for child in node.children:
            child.is_init = True

    # 创建子节点，同时进行backpropagate操作
    def recursive_create_child(
        self, 
        node: Type[MCTSNode],
        step_completion: str, 
        step_prefix_ids: List[int],
        step_logprobs,
        rollout_score: int,
    ) -> None:
        # 1. split the step_completion with self.split_sequence
        split_indices = kmp_search(step_completion, self.split_sequence)

        start_index = 0 # since `## Reasoning step 1:` is in prompt, we can directly use set initial start_index as 0
        parent = node
        cur_prefix = step_prefix_ids
        for index in split_indices + [len(step_completion)]:
            # create node
            child = MCTSNode(
                prefix_ids=cur_prefix,
                resp_ids=step_completion[start_index:index],
                is_terminal=index == len(step_completion),
                is_expand=start_index != 0, # only the first split node is add
                resp_logprob=step_logprobs[start_index:index].sum() if step_logprobs is not None else None,
                parent=parent,
                tag=f"{parent.tag}.{len(parent.children) + 1}",
            )
            parent.children.append(child)

            # update
            cur_prefix = cur_prefix + step_completion[start_index:index]
            start_index = index
            parent = child

        # 通过parsing进行一些改写
        # TODO: exception handler?
        # if parser_result is None:
        #     new_node.is_terminal = True
        #     new_node.state["text"] = step_result
        #     new_node.state["final_answer"] = NO_VALID_CHILD
        #     self.eval_final_answer(new_node)
        # elif parser_result["final_answer"]:
        #     new_node.is_terminal = True
        #     new_node.state["text"] = step_result
        #     new_node.state["final_answer"] = parser_result["final_answer"]
        #     self.eval_final_answer(new_node)
        # elif parser_result["action"]: #TODO: we may need to modify this
        #     observation = code_execution(node, parser_result)
        #     new_node.state["action"] = parser_result["action"]
        #     new_node.state["action_input"] = parser_result["action_input"]
        #     new_node.state["observation"] = observation
        #     if CODE_END in parser_result["action_input"]:
        #         observation = self.obs_wrap(observation)
        #         new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
        #     else:
        #         new_node.state["text"] = step_result

        #     if "error" in observation.lower():  # 错误处理逻辑
        #         new_node.consecutive_errors = node.consecutive_errors + 1
        #         if new_node.consecutive_errors >= self.config.errors_threshold:
        #             observation = self.obs_wrap(observation)
        #             step_result = step_result + CODE_END if CODE_END not in step_result else step_result
        #             new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
        #             new_node.is_terminal = True
        #             new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
        #             self.eval_final_answer(new_node)
        # else:
        #     new_node.state["text"] = step_result
        # # terminate new node if depth exceeds max_depth
        # if not new_node.is_terminal and new_node.depth > self.config.max_depth:
        #     new_node.is_terminal = True
        #     new_node.state["final_answer"] = TOO_MANY_STEPS
        #     self.eval_final_answer(new_node)


        """BackPropagation"""
        parent.update_recursive(rollout_score, root=self.root)




    # # 评估终止节点的答案是否正确，并沿path更新奖励
    # def eval_final_answer(self, node: Type[MCTSNode]) -> None:
    #     if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
    #         node.update(self.config.negative_reward)
    #         return 
    #     if self.config.is_sampling:
    #         final_answer = node.state["final_answer"]
    #         correct = random.random() < 0.3
    #         node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)
    #     else:
    #         self.candidate_nodes.append(node)

    # 中间过程记录答案及其 value 估计值
    def record_intermediate_metric(self, answer, value_estimate):
        self.intermediate_metric["question"] = self.question
        self.intermediate_metric["gt"] = self.ground_truth
        if self.intermediate_metric["rollout_indexs"] and self.rollout_idx in self.intermediate_metric["rollout_indexs"]:
            index = self.intermediate_metric["rollout_indexs"].index(self.rollout_idx)
            if value_estimate > self.intermediate_metric["value_estimate"][index]:
                self.intermediate_metric["answers"][index] = answer
                self.intermediate_metric["judgements"][index] = random.random() < 0.3
                self.intermediate_metric["value_estimate"][index] = value_estimate
        else:
            self.intermediate_metric["answers"].append(answer)
            self.intermediate_metric["judgements"].append(random.random() < 0.3)
            self.intermediate_metric["value_estimate"].append(value_estimate)
            self.intermediate_metric["rollout_indexs"].append(self.rollout_idx)

    # 选择下一步执行的节点（生成新的候选集）
    # FIXME: difference between self.selection and self.select_next_step(): nearly the same if output is None
    def select_next_step(self, from_root=False) -> None:
        """
        Args:
            outputs: List of outputs from the model, each is a return of vllm engine
            from_root: Whether it is for initial selection or not.
        """
        # self.search_node = self.current_nodes[0] if self.current_nodes else None
        self.current_nodes = []
        # if outputs:
        if False:
            assert False
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                if candidate_node.is_terminal and self.config.is_sampling:
                    continue
                value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                if output.value_estimate is None:
                    candidate_node.is_terminal = True

                if candidate_node.is_terminal and candidate_node.state["final_answer"]:
                    if candidate_node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
                        candidate_node.update(self.config.negative_reward)
                    else:
                        self.record_intermediate_metric(answer=candidate_node.state["final_answer"], value_estimate=value_estimate)
                        candidate_node.update_recursive(value_estimate, self.root)
                else:
                    if self.config.terminal_sample:
                        pass
                    else:
                        candidate_node.update(value_estimate)

                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)
        # selection_node = self.selection(start_node=self.root if from_root else self.search_node)
        # if selection_node is not None:
        #     self.current_nodes.append(selection_node)



        node = self.root
        # selection loop 
        while node.has_children():
            node = self._select_child(node)
            self.current_nodes.append(node)
        self.current_nodes.append(node) # add the leaf node
        valid_current_nodes = [node for node in self.current_nodes if not self.is_terminated_node(node)] # remove terminal nodes and nodes with depth > max_depth
        self.current_nodes = valid_current_nodes[-1:] # only keep the last valid leaf for expansion



    # 根据当前节点和模型输出，扩展当前节点，生成多个子节点
    def generate_next_step(self, outputs_lst: List[RequestOutput]) -> None:
        # self.candidate_nodes = [] # refresh candidate
        for current_node, outputs_object in zip(self.current_nodes, outputs_lst):
            # value_estimate = outputs_object.value_estimate # inherit from current_nodes, FIXME: change to assert value_estimate = current_node.get_reward()
            # assert value_estimate is not None, "value_estimate is None, should not be None"
            # assert value_estimate == current_node.get_reward(), "value_estimate is not equal to current_node.get_reward()"
            self.expand_and_simulate_node(outputs_object, current_node)
            # if self.config.update_leaf_value: # FIXME: useless
            #     for value_node in current_node.children:
            #         if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
            #             self.candidate_nodes.append(value_node) 

    # 返回整个搜索树中所有节点的状态
    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            states[node.tag]["q_value"] = node.q_value()
            states[node.tag]["visit_count"] = node.visit_count()
            if node.has_children():
                candidates.extend(node.children)
        return states

    def is_terminated_node(self, node: MCTSNode) -> bool: #TODO: is called
        return node.is_terminal or node.depth > self.max_depth

    # Check if any node in the current_nodes can be expanded
    def should_generate_next(self) -> bool: #TODO: is called 
        # need_generate = False
        # for step_node in self.current_nodes:
        #     if not self.is_terminated_node(step_node):
        #         need_generate = True
        #         break
        # return need_generate
        return len(self.current_nodes) > 0
    
    # def has_expanded(self) -> bool: #TODO: is called
    #     if not self.current_nodes:
    #         return False
    #     step_node = self.current_nodes[0] # FIXME: what is step_node, why only consider the first node in `current_nodes`?
    #     if step_node.has_children():
    #         return True
    #     return False

    # def get_rewards(self): # TODO: is called
    #     rewards = []
    #     for node in self.current_nodes:
    #         rewards.append(node.reward if node.reward is not None else 0) # default reward is 0
    #     return rewards

    def create_prompt(
        self,
    ) -> str: # TODO: is called
        prompts = []
        current_nodes = self.current_nodes
        for current_node in current_nodes:
            # partial_solution = self.collect_partial_solution(current_node)
            # prompt = self.prompt_wrap(
            #     self.question, 
            #     partial_solution,
            #     self.config
            # )
            # prompts.append(prompt)
            prompts.append({
                'prompt_token_ids': current_node.state['prefix_ids'] + current_node.state['resp_ids'],
            })
        return prompts
    
    def collect_partial_solution(self, node: MCTSNode) -> str: #TODO: is called # collect generation in parents nodes #TODO: modify to concat input_ids
        # from leaf to root, and reverse
        trajectory = []
        while node:
            if node.state['text']:
                trajectory.append(node.state['text'])
            node = node.parent
        return "".join(reversed(trajectory))
    
    def draw_tree(self, tree_id, node: MCTSNode=None) -> None:
        if node is None:
            node = self.root
        import matplotlib.pyplot as plt
        import matplotlib
        import networkx as nx
        from networkx.drawing.nx_pydot import graphviz_layout

        # matplotlib.rcParams["text.usetex"] = True

        G = nx.DiGraph()

        def add_nodes_edges(current_node):
            text = self.tokenizer.decode(current_node.state["resp_ids"]).replace(":"," ")
            node_label = f'{text}\nQ={current_node._value_sum}\nN={current_node._visit_count}'
            G.add_node(id(current_node), label=node_label)
            if current_node.parent:
                G.add_edge(id(current_node.parent), id(current_node))
            for child in current_node.children:
                add_nodes_edges(child)

        add_nodes_edges(node)

        pos = graphviz_layout(G, prog='dot')
        labels = nx.get_node_attributes(G, 'label')

        plt.figure(figsize=(30, 20))  # 增大图像尺寸
        nx.draw(G, pos, labels=labels, with_labels=True, 
                node_size=3000,  # 调整节点尺寸
                node_color='lightblue', 
                font_size=3,     # 缩小字体
                alpha=0.9,       # 半透明效果
                arrows=True,     # 显示箭头
                arrowsize=15,    # 调整箭头大小
                width=1.5)       # 调整边的宽度
        plt.title('MCTS Tree')
        plt.savefig(f'mcts/{tree_id}.png', dpi=600, bbox_inches='tight')
        plt.close()