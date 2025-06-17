# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO

from __future__ import annotations
import json
from textwrap import indent
from token import OP
import uuid
import os
import traceback
import torch
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from pydantic import field_validator
# from vllm.outputs import CompletionOutput, RequestOutput

from .mcts_node import MCTSNode, MeasureType
from verl.utils.reward_score.math_verify import compute_score as math_verify_compute_score

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
            result.append(i - j + 1)
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

    def __init__(self, 
                 data_id: int,
                 query_ids: List[int],
                 split_sequence: List[int],
                 max_depth=4, 
                 tokenizer=None, 
                 c_puct=2, 
                 ground_truth=None, # used for rollout reward estimation
                 compute_score: Callable = math_verify_compute_score,
                 measure_name: MeasureType = MeasureType.REWARD,
                 **kwargs) -> None:
        self.data_id = data_id
        self.root = MCTSNode(prefix_ids=[],
                            resp_ids=query_ids,
                            is_terminal=False,
                            resp_logprob=0,
                            parent=None,
                            tag="0")
        self.split_sequence = split_sequence
        self.max_depth = max_depth
        self.tokenizer = tokenizer
        self.c_puct = c_puct
        self.ground_truth = ground_truth
        self.compute_score = compute_score
        self.search_turn = 0
        self.measure_name = measure_name
        # self.terminate_tree = False  # 用于标记是否终止搜索树


         
        # self.prompt_wrap = trivial_prompt_wrap
        # self.obs_wrap = obs_wrap
        # self.step_unwrap = step_result_unwrap
        return
    
    # TODO:
    def dump_rollouts(self, outputs, rollout_idx: int) -> None:
        raise NotImplementedError("dump_rollouts is not implemented")



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

    # 根据 PUCT 值选择一个non-terminal的子节点
    def _select_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:  # 如果子节点是终止节点，则不能作为下一个展开节点，跳过
                continue
            if self.measure_name == MeasureType.REWARD:
                measure_value = child.puct(c_puct=self.c_puct)  # 计算当前节点的 puct 值
            else:
                measure_value = child.uct(c_puct=self.c_puct, measure=self.measure_name)  # 计算当前节点的 uct 值
            

            if measure_value == best_value:
                best_childs.append(child)
            elif measure_value > best_value:
                best_value = measure_value
                best_childs = [child]

        # return best_childs[0] if best_childs else None  # 返回唯一最佳子节点, 尽管可能有多个相同的 puct 值的子节点
        return random.choice(best_childs) if best_childs else None  # 随机选择一个最佳子节点

    # 基于生成结果，展开当前节点，一次性生成多个子节点
    # TODO: 每一个output都是一个完整的rollout, create_child需要调用多次
    def _expand_and_simulate_node(self, output_object: List[Dict[str, Any]], node: MCTSNode) -> None:

        for idx, output in enumerate(output_object):
            if output.get('output_ids') is not None:  # vllm engine output
                output_text = self.tokenizer.decode(output['output_ids'], skip_special_tokens=True) # type: ignore
                output_ids = output['output_ids']
            elif output.get('text') is not None:  # vllm engine output
                output_text = output['text']
                output_ids = [i[1] for i in  output['meta_info']['output_token_logprobs']]
            else:
                random_filename = f".cache/sglang_rtn/bad_{uuid.uuid4().hex}.tmp"
                with open(random_filename, "w") as f:
                    f.write(json.dumps(output, indent=2, ensure_ascii=False))
                continue
            score = self.compute_score(data_source=None, solution_str=output_text, ground_truth=self.ground_truth)['score']
            # score = random.random() > 0.98 # DEBUG:
            # if score == 1:
            #     self.terminate_tree = True
                

            self._recursive_create_child(
                node=node,
                step_completion_ids=output_ids,
                step_prefix_ids=self.create_prompt(node),
                step_logprobs=[i[0] for i in output['meta_info']['output_token_logprobs']],
                rollout_score=score,
                # all_resp_ids=output.token_ids,
                # all_logprobs=output.all
            )
        node.is_expand = True  # 标记当前节点为已展开

    # 创建子节点，同时进行backpropagate操作
    def _recursive_create_child(
        self, 
        node: MCTSNode,
        step_completion_ids: List[int],
        step_prefix_ids: List[int],
        step_logprobs,
        rollout_score: int,
    ) -> None:
        # 1. split the step_completion with self.split_sequence
        split_indices = kmp_search(step_completion_ids, self.split_sequence[0])
        split_indices += kmp_search(step_completion_ids, self.split_sequence[1])
        split_indices = sorted(set(split_indices))  # 去重并排序

        if len(split_indices) == 0:
            start_index = 0
        else:
            start_index = split_indices[0] # skip the content between [0, split_indices[0]) as it is mostly white space
            split_indices = split_indices[1:]  # 去掉第一个分割点，因为它已经被包含在第一个子节点中

        parent = node
        cur_prefix = step_prefix_ids
        for index in split_indices + [len(step_completion_ids)]:
            # create node
            child = MCTSNode(
                prefix_ids=cur_prefix,
                resp_ids=step_completion_ids[start_index:index],
                is_terminal=index == len(step_completion_ids), # only the last step is terminal / leaf
                resp_logprob=sum(step_logprobs[start_index:index]),
                parent=parent,
                tag=f"{parent.tag}.{len(parent.children) + 1}",
            )
            parent.children.append(child)

            # update
            cur_prefix = cur_prefix + step_completion_ids[start_index:index]
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
        parent.update_recursive(
            value={"reward": rollout_score, "nll": 0, "length": 0},
            root=self.root
        )




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


    # 选择下一步进行展开的节点（该节点必须是未被展开过）
    def select_next_step(self, from_root=False) -> Optional[MCTSNode]:
        """
        Args:
            outputs: List of outputs from the model, each is a return of vllm engine
            from_root: Whether it is for initial selection or not.
        """
        # self.search_node = self.current_nodes[0] if self.current_nodes else None
        self.current_nodes: List[MCTSNode] = []

        node = self.root
        # selection loop 
        # continue until we reach an unexpanded node
        self.current_nodes.append(node)
        # the last node is an unexpanded node or None, which is the child of an expanded node
        while node is not None and node.is_expand:
            node = self._select_child(node)
            self.current_nodes.append(node)
        valid_current_nodes = [node for node in self.current_nodes if not self.is_terminated_node(node)] # remove terminal nodes and nodes with depth > max_depth

        # if no valid nodes, stop the search
        if len(valid_current_nodes) == 0:
            return None

        self.current_nodes = valid_current_nodes[-1:] # only keep the last valid leaf for expansion
        return self.current_nodes[0]



    # 根据当前节点和模型输出，扩展当前节点，生成多个子节点
    def generate_next_step(self, outputs_lst: List[Dict[str, Any]]) -> None:
        self.search_turn += 1
        self._expand_and_simulate_node(outputs_lst, self.current_nodes[0])
        return


        # self.candidate_nodes = [] # refresh candidate
        for current_node, outputs_object in zip(self.current_nodes, outputs_lst):
            # value_estimate = outputs_object.value_estimate # inherit from current_nodes, FIXME: change to assert value_estimate = current_node.get_reward()
            # assert value_estimate is not None, "value_estimate is None, should not be None"
            # assert value_estimate == current_node.get_reward(), "value_estimate is not equal to current_node.get_reward()"
            self._expand_and_simulate_node(outputs_object, current_node)
            # if self.config.update_leaf_value: # FIXME: useless
            #     for value_node in current_node.children:
            #         if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
            #             self.candidate_nodes.append(value_node) 

    def is_terminated_node(self, node: MCTSNode) -> bool: #TODO: is called
        return node is None or node.is_terminal or node.depth > self.max_depth

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
        node: Optional[MCTSNode] = None,
    ) -> List[int]:
        if node is None:
            current_nodes = self.current_nodes
            assert len(current_nodes) == 1, "current_nodes is empty"
            current_node = current_nodes[0]
            node = current_node
        prompt_ids = node.state['prefix_ids'] + node.state['resp_ids']
        return prompt_ids
    
    # def collect_partial_solution(self, node: MCTSNode) -> str: #TODO: is called # collect generation in parents nodes #TODO: modify to concat input_ids
    #     # from leaf to root, and reverse
    #     trajectory = []
    #     while node:
    #         if node.state['text']:
    #             trajectory.append(node.state['text'])
    #         node = node.parent
    #     return "".join(reversed(trajectory))
    
    def bound_string_to_limited_width(self, long_string: str, width: int = 100) -> str:
        """
        将长字符串截断为指定宽度的字符串，保留完整的单词。
        如果字符串长度超过指定宽度，则从末尾开始换行"""
        if len(long_string) <= width:
            return long_string
        
        # Find all LaTeX equation blocks \(...\)
        equations = []
        equation_pattern = r'\\\(.*?\\\)'
        
        # Replace equations with placeholders and store them
        string_slices = []
        last_end = 0
        for i, match in enumerate(re.finditer(equation_pattern, long_string, re.DOTALL)):
            placeholder = f" __EQUATION__{i}__ "
            equations.append(match.group())
            # long_string = long_string[:match.start()] + placeholder + long_string[match.end():]
            
            string_slices.append(long_string[last_end:match.start()])
            string_slices.append(placeholder)
            last_end = match.end()
        string_slices.append(long_string[last_end:])  # Add the remaining part of the string
        long_string = ''.join(string_slices)


            
        
        # Split into words, but treat equation placeholders as single units
        words = long_string.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Check if this word is an equation placeholder
            if word.startswith("__EQUATION__") and word.endswith("__"):
                # Restore the original equation
                print(f"Restoring equation: {word}")
                eq_index = int(word.split("__")[2])
                actual_word = equations[eq_index]
            else:
                actual_word = word
            
            # If adding this word would exceed width, start a new line
            if current_line and len(current_line) + len(actual_word) + 1 > width:
                lines.append(current_line)
                current_line = actual_word
            else:
                # Add word to current line
                if current_line:
                    current_line += " " + actual_word
                else:
                    current_line = actual_word
        
        # Add the last line if it exists
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines)
    
    def draw_tree(self, node: Optional[MCTSNode]=None) -> None:
        if node is None:
            node = self.root
        import matplotlib.pyplot as plt
        import matplotlib
        import networkx as nx
        from networkx.drawing.nx_pydot import graphviz_layout

        # matplotlib.rcParams["text.usetex"] = True
        # plt.rcParams['axes.formatter.use_mathtext'] = False
        plt.rcParams['text.parse_math'] = False

        G = nx.DiGraph()

        def add_nodes_edges(current_node, depth=0):
            text = self.tokenizer.decode(current_node.state["resp_ids"]).replace(":"," ") # type: ignore
            text = self.bound_string_to_limited_width(text, width=70)  # 限制宽度为100字符
            
            # print(text)
            node_label = f'{text}\n\nQ={current_node._reward_sum};N={current_node._visit_count};PUCT={current_node.puct(self.c_puct):.2f};\nNLL={current_node.state["resp_nll"]:.2e};NLL_NORM={current_node.state["resp_nll_norm"]:.2e};\nE={(current_node._entropy_sum / current_node._visit_count) :.2e};E_NORM={(current_node._entropy_sum_length_norm / current_node._visit_count) :.2e}'
            G.add_node(id(current_node), label=node_label, depth=depth)
            if current_node.parent:
                G.add_edge(id(current_node.parent), id(current_node))
            for child in current_node.children:
                add_nodes_edges(child, depth=depth + 1)

        add_nodes_edges(node)

        # pos = graphviz_layout(G, prog='dot')
        pos = nx.nx_agraph.pygraphviz_layout(G, prog='twopi', root=id(self.root),
                                             args="-Goverlap=scale -Gsep=3")  # 使用twopi布局，根节点为self.root
        labels = nx.get_node_attributes(G, 'label')
        colors = [data.get('depth', -1) for _, data in G.nodes(data=True)]
        cmap = plt.cm.get_cmap('viridis', 15)  # 使用viridis颜色映射

        plt.figure(figsize=(30, 30))  # 增大图像尺寸
        nx.draw(G, pos, labels=labels, with_labels=True, 
                node_size=3000,  # 调整节点尺寸
                node_color=colors,
                cmap=cmap,  # 使用颜色映射
                font_size=1.5,     # 缩小字体
                alpha=0.7,       # 半透明效果
                arrows=False,     # 显示箭头
                # arrowsize=5,    # 调整箭头大小
                width=0.5)       # 调整边的宽度
        plt.title('MCTS Tree')
        try:
            os.makedirs(f'outputs/{self.data_id}/{self.measure_name}', exist_ok=True)
            # plt.savefig(f'outputs/{self.data_id}/{self.search_turn}.png', dpi=600, bbox_inches='tight')
            plt.savefig(f'outputs/{self.data_id}/{self.measure_name}/{self.search_turn}.pdf', dpi=600, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving figure: {e}")
            traceback.print_exc()
            with open(f'outputs/{self.data_id}/{self.search_turn}.txt', 'w') as f:
                f.write(f"Error saving figure: {e}\n")
                f.write(traceback.format_exc())
        plt.close()