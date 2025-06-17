# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from enum import Enum

import numpy as np

# class BaseNode():

#     state: Dict[str, str] = {"text": "", "extra_info": ""}
#     additional_state_keys: List[str] = []
#     parent: Optional[Any] = None
#     children: List[Any] = []
#     depth: int = 0
#     is_terminal: bool = False
#     reward: Optional[float] = None
#     value: Optional[float] = 0

#     tag: str = "0"
#     consecutive_errors: int = 0

#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)

#         for key in self.additional_state_keys:
#             self.state[key] = ""

#     def has_children(self) -> bool:
#         return self.children != []

#     def is_root(self) -> bool:
#         return self.parent is None
class MeasureType(Enum):
    REWARD = "reward"
    ENTROPY = "entropy"
    ENTROPY_NORM = "entropy_norm"


class MCTSNode():
    def __init__(self, 
                 prefix_ids: List[int],
                 resp_ids: List[int],
                 is_terminal: bool,
                 resp_logprob: float,
                 parent: Optional[MCTSNode] = None,
                 tag: str = "0",
    ):
        self.parent = parent
        self.tag = tag
        self.state = {
            "prefix_ids": prefix_ids,
            "resp_ids": resp_ids,
            "resp_nll": -resp_logprob,
            "resp_nll_norm": -resp_logprob / len(resp_ids),
            "resp_prob": np.exp(resp_logprob) if resp_logprob is not None else None,
            "extra_info": ""
        }
        if self.parent is not None:
            assert self.parent.state['prefix_ids'] + self.parent.state['resp_ids'] == prefix_ids
        self.is_terminal = is_terminal



        self.children: List[MCTSNode] = []
        self.depth = 0 if parent is None else parent.depth + 1
        self._visit_count = 0
        self._reward_sum = 0
        self._entropy_sum = 0 # sum of entropy: Sum(-\logp(y))
        self._entropy_sum_length_norm = 0 # expectation of length-normalized entropy: Sum(Mean_i(-\logp(y_i|y_<i)))
        
        self.input_ids_start_id = len(prefix_ids)
        self.input_ids_end_id = len(prefix_ids) + len(resp_ids)
        self.is_expand = False # NOTE: since we merge `expand` and `simulate`, only the first child is expanded. The others are expanded once their father are selected during selection stage.


        # self.reward = None
        # self.value = 0
        # self.consecutive_errors = 0

    def has_children(self) -> bool:
        return self.children != []

    def is_root(self) -> bool:
        return self.parent is None

    def q_value(self, measure:MeasureType=MeasureType.REWARD) -> float:
        if self._visit_count == 0:
            return 0
        if measure == MeasureType.ENTROPY:
            return self._entropy_sum / self._visit_count
        elif measure == MeasureType.ENTROPY_NORM:
            return self._entropy_sum_length_norm / self._visit_count
        elif measure == MeasureType.REWARD:
            return self._reward_sum / self._visit_count
        else:
            raise NotImplementedError(f"Measure {measure} is not implemented. Use one of {list(MeasureType)}")

    def visit_count(self) -> int:
        return self._visit_count

    def update(self, value: Dict) -> None:
        self._visit_count += 1
        self._reward_sum += value['reward']
        if value['length'] == 0:
            return
        self._entropy_sum += value['nll']
        self._entropy_sum_length_norm += (value['nll'] / value['length'])

    def update_recursive(self, value: Dict, root: MCTSNode) -> None:
        assert isinstance(value, Dict), "Value must be a dictionary with keys 'value' , 'nll', and 'length'."
        
        # update current node
        self.update(value)
        if self == root:
            return
        
        # update current value dict
        value['reward'] # reward remain the same
        value['nll'] += self.state['resp_nll'] # add the logprob of the response
        value['length'] += len(self.state['resp_ids']) # add the length of the response
        
        # update parent node
        assert self.parent is not None, "Parent node must exist to update recursively."
        self.parent.update_recursive(value, root)

    def uct(self, c_puct, measure: MeasureType=MeasureType.REWARD) -> float:
        assert measure in MeasureType, f"Measure must be one of {list(MeasureType)}"

        if not self.parent: return 0
        q_value = self.q_value(measure=measure)
        if self.parent.visit_count() == 0 or self.visit_count() == 0:
            u_value = 0
        else:
            u_value = c_puct * np.sqrt(np.log(self.parent.visit_count()) / (self.visit_count()))
        return q_value + u_value
    
    def puct(self, c_puct, measure: MeasureType=MeasureType.REWARD) -> float:
        assert measure == MeasureType.REWARD, "Only 'reward' measure is supported for puct calculation."
        assert self.state['resp_prob'] is not None, "resp_prob is not set when calculating puct"

        if not self.parent: return 0
        q_value = self.q_value(measure)
        if self.parent.visit_count() == 0 or self.visit_count() == 0:
            u_value = 0
        else:
            u_value = c_puct * np.sqrt(self.parent.visit_count()) / (self.visit_count()) * self.state['resp_prob']
        return q_value + u_value
        
    # def get_reward(self) -> float:
    #     return self.reward if self.reward is not None else 0
