# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/MARIO-Math-Reasoning/Super_MARIO
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

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


class MCTSNode():
    def __init__(self, 
                 prefix_ids: List[int],
                 resp_ids: List[int],
                 is_terminal: bool,
                 resp_logprob: float = None,
                 parent: Optional[MCTSNode] = None,
                 tag: str = "0",
    ):
        self.parent = parent
        self.tag = tag
        self.state = {
            "prefix_ids": prefix_ids,
            "resp_ids": resp_ids,
            "resp_prob": np.exp(resp_logprob) if resp_logprob is not None else None,
            "extra_info": ""
        }
        if self.parent is not None:
            assert self.parent.state['prefix_ids'] + self.parent.state['resp_ids'] == prefix_ids
        self.is_terminal = is_terminal



        self.children = []
        self.depth = 0 if parent is None else parent.depth + 1
        self._visit_count = 0
        self._value_sum = 0
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

    def q_value(self) -> float:
        if self._visit_count == 0:
            return 0
        return self._value_sum / self._visit_count

    def visit_count(self) -> int:
        return self._visit_count

    # def update_visit_count(self, count: int) -> None:
    #     self.__visit_count = count

    def update(self, value: float) -> None:
        # if self.inited is False:
        #     self.inited = True #not used at all
        #     self.value = value # not used at all
        self._visit_count += 1
        self._value_sum += value

    def update_recursive(self, value: float, root: MCTSNode) -> None:
        self.update(value)
        if self == root:
            return
        self.parent.update_recursive(value, root)

    def uct(self, c_puct) -> float:
        if not self.parent: return 0
        q_value = self.q_value()
        if self.parent.visit_count() == 0 or self.visit_count() == 0:
            u_value = 0
        else:
            u_value = c_puct * np.sqrt(np.log(self.parent.visit_count()) / (self.visit_count()))
        return q_value + u_value
    
    def puct(self, c_puct) -> float:
        assert self.state['resp_prob'] is not None, "resp_prob is not set when calculating puct"
        if not self.parent: return 0
        q_value = self.q_value()
        if self.parent.visit_count() == 0 or self.visit_count() == 0:
            u_value = 0
        else:
            u_value = c_puct * np.sqrt(self.parent.visit_count()) / (self.visit_count()) * self.state['resp_prob']
        return q_value + u_value
        
    # def get_reward(self) -> float:
    #     return self.reward if self.reward is not None else 0
