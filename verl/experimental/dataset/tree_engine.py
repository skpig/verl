from dataclasses import dataclass
from typing import List, Optional, Sized
import numpy as np
import torch
import ray
from omegaconf import DictConfig
from verl.protocol import DataProto


@dataclass
class TreeSpec:
    num_parents: int
    children_per_parent: List[int]

class TreeNode:
    """
    A class representing a node in a tree structure.
    """

    def __init__(self, 
                 item,
                 father_item: Optional[int] = None,
                 partial_rollout: Optional[list[int]] = None,
                 step_num=0):
        """
        Initialize the TreeNode with the given data.
        """
        self.item = item # a unique identifier for the node, also the one used to access the node in the dataset
        self.father_item = father_item  # the item index of parent node, None if it's the root node
        self.children_items = []  # list of item indices of child nodes

        self.step_num = step_num  # the number of steps in the training process

        self.partial_rollout = partial_rollout  # the length of the partial rollout

        assert step_num <= 0 or partial_rollout is not None and father_item is not None
    
    @property
    def partial_rollout_len(self):
        """
        The length of the partial rollout.
        If partial_rollout is None, return 0.
        """
        return len(self.partial_rollout)
    
    def depth(self, item2node):
        """
        The depth of the node in the tree.
        """
        if self.step_num <= 0:
            return 0
        father_node = item2node[self.father_item]
        return father_node.depth(item2node) + 1
    
    def get_original_ancestor_item(self, item2node):
        """
        Get the original ancestor of this node.
        The original ancestor is the root node of the tree.
        """
        if self.step_num <= 0:
            return self.item
        father_node = item2node[self.father_item]
        return father_node.get_original_ancestor_item(item2node)

    def add_child(self, child_item: int):
        """
        Add a child node to this node.
        """
        self.children_items.append(child_item)

    def __repr__(self):
        return f"TreeNode(item={self.item})"



class TreeEngine:
    def __init__(self, original_data_len, data_config):
        self.spec = TreeSpec(num_parents=original_data_len, children_per_parent=[])
        self.rng = np.random.default_rng()
        self.tree_config = data_config.tree_data
        self.original_datalength = original_data_len

        # Initialize an empty dataset for new data
        self.root = TreeNode(item=-1, father_item=None, step_num=-1)
        self.item2node = {-1: self.root}
        self.next_item = 0

        for i in range(self.original_datalength):
            node = TreeNode(item=i, father_item=-1, step_num=0)
            self.root.add_child(i)
            self.item2node[i] = node
            self.next_item += 1

    
    def __len__(self):
        return self.next_item
    
    def get_node(self, item: int) -> TreeNode:
        """
        Get the node of the given item.
        """
        return self.item2node.get(item, None)
    
    def get_original_ancestor_item(self, item: int) -> int:
        """
        Get the original ancestor item of the given item.
        """
        node = self.item2node[item]
        return node.get_original_ancestor_item(self.item2node)
    
    def state_dict(self):
        """
        Return the state dict of the dataset.
        """
        return {
            "tree_config": self.tree_config,
            "item2node": self.item2node,
            "next_item": self.next_item,
        }

    def load_state_dict(self, state_dict):
        """
        Load the state dict of the dataset.
        """
        assert self.tree_config == state_dict["tree_config"]
        self.item2node = state_dict["item2node"]
        self.next_item = state_dict["next_item"]
    
    def create_new_node(self, father_node: TreeNode, partial_rollout: List[int], step_num: int) -> None:
        """
        Create a new node with the given father node and partial rollout.
        """
        new_item = self.next_item
        self.item2node[new_item] = TreeNode(
            item=new_item,
            father_item=father_node.item,
            partial_rollout=partial_rollout,
            step_num=step_num
        )
        father_node.add_child(new_item)
        self.next_item += 1


    def update_data_source(self, batch: DataProto, step_num: int) -> None:
        """
        Update the dataset with the current batch.
        This method is called after each training batch.
        """
        items = torch.tensor(batch.non_tensor_batch['item'].astype(int))

        unique_indices, inverse_indices = torch.unique(items, return_inverse=True)

        all_scores = torch.tensor(batch.non_tensor_batch['score']) # raw score
        all_partial_rollout_len = torch.tensor(batch.non_tensor_batch['partial_rollout_len'].astype(int))
        # all_response_mask = batch.batch['response_mask_w_partial_rollouts'].bool()
        all_response_mask = batch.batch['response_mask'].bool()
        all_response_len = all_response_mask.sum(dim=-1).tolist()
        all_responses = batch.batch["responses"].clone() # (bsz, response_len)
        all_values = batch.batch["values"].clone() # (bsz, response_len)
        all_entropys = batch.batch["entropys"].clone() # (bsz, response_len)

        # We can select the item with highest score as the new node
        assert len(unique_indices) == len(set(items)), "Currently, items should be unique in the batch."
        # assert self.use_critic, "Currently only support use_critic=True for TreeDataset"

        # breakpoint()

        # metrics
        new_partial_rollout_len_lst = []
        new_partial_rollout_len_ratio_lst = []
        for i, index in enumerate(inverse_indices):
            item = unique_indices[index].item()
            father_node = self.item2node.get(item, None)
            assert father_node is not None, f"Item {item} not found in the dataset."

            if self.tree_config.correct_only and all_scores[i] == 0:
                continue

            father_depth = father_node.depth(self.item2node)
            if self.tree_config.root_only and father_depth > 0:
                continue
            elif father_depth > 0:
                father_node = self.item2node[father_node.get_original_ancestor_item(self.item2node)] # get the original ancestor node as father node

            # only use the first half of the response as partial rollout
            valid_position_start = all_partial_rollout_len[i]
            valid_position_end = int(all_response_len[i] * self.tree_config.partial_rollout_ratio)
            valid_length = valid_position_end - valid_position_start
            # if the partial rollout is too short, skip
            if valid_length <= self.tree_config.min_partial_rollout_len:
                continue

            valid_values = all_values[i, valid_position_start:valid_position_end]
            valid_entropys = all_entropys[i, valid_position_start:valid_position_end]
            # V1: Use the index with highest value as the new node, should assert critic_lam == 1
            if self.tree_config.name == "value":
                max_value_index = torch.argmax(valid_values).item()

                partial_rollout_len = max_value_index # the index with highest value should be excluded, since V[i] is the value of sequence x[:idx]
            
            # V2: Use the index with highest entropy as the new node
            elif self.tree_config.name == "entropy":
                max_entropy_index = torch.argmax(valid_entropys).item()
                partial_rollout_len = max_entropy_index # the index with highest entropy should be excluded, since H[i] is the entropy of sequence x[:idx]
            
            # V3: Use the index with highest value over 80-percentile entropy tokens
            elif self.tree_config.name == "mix":
                percentile_entropy = torch.kthvalue(valid_entropys, int(valid_length * 0.8))[0]  # kthvalue 从1开始计数
                masked_valid_values = torch.where(valid_entropys > percentile_entropy, valid_values, -float('inf')) # mask low entropy position
                partial_rollout_len = torch.argmax(masked_valid_values).item()
            
            # V4: Use the index with highest entropy over 90-percentile high-value tokens
            elif self.tree_config.name == "mix2":
                percentile_value = torch.kthvalue(valid_values, int(valid_length * 0.9))[0]  # kthvalue 从1开始计数
                masked_valid_entropys = torch.where(valid_values > percentile_value, valid_entropys, -float('inf')) # mask low value position
                partial_rollout_len = torch.argmax(masked_valid_entropys).item()
            
            else:
                raise NotImplementedError(f"Tree config name {self.tree_config.name} not implemented.") 

            # NOTE: partial_rollout_len might be zero here
            partial_rollout = all_responses[i, :valid_position_start + partial_rollout_len].tolist()


            """Create new node"""
            self.create_new_node(father_node, partial_rollout, step_num)

            # metrics
            new_partial_rollout_len_lst.append(partial_rollout_len)
            new_partial_rollout_len_ratio_lst.append(partial_rollout_len / all_response_len[i])
            
        # Remove some old rollouts if log_prob of partial rollout is too low under current policy
        return {
            "dataset/partial_rollout_len_mean": np.mean(new_partial_rollout_len_lst),
            "dataset/partial_rollout_len_std": np.std(new_partial_rollout_len_lst),
            "dataset/partial_rollout_len_max": np.max(new_partial_rollout_len_lst) if new_partial_rollout_len_lst else 0,
            "dataset/partial_rollout_len_min": np.min(new_partial_rollout_len_lst) if new_partial_rollout_len_lst else 0,
            "dataset/partial_rollout_len_ratio_mean": np.mean(new_partial_rollout_len_ratio_lst),
            "dataset/partial_rollout_len_ratio_std": np.std(new_partial_rollout_len_ratio_lst),
            "dataset/partial_rollout_zero_ratio": np.mean(np.array(new_partial_rollout_len_lst) == 0),
        }
    
    def update_posterior(self, item_lst: List[int], reward_lst: List[float]):
        pass

    def select_batch(self, batch_size: int) -> List[int]:
        pass

@ray.remote
class EpsilonRandomTreeEngine(TreeEngine):
    def __init__(self, original_data_len, data_config):
        super().__init__(original_data_len, data_config)
        self.epsilon = data_config.sampler.tree_sampler.epsilon
        self.pointer = 0
    
    def select_batch(self, batch_size: int) -> List[int]:
        batch = []
        # breakpoint()
        while True:
            while self.pointer < self.original_datalength:
                node = self.item2node[self.pointer]
                use_self = (self.rng.random() < self.epsilon) or (len(node.children_items) == 0)
                if use_self:
                    choice = self.pointer
                else:
                    # 注意：np.random.choice 对 Python 对象列表也可用，但更稳妥是从整数里抽
                    choice = self.rng.choice(node.children_items)

                batch.append(choice)
                self.pointer += 1

                if len(batch) == batch_size:
                    return batch
                
            # reset pointer to 0
            self.pointer = 0
        

@ray.remote
class EpsilonGreedyTreeEngine(TreeEngine):
    def __init__(self, original_data_len, data_config):
        super().__init__(original_data_len, data_config)
        self.epsilon = data_config.sampler.tree_sampler.epsilon
        self.N = [0.0] * original_data_len
        self.S = [0.0] * original_data_len
    
    def create_new_node(self, father_node: TreeNode, partial_rollout: List[int], step_num: int) -> None:
        super().create_new_node(father_node, partial_rollout, step_num)

        self.N.append(0.0)
        self.S.append(0.0)

        assert len(self.N) == len(self.S) == self.next_item
    
    def update_posterior(self, item_lst: List[int], reward_lst: List[float]):
        for item, reward in zip(item_lst, reward_lst):
            self.N[item] += 1
            self.S[item] += reward
    
    def select_batch(self, batch_size: int) -> List[int]:
        if self.rng.random() < self.epsilon:
            return list(self.rng.choice(self.next_item, size=batch_size, replace=False))
        
        N = np.array(self.N)
        S = np.array(self.S)
        acc = np.divide(S, np.maximum(1, N)) # 0-value items are not considered, [num_nodes, ]
        error = np.abs(acc - 0.5) # [num_nodes, ]

        idx = np.argsort(error)[:batch_size] # [batch_size, ]
        return list(idx)



@ray.remote
class PGTreeEngine:
    pass
