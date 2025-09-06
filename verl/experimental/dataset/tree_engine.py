from collections import defaultdict
import scipy
from dataclasses import dataclass, is_dataclass
import math
import random
from typing import Any, Dict, List, Optional, Sized, Tuple
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
        self.spec = TreeSpec(num_parents=original_data_len, children_per_parent=[0] * original_data_len)
        self.rng = np.random.default_rng()
        self.tree_config = data_config.tree_data
        self.original_datalength = original_data_len

        # Initialize an empty dataset for new data
        self.root = TreeNode(item=-1, father_item=None, step_num=-1)
        self.item2node = {-1: self.root}
        self.next_item = 0

        # 统计相关属性
        self.parent_selection_counts = [0] * original_data_len  # 每个parent node被选中的次数（包括它的孩子）

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
    
    def get_children_items(self, item: int) -> List[int]:
        """
        Get the children items of the given item.
        """
        node = self.item2node[item]
        return node.children_items
    
    # def get_father_item(self, item: int) -> int:
    #     """
    #     Get the father item of the given item.
    #     """
    #     node = self.item2node[item]
    #     return node.father_item
    
    def state_dict(self):
        """
        Return the state dict of the dataset.
        """
        return {
            "tree_config": self.tree_config,
            "item2node": self.item2node,
            "next_item": self.next_item,
            "parent_selection_counts": self.parent_selection_counts,
        }

    def load_state_dict(self, state_dict):
        """
        Load the state dict of the dataset.
        """
        assert self.tree_config == state_dict["tree_config"]
        self.item2node = state_dict["item2node"]
        self.next_item = state_dict["next_item"]
        self.root = self.item2node[-1]
        # 恢复统计信息，如果不存在则使用默认值
        if "parent_selection_counts" in state_dict:
            self.parent_selection_counts = state_dict["parent_selection_counts"]
        else:
            # 兼容旧版本，如果没有统计信息则初始化为0
            self.parent_selection_counts = [0] * self.original_datalength
    
    def create_new_node(self, father_node: TreeNode, partial_rollout: List[int], step_num: int, score: float) -> None:
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

        self.spec.children_per_parent[father_node.item] += 1

    def update_data_source(self, batch, step_num: int) -> Dict[str, float]:
        """
        扩展版：支持 items 在一个 batch 内重复。
        会按 item 分组，对同一 item 的多个样本作为一个小 batch 一次性处理。
        """
        # ---------- pull batch fields ----------
        # items
        items = torch.as_tensor(batch.non_tensor_batch["item"].astype(np.int32)).to(torch.long)

        # raw score ∈ {0,1}
        all_scores = torch.as_tensor(batch.non_tensor_batch["score"]).to(torch.long)
        assert torch.all((all_scores == 0) | (all_scores == 1)), \
            "Currently only support score in {0, 1}."

        # 已有的 partial rollout 起点（绝对位置）
        all_partial_rollout_len = torch.as_tensor(
            batch.non_tensor_batch["partial_rollout_len"].astype(np.int32)
        ).to(torch.long)

        # masks & lengths
        all_response_mask = batch.batch["response_mask"].to(torch.bool)
        all_response_len = all_response_mask.sum(dim=-1).to(torch.long)

        # sequences & per-token stats
        all_responses = batch.batch["responses"]          # (bsz, T)
        all_values    = batch.batch.get("values", None)  # (bsz, T)
        all_entropys  = batch.batch["entropys"]           # (bsz, T)

        bsz = items.numel()
        assert all_responses.size(0) == bsz, "batch dims mismatch"

        # ---------- group by item ----------
        # groups: item_value(int) -> list[int]
        groups: Dict[int, List[int]] = {}
        for i in range(bsz):
            key = int(items[i].item())
            groups.setdefault(key, []).append(i)

        # ---------- process each group ----------
        all_partial_lens: List[int] = []     # 相对窗口起点的长度 j
        all_partial_ratios: List[float] = [] # j / response_len  (保持与旧指标一致)

        for item_val, idx_list in groups.items():
            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=items.device)
            gmetrics = self._batch_create_nodes(
                item=item_val,
                idx=idx_tensor,
                all_scores=all_scores,
                all_partial_rollout_len=all_partial_rollout_len,
                all_response_len=all_response_len,
                all_responses=all_responses,
                all_values=all_values,
                all_entropys=all_entropys,
                step_num=step_num,
            )
            all_partial_lens.extend(gmetrics["partial_lens"])
            all_partial_ratios.extend(gmetrics["partial_ratios"])

        # ---------- aggregate metrics ----------
        def _safe_mean(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        def _safe_std(xs: List[float]) -> float:
            return float(np.std(xs)) if xs else 0.0

        return {
            "dataset/num_nodes": self.next_item,
            "dataset/partial_rollout_len_mean": np.mean(all_partial_lens),
            "dataset/partial_rollout_len_std": np.std(all_partial_lens),
            "dataset/partial_rollout_len_max": np.max(all_partial_lens) if all_partial_lens else 0,
            "dataset/partial_rollout_len_min": np.min(all_partial_lens) if all_partial_lens else 0,
            "dataset/partial_rollout_len_ratio_mean": np.mean(all_partial_ratios),
            "dataset/partial_rollout_len_ratio_std": np.std(all_partial_ratios),
            "dataset/partial_rollout_zero_ratio": np.mean(np.array(all_partial_lens) == 0),
        }
    # ----------------------------------------------------------------------

    def _batch_create_nodes(
        self,
        item: int,
        idx: torch.Tensor,               # 1D, dtype=long
        all_scores: torch.Tensor,                   # (bsz,)
        all_partial_rollout_len: torch.Tensor,      # (bsz,)
        all_response_len: torch.Tensor,             # (bsz,)
        all_responses: torch.Tensor,                # (bsz, T)
        all_values: torch.Tensor,                   # (bsz, T)
        all_entropys: torch.Tensor,                 # (bsz, T)
        step_num: int,
    ) -> Dict[str, Any]:
        """
        针对同一个 item 的一批样本统一处理，返回该组的指标。
        """
        # ----- read config with defaults -----
        cfg = self.tree_config
        name: str = cfg.name
        ratio: float = cfg.partial_rollout_ratio
        min_len: int = cfg.min_partial_rollout_len
        root_only: bool = cfg.root_only


        # ----- father node -----
        father_node = self.item2node[item]

        depth = father_node.depth(self.item2node)
        if root_only and depth > 0:
            # 仅允许在 root 挂子节点
            return {"partial_lens": [], "partial_ratios": []}
        elif depth > 0:
            # 将父节点提到原始祖先（root）
            root_item = father_node.get_original_ancestor_item(self.item2node)
            father_node = self.item2node[root_item]
        
        device = all_responses.device
        T = all_responses.size(1)
        m = idx.numel()

        # ----- gather group tensors -----
        responses_g = all_responses.index_select(0, idx)                # (m, T)
        scores_g   = all_scores.index_select(0, idx)                    # (m,)
        start_g    = torch.clamp_min(all_partial_rollout_len.index_select(0, idx), min=10)       # (m,) We don't want a too short partial rollout
        rlen_g     = all_response_len.index_select(0, idx)              # (m,)
        values_g   = all_values.index_select(0, idx) if all_values is not None else None # (m, T)
        entropies_g= all_entropys.index_select(0, idx) # (m, T)

        # ----- filter out invalid rows -----
        # 仅对 score=1 的行抽样
        valid_row = torch.ones(m, dtype=torch.bool, device=device)
        keep_incorrect_prob: float = cfg.keep_incorrect_prob
        with torch.no_grad():
            rand = torch.rand(m, device=device)
        valid_row &= torch.where(scores_g == 0, rand <= keep_incorrect_prob, torch.ones_like(rand, dtype=torch.bool))

        # 有效窗口长度

        end_g = torch.floor(rlen_g.to(torch.float32) * ratio).to(torch.long)  # (m,), we need to ensure a sufficient long response space
        valid_len = end_g - start_g                                           # (m,)
        valid_row &= (valid_len > min_len)

        # ----- build window mask (m, T) -----
        # mask[i, t] = (start_i <= t < end_i)
        arange_T = torch.arange(T, device=device).view(1, T)  # (1, T)
        start_exp = start_g.view(m, 1)
        end_exp   = end_g.view(m, 1)
        mask_win = (arange_T >= start_exp) & (arange_T < end_exp)        # (m, T)
        mask_win &= valid_row.view(m, 1) # mask out invalid rows
        num_valid_tokens = mask_win.sum().item()

        if num_valid_tokens == 0:
            return {"partial_lens": [], "partial_ratios": []}

        # 基础掩码后的张量
        mv = torch.where(mask_win, values_g, -float('inf')) if values_g is not None else None # masked values of (m, T)
        me = torch.where(mask_win, entropies_g, -float('inf'))  # masked entropies of (m, T)

        # ----- strategy-wise batched selection -----
        if name == "value":
            # sel_score, pos = mv.max(dim=1)   # row-wise argmax over window
            pos = mv.argmax()
            row_id, col_id = torch.unravel_index(pos, mv.shape)
        elif name == "entropy":
            pos = me.argmax()
            row_id, col_id = torch.unravel_index(pos, me.shape)
        elif name == "mix":
            flattened_entropy = torch.masked_select(me, mask_win)
            percentile_entropy = torch.kthvalue(flattened_entropy, int(num_valid_tokens * 0.8))[0]  # kthvalue 从1开始计数
            masked_valid_values = torch.where(me > percentile_entropy, mv, -float('inf')) # mask low entropy position
            pos = masked_valid_values.argmax()
            row_id, col_id = torch.unravel_index(pos, masked_valid_values.shape)
        elif name == "mix2":
            flattened_value = torch.masked_select(mv, mask_win)
            percentile_value = torch.kthvalue(flattened_value, int(num_valid_tokens * 0.8))[0]  # kthvalue 从1开始计数
            masked_valid_values = torch.where(mv > percentile_value, me, -float('inf')) # mask low value position
            pos = masked_valid_values.argmax()
            row_id, col_id = torch.unravel_index(pos, masked_valid_values.shape)
        else:
            raise ValueError(f"Invalid tree config name: {name}")
        
        # ----- select batch -----
        partial_rollout_len = (col_id).item()
        partial_rollout_ratio = partial_rollout_len / rlen_g[row_id].item()
        partial_rollout = responses_g[row_id, :partial_rollout_len].tolist()


        # ----- create new node -----
        self.create_new_node(father_node, partial_rollout, step_num, scores_g[row_id].item())

        # metrics
        return {"partial_lens": [partial_rollout_len], "partial_ratios": [partial_rollout_ratio]}


        

    def update_posterior(self, item_lst: List[int], reward_lst: List[float], step_num: int):
        return {}

    def select_batch(self, batch_size: int, step_num: int) -> Tuple[List[int], Dict[str, float]]:
        raise NotImplementedError

    
    def _get_batch_statistics(self, selected_items: List[int], step_num: int) -> dict:
        """
        获取batch的统计信息
        """
        # 统计unique的parent node数量
        unique_parents = set()
        for item in selected_items:
            original_ancestor = self.get_original_ancestor_item(item)
            unique_parents.add(original_ancestor)
            self.parent_selection_counts[original_ancestor] += 1

        return {
            "sampler/unique_parent_nodes_in_batch": len(unique_parents),
        }
    
    def async_wrap_all(self, batch: DataProto, step_num: int, bsz: int):
        posterior_merics = self.update_posterior(batch.non_tensor_batch["item"].tolist(), batch.non_tensor_batch["score"].tolist(), step_num)
        data_metrics = self.update_data_source(batch, step_num)
        batch, selection_metrics = self.select_batch(bsz, step_num)
        metrics = {
            **posterior_merics,
            **data_metrics,
            **selection_metrics,
        }
        return batch, metrics

@ray.remote
class EpsilonRandomTreeEngine(TreeEngine):
    def __init__(self, original_data_len, data_config):
        super().__init__(original_data_len, data_config)
        self.epsilon = data_config.sampler.tree_sampler.epsilon
        self.pointer = 0
    
    def select_batch(self, batch_size: int, step_num: int) -> List[int]:
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
                    # 返回batch和统计信息
                    return batch, self._get_batch_statistics(batch)
                
            # reset pointer to 0
            self.pointer = 0
        

@ray.remote
class EpsilonGreedyTreeEngine(TreeEngine):
    def __init__(self, original_data_len, data_config):
        super().__init__(original_data_len, data_config)
        self.epsilon = data_config.sampler.tree_sampler.epsilon
        self.N = [0.0] * original_data_len
        self.S = [0.0] * original_data_len
    
    def create_new_node(self, father_node: TreeNode, partial_rollout: List[int], step_num: int, score: float) -> None:
        super().create_new_node(father_node, partial_rollout, step_num, score)

        self.N.append(0.0)
        self.S.append(0.0)

        assert len(self.N) == len(self.S) == self.next_item
    
    def update_posterior(self, item_lst: List[int], reward_lst: List[float], step_num: int):
        for item, reward in zip(item_lst, reward_lst):
            self.N[item] += 1
            self.S[item] += reward
        return {}
    
    def select_batch(self, batch_size: int, step_num: int) -> List[int]:
        if self.rng.random() < self.epsilon:
            batch = list(self.rng.choice(self.next_item, size=batch_size, replace=False))
            # 返回batch和统计信息
            return batch, self._get_batch_statistics(batch)
        
        N = np.array(self.N)
        S = np.array(self.S)
        acc = np.divide(S, np.maximum(1, N)) # 0-value items are not considered, [num_nodes, ]
        error = np.abs(acc - 0.5) # [num_nodes, ]

        idx = np.argsort(error)[:batch_size] # [batch_size, ]
        batch = list(idx)

        # 返回batch和统计信息
        return batch, self._get_batch_statistics(batch)



@ray.remote
class PGTreeEngine(TreeEngine):
    def __init__(self, original_data_len, data_config):
        super().__init__(original_data_len, data_config)

        # Fixed parameters
        self.use_warmup = data_config.sampler.tree_sampler.use_warmup
        self.diverse_threshold = int(data_config.sampler.tree_sampler.diverse_threshold)
        self.father_only_ratio = data_config.sampler.tree_sampler.father_only_ratio
        self.mu0 = float(data_config.sampler.tree_sampler.mu0)
        self.tau0 = float(data_config.sampler.tree_sampler.tau0)
        self.sigma0 = float(data_config.sampler.tree_sampler.sigma0) if data_config.sampler.tree_sampler.sigma0 is not None else None
        self.delta = data_config.sampler.tree_sampler.delta
        self.gamma = data_config.sampler.tree_sampler.gamma
        self.gibbs_sweeps = int(max(1, data_config.sampler.tree_sampler.gibbs_sweeps))
        self.rng = np.random.default_rng()

        self.tau0_2 = self.tau0 ** 2

        # State variables
        self.psi = self.rng.normal(loc=self.mu0, scale=self.tau0, size=self.original_datalength)
        # NOTE: for parents, self.variance[i] is the current variance of \psi_i; but for children, it is the **INITIAL** variance of p(\psi_i | \psi_parent)
        self.variance = np.ones(self.original_datalength) * self.tau0_2
        self.s = np.zeros(self.original_datalength)
        self.n = np.zeros(self.original_datalength)
        self.last_touch = np.zeros(self.original_datalength)
        self.father_last_touch = np.zeros(self.original_datalength)
        self.select_num = np.zeros(self.original_datalength)
        self.father_select_num = np.zeros(self.original_datalength)

        # ---- Polya-Gamma sampler backends (pypolyagamma -> polyagamma -> truncated series) ----
        self._pg_engine = None  # (kind, handle)
        self._init_pg_engine(data_config.train_batch_size)

    def _init_pg_engine(self, batch_size: int, force: Optional[str] = None):
        """Initialize PG backend once. force in {"pypolyagamma","polyagamma","trunc"}."""
        from polyagamma import random_polyagamma  # functional API
        self._pg_engine = ("polyagamma", random_polyagamma)
        # from pypolyagamma import PyPolyaGamma  # class + pgdraw
        # self._pg_engine_lst = [PyPolyaGamma(seed=i) for i in range(batch_size)]
        # if self._pg_engine is not None:
        #     return
        # import os
        # if force is None:
        #     force = os.environ.get("HIER_TS_PG_BACKEND", None)

        # def set_engine(kind, handle):
        #     self._pg_engine = (kind, handle)

        # if force in ("pypolyagamma", None):
        #     try:
        #         from pypolyagamma import PyPolyaGamma  # class + pgdraw
        #         set_engine("pypolyagamma", PyPolyaGamma())
        #         return
        #     except Exception:
        #         force = "polyagamma"
        # if force in ("polyagamma", None):
        #     from polyagamma import random_polyagamma  # functional API
        #     set_engine("polyagamma", random_polyagamma)
        #     return

        # raise NotImplementedError(f"PG backend {force} not implemented.")

    def sample_pg(self, b: float, c: float, rng: Optional[np.random.Generator] = None, trunc: int = 200) -> float:
        if b <= 0:
            return 0.0
        kind, eng = self._pg_engine
        if kind == "pypolyagamma":
            pg = eng
            return float(pg.pgdraw(b, c))
        elif kind == "polyagamma":
            fn = eng
            return float(fn(b, c, random_state=rng)) if rng is not None else float(fn(b, c))



    def create_new_node(self, father_node: TreeNode, partial_rollout: List[int], step_num: int, score: float) -> None:
        super().create_new_node(father_node, partial_rollout, step_num, score)

        # based on partial_rollout, prelocate \psi
        father_item = father_node.item
        father_psi = self.psi[father_item]
        father_variance = self.variance[father_item]
        if self.sigma0 is None:
            father_p = 1 / (1 + np.exp(-father_psi))
            p_low = max(0.01, father_p - self.delta)
            p_high = min(0.99, father_p + self.delta)
            psi_low = np.log(p_low / (1 - p_low))
            psi_high = np.log(p_high / (1 - p_high))
            sigma_low = (father_psi - psi_low) / 1.96
            sigma_high = (psi_high - father_psi) / 1.96
            sigma = max(sigma_low, sigma_high)

            final_sigma = max(sigma, 0.02)
            print("[PG Engine] Create new node: father_item={}, father_psi={}, father_variance={}, final_sigma={}, final_cliped_sigma={}".format(father_item, father_psi, father_variance, sigma, final_sigma))
        else:
            final_sigma = self.sigma0
            print("[PG Engine] Create new node: father_item={}, father_psi={}, father_variance={}, final_sigma={}".format(father_item, father_psi, father_variance, final_sigma))
        

        # add new node to the tree
        cur_psi = self.rng.normal(loc=father_psi, scale=final_sigma)
        self.psi = np.append(self.psi, cur_psi)
        # self.s = np.append(self.s, score)
        # self.n = np.append(self.n, 1.0)
        self.s = np.append(self.s, 0.0)
        self.n = np.append(self.n, 0.0)
        self.variance = np.append(self.variance, final_sigma ** 2)
        self.last_touch = np.append(self.last_touch, step_num)
        self.select_num = np.append(self.select_num, 0)
        self.father_last_touch[int(father_item)] = step_num
        self.spec.children_per_parent[father_item] += 1
    
    def update_posterior(self, item_lst: List[int], reward_lst: List[float], step_num: int):
        metrics = {}
        items = np.array(item_lst)
        rewards = np.array(reward_lst)

        # update observations
        # lazy update
        # time_step = step_num - self.last_touch[items]
        # self.last_touch[items] = step_num
        # self.s[items] *= discount
        # self.n[items] *= discount
        # self.s[items] += rewards
        # self.n[items] += 1

        discount = self.gamma
        self.s *= discount
        self.n *= discount
        # BUGGY: this is not correct, since the items might not be unique
        # self.s[items] += rewards
        # self.n[items] += 1
        item2rewardlst = defaultdict(list)
        for item, reward in zip(items, rewards):
            self.s[item] += reward
            self.n[item] += 1
            item2rewardlst[item].append(reward)
        item2acc = {item: np.mean(reward_lst).item() for item, reward_lst in item2rewardlst.items()}
        item2theta = {item: 1 / (1 + np.exp(-self.psi[item])).item() for item in item2rewardlst.keys()}
        # calculate correlations & error
        if len(item2acc) > 1:
            accs = np.array(list(item2acc.values()))
            thetas = np.array(list(item2theta.values()))
            r, pvalue = scipy.stats.pearsonr(accs, thetas)
            error = np.mean(np.abs(accs - thetas))
            metrics.update({
                "sampler/pg_correlation": r,
                "sampler/pg_pvalue": pvalue,
                "sampler/pg_error": error,
            })
            print("[PG Engine] Step {}: correlation={}, error={}".format(step_num, r, error))


        # update last touch
        self.last_touch[items] = step_num
        for item in items:
            father_item = self.get_original_ancestor_item(item)
            self.father_last_touch[int(father_item)] = step_num

        # group all items by parent
        parent_items = list(range(self.original_datalength))
        # parent_items = self.get_father_item(items)
        # parent_items = np.unique(parent_items)
        for _ in range(self.gibbs_sweeps):
            self._gibbs_one_sweep_selected(parent_items)

        return metrics

    def _gibbs_one_sweep_selected(self, p_lst):
        inv_tau02 = 1.0 / self.tau0_2

        # Leaves
        b_lst = [] # Sample omega in parallel TODO:
        c_lst = []
        sum_inv_sigma2_lst = dict()
        sum_inv_sigma2_w_psi_lst = dict()
        for p in p_lst:
            sum_inv_sigma2 = 0
            sum_inv_sigma2_w_psi = 0
            for j in self.get_children_items(p):
                n_ = float(self.n[j])
                s_ = float(self.s[j])
                kappa = s_ - n_ / 2.0
                psi_cur = float(self.psi[j])
                omega = self.sample_pg(b=n_, c=psi_cur, rng=self.rng)
                # b_lst.append(n_)
                # c_lst.append(psi_cur)
                inv_sigma2 = 1.0 / self.variance[j]
                V = 1.0 / (inv_sigma2 + omega)
                m = V * (inv_sigma2 * self.psi[p] + kappa)
                self.psi[j] = self.rng.normal(loc=m, scale=math.sqrt(V))

                # self.variance[j] = V # NOTE: should not update variance of children
                sum_inv_sigma2 += inv_sigma2
                sum_inv_sigma2_w_psi += inv_sigma2 * self.psi[j]
            sum_inv_sigma2_lst[p] = sum_inv_sigma2
            sum_inv_sigma2_w_psi_lst[p] = sum_inv_sigma2_w_psi

        # Roots
        for p in p_lst:
            n_ = float(self.n[p])
            s_ = float(self.s[p])
            kappa = s_ - n_ / 2.0
            psi_cur = float(self.psi[p])
            omega = self.sample_pg(b=n_, c=psi_cur, rng=self.rng) if n_ > 0 else 0.0

            V0 = 1.0 / (inv_tau02 + sum_inv_sigma2_lst[p] + omega)
            m0 = V0 * (inv_tau02 * self.mu0 + sum_inv_sigma2_w_psi_lst[p] + kappa)

            self.psi[p] = self.rng.normal(loc=m0, scale=math.sqrt(V0)) 
            self.variance[p] = V0

    def select_batch(self, batch_size: int, step_num: int) -> Tuple[List[int], Dict[str, float]]:
        thetas = 1 / (1 + np.exp(-self.psi)) # [num_nodes, ]

        if self.use_warmup and step_num < self.original_datalength / batch_size:
            return [i % self.original_datalength for i in range(batch_size * step_num, batch_size * (step_num + 1))], {}
        
        # father_only_ratio = self.tree_config.father_only_ratio
        if self.father_only_ratio is not None:
            if (self.rng.random() < self.father_only_ratio or step_num < self.original_datalength / batch_size):
                father_only_round = True
            else:
                father_only_round = False
        else:
            father_only_round = None
        
        # test diverse_threshold
        if (step_num - self.father_last_touch > self.diverse_threshold).sum() < batch_size:
            diverse_enable = False
        else:
            diverse_enable = True

        error = np.abs(thetas - 0.5)
        ids = np.argsort(error)
        batch = []
        parent_set = set()
        for idx in ids:
            parent = self.get_original_ancestor_item(idx)
            # one father at a time to ensure diveristy
            if parent in parent_set:
                continue
            # if the father has been selected too recently, skip it
            # step_num - self.father_last_touch[parent] == 0 indicates the father has just been selected last time
            if diverse_enable and step_num - self.father_last_touch[parent] < self.diverse_threshold:
                continue
            
            if father_only_round is not None:
                if father_only_round and idx != parent: # skip child nodes
                    continue
                if not father_only_round and idx == parent: # skip father nodes
                    continue

            parent_set.add(parent)
            batch.append(int(idx))
            self.select_num[idx] += 1
            self.father_select_num[parent] += 1
            if len(batch) == batch_size:
                break
            
        else:
            raise ValueError(f"Only {len(batch)} is collected")

        metrics = self._get_batch_statistics(batch, step_num)

        """add psi infomation"""

        # a fixed set for comparison between different methods
        seed = 42
        rng = np.random.default_rng(seed)
        random_parent_ids = rng.choice(list(range(self.original_datalength)), size=20, replace=False)
        fixed_thetas = []
        for p in random_parent_ids:
            children_thetas = []
            for j in self.get_children_items(p):
                children_thetas.append(thetas[j].item())
            fixed_thetas.append(children_thetas)
        metrics.update({
            "sampler/fixed_thetas": fixed_thetas,
        })

        # psi of current batch
        metrics.update({
            "sampler/thetas": thetas.tolist(), # [num_nodes, ]
            "sampler/father_thetas": thetas[:self.original_datalength].tolist(), # [num_fathers, ]
            "sampler/selected_thetas": thetas[batch].tolist(), # [batch_size, ]
        })

        # 返回batch和统计信息
        return batch, metrics
    
    def _get_batch_statistics(self, selected_items: List[int], step_num: int) -> dict:
        """
        获取batch的统计信息
        """
        parent_metrics = super()._get_batch_statistics(selected_items, step_num)

        time_not_selected = step_num - self.last_touch[selected_items] # [num_items, ]
        time_not_selected_mean = np.mean(time_not_selected)
        time_not_selected_std = np.std(time_not_selected)
        time_not_selected_max = np.max(time_not_selected)
        time_not_selected_min = np.min(time_not_selected)

        father_items = [self.get_original_ancestor_item(item) for item in selected_items]
        time_not_selected_father = step_num - self.father_last_touch[father_items] # [num_items, ]

        parent_metrics.update({
            "sampler/time_not_selected_mean": time_not_selected_mean,
            "sampler/time_not_selected_std": time_not_selected_std,
            "sampler/time_not_selected_max": time_not_selected_max,
            "sampler/time_not_selected_min": time_not_selected_min,
            "sampler/continuous_selected_num": np.sum(time_not_selected == 0),
            "sampler/father/time_not_selected_mean": np.mean(time_not_selected_father),
            "sampler/father/time_not_selected_std": np.std(time_not_selected_father),
            "sampler/father/time_not_selected_max": np.max(time_not_selected_father),
            "sampler/father/time_not_selected_min": np.min(time_not_selected_father),
            "sampler/father/continuous_selected_num": np.sum(time_not_selected_father == 0)
        })

        for i, threshold in enumerate([10, 20, 50, 100, 150, 200]):
            mask = time_not_selected > threshold
            parent_metrics.update({
                f"sampler/time_not_selected_gt_{threshold}_num": np.sum(mask),
                f"sampler/time_not_selected_gt_{threshold}_ratio": np.sum(mask) / len(time_not_selected),
            })
        
        for i, threshold in enumerate([10, 20, 50, 100, 150, 200]):
            mask = self.select_num > threshold
            father_mask = self.father_select_num > threshold
            parent_metrics.update({
                f"sampler/select_num/{i}selectnum_gt_{threshold}_num": np.sum(mask),
                f"sampler/father/select_num/{i}selectnum_gt_{threshold}_num": np.sum(father_mask),
            })
        
        total_select_num = np.sum(self.select_num)
        sort_select_num = np.sort(self.select_num)[::-1]
        sort_father_select_num = np.sort(self.father_select_num)[::-1]
        for ratio in [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]:
            largest_k = int(len(self.select_num) * ratio)
            largest_k_select_num = np.sum(sort_select_num[:largest_k])
            parent_metrics.update({
                f"sampler/coverage/top_{int(ratio * 100)}%_ratio": largest_k_select_num / total_select_num,
            })
            # for father
            largest_k = int(len(self.father_select_num) * ratio)
            largest_k_father_select_num = np.sum(sort_father_select_num[:largest_k])
            parent_metrics.update({
                f"sampler/father/coverage/top_{int(ratio * 100)}%_ratio": largest_k_father_select_num / total_select_num,
            })
        

        
        return parent_metrics

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["psi"] = self.psi
        state_dict["variance"] = self.variance
        state_dict["s"] = self.s
        state_dict["n"] = self.n
        state_dict["last_touch"] = self.last_touch
        state_dict["father_last_touch"] = self.father_last_touch
        state_dict["select_num"] = self.select_num
        state_dict["father_select_num"] = self.father_select_num
        return state_dict
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.psi = state_dict["psi"]
        self.variance = state_dict["variance"]
        self.s = state_dict["s"]
        self.n = state_dict["n"]
        self.last_touch = state_dict["last_touch"]
        self.father_last_touch = state_dict["father_last_touch"]
        self.select_num = state_dict["select_num"]
        self.father_select_num = state_dict["father_select_num"]