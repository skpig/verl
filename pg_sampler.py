from ast import Set
from collections import defaultdict
from email.policy import default
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# ===================== Utilities =====================

def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

# ---- Polya-Gamma sampler backends (pypolyagamma -> polyagamma -> truncated series) ----
_pg_engine = None  # (kind, handle)

def _init_pg_engine(force: Optional[str] = None):
    """Initialize PG backend once. force in {"pypolyagamma","polyagamma","trunc"}."""
    global _pg_engine
    if _pg_engine is not None:
        return
    import os
    if force is None:
        force = os.environ.get("HIER_TS_PG_BACKEND", None)

    def set_engine(kind, handle):
        global _pg_engine
        _pg_engine = (kind, handle)

    if force == "trunc":
        set_engine("trunc", None)
        return
    if force in ("pypolyagamma", None):
        try:
            from pypolyagamma import PyPolyaGamma  # class + pgdraw
            set_engine("pypolyagamma", PyPolyaGamma())
            return
        except Exception:
            if force == "pypolyagamma":
                set_engine("trunc", None)
                return
    if force in ("polyagamma", None):
        try:
            from polyagamma import random_polyagamma  # functional API
            set_engine("polyagamma", random_polyagamma)
            return
        except Exception:
            if force == "polyagamma":
                set_engine("trunc", None)
                return
    set_engine("trunc", None)

def _sample_pg_truncated(b: float, c: float, trunc: int = 200, rng: Optional[np.random.Generator] = None) -> float:
    if b <= 0:
        return 0.0
    if rng is None:
        rng = np.random.default_rng()
    k = np.arange(1, trunc + 1, dtype=float) - 0.5
    denom = k * k + (c / (2.0 * math.pi)) ** 2
    gammas = rng.gamma(shape=b, scale=1.0, size=trunc)
    w = (1.0 / (2.0 * math.pi ** 2)) * np.sum(gammas / denom)
    return float(w)

def sample_pg(b: float, c: float, rng: Optional[np.random.Generator] = None, trunc: int = 200) -> float:
    _init_pg_engine()
    if b <= 0:
        return 0.0
    kind, eng = _pg_engine
    if kind == "pypolyagamma":
        pg = eng
        return float(pg.pgdraw(b, c))
    elif kind == "polyagamma":
        fn = eng
        return float(fn(b, c, random_state=rng)) if rng is not None else float(fn(b, c))
    else:
        return _sample_pg_truncated(b, c, trunc=trunc, rng=rng)

# ===================== Environment =====================

@dataclass
class TreeSpec:
    num_parents: int
    children_per_parent: List[int]

class TreeBanditEnv:
    """Two-layer tree Bernoulli bandit environment.
    Truth: psi_root_true[p] ~ N(mu_env, tau_env^2),
           psi_leaf_true[p,j] ~ N(psi_root_true[p], sigma_env^2),
           theta_true = sigmoid(psi_leaf_true).
    Arms are leaves; we index them by a flat arm_id = 0..num_arms-1.
    """
    def __init__(self, spec: TreeSpec, mu_env=0.0, tau_env=1.0, sigma_env=0.75, seed: int = 42):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.mu_env = float(mu_env)
        self.tau_env2 = float(tau_env) ** 2
        self.sigma_env2 = float(sigma_env) ** 2

        # Sample truth
        self.psi_root_true = self.rng.normal(loc=self.mu_env, scale=math.sqrt(self.tau_env2), size=spec.num_parents)
        self.psi_leaf_true: List[np.ndarray] = []
        self.theta_true: List[np.ndarray] = []
        for p in range(spec.num_parents):
            m = spec.children_per_parent[p]
            psi_children = self.rng.normal(loc=self.psi_root_true[p], scale=math.sqrt(self.sigma_env2), size=m)
            self.psi_leaf_true.append(psi_children)
            self.theta_true.append(sigmoid(psi_children))

        # Mapping between flat arm_id and (p,j)
        self.arm_of: List[Tuple[int, int]] = []     # arm_id -> (p,j)
        self.arm_index: Dict[Tuple[int,int], int] = {}
        aid = 0
        for p in range(spec.num_parents):
            for j in range(spec.children_per_parent[p]):
                self.arm_of.append((p, j))
                self.arm_index[(p, j)] = aid
                aid += 1
        self.num_arms = aid

        # Precompute sorted true thetas for top-K oracle regret
        self._true_thetas_flat = np.array([self.theta_true[p][j] for (p,j) in self.arm_of], dtype=float)
        self._true_sorted_desc = np.sort(self._true_thetas_flat)[::-1]

    # -------- Interaction logic (moved inside env) --------
    def pull_arm(self, arm_id: int) -> int:
        p, j = self.arm_of[arm_id]
        theta = float(self.theta_true[p][j])
        return int(self.rng.random() < theta)

    def pull_arms(self, arm_ids: List[int]) -> np.ndarray:
        return np.array([self.pull_arm(a) for a in arm_ids], dtype=int)

    def step(self, method: "MABMethod", t: int, k: int = 1) -> Tuple[List[int], np.ndarray]:
        """One interaction round controlled by env: ask method to select k arms, pull them, then notify method."""
        k = max(1, int(k))
        actions = method.select(self, t, k)
        # Ensure unique and valid
        actions = list(dict.fromkeys([int(a) for a in actions]))  # dedup, keep order
        if len(actions) > k:
            actions = actions[:k]
        rewards = self.pull_arms(actions)
        method.observe(self, actions, rewards)
        return actions, rewards

    # -------- Oracle helpers --------
    def topk_true_sum(self, k: int) -> float:
        k = max(1, min(int(k), self.num_arms))
        return float(np.sum(self._true_sorted_desc[:k]))

    def best_arm(self) -> Tuple[int, float]:
        idx = int(np.argmax(self._true_thetas_flat))
        return idx, float(self._true_thetas_flat[idx])

# ===================== MAB Methods (abstract + baselines + PG-TS) =====================

class MABMethod:
    name: str = "base"
    def reset(self, env: TreeBanditEnv, seed: int = 0):
        pass
    def select(self, env: TreeBanditEnv, t: int, k: int = 1) -> List[int]:
        raise NotImplementedError
    def observe(self, env: TreeBanditEnv, actions: List[int], rewards: np.ndarray):
        pass

class RandomPolicy(MABMethod):
    name = "random"
    def reset(self, env: TreeBanditEnv, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.num_arms = env.num_arms
    def select(self, env: TreeBanditEnv, t: int, k: int = 1) -> List[int]:
        k = max(1, min(int(k), self.num_arms))
        return list(self.rng.choice(self.num_arms, size=k, replace=False))

class EpsilonGreedy(MABMethod):
    name = "eps-greedy"
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = float(epsilon)
    def reset(self, env: TreeBanditEnv, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.num_arms = env.num_arms
        self.N = np.zeros(self.num_arms, dtype=int)
        self.S = np.zeros(self.num_arms, dtype=float)
    def select(self, env: TreeBanditEnv, t: int, k: int = 1) -> List[int]:
        k = max(1, min(int(k), self.num_arms))
        if self.rng.random() < self.epsilon:
            return list(self.rng.choice(self.num_arms, size=k, replace=False))
        # exploit
        means = np.divide(self.S, np.maximum(1, self.N))
        idx = np.argsort(means)[-k:][::-1]
        return list(idx)
    def observe(self, env: TreeBanditEnv, actions: List[int], rewards: np.ndarray):
        for a, r in zip(actions, rewards):
            self.N[a] += 1
            self.S[a] += float(r)

class UCB1(MABMethod):
    name = "ucb1"
    def reset(self, env: TreeBanditEnv, seed: int = 0):
        self.num_arms = env.num_arms
        self.N = np.zeros(self.num_arms, dtype=int)
        self.S = np.zeros(self.num_arms, dtype=float)
        self.total_pulls = 0
    def select(self, env: TreeBanditEnv, t: int, k: int = 1) -> List[int]:
        k = max(1, min(int(k), self.num_arms))
        # Ensure each arm once
        not_tried = np.where(self.N == 0)[0]
        if len(not_tried) > 0:
            take = list(not_tried[:k]) if len(not_tried) >= k else list(not_tried) + list(np.argsort(self.N)[- (k-len(not_tried)) :])
            return take
        means = self.S / self.N
        bonus = np.sqrt(2.0 * np.log(max(1, self.total_pulls)) / self.N)
        ucb = means + bonus
        idx = np.argsort(ucb)[-k:][::-1]
        return list(idx)
    def observe(self, env: TreeBanditEnv, actions: List[int], rewards: np.ndarray):
        for a, r in zip(actions, rewards):
            self.N[a] += 1
            self.S[a] += float(r)
            self.total_pulls += 1

class ThompsonBetaBernoulli(MABMethod):
    name = "ts-beta"
    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0):
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
    def reset(self, env: TreeBanditEnv, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.num_arms = env.num_arms
        self.alpha = np.full(self.num_arms, self.alpha0)
        self.beta = np.full(self.num_arms, self.beta0)
    def select(self, env: TreeBanditEnv, t: int, k: int = 1) -> List[int]:
        k = max(1, min(int(k), self.num_arms))
        samples = self.rng.beta(self.alpha, self.beta)
        idx = np.argsort(samples)[-k:][::-1]
        return list(idx)
    def observe(self, env: TreeBanditEnv, actions: List[int], rewards: np.ndarray):
        for a, r in zip(actions, rewards):
            if r == 1:
                self.alpha[a] += 1.0
            else:
                self.beta[a] += 1.0

# ----- Hierarchical PG-augmented Thompson Sampling -----
class HierarchicalPGTS(MABMethod):
    name = "pg-hts"
    def __init__(self, mu0=0.0, tau0=3.0, sigma=1.0, gibbs_sweeps: int = 1, pg_trunc: int = 200, seed: int = 0):
        self.mu0 = float(mu0)
        self.tau0 = float(tau0)
        self.sigma = float(sigma)
        self.gibbs_sweeps = int(max(1, gibbs_sweeps))
        self.pg_trunc = int(pg_trunc)
        self._seed = int(seed)
    def reset(self, env: TreeBanditEnv, seed: int = 0):
        rng_seed = self._seed if seed == 0 else seed
        self.rng = np.random.default_rng(rng_seed)
        self.spec = env.spec
        self.sigma2 = self.sigma ** 2
        self.tau0_2 = self.tau0 ** 2
        # State
        self.psi_root = self.rng.normal(loc=self.mu0, scale=self.tau0, size=self.spec.num_parents)
        self.psi_leaf: List[np.ndarray] = []
        for p in range(self.spec.num_parents):
            m = self.spec.children_per_parent[p]
            self.psi_leaf.append(self.rng.normal(loc=self.psi_root[p], scale=self.sigma, size=m))
        # Online counts per leaf
        self.s = [np.zeros(self.spec.children_per_parent[p], dtype=int) for p in range(self.spec.num_parents)]
        self.n = [np.zeros(self.spec.children_per_parent[p], dtype=int) for p in range(self.spec.num_parents)]

    def _gibbs_one_sweep(self):
        inv_sigma2 = 1.0 / self.sigma2
        inv_tau02 = 1.0 / self.tau0_2

        # Leaves
        for p in range(self.spec.num_parents):
            for j in range(self.spec.children_per_parent[p]):
                n_ = int(self.n[p][j])
                s_ = int(self.s[p][j])
                kappa = s_ - n_ / 2.0
                psi_cur = float(self.psi_leaf[p][j])
                omega = sample_pg(b=n_, c=psi_cur, rng=self.rng, trunc=self.pg_trunc)
                V = 1.0 / (inv_sigma2 + omega)
                m = V * (inv_sigma2 * self.psi_root[p] + kappa)
                self.psi_leaf[p][j] = self.rng.normal(loc=m, scale=math.sqrt(V))

        # Roots
        for p in range(self.spec.num_parents):
            Kp = self.spec.children_per_parent[p]
            sum_children = float(np.sum(self.psi_leaf[p]))
            V0 = 1.0 / (inv_tau02 + Kp * (1.0 / self.sigma2))
            m0 = V0 * (inv_tau02 * self.mu0 + (1.0 / self.sigma2) * sum_children)
            self.psi_root[p] = self.rng.normal(loc=m0, scale=math.sqrt(V0))

    def _gibbs_one_sweep_selected(self, p2js):
        inv_sigma2 = 1.0 / self.sigma2
        inv_tau02 = 1.0 / self.tau0_2

        # Leaves
        for p, js in p2js.items():
            for j in js:
                n_ = int(self.n[p][j])
                s_ = int(self.s[p][j])
                kappa = s_ - n_ / 2.0
                psi_cur = float(self.psi_leaf[p][j])
                omega = sample_pg(b=n_, c=psi_cur, rng=self.rng, trunc=self.pg_trunc)
                V = 1.0 / (inv_sigma2 + omega)
                m = V * (inv_sigma2 * self.psi_root[p] + kappa)
                self.psi_leaf[p][j] = self.rng.normal(loc=m, scale=math.sqrt(V))

        # Roots
        for p in p2js.keys():
            Kp = self.spec.children_per_parent[p]
            sum_children = float(np.sum(self.psi_leaf[p]))
            V0 = 1.0 / (inv_tau02 + Kp * (1.0 / self.sigma2))
            m0 = V0 * (inv_tau02 * self.mu0 + (1.0 / self.sigma2) * sum_children)
            self.psi_root[p] = self.rng.normal(loc=m0, scale=math.sqrt(V0))

    def select(self, env: TreeBanditEnv, t: int, k: int = 1) -> List[int]:
        # compute theta tilde for all leaves and pick top-k
        thetas = []
        for p in range(self.spec.num_parents):
            thetas_p = sigmoid(self.psi_leaf[p])
            for j in range(self.spec.children_per_parent[p]):
                aid = env.arm_index[(p, j)]
                thetas.append((aid, float(thetas_p[j])))
        thetas.sort(key=lambda x: x[1], reverse=True)
        k = max(1, min(int(k), env.num_arms))
        return [aid for (aid, _) in thetas[:k]]
    def observe(self, env: TreeBanditEnv, actions: List[int], rewards: np.ndarray):
        p2j = defaultdict(set)
        for a, r in zip(actions, rewards):
            p, j = env.arm_of[a]
            self.n[p][j] += 1
            if int(r) == 1:
                self.s[p][j] += 1

            p2j[p].add(j)

        self._gibbs_one_sweep_selected(p2j)

# ===================== Evaluation / Main Loop =====================

@dataclass
class EvalLog:
    cum_reward: np.ndarray
    cum_regret: np.ndarray
    avg_reward_per_pull: np.ndarray

def run_method(env: TreeBanditEnv, method: MABMethod, rounds: int, pulls_per_round: int, seed: int = 0) -> EvalLog:
    method.reset(env, seed=seed)
    T = int(rounds)
    K = max(1, int(pulls_per_round))
    cum_reward = 0.0
    cum_regret = 0.0
    cr_traj = np.zeros(T)
    cg_traj = np.zeros(T)
    avg_traj = np.zeros(T)
    oracle_k_sum = env.topk_true_sum(K)
    for t in range(1, T+1):
        actions, rewards = env.step(method, t, k=K)
        sum_r = float(np.sum(rewards))
        cum_reward += sum_r
        # regret vs top-K oracle (stationary env): K * sum of best-K thetas per round
        cum_regret += (oracle_k_sum - sum_r)
        idx = t-1
        cr_traj[idx] = cum_reward
        cg_traj[idx] = cum_regret
        avg_traj[idx] = cum_reward / (t * K)
    return EvalLog(cum_reward=cr_traj, cum_regret=cg_traj, avg_reward_per_pull=avg_traj)

# ===================== Comparison / Plotting =====================

def compare_methods(seed: int = 2025,
                    parents: int = 3,
                    children_minmax: Tuple[int,int] = (2,5),
                    rounds: int = 1000,
                    pulls_per_round: int = 1,
                    mu_env=0.0, tau_env=1.0, sigma_env=0.75,
                    methods: Optional[List[MABMethod]] = None) -> Dict[str, EvalLog]:
    rng = np.random.default_rng(seed)
    children_per_parent = [int(rng.integers(children_minmax[0], children_minmax[1]+1))
                           for _ in range(parents)]
    spec = TreeSpec(num_parents=parents, children_per_parent=children_per_parent)
    env = TreeBanditEnv(spec, mu_env=mu_env, tau_env=tau_env, sigma_env=sigma_env, seed=seed+1)

    if methods is None:
        methods = [
            RandomPolicy(),
            EpsilonGreedy(0.1),
            UCB1(),
            ThompsonBetaBernoulli(1.0, 1.0),
            HierarchicalPGTS(mu0=0.0, tau0=3.0, sigma=1.0, gibbs_sweeps=2, pg_trunc=150, seed=seed+3),
        ]

    results: Dict[str, EvalLog] = {}
    for i, m in enumerate(methods):
        name = getattr(m, 'name', f'method_{i}')
        print(f"Running {name} ...")
        results[name] = run_method(env, m, rounds=rounds, pulls_per_round=pulls_per_round, seed=seed+10+i)
    return results


def plot_comparison(results: Dict[str, EvalLog]):
    import matplotlib.pyplot as plt
    T = None
    # Cumulative Regret
    plt.figure()
    for name, log in results.items():
        if T is None:
            T = len(log.cum_regret)
        plt.plot(np.arange(1, len(log.cum_regret)+1), log.cum_regret, label=name)
    plt.title("Cumulative Regret (lower is better)")
    plt.xlabel("round")
    plt.ylabel("cumulative regret")
    plt.legend()
    plt.tight_layout()

    # Average reward per pull
    plt.figure()
    for name, log in results.items():
        plt.plot(np.arange(1, len(log.avg_reward_per_pull)+1), log.avg_reward_per_pull, label=name)
    plt.title("Average Reward per Pull (higher is better)")
    plt.xlabel("round")
    plt.ylabel("avg reward per pull")
    plt.legend()
    plt.tight_layout()

    plt.show()

# ===================== Entry =====================
if __name__ == "__main__":
    results = compare_methods(
        seed=205,
        parents=1700,
        children_minmax=(10,50),
        rounds=1000,
        pulls_per_round=409,  # set K here
        mu_env=0.0, tau_env=100.0, sigma_env=1,
        methods=None,  # use defaults
    )
    plot_comparison(results)
