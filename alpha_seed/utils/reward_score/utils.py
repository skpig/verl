import os
import hashlib
from typing import Any, Dict, Type
import ray
import numpy as np
from alpha_seed.utils.reward_score import response_post_proc

from alpha_seed.workers.agents import load_external_module


def post_process_solution_str(config, solution_str, reward_style, eos_token):
    if solution_str.endswith(eos_token):
        solution_str = solution_str[:-len(eos_token)]
    solution_str = solution_str.split(eos_token, 1)[-1]
    if reward_style == "code-sandbox" and config.reward_model.use_last_response == 'lastcodeblock':
        solution_str_post_proc = response_post_proc.last_codeblock_postprocess(
            solution_str,
            codeblock_seps=config.reward_model.last_response_sep,
            last_response_strict=config.reward_model.last_response_strict)
    else:
        solution_str_post_proc = solution_str
    return solution_str_post_proc


class Verifier:

    _registry: Dict[str, Type["Verifier"]] = {}
    _named_verifiers = {}

    def __init__(self, config=None, tokenizer=None) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self._handle = None

    def handler(self, req_id):
        if self._handle is None:
            self._handle = [
                ray.get_actor(f'remote_client_{idx}') for idx in range(self.config.reward_model.num_remote_client)
            ]
        assert self.is_remote()
        idx = int(hashlib.md5(req_id.encode()).hexdigest(), 16) % len(self._handle)
        return self._handle[idx]

    # 子类定义时自动调用，完成注册
    def __init_subclass__(cls, reward_style: str, **kwargs):
        super().__init_subclass__(**kwargs)
        # 将子类注册到映射表
        if reward_style in cls._registry:
            raise ValueError(f"reward_style '{reward_style}' 已被注册，避免重复")
        cls._registry[reward_style] = cls
        # 子类保存自身对应的reward_style
        cls.reward_style = reward_style

    # 静态方法：根据reward_style获取对应的子类
    @staticmethod
    def get_verifier(reward_style: str, config=None, tokenizer=None) -> Type["Verifier"]:
        external_module = load_external_module(package_name=reward_style,
                                               external_lib=config.reward_model.external_lib,
                                               external_path=os.environ.get('EXTERNAL_REWARD_FN_PATH', None))
        if reward_style in Verifier._named_verifiers:
            return Verifier._named_verifiers[reward_style]

        if external_module is not None:
            verifier = ExternalVerifier(config=config,
                                        tokenizer=tokenizer,
                                        reward_style=reward_style,
                                        external_module=external_module)
        elif reward_style not in Verifier._registry:
            raise ValueError(
                f"cannot find verifier for reward_style '{reward_style}', please check the reward_style is correct, or the dependency is installed"
            )
        else:
            verifier = Verifier._registry[reward_style](config=config, tokenizer=tokenizer)
        Verifier._named_verifiers[reward_style] = verifier
        return verifier

    def is_remote(self):
        return False

    def preprocess(self, *args, **kwargs):
        ground_truth = kwargs['ground_truth']
        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
            input_ids = np.array(input_ids)
            input_ids = input_ids[input_ids >= 0].tolist()
            solution_str = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            solution_str = solution_str.split("assistant\n")[-1]
            solution_str_post_proc = post_process_solution_str(self.config,
                                                               solution_str,
                                                               self.reward_style,
                                                               eos_token=self.tokenizer.eos_token)
        else:
            solution_str_post_proc = kwargs['solution_str']
        return solution_str_post_proc, ground_truth

    def get_remote_score(self, data_uid):
        # TODO: 统一在reward_manager调用ray.get
        return ray.get(self.handler(data_uid).get_results.remote(data_uid, self.reward_style))

    def add_requests(self, req_id, *args, **kwargs):
        self.handler(req_id).add_requests.remote(req_id, *args, **kwargs)

    def compute_score_client(self, *args, **kwargs) -> float:
        score = None
        if self.is_remote():
            data_uid = kwargs.get('data_uid', None)
            assert data_uid is not None, "data_uid is required for remote compute_score_client"
            score = self.get_remote_score(data_uid)

        if score is None:
            if self.is_remote():
                return self.compute_score_remote(*args, **kwargs)
            score = self.compute_score(*args, **kwargs)
        return score

    @staticmethod
    def compute_score(*args, **kwargs) -> float:
        pass

    def compute_score_remote(self, *args, **kwargs):
        return self.compute_score(*self.preprocess(*args, **kwargs))


class ExternalVerifier(Verifier, reward_style="external"):

    def __init__(self, config=None, tokenizer=None, reward_style="external", external_module=None) -> None:
        super().__init__(config=config, tokenizer=tokenizer)
        if reward_style is not None:
            self.reward_style = reward_style
        else:
            reward_style = self.reward_style
        compute_score_fn = getattr(external_module, 'compute_score')
        assert hasattr(compute_score_fn, '__call__'), f"{reward_style}.compute_score is not a function"
        self.compute_score_fn = compute_score_fn

    def compute_score_client(self, *args, **kwargs) -> float:
        return self.compute_score_fn(*args, **kwargs)
