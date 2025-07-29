import ray
import torch
import random
import traceback
import asyncio
from typing import Dict, List
from transformers import AutoTokenizer
import re
from functools import wraps
from alpha_seed.utils.reward_score.llm_remote_client import BaseLLMRemoteClient
from alpha_seed.utils.ckpt import download_minimal_required_files

GRM_INVALID_SCORE = -100.0
MAX_RETRIES = 3
REQUEST_DELAY = 1.0


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):

    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):  # 包含首次尝试
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries:
                        print(f"[GRM Retry] {func.__name__} 第 {attempt+1} 次重试，错误：{str(e)}")
                        await asyncio.sleep(delay)
            print(f"[GRM] 获取最终结果失败")
            return "", GRM_INVALID_SCORE  # 所有重试失败后返回默认值

        return wrapper

    return decorator


@ray.remote
class GRMService:

    def __init__(self, config, tokenizer_path):
        self.config = config
        local_path = download_minimal_required_files(tokenizer_path, from_scratch=False, rank=0, world_size=1)
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)

        self.results = []
        self.score_weight = config.reward_model.grm.get("score_weight", [1])
        self.rm_num = len(self.score_weight)
        for _ in range(self.rm_num):
            self.results.append({})

        self.actor_pool = self._init_llm_remote_client()
        special_tokens = config.data.special_tokens

        self.grm_system_prompts = config.reward_model.grm.get("system_prompt_list", [''])
        self.response_postprocess_mode = config.reward_model.grm.get("response_postprocess_mode", [0])
        self.prepare_grm_prompt_mode = config.reward_model.grm.get("prepare_grm_prompt_mode", [0])
        self.score_parser_version = config.reward_model.grm.get("score_parser", ['v1'])
        self.empty_response_default_score = config.reward_model.grm.get("empty_response_default_score", [0])
        assert len(
            self.grm_system_prompts
        ) == self.rm_num, f"rm_num is {self.rm_num}, but length of system_prompt_list is {len(self.grm_system_prompts)}"
        assert len(
            self.response_postprocess_mode
        ) == self.rm_num, f"rm_num is {self.rm_num}, but length of response_postprocess_mode is {len(self.response_postprocess_mode)}"
        assert len(
            self.prepare_grm_prompt_mode
        ) == self.rm_num, f"rm_num is {self.rm_num}, but length of prepare_grm_prompt_mode is {len(self.prepare_grm_prompt_mode)}"
        assert len(
            self.score_parser_version
        ) == self.rm_num, f"rm_num is {self.rm_num}, but length of score_parser is {len(self.score_parser_version)}"
        assert len(
            self.empty_response_default_score
        ) == self.rm_num, f"rm_num is {self.rm_num}, but length of empty_response_default_score is {len(self.empty_response_default_score)}"

        self.think_begin = special_tokens.think_begin
        self.think_end = special_tokens.think_end
        self.assistant_begin = "assistant\n"

    def _init_llm_remote_client(self):
        actor_config = self.config.reward_model.grm.server
        return [
            BaseLLMRemoteClient.options(name=f"grm_remote_client.{idx}").remote(
                psm=actor_config.llm_serving_psm,
                idc=actor_config.llm_serving_idc,
                cluster=actor_config.llm_serving_cluster,
                model_name=actor_config.model_name,
                max_response_length=self.config.reward_model.grm.max_response_length,
            ) for idx in range(actor_config.ray_actor_pool_size)
        ]

    def clear(self):
        # for some cases, the results won't be claimed. So we need to clear the results.
        self.results = []
        for _ in range(self.rm_num):
            self.results.append({})

    def get_num_pending_outputs(self):
        """Return the number of outputs, whose result is not claimed"""
        return sum([len(res) for res in self.results])

    async def add_requests(self, req_id: str, response_ids: List[int], grm_pre_ids: List[int],
                           grm_post_ids: List[int]) -> None:
        """调用grm服务请求

        Args:
            req_id: 唯一请求标识符
            response_ids: 需要评分的响应文本的token id列表
            grm_pre_ids: grm正向前缀token id
            grm_post_ids: grm正向后缀token id
        """

        pre_text = self.tokenizer.decode(grm_pre_ids, skip_special_tokens=True)
        post_text = self.tokenizer.decode(grm_post_ids, skip_special_tokens=True)

        response_ids = [
            t for t in response_ids
            if t not in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        ]

        # TODO: Support VLM-GRM later, temporarily remove VLM placeholder -100
        response_ids = [t for t in response_ids if t != -100]

        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        response_text = response_text.split(self.assistant_begin, 1)[-1]

        for ind in range(self.rm_num):
            grm_sp = self.grm_system_prompts[ind]
            response_mode = self.response_postprocess_mode[ind]
            prompt_mode = self.prepare_grm_prompt_mode[ind]

            prompt, response_empty_flag = await asyncio.to_thread(self._preprocess_grm, pre_text, response_text,
                                                                  post_text, prompt_mode, response_mode)

            # grm forward
            actor = random.choice(self.actor_pool)
            if response_empty_flag:
                self.results[ind][f'{req_id}'] = float(self.empty_response_default_score[ind])
            else:
                self.results[ind][f'{req_id}'] = actor.generate.options(enable_task_events=False).remote(
                    prompt, grm_sp, self.config.reward_model.grm.server.max_retry,
                    self.config.reward_model.grm.server.retry_interval)

    async def get_results(self, req_id: str) -> [str, float]:
        try:
            grm_resps = []
            grm_scores = []
            for ind in range(self.rm_num):
                resp, s = await self._process_single_result(f'{req_id}', self.results[ind],
                                                            self.score_parser_version[ind])
                grm_resps.append(resp)
                grm_scores.append(s)

            response_str = "\n---\n".join(grm_resps)
            response_str += "\n---\n" + ",".join([str(s) for s in grm_scores])

            fscore = 0
            for ind, s in enumerate(grm_scores):
                if s == GRM_INVALID_SCORE:
                    fscore = GRM_INVALID_SCORE
                    break
                else:
                    fscore += s * self.score_weight[ind]
            return response_str, fscore
        except Exception as e:
            print(f"[GRM] Fail: {traceback.format_exc()}")
            return "", GRM_INVALID_SCORE

    @retry_on_failure(max_retries=MAX_RETRIES, delay=REQUEST_DELAY)
    async def _process_single_result(self, req_id: str, result_map: Dict, score_parser_version: str) -> [str, float]:
        try:
            if req_id not in result_map:
                raise ValueError(f"未找到对应的请求ID: {req_id}")

            future = result_map.get(req_id)
            if isinstance(future, float):
                return "空结果", future

            result = await future
            if not isinstance(result, dict):
                raise TypeError("返回结果格式异常")

            raw_response = result.get("response", "")
            if not raw_response:
                raise ValueError("response请求失败")

            # 动态获取parser
            parser_name = f"_parse_score_{score_parser_version}"
            parser = getattr(self, parser_name, self._parse_score_v1)
            if not callable(parser):
                raise AttributeError(f"无效解析器: {parser_name}")
            return raw_response, parser(raw_response)

        except (asyncio.TimeoutError, KeyError, AttributeError) as e:
            raise

    def _postprocess_response(self, input_str, postprocess_mode):
        start_idx = input_str.find(self.think_begin)
        end_idx = input_str.find(self.think_end, start_idx)

        if postprocess_mode == 1:  # remove thinking
            if start_idx != -1 and end_idx != -1:
                end_idx += len(self.think_end)
                output_str = input_str[:start_idx] + input_str[end_idx:]
            elif end_idx != -1:
                end_idx += len(self.think_end)
                output_str = input_str[end_idx:]
            else:
                output_str = ""
        elif postprocess_mode == 2:  # remove response
            if start_idx != -1 and end_idx != -1:
                start_idx += len(self.think_begin)
                output_str = input_str[start_idx:end_idx]
            elif start_idx == -1 and end_idx != -1:
                output_str = input_str[:end_idx]
            elif start_idx != -1 and end_idx == -1:
                start_idx += len(self.think_begin)
                output_str = input_str[start_idx:]
            else:
                output_str = input_str
        else:
            raise NotImplementedError
        return output_str

    def _preprocess_grm(self, pre_text, response_text, post_text, prompt_mode, response_mode):
        response_empty_flag = False
        if response_mode > 0:  # 0 = no postprocess
            response_text = self._postprocess_response(response_text, response_mode)
        if response_text == "":
            response_empty_flag = True

        if prompt_mode == 0:
            grm_prompt = pre_text + response_text + post_text
        elif prompt_mode == 1:
            grm_prompt = response_text
        else:
            raise NotImplementedError

        if random.random() < 0.01:
            print(f"[grm debug] service request, prompt_mode: {prompt_mode}, "
                  f"response_mode: {response_mode}, {repr(grm_prompt)}")

        return grm_prompt, response_empty_flag

    def _parse_score_v1(self, raw_response):
        # 预处理
        answer_part = raw_response.split(self.config.data.special_tokens.think_end)[-1]

        # 提取分数
        score_pattern = r"回答总得分(?:.*?=)?[^\d]*(\d+\.?\d*)"
        score_matches = re.findall(score_pattern, answer_part, re.DOTALL)
        if not score_matches:
            return GRM_INVALID_SCORE

        final_score = float(score_matches[-1])
        if 0 <= final_score <= 1:
            return final_score
        else:
            return GRM_INVALID_SCORE

    def _parse_score_v2(self, raw_response):
        # 预处理
        answer_part = raw_response.split(self.config.data.special_tokens.think_end)[-1]

        # 提取分数
        score_pattern = r"(?:回答|答案|Answer|answer)[ ]?(?:总|Total |total )(?:得分|Score|score)(?:.*?[=≈:：])?[^\d]*(\d+\.?\d*)[ ]?(?:分|point)"
        score_matches = re.findall(score_pattern, answer_part, re.DOTALL)
        if not score_matches:
            return GRM_INVALID_SCORE

        final_score = float(score_matches[-1])
        if 0 <= final_score <= 1:
            return final_score
        else:
            return GRM_INVALID_SCORE


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = OmegaConf.load("tasks/config/ppo_trainer.yaml")
    conf.reward_model.grm.server.llm_serving_psm = "data.seed.rl_human_eval"
    conf.reward_model.grm.server.llm_serving_idc = "lf"
    conf.reward_model.grm.server.llm_serving_cluster = (
        "Seed-14B-SFT-P8.0.0_D7.3.0_T17000B-SFT33.0.0-C3.0.1-grm.w8a8tp1")
    conf.reward_model.grm.server.model_name = ("Seed-14B-SFT-P8.0.0_D7.3.0_T17000B-SFT33.0.0-C3.0.1-grm.0420k8.w8a8tp1")
    conf.reward_model.grm.score_parser = "v1"
    conf.reward_model.grm.server.client_pool_size = 16
    conf.reward_model.grm.server.ray_actor_pool_size = 16
    conf.reward_model.grm.max_prompt_length = 20480
    conf.reward_model.grm.max_response_length = 4096
    tokenizer_path = "bbpe155k-v6.4.3-ml.pret_v3.3_20250210"

    grm_service = GRMService.remote(conf, tokenizer_path)

    ray.get(
        grm_service.add_requests.remote(req_id=123,
                                        response_ids=tokenizer("你好，你是一只猫")["input_ids"],
                                        grm_pre_ids=torch.tensor(tokenizer("pre")["input_ids"]),
                                        grm_post_ids=torch.tensor(tokenizer("post")["input_ids"])))

    response, score = ray.get(grm_service.get_results.remote(123))
    print(response)
