import re
import time
import asyncio
import math
import json
import torch
import random
import pandas as pd
import ray
import logging
import numpy as np
from uuid import uuid4
from alpha_seed.workers.xperf_rollout.component.query import Query
from mono_rl.utils.infer.client import GRMServingClient
from mono_rl.utils.dataset.dist_data_util import get_dist_data_manager
from typing import Dict, List
from langchain.schema import HumanMessage, SystemMessage
from bytedagi.model_io import InferenceRequest, ModelIO
from langchain.schema import HumanMessage
from alpha_seed.utils.reward_score.rm_utils import wait_remote_server_ready, check_nan, _extract_conversation, _process_history, decode_with_image_tag, replace_image_tag, get_query_imgs
from alpha_seed.utils.reward_score.utils import Verifier

# NOTE: GRM,QRM,ORM has same invalid score
RM_INVALID_SCORE = -100.0

logging.disable(logging.INFO)


def prepare_vlm_grm_input(
    tokenizer,
    row_dict,
    prompt_key,
    is_image=False,
    max_prompt_len=4096,
    max_resp_len=24576,
):
    """
        Args:
            prompts: [{"role": "user", "content": "..."}, ...]
            answer: 正确答案
            max_prompt_len: 输入最大长度
            max_resp_len: 响应最大长度
        """

    def pad_sequence(seq, target_len):
        seq = seq[:target_len]
        out = torch.full((target_len,), tokenizer.pad_token_id, dtype=torch.long)
        out[-len(seq):] = torch.tensor(seq, dtype=torch.long)
        return out

    system_prompt = row_dict.get('system_prompt', '')
    ability = row_dict.get("ability", "unknown")
    # build system prompt
    if ability in ["verifiable_function_call"]:
        sp = ''
        for prompt_turn in json.loads(system_prompt):
            if 'name' in prompt_turn and prompt_turn['name'] is not None and prompt_turn['name'] != '':
                sp += f'{prompt_turn["role"]} name={prompt_turn["name"]}\n{prompt_turn["content"]}\n'
            else:
                sp += f'{prompt_turn["role"]}\n{prompt_turn["content"]}\n'
    else:
        sp = row_dict.get('system_prompt', '')
        if row_dict.get('remark', '') and ('answer' in row_dict['remark'] or 'dypcot' in row_dict['remark']):
            if 'dypcot' in row_dict['remark']:
                sp += "\n\n请参考以下内容进行回答: \n\n" + row_dict['remark'][row_dict['remark'].find("dypcot:") + 7:]
            else:
                sp += "\n\n请参考以下内容进行回答: \n\n" + row_dict['remark'][row_dict['remark'].find("answer"):]

    prompts = row_dict[prompt_key]
    if is_image:  # [FIXME] Hack for text input
        prompts = [raw_ctx.replace("<image>", "<|image|>") for raw_ctx in prompts]

    # # GRM reference response
    grm_reference_resp = row_dict.get('rm_reference_response', None)
    if grm_reference_resp is None or check_nan(grm_reference_resp) or len(grm_reference_resp) == 0:
        grm_reference_resp = row_dict.get('ground_truth', None)  # Legency字段 (remove in future)
        if grm_reference_resp is None or check_nan(grm_reference_resp) or len(grm_reference_resp) == 0:
            grm_reference_resp = "No ground truth is found."

    assert not pd.isna(grm_reference_resp)

    if sp is None:
        sp = ""
    pre_context = ""
    if sp != "":
        pre_context = f"<对话场景设定>\n{sp}\n</对话场景设定>\n\n"
    else:
        pre_context = f"<对话场景设定>\n无\n</对话场景设定>\n\n"
    pre_context = tokenizer(pre_context)["input_ids"]
    history, final_question = _extract_conversation(prompts)
    final_question = final_question["content"]

    # context
    context = tokenizer(
        f"<当前user问题>\n{final_question}\n</当前user问题>\n\n"
        f"<回答1>\n{grm_reference_resp}\n</回答1>\n\n"
        f"<回答2>\n",
    )["input_ids"]
    post_context = tokenizer("\n</回答2>")["input_ids"]

    history_ids = _process_history(tokenizer,
                                   history,
                                   base_length=len(pre_context) + len(context) + len(post_context),
                                   max_total=max_prompt_len)

    return {
        "rm_pre_ids": pad_sequence(pre_context + history_ids + context, max_prompt_len),
        "rm_post_ids": torch.tensor(post_context)
    }


def prepare_grm_input(
    prompts,
    answer,
    tokenizer,
    max_prompt_len=4096,
    max_resp_len=24576,
):
    """
        Args:
            prompts: [{"role": "user", "content": "..."}, ...]
            answer: 正确答案
            max_prompt_len: 输入最大长度
            max_resp_len: 响应最大长度
        """

    def pad_sequence(seq, target_len):
        return torch.tensor([tokenizer.pad_token_id] * (target_len - len(seq)) + seq[:target_len])

    assert not pd.isna(answer)
    # system prompt
    system_prompt = next((prompt["content"] for prompt in prompts if prompt["role"] == "system"), "")
    pre_context = ""
    if system_prompt != "":
        pre_context = f"<场景设定>\n{system_prompt}\n</场景设定>\n\n"
    pre_context = tokenizer(pre_context)["input_ids"]
    # history
    history, final_question = _extract_conversation(prompts)
    # context
    context = tokenizer(
        f"<问题>\n{final_question}\n</问题>\n\n"
        f"<标准答案>\n{answer}\n</标准答案>\n\n"
        f"<回答>\n",
    )["input_ids"]
    post_context = tokenizer("\n</回答>")["input_ids"]
    # 历史对话
    history_ids = _process_history(tokenizer,
                                   history,
                                   base_length=len(pre_context) + len(context) + len(post_context),
                                   max_total=max_prompt_len)
    return {
        "rm_pre_ids": pad_sequence(pre_context + history_ids + context, max_prompt_len),
        "rm_post_ids": torch.tensor(post_context)
    }


class GrmVerifier(Verifier, reward_style="grm_service"):

    def __init__(self, config=None, tokenizer=None):
        super().__init__(config=config, tokenizer=tokenizer)

    def is_remote(self):
        return True

    def compute_score_client(self, data_uid, *args, **kwargs) -> float:
        cur_time = time.time()
        result = None
        if self.is_remote():
            result = self.get_remote_score(data_uid)
            if isinstance(result, ray.ObjectRef):
                result = ray.get(result)
        wait_time = time.time() - cur_time
        if result is None:
            # rsp, score, wait_cost, total_time , retry_cnt
            return "", 0, -1, -1, -1
        return result['grm_resp'], result['score'], wait_time, result['time_cost'], result['retry_cnt']

    def compute_score_remote(self, *args, **kwargs) -> float:
        remote_service = kwargs['remote_service']
        actor = random.choice(remote_service)
        return actor.call.remote(*args, **kwargs)

    def merge_score(self, scores_lst, merge_type="v1", **kwargs):
        rm_score, raw_score = scores_lst[0], scores_lst[1]
        if merge_type == 'v1':  # verifier基础上线性融合一定权重grm score
            if rm_score == RM_INVALID_SCORE:
                score = raw_score
            elif raw_score > 0:
                score = raw_score * 0.7 + rm_score * 0.3
            else:
                score = raw_score
        elif merge_type == 'v2':  # 主要用grm分数，verifier raw_score只做兜底
            if rm_score == RM_INVALID_SCORE:
                score = raw_score
            elif rm_score > 0:
                score = 0.7 + rm_score * 0.3
            else:
                score = -1
        else:
            raise NotImplementedError
        return score

    def merge_vlm_score(self, verifier_score, rm_score, merge_type="v1", **kwargs):
        verify_index_set = kwargs.get("verify_index_set", set())
        verify_fusion_rule_dict = kwargs.get("verify_fusion_rule_dict", {})
        ab_idx = kwargs.get("ab_idx", -1)
        is_valid_format = kwargs.get("is_valid_format", True)

        if ab_idx not in verify_index_set:
            verifier_score = -1.0
            if rm_score == RM_INVALID_SCORE:
                score = 0.5
            else:
                score = rm_score
            if not is_valid_format:
                score = -0.1
        else:
            rescaled_verifier_score = (verifier_score + 1) / 2  # Map verifier_score from [-1, 1] to [0, 1]
            if ab_idx in verify_fusion_rule_dict["code_switch"]:
                if rm_score == RM_INVALID_SCORE:
                    score = 0.5
                else:
                    score = rm_score * (0.1 + 0.9 * rescaled_verifier_score)
            elif ab_idx in verify_fusion_rule_dict["instrruler"]:
                if rm_score == RM_INVALID_SCORE:
                    score = rescaled_verifier_score
                else:
                    score = rm_score * rescaled_verifier_score
            elif ab_idx in verify_fusion_rule_dict["function_call"]:
                if rm_score == RM_INVALID_SCORE:
                    score = rescaled_verifier_score
                else:
                    score = min(rm_score + rescaled_verifier_score * 0.5, 1.0)
            else:
                score = rescaled_verifier_score
        score = (score * 2) - 1  # Rescale back from [0, 1] to [-1, 1]
        return score


def init_grm_server(config, **kwargs):
    rm_conf = config.reward_model
    tokenizer = kwargs.get("tokenizer", None)
    vlm_grm_clients = RemoteGRMServingClient.create_remote_clients(
        psm=rm_conf.rm_server.llm_serving_psm,
        idc=rm_conf.rm_server.llm_serving_idc,
        cluster=rm_conf.rm_server.llm_serving_cluster,
        model_name=rm_conf.rm_server.model_name,
        max_response_length=rm_conf.grm.max_response_length,
        inner_pool_size=rm_conf.rm_server.client_pool_size,
        retry=rm_conf.rm_server.max_retry,
        retry_interval=rm_conf.rm_server.retry_interval,
        timeout=rm_conf.rm_server.timeout,
        top_p=rm_conf.rm_server.top_p,
        tokenizer=tokenizer,
        config=config,
    )
    print("[GRM INFO] build grm server success in RemoteClient")
    return vlm_grm_clients


def swap_std_ans(text: str) -> str:
    STD_OPEN, STD_CLOSE = "<回答1>", "</回答1>"
    ANS_OPEN, ANS_CLOSE = "<回答2>", "</回答2>"

    std_m = re.search(rf"{STD_OPEN}(.*?){STD_CLOSE}", text, flags=re.DOTALL)
    ans_m = re.search(rf"{ANS_OPEN}(.*?){ANS_CLOSE}", text, flags=re.DOTALL)
    if not (std_m and ans_m):
        return text  # 任一缺失就原样返回，或你也可 raise

    std, ans = std_m.group(1), ans_m.group(1)

    # 2) 用占位符避免相互覆盖
    placeholder = f"__SWAP_PLACEHOLDER_{uuid4().hex}__"

    text = re.sub(r"(<回答1>)(.*?)(</回答1>)",
                  lambda m: m.group(1) + placeholder + m.group(3),
                  text,
                  count=1,
                  flags=re.DOTALL)

    text = re.sub(r"(<回答2>)(.*?)(</回答2>)", lambda m: m.group(1) + std + m.group(3), text, count=1, flags=re.DOTALL)

    text = text.replace(placeholder, ans)
    return text


class RemoteGRMServingClient(GRMServingClient):

    def __init__(self,
                 psm,
                 idc,
                 cluster="default",
                 model_name="",
                 tokenizer=None,
                 max_response_length=1024,
                 top_p=1.0,
                 timeout=120,
                 retry=0,
                 retry_interval=0,
                 pool_size=0,
                 random_rsp=False,
                 **kwargs):
        super().__init__(psm=psm,
                         idc=idc,
                         cluster=cluster,
                         model_name=model_name,
                         max_response_length=max_response_length,
                         tokenizer=tokenizer,
                         top_p=top_p,
                         timeout=timeout,
                         retry=retry,
                         retry_interval=retry_interval,
                         pool_size=pool_size,
                         random_rsp=False)
        wait_remote_server_ready(psm)
        self.config = kwargs.get("config", None)
        self.tokenizer = tokenizer
        self.timeout = timeout
        self.dist_data_manager = get_dist_data_manager()
        self.score_weight = self.config.reward_model.grm.get("score_weight", [1])
        self.grm_system_prompts = self.config.reward_model.grm.get("system_prompt_list", [''])
        self.response_postprocess_mode = self.config.reward_model.grm.get("response_postprocess_mode", [0])
        self.prepare_grm_prompt_mode = self.config.reward_model.grm.get("prepare_grm_prompt_mode", [0])
        self.score_parser_version = self.config.reward_model.grm.get("score_parser", ['v1'])
        self.empty_response_default_score = self.config.reward_model.grm.get("empty_response_default_score", [0])
        self.use_grm_reverse = self.config.reward_model.grm.get("use_grm_reverse", False)
        special_tokens = self.config.data.special_tokens
        self.think_begin = special_tokens.think_begin
        self.think_end = special_tokens.think_end
        self.assistant_begin = "assistant\n"
        self.img_tag = "<|image|>"

    def _preprocess_grm_data(self, rm_pre_ids, response_ids, rm_post_ids, rm_method, **kwargs):
        # for preprocess function, just always used by one data
        system_prompt = self.grm_system_prompts[rm_method]
        response_mode = self.response_postprocess_mode[rm_method]
        prompt_mode = self.prepare_grm_prompt_mode[rm_method]
        images_bytes_lst = get_query_imgs(self.dist_data_manager, **kwargs)

        grm_pre_text = decode_with_image_tag(self.tokenizer, rm_pre_ids)
        grm_post_text = decode_with_image_tag(self.tokenizer, rm_post_ids)
        if response_ids is not None:
            response_ids = [
                t for t in response_ids
                if t not in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
            ]
            response_text = decode_with_image_tag(self.tokenizer, response_ids, skip_special_tokens=False)
        else:
            input_ids = kwargs.get('input_ids')
            input_ids = [
                t for t in input_ids
                if t not in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
            ]
            input_ids_text = decode_with_image_tag(self.tokenizer, input_ids, skip_special_tokens=False)
            response_text = input_ids_text.rsplit(self.assistant_begin, 1)[-1]

        response_empty_flag = False
        if response_mode > 0:  # 0 = no postprocess
            response_text = self._postprocess_response(response_text, response_mode)
        if response_text == "":
            response_empty_flag = True

        if prompt_mode == 0:
            grm_prompt = grm_pre_text + response_text + grm_post_text
        elif prompt_mode == 1:
            grm_prompt = response_text
        else:
            raise NotImplementedError

        ref_grm_prompt_w_img = replace_image_tag(grm_prompt, images_bytes_lst, img_tag=self.img_tag)
        rev_grm_prompt, rev_grm_prompt_w_img = "", ""
        if self.use_grm_reverse and not response_empty_flag:
            rev_grm_prompt = swap_std_ans(grm_prompt)
            rev_grm_prompt_w_img = replace_image_tag(rev_grm_prompt, images_bytes_lst, img_tag=self.img_tag)

        if random.random() < 0.01:
            debug_info = f"[GRM REVERSE DEBUG] grm prompt: {grm_prompt}"
            if self.use_grm_reverse and not response_empty_flag:
                debug_info = debug_info + f"\n grm reverse prompt: {rev_grm_prompt}"
            print(debug_info)
        grm_pmp = [grm_prompt, rev_grm_prompt]
        return system_prompt, ref_grm_prompt_w_img, rev_grm_prompt_w_img, response_empty_flag, grm_pmp

    async def get_result(self, predict_output, rm_method):
        response_buffer = []
        async for msg in predict_output.astream_output:
            response_buffer.append(msg.content)
        response = "".join(response_buffer)

        parser_name = f"_parse_score_{self.score_parser_version[rm_method]}"
        parser = getattr(self, parser_name, self._parse_score_v1)
        if not callable(parser):
            raise AttributeError(f"无效解析器: {parser_name}")

        score = parser(response)
        if score == RM_INVALID_SCORE:
            return response, RM_INVALID_SCORE
        weight_score = score * self.score_weight[rm_method]
        return response, weight_score

    async def call(self, reward_model=None, response_ids=None, rm_method=0, **kwargs):
        cur_time = time.time()
        rm_pre_ids = reward_model.get("rm_pre_ids", None)
        rm_post_ids = reward_model.get("rm_post_ids", None)
        images_bytes_ref = reward_model.get("images_bytes_ref", None)

        system_prompt, ref_prompt, rev_prompt, response_empty_flag, grm_prompt = \
            await asyncio.to_thread(
                self._preprocess_grm_data,
                rm_pre_ids=rm_pre_ids,
                response_ids=response_ids,
                rm_post_ids=rm_post_ids,
                rm_method=rm_method,
                images_bytes_ref=images_bytes_ref,
                **kwargs
            )

        if response_empty_flag:
            default_score = self.empty_response_default_score[rm_method]
            return {
                "score": default_score,
                "status": "empty response",
                "grm_prompt": grm_prompt,
                "grm_resp": "",
                "retry_cnt": 0,
                "time_cost": time.time() - cur_time
            }

        total_retry_cnt = 0
        grm_resps = []
        rev_resp, rev_grm_score, retry_cnt = "", RM_INVALID_SCORE, 0

        tasks = [self._call(ref_prompt, system_prompt, rm_method)]
        if self.use_grm_reverse:
            tasks.append(self._call(rev_prompt, system_prompt, rm_method))
        results = await asyncio.gather(*tasks)
        (ref_resp, ref_grm_score, retry_cnt), *rest = results
        total_retry_cnt += retry_cnt

        if self.use_grm_reverse:
            (rev_resp, rev_grm_score, retry_cnt) = rest[0]
            total_retry_cnt += retry_cnt

        if ref_grm_score == RM_INVALID_SCORE and rev_grm_score == RM_INVALID_SCORE:
            final_score = RM_INVALID_SCORE
        elif ref_grm_score == RM_INVALID_SCORE:
            final_score = 1 - rev_grm_score
        elif rev_grm_score == RM_INVALID_SCORE:
            final_score = ref_grm_score
        else:
            final_score = (ref_grm_score + (1 - rev_grm_score)) / 2
        assert (final_score == RM_INVALID_SCORE) or (
            0.0 <= final_score <= 1.0), f"score={final_score} is invalied, grm_scores={[ref_grm_score, rev_grm_score]}"
        grm_resps.append(f"FORWARD:\n{ref_resp}\nREVERSE:\n{rev_resp}\n")

        if final_score == RM_INVALID_SCORE:
            # fail to get result
            return {
                "score": RM_INVALID_SCORE,
                "status": "fail",
                "grm_prompt": grm_prompt,
                "grm_resp": "",
                "retry_cnt": total_retry_cnt,
                "time_cost": time.time() - cur_time
            }
        else:
            return {
                "score": final_score,
                "status": "success",
                "grm_prompt": grm_prompt,
                "grm_resp": "\n".join(grm_resps),
                "retry_cnt": total_retry_cnt,
                "time_cost": time.time() - cur_time
            }

    async def _call(self, prompt, system_prompt=None, rm_method=0):
        for attempt in range(self.retry + 1):
            astream_task, predict_output = None, None
            try:
                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=prompt))
                predict_input = InferenceRequest(messages=messages)

                # get result
                predict_output = await self.clients[random.randrange(len(self.clients))].astream(predict_input)
                grm_resp, grm_score = await self.get_result(predict_output, rm_method)
                # astream_task = asyncio.create_task(self.get_result(predict_output, rm_method))
                # grm_resp, grm_score = await asyncio.wait_for(asyncio.shield(astream_task), self.timeout)

                if grm_score == RM_INVALID_SCORE:
                    # get invalid score, retry
                    print(f"[GRM Retry] Attempt {attempt+1}: 解析分数失败，\n====\n{grm_resp}\n====\n")
                    if attempt < self.retry:
                        continue
                    else:
                        break
                return grm_resp, grm_score, attempt

            except Exception as e:
                if astream_task:
                    astream_task.cancel()
                if predict_output and hasattr(predict_output, 'response_iterator'):
                    predict_output.response_iterator.cancel()
                if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                    print(f"[GRM Request Timeout] ... ")
                    return "", RM_INVALID_SCORE, attempt  # 超时不重试
                else:
                    print(f"[GRM Request] Attempt {attempt+1} failed: {e}")
                    if attempt < self.retry:
                        await asyncio.sleep(self.retry_interval)
        return "", RM_INVALID_SCORE, self.retry

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
        elif postprocess_mode == 3:  # compatibility mode
            if start_idx == -1 and end_idx == -1:  # NoThink Mode
                output_str = input_str
            elif start_idx != -1 and end_idx != -1:
                end_idx += len(self.think_end)
                output_str = input_str[:start_idx] + input_str[end_idx:]
            elif end_idx != -1:
                end_idx += len(self.think_end)
                output_str = input_str[end_idx:]
            else:
                output_str = ""
        else:
            raise NotImplementedError
        return output_str

    def _parse_score_v1(self, raw_response):
        # 预处理
        answer_part = raw_response.split(self.config.data.special_tokens.think_end)[-1]

        # 提取分数
        score_pattern = r"回答总得分(?:.*?=)?[^\d]*(\d+\.?\d*)"
        score_matches = re.findall(score_pattern, answer_part, re.DOTALL)
        if not score_matches:
            return RM_INVALID_SCORE

        final_score = float(score_matches[-1])
        if 0 <= final_score <= 1:
            return final_score
        else:
            return RM_INVALID_SCORE

    def _parse_score_v2(self, raw_response):
        # 预处理
        answer_part = raw_response.split(self.config.data.special_tokens.think_end)[-1]

        # 提取分数
        score_pattern = r"(?:回答|答案|Answer|answer)[ ]?(?:总|Total |total )(?:得分|Score|score)(?:.*?[=≈:：])?[^\d]*(\d+\.?\d*)[ ]?(?:分|point)"
        score_matches = re.findall(score_pattern, answer_part, re.DOTALL)
        if not score_matches:
            return RM_INVALID_SCORE

        final_score = float(score_matches[-1])
        if 0 <= final_score <= 1:
            return final_score
        else:
            return RM_INVALID_SCORE

    def _parse_score_v3(self, raw_response):

        def simple_parse(raw_response):
            predict_0 = predict_1 = -1
            if "回答1对比回答2胜出" in raw_response:
                predict_0 = 1
            if "回答1对比回答2落败" in raw_response:
                predict_1 = 1

            if predict_0 == 1 and predict_1 == -1:
                return 0
            elif predict_1 == 1 and predict_0 == -1:
                return 1
            else:
                return RM_INVALID_SCORE

        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        # 预处理
        res = raw_response.split(self.config.data.special_tokens.think_end)[-1]
        res = res.replace(" ", "")
        res = res.replace("Answer1", "回答1")
        res = res.replace("Answer2", "回答2")

        matches = re.findall(r"对比(.*?)结论", res, re.DOTALL)
        if not matches:
            return simple_parse(res)
        res_match = matches[-1].replace("**", "")
        pattern = r'回答1总得分：(?:.*?=)?[^\d]*(\d+\.?\d*).*?回答2总得分：(?:.*?=)?[^\d]*(\d+\.?\d*)'
        matches = re.findall(pattern, res_match, re.DOTALL)
        if not matches:
            return simple_parse(res)
        else:
            score1, score2 = matches[-1]
            score1 = float(score1)
            score2 = float(score2)
            return sigmoid(score2 - score1)
