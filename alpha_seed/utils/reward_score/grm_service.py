import re
import time
import asyncio
import math
import torch
import random
import traceback
import pandas as pd
import ray
import logging
from alpha_seed.workers.xperf_rollout.component.query import Query
from mono_rl.utils.infer.client import GRMServingClient
from mono_rl.utils.dataset.dist_data_util import get_dist_data_manager
from typing import Dict, List
from langchain.schema import HumanMessage, SystemMessage
from bytedagi.model_io import InferenceRequest, ModelIO
from langchain.schema import HumanMessage
from bytedance import servicediscovery
from servicediscovery import ServiceDiscoveryError
from alpha_seed.utils.reward_score.utils import Verifier

# NOTE: GRM,QRM,ORM has same invalid score
RM_INVALID_SCORE = -100.0
MAX_RETRIES = 3
REQUEST_DELAY = 1.0


class GrmVerifier(Verifier, reward_style="grm"):

    def __init__(self, config=None, tokenizer=None):
        super().__init__(config=config, tokenizer=tokenizer)

    def is_remote(self):
        return True

    def get_remote_score(self, data_uid):
        score_dict = super().get_remote_score(data_uid)
        return score_dict['grm_prompt'], score_dict['grm_resp'], score_dict['score']

    def compute_score_client(self, data_uid, *args, **kwargs) -> float:
        result = None
        if self.is_remote():
            result = self.get_remote_score(data_uid)
        if result is None:
            return None, None, None
        return result

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


def wait_remote_server_ready(psm: str):
    fail_time = 0
    while fail_time < 20:
        try:
            sd_result = servicediscovery.get_one(psm, address_family="dual-stack")
            print(f"[GRM SERVER INFO] psm {psm} ready !!!")
            return
        except ServiceDiscoveryError:
            print(f"[GRM SERVER WARNING] waitting psm {psm} ready, cnt={fail_time}, begin sleep 60s")
            time.sleep(60)
            fail_time += 1
    raise ServiceDiscoveryError(f"psm {psm} not ready after 30 times retry, please check remote rm log")


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


def _extract_conversation(prompts):
    conversation = [p["content"] for p in prompts if p["role"] in ("user", "assistant")]
    assert len(conversation) % 2 == 1, f"invalid conversation: {prompts}"
    return conversation[:-1], conversation[-1]


def _process_history(tokenizer, history, base_length, max_total):
    history_tag_ids = tokenizer("<对话历史>\n")["input_ids"]
    history_end_tag_ids = tokenizer("\n</对话历史>\n\n")["input_ids"]
    available = (max_total - base_length - len(history_tag_ids) - len(history_end_tag_ids))
    buffer = []
    current_len = 0

    # 逆向处理历史对话
    for message in reversed(history):
        role = message["role"]
        content = message["content"]
        new_content = f"{role}\n{content}\n"
        new_tokens = tokenizer(new_content)["input_ids"]

        if current_len + len(new_tokens) > available:
            break

        buffer.append(new_tokens)
        current_len += len(new_tokens)

    if buffer:
        history_tokens = sum(reversed(buffer), [])
        return (history_tag_ids + history_tokens + history_end_tag_ids)
    else:
        return []


def init_grm_server(config, **kwargs):
    rm_conf = config.reward_model
    tokenizer = kwargs.get("tokenizer", None)
    vlm_grm_clients = RemoteGRMServingClient.create_remote_clients(
        psm=rm_conf.rm_server.llm_serving_psm,
        idc=rm_conf.rm_server.llm_serving_idc,
        cluster=rm_conf.rm_server.llm_serving_cluster,
        model_name=rm_conf.rm_server.model_name,
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


def decode_with_image_tag(tokenizer, ids, skip_special_tokens=True, image_tag="<image>"):
    seq = torch.as_tensor(ids, dtype=torch.long, device="cpu").tolist()
    if -100 not in seq:  # pure text
        return tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)
    out, i, n = [], 0, len(seq)
    while i < n:
        if seq[i] == -100:
            while i < n and seq[i] == -100:
                i += 1
            out.append(image_tag)
        else:
            j = i
            while j < n and seq[j] != -100:
                j += 1
            out.append(tokenizer.decode(seq[i:j], skip_special_tokens=skip_special_tokens))
            i = j

    return "".join(out)


def replace_image_tag(prompt, images_bytes_lst=None, img_tag="<image>"):
    pattern = rf"({re.escape(img_tag)})"
    prompt_chunks = re.split(pattern, prompt)
    image_tag_count = sum(1 for chunk in prompt_chunks if chunk == img_tag)
    assert image_tag_count == len(images_bytes_lst), (
        f"Mismatch between image tags ({image_tag_count}) and provided images ({len(images_bytes_lst)})")

    content = []
    image_idx = 0
    for chunk in prompt_chunks:
        if len(chunk) == 0:
            continue
        if chunk == "<image>":
            content.append({"type": "image_binary", "image_binary": {"binary": images_bytes_lst[image_idx]}})
            image_idx += 1
        else:
            content.append({"type": "text", "text": chunk})
    return content


def get_query_imgs(dist_data_manager, **kwargs):
    images_bytes_lst = kwargs.get("images_bytes_lst", [])
    images_bytes_ref = kwargs.get('images_bytes_ref_lst', [])
    if images_bytes_lst:
        return images_bytes_lst
    if images_bytes_ref:
        images_bytes_ref_lst = [images_bytes_ref]
        image_lst = ray.get(ray.get(dist_data_manager.get_refs.remote(images_bytes_ref_lst)))
        return image_lst[0]  # np.array
    return []


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
                         tokenizer=tokenizer,
                         top_p=top_p,
                         timeout=timeout,
                         retry=retry,
                         retry_interval=retry_interval,
                         pool_size=pool_size,
                         random_rsp=False)
        # wait_remote_server_ready(psm)
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
        special_tokens = self.config.data.special_tokens
        self.think_begin = special_tokens.think_begin
        self.think_end = special_tokens.think_end
        self.assistant_begin = "assistant\n"
        self.img_tag = "<image>"

    def _preprocess_grm_data(self, rm_pre_ids, rollout_ids, rm_post_ids, images_bytes_lst, rm_method):
        # for preprocess function, just always used by one data
        system_prompt = self.grm_system_prompts[rm_method]
        response_mode = self.response_postprocess_mode[rm_method]
        prompt_mode = self.prepare_grm_prompt_mode[rm_method]

        grm_pre_text = decode_with_image_tag(self.tokenizer, rm_pre_ids)
        grm_post_text = decode_with_image_tag(self.tokenizer, rm_post_ids)
        response_text = decode_with_image_tag(self.tokenizer, rollout_ids)

        response_empty_flag = False
        if len(images_bytes_lst) > 0:  # 强制使用拼接模式
            prompt_mode = 1
            response_mode = 0

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

        grm_prompt = replace_image_tag(grm_prompt, images_bytes_lst, img_tag=self.img_tag)
        return system_prompt, grm_prompt, response_empty_flag

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
        weight_score = score * self.score_weight[rm_method]
        return response, weight_score

    async def call(self, reward_model=None, response_ids="", rm_method=0, **kwargs):
        cur_time = time.time()
        logging.disable(logging.INFO)
        rm_pre_ids = reward_model.get("rm_pre_ids", None)
        rm_post_ids = reward_model.get("rm_post_ids", None)
        images_bytes_lst = get_query_imgs(self.dist_data_manager, **kwargs)
        system_prompt, prompt, response_empty_flag = self._preprocess_grm_data(rm_pre_ids, response_ids, rm_post_ids,
                                                                               images_bytes_lst, rm_method)
        if response_empty_flag:
            default_score = self.empty_response_default_score[rm_method]
            return {
                "score": default_score,
                "status": "empty response",
                "grm_prompt": prompt,
                "grm_resp": "",
                "retry_cnt": 0,
                "time_cost": time.time() - cur_time
            }

        for attempt in range(self.retry + 1):
            try:
                astream_task, predict_output = None, None

                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=prompt))
                predict_input = InferenceRequest(messages=messages)

                # 获得结果
                predict_output = await self.clients[random.randrange(len(self.clients))].astream(predict_input)
                astream_task = asyncio.create_task(self.get_result(predict_output, rm_method))
                grm_resp, grm_score = await asyncio.wait_for(asyncio.shield(astream_task), self.timeout)

                return {
                    "score": grm_score,
                    "status": "success",
                    "grm_prompt": prompt,
                    "grm_resp": grm_resp,
                    "retry_cnt": attempt,
                    "time_cost": time.time() - cur_time
                }

            except Exception as e:
                if astream_task:
                    astream_task.cancel()
                if predict_output and hasattr(predict_output, 'response_iterator'):
                    predict_output.response_iterator.cancel()
                if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                    print(f"[GRM Request Timeout] ... ")
                    return {
                        "score": RM_INVALID_SCORE,
                        "status": "timeout fail",
                        "grm_prompt": prompt,
                        "grm_resp": "",
                        "retry_cnt": attempt,
                        "time_cost": time.time() - cur_time
                    }  # 超时不重试
                else:
                    print(f"[GRM Request] Attempt {attempt+1} failed: {e}")
                    if attempt < self.retry:
                        await asyncio.sleep(self.retry_interval)

        return {
            "score": RM_INVALID_SCORE,
            "status": "retry_max fail",
            "grm_prompt": prompt,
            "grm_resp": "",
            "retry_cnt": attempt,
            "time_cost": time.time() - cur_time
        }

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
