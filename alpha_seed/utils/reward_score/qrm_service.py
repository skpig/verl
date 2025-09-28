import io
import re
import json
import time
import asyncio
import math
import torch
import random
import ray
import logging
from transformers import AutoTokenizer
from alpha_seed.workers.xperf_rollout.component.query import Query
from alpha_seed.utils.reward_score.grm_service import RM_INVALID_SCORE
from alpha_seed.utils.reward_score.rm_utils import wait_remote_server_ready, check_nan, _extract_conversation, _process_history, decode_with_image_tag, replace_image_tag, get_query_imgs
from mono_rl.utils.infer.client import QRMServingClient
from mono_rl.utils.dataset.dist_data_util import get_dist_data_manager

from bytedagi.model_io import InferenceRequest, ModelIO
from bytedagi.schema.param import LLMServerParamMixin
from langchain.schema import HumanMessage
from bytedance import servicediscovery
from servicediscovery import ServiceDiscoveryError
from .utils import Verifier


class QrmVerifier(Verifier, reward_style="qrm_service"):

    def __init__(self, config=None, tokenizer=None):
        super().__init__(config=config, tokenizer=tokenizer)
        self.config = config
        self.tokenizer = tokenizer

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
        return "", result['score'], wait_time, result['time_cost'], result['retry_cnt']

    def compute_score_remote(self, *args, **kwargs) -> float:
        remote_service = kwargs['remote_service']
        actor = random.choice(remote_service)
        return actor.call.remote(*args, **kwargs)

    def merge_score(self, scores_lst, merge_type="mean", **kwargs):
        if merge_type == "mean":
            return sum(scores_lst) / len(scores_lst)
        else:
            raise NotImplementedError(f"{merge_type=} not implemented")


def init_qrm_server(config, **kwargs):
    rm_conf = config.reward_model
    tokenizer = kwargs.get("tokenizer", None)
    vlm_qrm_clients = VLMQRMServingClient.create_remote_clients(
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
    )
    print("[QRM INFO] build qrm server success in RemoteClient")
    return vlm_qrm_clients


def prepare_qrm_input(prompts, answer, tokenizer, is_image=False):
    text_before_resp1 = "\n针对上述问题，已有回复：\n"
    text_before_resp2 = "\n相比之下，请回答下面的回复是否更好：\n"
    text_after_instruct = "回答是或否。[EOS]assistant\n"
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    qrm_pre_prompt = f"{bos}{prompts}{eos}{text_before_resp1}{answer}{text_before_resp2}"
    if is_image:  # [FIXME] Hack for text input
        prompts = [raw_ctx.replace("<image>", "<|image|>") for raw_ctx in qrm_pre_prompt]
    qrm_post_prompt = f"{text_after_instruct}"
    qrm_pre_ids = tokenizer(qrm_pre_prompt)["input_ids"]
    qrm_post_ids = tokenizer(qrm_post_prompt)["input_ids"]

    return {"rm_pre_ids": qrm_pre_ids, "rm_post_ids": qrm_post_ids}


class VLMQRMServingClient(QRMServingClient):

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
        self.tokenizer = tokenizer
        self.timeout = timeout
        self.logprob_tokens = tokenizer.encode("是")
        self.dist_data_manager = get_dist_data_manager()
        self.bos = self.tokenizer.bos_token
        self.eos = self.tokenizer.eos_token
        self.text_before_resp1 = "\n针对上述问题，已有回复：\n"
        self.text_before_resp2 = "\n相比之下，请回答下面的回复是否更好：\n"
        self.text_after_instruct = "回答是或否。[EOS]assistant\n"
        self.img_tag = "<|image|>"
        wait_remote_server_ready(psm)

    def _preprocess_qrm_data(self, rm_pre_ids, response_ids, rm_post_ids, **kwargs):
        images_bytes_lst = get_query_imgs(self.dist_data_manager, **kwargs)
        rm_pre_ids = torch.as_tensor(rm_pre_ids, dtype=torch.long, device="cpu")
        rm_post_ids = torch.as_tensor(rm_post_ids, dtype=torch.long, device="cpu")
        rm_pre_ids = self.trim_tensor(rm_pre_ids)
        rm_post_ids = self.trim_tensor(rm_post_ids)
        qrm_pre_text = decode_with_image_tag(self.tokenizer, rm_pre_ids, skip_special_tokens=False)
        qrm_post_text = decode_with_image_tag(self.tokenizer, rm_post_ids, skip_special_tokens=False)
        response_ids = [
            t for t in response_ids
            if t not in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        ]
        response_text = decode_with_image_tag(self.tokenizer, response_ids, skip_special_tokens=False)

        qrm_prompt = f"{qrm_pre_text}{response_text}{qrm_post_text}"
        content = replace_image_tag(qrm_prompt, images_bytes_lst, img_tag=self.img_tag)

        return content

    async def get_qrm_logprob(self, output):
        length = 0
        values = 0
        async for msg in output.astream_output:
            length += 1
            try:
                values = msg.chat_info["choice"]["struct"]["fields"]["logprobs"]["struct"]["fields"]["token_logprobs"][
                    "floatList"]["values"][0]

                yes_token = msg.chat_info["choice"]["struct"]["fields"]["logprobs"]["struct"]["fields"]["tokens"][
                    "stringList"]["values"][0]

                assert yes_token == "是", (
                    f"token_id 是 in rm server tokenizer({yes_token}) and driver tokenizer is different")

            except Exception as e:
                print("DEBUG", msg.content, msg.chat_info)
                raise RuntimeError(f"Failed to extract logprobs: {e}")
        assert length == 1, "in qrl mode, only output one token 是/否， something must be wrong"
        return math.exp(values)

    async def call(self, reward_model=None, response_ids="", **kwargs):
        # 理解为每次处理单条数据
        logging.disable(logging.INFO)
        cur_time = time.time()
        rm_pre_ids = reward_model.get("rm_pre_ids", None)
        rm_post_ids = reward_model.get("rm_post_ids", None)
        images_bytes_ref = reward_model.get("images_bytes_ref", None)

        data = await asyncio.to_thread(self._preprocess_qrm_data,
                                       rm_pre_ids=rm_pre_ids,
                                       response_ids=response_ids,
                                       rm_post_ids=rm_post_ids,
                                       images_bytes_ref=images_bytes_ref,
                                       **kwargs)

        for attempt in range(self.retry + 1):
            try:
                astream_task = None
                predict_output = None

                predict_input = InferenceRequest(messages=[
                    HumanMessage(content=data),
                ])

                predict_input.param = LLMServerParamMixin()
                predict_input.param.logprob_tokens = self.logprob_tokens

                # 获得结果
                predict_output = await self.clients[random.randrange(len(self.clients))].astream(predict_input)
                astream_task = asyncio.create_task(self.get_qrm_logprob(predict_output))

                probability = await asyncio.wait_for(asyncio.shield(astream_task), self.timeout)
                return {
                    "score": probability,
                    "status": "success",
                    "qrm_prompt": data,
                    "retry_cnt": attempt,
                    "time_cost": time.time() - cur_time
                }

            except Exception as e:
                if astream_task:
                    astream_task.cancel()
                if predict_output and hasattr(predict_output, 'response_iterator'):
                    predict_output.response_iterator.cancel()
                if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                    print(f"[QRM Request Timeout] prompt: {data}")
                    return {
                        "score": RM_INVALID_SCORE,
                        "status": "timeout fail",
                        "qrm_prompt": data,
                        "retry_cnt": attempt,
                        "time_cost": time.time() - cur_time
                    }  # 超时不重试
                else:
                    print(f"[QRM Request] Attempt {attempt+1} failed: {e}")
                    if attempt < self.retry:
                        await asyncio.sleep(self.retry_interval)

        return {
            "score": RM_INVALID_SCORE,
            "status": "retry_max fail",
            "qrm_prompt": data,
            "retry_cnt": attempt,
            "time_cost": time.time() - cur_time
        }
