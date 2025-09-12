import io
import re
import json
import time
import asyncio
import math
import torch
import random
import ray
from transformers import AutoTokenizer
from alpha_seed.workers.xperf_rollout.component.query import Query
from mono_rl.utils.infer.client import QRMServingClient
from mono_rl.utils.infer.cli.base_infer_cli import InferCli

from bytedagi.model_io import InferenceRequest, ModelIO
from bytedagi.schema.param import LLMServerParamMixin
from langchain.schema import HumanMessage
from bytedance import servicediscovery
from servicediscovery import ServiceDiscoveryError

QRM_INVALID_SCORE = -100.0


def wait_remote_server_ready(psm: str):
    fail_time = 0
    while fail_time < 20:
        try:
            sd_result = servicediscovery.get_one(psm, address_family="dual-stack")
            print(f"[QRM SERVER INFO] psm {psm} ready !!!")
            return
        except ServiceDiscoveryError:
            print(f"[QRM SERVER WARNING] waitting psm {psm} ready, cnt={fail_time}, begin sleep 60s")
            time.sleep(60)
            fail_time += 1
    raise ServiceDiscoveryError(f"psm {psm} not ready after 30 times retry, please check remote rm log")


def get_qrm_score(data_uid):
    handler = ray.get_actor('remote_client')
    score_dict = ray.get(handler.get_remote_rm_results.remote(data_uid))
    if score_dict is None:
        return None, None
    return score_dict['response']


def qrm_merge_score(scores_lst, merge_type="mean", **kwargs):
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
        self.dist_data_manager = kwargs.get("dist_data_manager", None)
        self.tokenizer = tokenizer
        self.timeout = timeout
        self.logprob_tokens = tokenizer.encode("是")
        self.bos = "<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"
        self.eos = "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>"
        self.text_before_resp1 = "\n针对上述问题，已有回复：\n"
        self.text_before_resp2 = "\n相比之下，请回答下面的回复是否更好：\n"
        self.text_after_instruct = "回答是或否。[EOS]assistant\n"
        self.img_tag = "<image>"
        wait_remote_server_ready(psm)

    def decode_with_image_tag(self, ids, skip_special_tokens=True, image_tag=None):
        if image_tag is None:
            image_tag = self.img_tag
        seq = torch.as_tensor(ids, dtype=torch.long, device="cpu").tolist()
        if -100 not in seq:  # pure text
            return self.tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)
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
                out.append(self.tokenizer.decode(seq[i:j], skip_special_tokens=skip_special_tokens))
                i = j

        return "".join(out)

    def select_qrm_data(self, data):
        if isinstance(data, Query):
            # only support eos callback
            user_prompt = data.meta_info['chat'].item()
            ground_truth_ans = data.meta_info['reward_model']['ground_truth']
            rollout_ans = self.decode_with_image_tag(data.new_token_ids)
            images_bytes_lst = data.meta_info.get('images_bytes')
            if images_bytes_lst is None:
                images_bytes_lst = []
            return user_prompt, ground_truth_ans, rollout_ans, images_bytes_lst
        else:
            raise ValueError(f"unsupported data type {type(data)}")

    def replace_image_tag(self, prompt, images_bytes_lst=None):
        pattern = rf"({re.escape(self.img_tag)})"
        prompt_chunks = re.split(pattern, prompt)
        image_tag_count = sum(1 for chunk in prompt_chunks if chunk == self.img_tag)
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

    def _preprocess_qrm_data(self, data):

        user_prompt, ground_truth_ans, rollout_ans, images_bytes_lst = self.select_qrm_data(data)

        # QRM conversation concat
        qrm_prompt = self.bos + user_prompt + self.eos + self.text_before_resp1 + ground_truth_ans + self.text_before_resp2 + rollout_ans + self.text_after_instruct + self.eos

        content = self.replace_image_tag(qrm_prompt, images_bytes_lst)

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

    async def call(self, input_ids, ground_truth, reward_style, processed=False, **kwargs):
        # 理解为每次处理单条数据
        if not processed:
            data = kwargs.get("query", None)
            data = self._preprocess_qrm_data(data)

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
                return {"response": probability, "status": "success", "qrm_prompt": data}

            except Exception as e:
                if astream_task:
                    astream_task.cancel()
                if predict_output and hasattr(predict_output, 'response_iterator'):
                    predict_output.response_iterator.cancel()
                if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                    print(f"[QRM Request Timeout] prompt: {data}")
                    return {"response": QRM_INVALID_SCORE, "status": "timeout fail", "qrm_prompt": data}  # 超时不重试
                else:
                    print(f"[QRM Request] Attempt {attempt+1} failed: {e}")
                    if attempt < self.retry:
                        await asyncio.sleep(self.retry_interval)

        return {"response": QRM_INVALID_SCORE, "status": "retry_max fail", "qrm_prompt": data}
