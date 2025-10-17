import time
import ray
import asyncio
import random
import logging
from alpha_seed.utils.reward_score.grm_service import RM_INVALID_SCORE
from alpha_seed.utils.reward_score.rm_utils import wait_remote_server_ready, decode_with_image_tag, replace_image_tag, get_query_imgs
from mono_rl.utils.infer.client import ORMServingClient
from mono_rl.utils.dataset.dist_data_util import get_dist_data_manager
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from bytedagi.model_io import InferenceRequest
from .utils import Verifier


class SecRmVerifier(Verifier, reward_style="secrm_service"):

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
        return "", result['score_lst'], wait_time, result['time_cost'], result['retry_cnt']

    def compute_score_remote(self, *args, **kwargs) -> float:
        remote_service = kwargs['secrm_service']
        actor = random.choice(remote_service)
        return actor.call.remote(*args, **kwargs)

    def merge_score(self, scores_lst, merge_type="mean", **kwargs):
        if merge_type == "mean":
            return sum(scores_lst) / len(scores_lst)
        else:
            raise NotImplementedError(f"{merge_type=} not implemented")

    def merge_vlm_score(self, scores_lst, merge_type="mean", **kwargs):
        return RM_INVALID_SCORE


def init_secrm_server(config, **kwargs):
    from omegaconf import OmegaConf
    rm_conf = config.reward_model.rm_server
    secrm_conf = config.reward_model.secrm_server
    secrm_conf = OmegaConf.merge(rm_conf, secrm_conf)

    tokenizer = kwargs.get("tokenizer", None)
    sec_rm_clients = RemoteSecRMServingClient.create_remote_clients(
        psm=secrm_conf.llm_serving_psm,
        idc=secrm_conf.llm_serving_idc,
        cluster=secrm_conf.llm_serving_cluster,
        model_name=secrm_conf.model_name,
        pool_size=secrm_conf.ray_actor_pool_size,
        inner_pool_size=secrm_conf.client_pool_size,
        retry=secrm_conf.max_retry,
        retry_interval=secrm_conf.retry_interval,
        timeout=secrm_conf.timeout,
        top_p=secrm_conf.top_p,
        tokenizer=tokenizer,
        config=config,
    )
    print("[SECRM INFO] build orm server success in RemoteClient")
    return sec_rm_clients


class RemoteSecRMServingClient(ORMServingClient):

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
                 pool_size=10,
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
        wait_remote_server_ready(psm)
        self.dist_data_manager = get_dist_data_manager()
        self.img_tag = "<|image|>"
        self.sem = asyncio.Semaphore(max(pool_size, 1) * 2)

    def preprocess_sec_rm_data(self, input_ids, attention_mask, **kwargs):
        images_bytes_lst = get_query_imgs(self.dist_data_manager, **kwargs)
        input_ids = self._ignore_cot_tokens(input_ids, attention_mask)
        sec_rm_text = decode_with_image_tag(self.tokenizer, input_ids, skip_special_tokens=False)
        sec_prompt_w_img = replace_image_tag(sec_rm_text,
                                             images_bytes_lst,
                                             img_tag=self.img_tag,
                                             tokenizer=self.tokenizer)
        return sec_prompt_w_img

    async def get_result(self, predict_output):
        length = 0
        values = 0
        async for msg in predict_output.astream_output:
            length += 1
            values = msg.chat_info["logprobs"]["floatList"]["values"]
        assert length == 1, "in orl mode, only output one token 是/否， something must be wrong"
        return values

    async def call(self, input_ids, attention_mask, reward_model=None, **kwargs):
        logging.disable(logging.INFO)
        cur_time = kwargs.get("call_fn_time", time.time())
        images_bytes_ref = reward_model.get("images_bytes_ref", None)
        sec_prompt_w_img = self.preprocess_sec_rm_data(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       images_bytes_ref=images_bytes_ref,
                                                       **kwargs)
        values, attempt = await self._call(sec_prompt_w_img)

        return {"score_lst": values, "retry_cnt": attempt, "time_cost": time.time() - cur_time}

    async def _call(self, prompt):
        for attempt in range(self.retry + 1):
            astream_task, predict_output = None, None
            try:
                predict_input = InferenceRequest(messages=[
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=prompt),
                    AIMessage(content=[])
                ])
                async with self.sem:
                    predict_output = await self.clients[random.randrange(len(self.clients))].astream(predict_input)
                    values = await self.get_result(predict_output)

                return values, attempt

            except Exception as e:
                if astream_task:
                    astream_task.cancel()
                if predict_output and hasattr(predict_output, 'response_iterator'):
                    predict_output.response_iterator.cancel()
                if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                    print(f"[SECRM Request Timeout] ... ")
                    return RM_INVALID_SCORE, attempt  # 超时不重试
                else:
                    print(f"[SECRM Request] Attempt {attempt+1} failed: {e}")
                    if attempt < self.retry:
                        await asyncio.sleep(self.retry_interval)
        return RM_INVALID_SCORE, self.retry
