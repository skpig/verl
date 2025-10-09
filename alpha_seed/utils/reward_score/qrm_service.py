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
from alpha_seed.utils.reward_score.rm_utils import wait_remote_server_ready, check_nan, _extract_conversation, process_qrm_history, decode_with_image_tag, replace_image_tag, get_query_imgs, swap_std_ans
from mono_rl.utils.infer.client import QRMServingClient
from mono_rl.utils.dataset.dist_data_util import get_dist_data_manager

from bytedagi.model_io import InferenceRequest, ModelIO
from bytedagi.schema.param import LLMServerParamMixin
from langchain.schema import HumanMessage, SystemMessage
from .utils import Verifier

TEXT_BEFORE_RESP1 = "\n针对上述问题，已有回复：\n"
TEXT_BEFORE_RESP2 = "\n相比之下，请回答下面的回复是否更好：\n"
TEXT_AFTER_INSTRUCT = "回答是或否。[EOS]assistant\n"


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

    def merge_vlm_score(self, verifier_score, rm_score, **kwargs):
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
            elif ab_idx in verify_fusion_rule_dict["grm"]:
                if rm_score == RM_INVALID_SCORE:
                    score = rescaled_verifier_score
                else:
                    score = rm_score + 0.4 * rescaled_verifier_score - 0.2
            else:
                score = rescaled_verifier_score
        score = (score * 2) - 1  # Rescale back from [0, 1] to [-1, 1]
        return score


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
        config=config,
    )
    print("[QRM INFO] build qrm server success in RemoteClient")
    return vlm_qrm_clients


def prepare_qrm_input(prompts, answer, tokenizer, max_prompt_len=4096, max_resp_len=24576):
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    qrm_pre_prompt = f"{bos}{prompts}{eos}{TEXT_BEFORE_RESP1}{answer}{TEXT_BEFORE_RESP2}"
    qrm_post_prompt = f"{eos}{TEXT_AFTER_INSTRUCT}"
    qrm_pre_ids = tokenizer(qrm_pre_prompt)["input_ids"]
    qrm_post_ids = tokenizer(qrm_post_prompt)["input_ids"]

    return {"rm_pre_ids": qrm_pre_ids, "rm_post_ids": qrm_post_ids}


def prepare_vlm_qrm_input(
    tokenizer=None,
    row_dict=None,
    prompt_key=None,
    is_image=False,
    max_prompt_len=4096,
    max_resp_len=24576,
):
    """
        [BOS]system\nsp[EOS]
        [BOS]user\nXXXXX[EOS]
        [BOS]assistant\nXXXXX[EOS]
        [BOS]user\nXXXXX[EOS]\n已有回复xxxxxx
    """

    def pad_sequence(seq, target_len):
        seq = seq[:target_len]
        out = torch.full((target_len,), tokenizer.pad_token_id, dtype=torch.long)
        out[-len(seq):] = torch.tensor(seq, dtype=torch.long)
        return out

    bos = tokenizer.bos_token
    eos = tokenizer.eos_token

    system_prompt = row_dict.get('system_prompt', '')
    ability = row_dict.get("ability", "unknown")

    # build system prompt
    pre_context = ''
    if ability in ["verifiable_function_call"]:
        sp = ''
        for prompt_turn in json.loads(system_prompt):
            if 'name' in prompt_turn and prompt_turn['name'] is not None and prompt_turn['name'] != '':
                sp += f'{bos}{prompt_turn["role"]} name={prompt_turn["name"]}\n{prompt_turn["content"]}{eos}'
            else:
                sp += f'{bos}{prompt_turn["role"]}\n{prompt_turn["content"]}{eos}'
        pre_context = sp
    else:
        sp = row_dict.get('system_prompt', '')
        if row_dict.get('remark', '') and ('answer' in row_dict['remark'] or 'dypcot' in row_dict['remark']):
            if 'dypcot' in row_dict['remark']:
                sp += "\n\n请参考以下内容进行回答: \n\n" + row_dict['remark'][row_dict['remark'].find("dypcot:") + 7:]
            else:
                sp += "\n\n请参考以下内容进行回答: \n\n" + row_dict['remark'][row_dict['remark'].find("answer"):]
        if sp != "":
            pre_context = f"{bos}system\n{sp}{eos}"

    pre_context = tokenizer(pre_context)["input_ids"]

    prompts = row_dict[prompt_key]
    if is_image:  # [FIXME] Hack for text input
        prompts = [raw_ctx.replace("<image>", "<|image|>") for raw_ctx in prompts]

    history, final_question = _extract_conversation(prompts)
    final_question = final_question["content"]

    qrm_reference_resp = row_dict.get('rm_reference_response', None)
    if qrm_reference_resp is None or check_nan(qrm_reference_resp) or len(qrm_reference_resp) == 0:
        qrm_reference_resp = row_dict.get('ground_truth', None)  # Legency字段 (remove in future)
        if qrm_reference_resp is None or check_nan(qrm_reference_resp) or len(qrm_reference_resp) == 0:
            qrm_reference_resp = "No ground truth is found."

    # context
    context = tokenizer(
        f"{bos}user\n{final_question}{eos}"
        f"{TEXT_BEFORE_RESP1}{qrm_reference_resp}"
        f"{TEXT_BEFORE_RESP2}",
    )["input_ids"]

    post_context = tokenizer(f"{eos}{TEXT_AFTER_INSTRUCT}")["input_ids"]

    history_ids = process_qrm_history(tokenizer,
                                      history,
                                      base_length=len(pre_context) + len(context) + len(post_context),
                                      max_total=max_prompt_len)

    return {
        "rm_pre_ids": pad_sequence(pre_context + history_ids + context, max_prompt_len),
        "rm_post_ids": torch.tensor(post_context)
    }


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
        wait_remote_server_ready(psm)
        self.config = kwargs.get("config", None)
        self.tokenizer = tokenizer
        self.timeout = timeout
        self.logprob_tokens = tokenizer.encode("是")
        self.dist_data_manager = get_dist_data_manager()
        self.bos = self.tokenizer.bos_token
        self.eos = self.tokenizer.eos_token
        self.img_tag = "<|image|>"
        self.use_rm_reverse = self.config.reward_model.qrm.get("use_rm_reverse", False)
        special_tokens = self.config.data.special_tokens
        self.think_begin = special_tokens.think_begin
        self.think_end = special_tokens.think_end
        self.sem = asyncio.Semaphore(pool_size * 2)

    def _preprocess_qrm_data(self, rm_pre_ids, response_ids, rm_post_ids, **kwargs):
        use_token_ids = kwargs.get("use_token_ids", True)
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

        # remove think tag
        def trim_think_tag(response_text):
            start = response_text.find(self.think_begin)
            if start == -1:
                return response_text
            inner_start = start + len(self.think_begin)
            end = response_text.find(self.think_end, inner_start)
            if end == -1:
                return response_text
            return response_text[inner_start:end]

        output_str = trim_think_tag(response_text)
        response_empty_flag = False
        if output_str == "":
            response_empty_flag = True

        qrm_prompt = f"{qrm_pre_text}{output_str}{qrm_post_text}"
        ref_qrm_prompt_w_img = replace_image_tag(qrm_prompt,
                                                 images_bytes_lst,
                                                 img_tag=self.img_tag,
                                                 use_token_ids=use_token_ids,
                                                 tokenizer=self.tokenizer)
        rev_qrm_prompt, rev_qrm_prompt_w_img = "", ""
        if self.use_rm_reverse and not response_empty_flag:
            rev_qrm_prompt = swap_std_ans(qrm_prompt,
                                          std_open=TEXT_BEFORE_RESP1,
                                          std_close=TEXT_BEFORE_RESP2,
                                          ans_open=TEXT_BEFORE_RESP2,
                                          ans_close=TEXT_AFTER_INSTRUCT)
            rev_qrm_prompt_w_img = replace_image_tag(rev_qrm_prompt,
                                                     images_bytes_lst,
                                                     img_tag=self.img_tag,
                                                     use_token_ids=use_token_ids,
                                                     tokenizer=self.tokenizer)

        if random.random() < 0.01:
            debug_info = f"[QRM REVERSE DEBUG] qrm prompt: {qrm_prompt}"
            if self.use_rm_reverse and not response_empty_flag:
                debug_info = debug_info + f"\n qrm reverse prompt: {rev_qrm_prompt}"

            print(debug_info)

        qrm_pmp = [qrm_prompt, rev_qrm_prompt]
        return ref_qrm_prompt_w_img, rev_qrm_prompt_w_img, response_empty_flag, qrm_pmp

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

        ref_qrm_prompt_w_img, rev_qrm_prompt_w_img, response_empty_flag, qrm_prompt = \
            await asyncio.to_thread(
                self._preprocess_qrm_data,
                rm_pre_ids=rm_pre_ids,
                response_ids=response_ids,
                rm_post_ids=rm_post_ids,
                images_bytes_ref=images_bytes_ref,
                **kwargs
            )

        if response_empty_flag:
            default_score = 0
            return {
                "score": default_score,
                "status": "empty response",
                "qrm_prompt": qrm_prompt,
                "qrm_resp": "",
                "retry_cnt": 0,
                "time_cost": time.time() - cur_time
            }

        total_retry_cnt = 0
        rev_qrm_score, retry_cnt = RM_INVALID_SCORE, 0

        tasks = [self._call(ref_qrm_prompt_w_img)]
        if self.use_rm_reverse:
            tasks.append(self._call(rev_qrm_prompt_w_img))
        results = await asyncio.gather(*tasks)
        (ref_qrm_score, retry_cnt), *rest = results
        total_retry_cnt += retry_cnt

        if self.use_rm_reverse:
            (rev_qrm_score, retry_cnt) = rest[0]
            total_retry_cnt += retry_cnt

        if ref_qrm_score == RM_INVALID_SCORE and rev_qrm_score == RM_INVALID_SCORE:
            final_score = RM_INVALID_SCORE
        elif ref_qrm_score == RM_INVALID_SCORE:
            final_score = 1 - rev_qrm_score
        elif rev_qrm_score == RM_INVALID_SCORE:
            final_score = ref_qrm_score
        else:
            final_score = (ref_qrm_score + (1 - rev_qrm_score)) / 2
        assert (final_score == RM_INVALID_SCORE) or (
            0.0 <= final_score <= 1.0), f"score={final_score} is invalied, qrm_scores={[ref_qrm_score, rev_qrm_score]}"

        if final_score == RM_INVALID_SCORE:
            # fail to get result
            return {
                "score": RM_INVALID_SCORE,
                "status": "fail",
                "qrm_prompt": qrm_prompt,
                "retry_cnt": total_retry_cnt,
                "time_cost": time.time() - cur_time
            }
        else:
            return {
                "score": final_score,
                "status": "success",
                "qrm_prompt": qrm_prompt,
                "retry_cnt": total_retry_cnt,
                "time_cost": time.time() - cur_time
            }

    async def _call(self, prompt):
        for attempt in range(self.retry + 1):
            astream_task, predict_output = None, None
            try:
                predict_input = InferenceRequest(messages=[
                    HumanMessage(content=prompt),
                ])

                predict_input.param = LLMServerParamMixin()
                predict_input.param.logprob_tokens = self.logprob_tokens

                async with self.sem:  # Add a rate limiter to prevent overwhelming downstream services.
                    predict_output = await self.clients[random.randrange(len(self.clients))].astream(predict_input)
                    probability = await self.get_qrm_logprob(predict_output)

                return probability, attempt
            except Exception as e:
                if astream_task:
                    astream_task.cancel()
                if predict_output and hasattr(predict_output, 'response_iterator'):
                    predict_output.response_iterator.cancel()
                if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                    print(f"[QRM Request Timeout] ...")
                    return RM_INVALID_SCORE, attempt
                else:
                    print(f"[QRM Request] Attempt {attempt+1} failed: {e}")
                    if attempt < self.retry:
                        await asyncio.sleep(self.retry_interval)
        return RM_INVALID_SCORE, attempt
