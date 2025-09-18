import io
import re
import asyncio
import torch
import random
import ray
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import torch.nn.functional as F
from mono_rl import DataProto
from typing import Iterable, Iterator, Tuple, Dict, Any
from alpha_seed.utils.reward_score.qrm_service import RM_INVALID_SCORE, init_qrm_server
from alpha_seed.utils.reward_score.grm_service import decode_with_image_tag, RM_INVALID_SCORE


def iter_qrl_files(
    file_last_str: Iterable[str],
    base_dir: str = "/opt/tiger/250902_seedrl_qrm",
) -> Iterator[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[int, bytes]]]:
    for last in file_last_str:
        qrl_input = torch.load(f"{base_dir}/qrl_infer_input_ref_prompt_20250902-{last}.pt")
        qrl_reverse_input = torch.load(f"{base_dir}/qrl_infer_input_reverse_ref_prompt_20250902-{last}.pt")
        imgs = torch.load(f"{base_dir}/img_20250902-{last}.pt")
        yield qrl_input, qrl_reverse_input, imgs


def load_baseline(file_path):
    train_out = DataProto.load_from_disk(file_path)
    raw_data_idx = train_out.batch['global_indices']
    raw_ref_score = train_out.batch['raw_scores_0']
    raw_rev_score = train_out.batch['raw_scores_reverse_0']
    return raw_data_idx, raw_ref_score, raw_rev_score


def get_qrm_server(conf, tokenizer):
    from alpha_seed.utils.reward_score import select_remote_rm_fn
    remote_rm_cls = select_remote_rm_fn(conf)
    kwargs = {
        "config": conf,
        "tokenizer": tokenizer,
    }
    remote_rm_service = init_qrm_server(**kwargs)
    return remote_rm_service


text_before_resp1 = "\n针对上述问题，已有回复：\n"
text_before_resp2 = "\n相比之下，请回答下面的回复是否更好：\n"
text_after_instruct = "回答是或否。[EOS]assistant\n"


def split_three_by_regex(full: str, left: str, right: str):
    pat = re.compile(re.escape(left) + r"(.*?)" + re.escape(right), flags=re.S)
    m = pat.search(full)
    if not m:
        raise ValueError("未按顺序找到两个锚点")

    left_part = full[:m.start()] + left  # 锚点 left 之前
    middle = m.group(1)  # 两锚点之间
    right_part = right + full[m.end():]  # 锚点 right 之后
    return left_part, middle, right_part


def _preprocess_qrm_data(tokenizer, inputs_ids):
    prompt = decode_with_image_tag(tokenizer, inputs_ids, skip_special_tokens=True)
    left_part, middle, right_part = split_three_by_regex(prompt, text_before_resp2, text_after_instruct)
    data_dict = {}
    data_dict['rm_pre_ids'] = tokenizer.encode(left_part, add_special_tokens=False)
    data_dict['rm_post_ids'] = tokenizer.encode(right_part, add_special_tokens=False)
    rollout_ids = tokenizer.encode(middle, add_special_tokens=False)
    return rollout_ids, data_dict


async def test_main():
    # 修改rm server的配置情况
    conf = OmegaConf.load("tasks/config/ppo_trainer.yaml")
    conf.trainer.remote_rm_type = "qrm"
    conf.reward_model.rm_server.llm_serving_psm = "data.aml.arnold_inference_58339313"  # always change this for new trial
    conf.reward_model.rm_server.llm_serving_idc = "lq,lf,hl,yg,gl,wlby"
    conf.reward_model.rm_server.llm_serving_cluster = "default"
    conf.reward_model.rm_server.model_name = "vlm/RM-M8-2.5B-MoE-m8v6_2b5_navit_3t_phase1_text_182kCT2_123kRationale_158kMath_phase2_hist26k_online18k_complex1k_ref_replay_qrm-S1125"
    tokenizer_path = "/opt/tiger/bbpe155k-v6.4.3-ml.pret_v4_20241227"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 服务初始化
    remote_rm_cli = get_qrm_server(conf, tokenizer)

    # 加载seedrl数据&打分
    raw_data_idx, raw_ref_score, raw_rev_score = load_baseline("/opt/tiger/250902_seedrl_qrm/step_train_batch.pt")

    all_ref_score_cached = []
    all_rev_score_cached = []
    all_ref_score_remote = []
    all_rev_score_remote = []

    file_last_str = ['202524', '202533', '202541', '202558', '202620', '202703']
    for qrl_input, qrl_reverse_input, imgs in iter_qrl_files(file_last_str, base_dir="/opt/tiger/250902_seedrl_qrm"):
        for mini_idx in range(len(qrl_input['global_indices'])):
            # 获取基线打分
            gid = qrl_input['global_indices'][mini_idx]
            img_bytes_lst = imgs[gid.item()]  # list[bytes]
            if img_bytes_lst is None:
                img_bytes_lst = []
            raw_dataproto_idx = torch.where(raw_data_idx == gid)[0].item()
            ref_score_cached = raw_ref_score[raw_dataproto_idx]
            rev_score_cached = raw_rev_score[raw_dataproto_idx]
            all_ref_score_cached.append(ref_score_cached)
            all_rev_score_cached.append(rev_score_cached)

            ref_imputs_ids = qrl_input['input_ids'][mini_idx]
            rev_imputs_ids = qrl_reverse_input['input_ids'][mini_idx]

            ref_rollout_ids, ref_data_dict = _preprocess_qrm_data(tokenizer, ref_imputs_ids)
            rev_rollout_ids, rev_data_dict = _preprocess_qrm_data(tokenizer, rev_imputs_ids)

            ref_remote_score = ray.get(remote_rm_cli[0].call.remote(rollout_ids=ref_rollout_ids,
                                                                    reward_model=ref_data_dict,
                                                                    images_bytes_lst=img_bytes_lst))['score']
            rev_remote_score = ray.get(remote_rm_cli[0].call.remote(rollout_ids=rev_rollout_ids,
                                                                    reward_model=rev_data_dict,
                                                                    images_bytes_lst=img_bytes_lst))['score']

            print(f"[REF] 本地部署: {ref_score_cached:.4f}")
            try:
                all_ref_score_remote.append(ref_remote_score)
                print(f"[REF] 伴生调用: {ref_remote_score:.4f}" if ref_remote_score !=
                      RM_INVALID_SCORE else "[REF] Remote Score: INVALID")
            except Exception as e:
                all_ref_score_remote.append(RM_INVALID_SCORE)
                print(f"[REF] Remote QRM call failed: {e}")
            print("-" * 30)
            print(f"[REV] 本地部署: {rev_score_cached:.4f}")
            try:
                all_rev_score_remote.append(rev_remote_score)
                print(f"[REV] 伴生调用: {rev_remote_score:.4f}" if rev_remote_score !=
                      RM_INVALID_SCORE else "[REV] Remote Score: INVALID")
            except Exception as e:
                all_rev_score_remote.append(RM_INVALID_SCORE)
                print(f"[REV] Remote QRM call failed: {e}")

    ############## print metric ##############
    ref_cached_tensor = torch.stack(all_ref_score_cached)
    ref_remote_tensor = torch.as_tensor(all_ref_score_remote)
    ref_mae = F.l1_loss(ref_cached_tensor, ref_remote_tensor, reduction="mean")  # == (A - B).abs().mean()

    rev_cached_tensor = torch.stack(all_rev_score_cached)
    rev_remote_tensor = torch.as_tensor(all_rev_score_remote)
    rev_mae = F.l1_loss(rev_cached_tensor, rev_remote_tensor, reduction="mean")  # == (A - B).abs().mean()

    print(
        f"total ref msg: {len(all_ref_score_cached)}, total rev msg: {len(all_rev_score_cached)}, ref score mae: {ref_mae.item():.5f}, rev score mae: {rev_mae.item():.5f}, total mae: {(ref_mae + rev_mae).item() / 2 :.5f}"
    )


if __name__ == "__main__":
    """
    [IMPORTANT!!!] DOWNLOAD FILE FROM HDFS & START QRM SERVER
    hdfs dfs get hdfs://haruna/home/byte_data_seed/ssd_hldy/user/sunzewei.v/tokenizers/bbpe155k-v6.4.3-ml.pret_v4_20241227 /opt/tiger/
    hdfs dfs get hdfs://haruna/home/byte_data_seed/hl_lf/user/xiangyongan/data/250902_seedrl_qrm /opt/tiger/
    
    python3 ./alpha_seed/utils/reward_score/qrm_service.py 
    """
    ray.init()
    asyncio.run(test_main())
