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
from alpha_seed.utils.reward_score.qrm_service import QRM_INVALID_SCORE, init_qrm_server


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


def _preprocess_qrm_data(clinet, data):
    inputs_ids, img_bytes_lst = data
    prompt = clinet.decode_with_image_tag.remote(inputs_ids, skip_special_tokens=True)
    prompt_w_img = clinet.replace_image_tag.remote(prompt, img_bytes_lst)
    return prompt_w_img


async def test_main():
    # 修改rm server的配置情况
    conf = OmegaConf.load("tasks/config/ppo_trainer.yaml")
    conf.trainer.remote_rm_type = "qrm"
    conf.reward_model.rm_server.llm_serving_psm = "data.aml.arnold_inference_57504941"  # always change this for new trial
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

    # file_last_str = ['202524','202533','202541','202558','202620','202703','202835','202845']
    file_last_str = ['202524']
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

            ref_prompt = _preprocess_qrm_data(remote_rm_cli[0], (ref_imputs_ids, img_bytes_lst))
            rev_prompt = _preprocess_qrm_data(remote_rm_cli[0], (rev_imputs_ids, img_bytes_lst))

            ref_remote_score = ray.get(remote_rm_cli[0].call.remote(ref_prompt, processed=True))['response']
            rev_remote_score = ray.get(remote_rm_cli[0].call.remote(rev_prompt, processed=True))['response']

            print(f"[REF] 本地部署: {ref_score_cached:.4f}")
            try:
                all_ref_score_remote.append(ref_remote_score)
                print(f"[REF] 伴生调用: {ref_remote_score:.4f}" if ref_remote_score !=
                      QRM_INVALID_SCORE else "[REF] Remote Score: INVALID")
            except Exception as e:
                all_ref_score_remote.append(QRM_INVALID_SCORE)
                print(f"[REF] Remote QRM call failed: {e}")

            print(f"[REV] 本地部署: {rev_score_cached:.4f}")
            try:
                all_rev_score_remote.append(rev_remote_score)
                print(f"[REV] 伴生调用: {rev_remote_score:.4f}" if rev_remote_score !=
                      QRM_INVALID_SCORE else "[REV] Remote Score: INVALID")
            except Exception as e:
                all_rev_score_remote.append(QRM_INVALID_SCORE)
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
