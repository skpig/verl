# model path: hdfs://haruna/home/byte_data_seed/ssd_hldy/evals_pipeline/checkpoints/20250703/home/byte_data_seed/ssd_hldy/p0/ckpts/shrub/m11_8b_D9fix2_master_wteoe_nowd_all_bf16_64M/checkpoints/global_step_313000/megatron_merge_states.pt
# tokenizer path: hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/M8_680m_SFT_hf
# torchrun --nproc_per_node=4 --standalone run_xperf_m11.py
# requires xperf_gpt version 1131

import os
import torch
import xperf_gpt
import logging
from xperf_gpt.inference.session import InferenceSession
from xperf_gpt.utils import logging_rank, logging_rank_only

os.environ["XGPT_TUNER_ENABLE"] = "1"

if __name__ == "__main__":
    model_path = '/opt/tiger/megatron_merge_states.pt'
    config_path = 'tests/infer/m11_8b_xperf_config.json'
    tokenizer_path = '/opt/tiger/M8_680m_SFT_hf'

    xperf_gpt.load_xperf_gpt()
    max_batch_size = 16
    max_length = 2048
    max_new_tokens = 128
    slot_size = 256
    num_slots = max_batch_size * max_length // slot_size
    inference_sess = InferenceSession(
        num_slots=num_slots,
        max_batch_size=max_batch_size,
        max_length=max_length,
        slot_block_size=slot_size,
        use_vllm=True,
        vocab_tp=True,
        # context_split_len=3,
        context_limit_bs=2,
        enable_truncation=False)
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_k=1,
        top_p=0.7,
        temperature=1.0,
        context_only=False,
    )
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        model_dict = torch.load(model_path)
    else:
        model_dict = {}
    inference_sess.init_inference_engine(
        config_path,
        generate_kwargs,
        state_dict=model_dict,
        # vanilla_checkpoint_path="/opt/tiger/models/m10/m10_new_key_merged_ckpt_step_47000",
        tokenizer_path=tokenizer_path,
        enable_metrics=True,
        use_ep=True,
        # save_mp_checkpoint_path="/opt/tiger/models/m11/preshard-mtp",
        # checkpoint_path="/opt/tiger/models/m11/preshard-mtp",
    )
    query_pool = [
        "小炒肉怎么做",
        "三天不洗头，头上会长草吗？",
        "周杰伦",
        "字节跳动什么时候上市",
    ]
    inference_sess.execute(query_pool)
    for v in inference_sess.get_inorder_responses():
        logging_rank_only(
            logging.info,
            0,
            f"{v.input_prompt}\n"
            f"{v.output_prompt}\n"
            f"{v.accepted_len}\n"
            f"{v.idx}\n"
            f"{'='*64}",
        )
        if inference_sess.context_only:
            print(v.logits, v.last_hidden_states)
