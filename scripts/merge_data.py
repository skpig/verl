import json
import pandas as pd
import numpy as np
from collections import defaultdict
import os

paths = """
# base reasoning
## math
hdfs://haruna/home/byte_data_seed/hdd_hldy/user/wangchengyi.01/data/o1_proj/rl/release6.1.parquet, agent/base_reasoning_handler, 6
## code
hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/liuzhicheng.lzc/alphaseed/rl_prompts/merge_0710/merge_code_rm_easy_hard_short_add_hard_0710_x8_shuf.parquet, agent/base_reasoning_handler, 1
## puzle
hdfs://haruna/home/byte_data_seed/ssd_hldy/user/jiangjiec/alphaseed/rl/data/training/puzzle_release_3.1_250704.parquet, agent/base_reasoning_handler, 6
## aider
hdfs://haruna/home/byte_data_seed/ssd_ygdt/alphaseed/xjj_dev/rl/r1_debug_prompt, agent/base_reasoning_handler, 20
hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/aider_v2_singleround_train_rep16.parquet, agent/base_reasoning_handler, 10
## gaokao vlm
hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yueyu/alphaseed_workspace/data/Gaokao/mixrl/gaokao_llm_format_train.parquet, agent/base_reasoning_handler, 1
# agent
## swe-bench
hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yueyu/alphaseed_workspace/data/Gaokao/mixrl/F0624_gym405_sbt353_smith500_rebench1952_all3210.fix_v2.parquet, agent/agentbench/agentless, 30
## search && textbrowser
hdfs://harunawl/home/byte_data_seed_wl/alphaseed/user/zuoxiaochen.221/data/250505_rl_gaia_browsecomp_v4_nolinkreader.asformat.fix.parquet, agent/tool/search_and_text_browser, 1
## ci
hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/qiying.01/projects/agent/datasets/0703_mathci_train_intonly_ci_dataset.parquet, agent/ci, 1
"""
save_path = "base_reaoning_swebench_search_ci_train_0731.parquet"
is_eval = False
# paths = """
# # base reasoning
# ## math && stem
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/aime2024_rule_verifier.parquet, agent/base_reasoning_handler, 32
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/aime2025_rule_verifier.parquet, agent/base_reasoning_handler, 32
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/beyondaime_rule_verifier_v3.parquet, agent/base_reasoning_handler, 16
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/gpqa_diamond_rule_verifier.parquet, agent/base_reasoning_handler, 8
# ## code
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/aider_v2_singleround_eval_rep1.parquet, agent/base_reasoning_handler, 2
# hdfs://haruna/home/byte_data_seed/ssd_ygdt/user/gracexu/xjj_dev/rl_livecodebench_v6_start_0201_end_0501, agent/base_reasoning_handler, 2
# ## puzle
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/puzzle-ood_test_arc_agi.parquet, agent/base_reasoning_handler, 1
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/alphaseed/data/rl_dataset/puzzle-iid_test.parquet, agent/base_reasoning_handler, 1
# ## vlm reasoning
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yueyu/alphaseed_workspace/data/Gaokao/mixrl/zerobench_subquestions_llm_format_eval.parquet, agent/base_reasoning_handler, 1
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yueyu/alphaseed_workspace/data/Gaokao/mixrl/zerobench_llm_format_eval.parquet, agent/base_reasoning_handler, 1
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yueyu/alphaseed_workspace/data/Gaokao/mixrl/emma_mini_llm_format_eval.parquet, agent/base_reasoning_handler, 1
# # agent
# ## swe-bench
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yueyu/alphaseed_workspace/data/Gaokao/mixrl/alphaseed_swe_bench_verified_agentless_500.fix_v2.parquet, agent/agentbench/agentless, 1
# ## search && textbrowser
# hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/qiying.01/projects/agent/datasets/test_euler_100_qiying1.parquet, agent/ci, 1
# hdfs://haruna/home/byte_data_seed/ssd_wlcb/user/qiying.01/projects/agent/datasets/aime_beyondaime_citest_x10_qiying1.parquet, agent/ci, 1
# ## ci
# hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yueyu/alphaseed_workspace/data/Gaokao/mixrl/allsearch.newsp.fix.parquet, agent/tool/search_and_text_browser, 1
# """
# save_path = "base_reaoning_swebench_search_ci_eval_0731.parquet"
# is_eval = True

# 把各来源数据join到一块
paths = list(map(lambda x: x.split(","), filter(lambda x: x.strip(), paths.split("\n"))))
df = pd.DataFrame()
necessary_keys = ['prompt', 'data_source', 'ability', 'reward_model', 'img']
cur_len = 0
for path in paths:
    if path[0].startswith("#"):
        continue
    path, agent_handler, repeat_num = path
    path = path.strip()
    agent_handler = agent_handler.strip()
    repeat_num = int(repeat_num)
    cur_df = pd.read_parquet(path)
    print(path, len(cur_df), cur_df.keys().tolist())
    swe_flag = False
    if agent_handler == "agent/agentbench/agentless":
        swe_flag = True
    if swe_flag:
        cur_necessary_keys = necessary_keys + ["extra_info"]
    else:
        cur_necessary_keys = necessary_keys
    for key in cur_necessary_keys:
        if key == "ability" and key not in cur_df.keys():
            assert agent_handler == "agent/tool/search_and_text_browser"
            cur_df.loc[:, "ability"] = np.array(["Search" for _ in range(len(cur_df))])
        if key == "img" and key not in cur_df.keys():
            cur_df.loc[:, "img"] = [[] for _ in range(len(cur_df))]
        assert key in cur_df.keys()

    not_necessary_keys = list(set(cur_df.keys()) - set(cur_necessary_keys))
    cur_df = cur_df.drop(not_necessary_keys, axis=1)
    # print(path, len(cur_df), cur_df.keys())
    # 设置index，用于后续算法的group操作
    if agent_handler == "agent/ci":
        cur_df.loc[:, "data_source"] = np.array(["ci_" + cur_df.iloc[i]["data_source"] for i in range(len(cur_df))])
    if not swe_flag:
        cur_df.loc[:, "extra_info"] = np.array([{
            "index": f"{cur_df.iloc[i]['data_source']}-{i + cur_len}"
        } for i in range(len(cur_df))])
    else:
        cur_df.loc[:, "extra_info"] = np.array([{
            "index": f"{cur_df.iloc[i]['data_source']}-{cur_df.iloc[i]['extra_info']['index']}"
        } for i in range(len(cur_df))])
    cur_len += len(cur_df)
    cur_df.loc[:, "agent_handler"] = np.array([agent_handler for i in range(len(cur_df))])
    df = pd.concat([df] + [cur_df] * repeat_num, ignore_index=True)
    print(path, len(cur_df), cur_df.keys().tolist())
    # print(cur_df.iloc[0].to_dict())
# 打散
if is_eval:
    print(len(df))
    filter_source = [
        "BeyondAIME_Verifier_v3", "AIME2025_Verifier", "AIME2024_Verifier", "GPQA_Verifier", "PUZZLE##in-domain",
        "PUZZLE##arc_agi_2", "PUZZLE##KORBench"
    ]
    for x in filter_source:
        df = df[df["data_source"] != x]
    print(len(df))
df = df.sample(frac=1).reset_index(drop=True)

# 设置verify_type, 有<answer><\answer>格式的设置成3，否则设置成4
data_source_verify_type_cnt = defaultdict(float)
verify_type = 4
reward_model = df["reward_model"]
new_reward_model = []
for rm, data_source in zip(reward_model, df["data_source"]):
    if "swe" in data_source:
        new_reward_model.append(rm)
        continue
    if "ground_truth" in rm and "verify_type" in rm["ground_truth"]:
        gt = json.loads(rm["ground_truth"])
        gt["verify_type"] = verify_type
        rm["ground_truth"] = json.dumps(gt, ensure_ascii=False)
        data_source_verify_type_cnt[data_source] += 1
    new_reward_model.append(rm)
df.loc[:, "reward_model"] = new_reward_model

# 确认verify_type设置正确
verify_type_cnt = defaultdict(float)
styles_cnt = defaultdict(float)
for rm in df["reward_model"]:
    if "ground_truth" not in rm:
        continue
    if "verify_type" in rm["ground_truth"]:
        verify_type = json.loads(rm["ground_truth"])["verify_type"]
        verify_type_cnt[verify_type] += 1
    styles_cnt[rm["style"]] += 1
print(verify_type_cnt)
print(styles_cnt)

# 统计各来源数据的分布
abilitys = defaultdict(int)
data_sources = defaultdict(int)
for ability, data_source in zip(df["ability"], df["data_source"]):
    abilitys[ability] += 1
    data_sources[data_source] += 1
print(abilitys)
print(data_sources)

# 保存数据
df.to_parquet(save_path, row_group_size=10000)
print(len(df))

# 确认数据保存正确，读取测试一下
df = pd.read_parquet(save_path)
print("load succeed")

os.system(
    f"hadoop fs -put {save_path} hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yueyu/alphaseed_workspace/data/agent")
