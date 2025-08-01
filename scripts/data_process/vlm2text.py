# Script to reverse the VLM format transformation back to original format

import pandas as pd
import os
from collections import defaultdict
import numpy as np

# Input path - the VLM formatted file
input_path = 'hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yueyu/alphaseed_workspace/data/Gaokao/mixrl/general_reasoning_search_ci_train_0721_vlm_format.filter.parquet'
input_path = 'hdfs://haruna/home/byte_data_seed/ssd_hldy/user/yueyu/alphaseed_workspace/data/Gaokao/mixrl/general_reasoning_search_ci_eval_0721_vlm_format.filter.parquet'

# Output path - reversed back to original format
output_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/qiying.01/projects/agent/dataset/general_reasoning_search_ci_train_0721_vlm_format.filter.text_format.parquet'
output_path = 'hdfs://haruna/home/byte_data_seed/lf_lq/user/qiying.01/projects/agent/dataset/general_reasoning_search_ci_eval_0721_vlm_format.filter.text_format.parquet'

# Read the VLM formatted data
print("Reading VLM formatted data...")
df = pd.read_parquet(input_path)
print(f"Total records: {len(df)}")
print(f"Columns: {df.keys()}")
print(f"First record keys: {df.iloc[0].keys()}")

# Keys that were preserved from original
preserved_keys = ["data_source", "extra_info", "agent_handler"]

# Reverse the transformation
datas = []
for idx in range(len(df)):
    row = df.iloc[idx].to_dict()
    res = {}

    # Copy preserved keys as-is
    for k in preserved_keys:
        res[k] = row[k]

    # Reverse prompt transformation
    # Original: res["prompt"] = [x["prompt"][0]["content"]]
    # Reverse: create the original structure [{"content": ...}]
    if (isinstance(row["prompt"], list) or isinstance(row["prompt"], np.ndarray)) and len(row["prompt"]) > 0:
        res["prompt"] = [{"content": row["prompt"][0], 'role': 'user'}]
    else:
        # Handle edge case where prompt might be a single string
        res["prompt"] = [{"content": row["prompt"], 'role': 'user'}]

    # Recreate reward_model dictionary
    # Original: res["ability"] = x["reward_model"]["style"]
    # Original: res["verifier_feature"] = x["reward_model"]["ground_truth"]
    res["reward_model"] = {"style": row["ability"], "ground_truth": row["verifier_feature"]}

    # Note: system_prompt and img were added as empty values,
    # so we don't need to include them in the reversed data

    datas.append(res)

# Show first record for verification
print("\nFirst reversed record:")
for k in datas[0]:
    print(f"{k}: {type(datas[0][k])}")
    print(f"  {datas[0][k]}")

# Create DataFrame and save
reversed_df = pd.DataFrame(datas)
reversed_df.to_parquet(output_path)
print(f"\nReversed data saved to: {output_path}")
print(f"Total records: {len(reversed_df)}")

# Statistics (similar to original script)
print("\nData statistics:")
reward_styles = defaultdict(int)
data_sources = defaultdict(int)
for idx in range(len(reversed_df)):
    reward_styles[reversed_df.iloc[idx]["reward_model"]["style"]] += 1
    data_sources[reversed_df.iloc[idx]["data_source"]] += 1

print("Reward model styles distribution:")
print(dict(reward_styles))
print("\nData sources distribution:")
print(dict(data_sources))

# Verify the reversed data can be loaded
print("\nVerifying reversed data...")
test_df = pd.read_parquet(output_path)
print("Load successful!")
print(f"Columns: {test_df.keys()}")
print(f"First record structure: {test_df.iloc[0].keys()}")

# Optional: Upload to HDFS
# os.system(f"hadoop fs -put {output_path} hdfs://haruna/home/byte_data_seed/lf_lq/user/qiying.01/projects/agent/dataset/")
