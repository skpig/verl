import pickle
import glob
import os
from collections import defaultdict
import numpy as np
import json
import pandas as pd
from transformers import AutoTokenizer

def load_acc(dir, outname, bon=32):
    item2scores = defaultdict(list)
    for child_dir in glob.glob(f"{dir}/global_step*"):
        path = os.path.join(child_dir, "scores.pkl")
        with open(path, "rb") as f:
            scores = pickle.load(f)
        
        for score, item in scores:
            item2scores[item].append(score)
    assert all(len(scores) == bon for scores in item2scores.values())
    item2acc = {item.item(): np.mean(scores).item() for item, scores in item2scores.items()}
    
    with open(f"{outname}_item2acc.json", "w") as f:
        json.dump(item2acc, f, indent=4)

def load_case(dir, bon=64):
    for child_dir in glob.glob(f"{dir}/global_step*"):
        path = os.path.join(child_dir, "scores.pkl")
        with open(path, "rb") as f:
            scores = pickle.load(f)
        items = [item for _, item in scores]
        path = os.path.join(child_dir, "prompts.pkl")
        with open(path, "rb") as f:
            prompts = pickle.load(f)
        path = os.path.join(child_dir, "responses.pkl")
        with open(path, "rb") as f:
            responses = pickle.load(f)
    
    # df = pd.DataFrame({"prompt": prompts, "response": responses, "item": items})

    tokenizer = AutoTokenizer.from_pretrained("/mnt/hdfs/huangbaizhou/tmp/pretrain/Qwen/Qwen3-4B-Base/")
    response_tokenized = tokenizer(responses, padding=True, truncation=True, return_tensors="pt")

    idx = 0
    while True:
        assert set(prompts[idx:idx+bon]) == 1
        # calculate response similarity
        min_length = min(len(i) for i in responses[idx:idx+bon])
        tokenized_
        for length in range(500, min_length, 500):
            part_prompts = prompts[idx:idx+bon][:length]



        



if __name__ == '__main__':
    # Use base to calculate offline acc
    BASE_LIMR = "/mnt/hdfs/huangbaizhou/tmp/ckpt/debug_hbz2/LIMR_Qwen3-4B-Base_bon32/"
    load_acc(BASE_LIMR, outname="BASE_LIMR")
    BASE_LIMR = "/mnt/hdfs/huangbaizhou/tmp/ckpt/debug_hbz2/DAPO_Qwen3-8B-Base_bon32/" # 命名错误了
    load_acc(BASE_LIMR, outname="BASE_DAPO")

    # Use id80 to calculate case similarity
    ID80_LIMR = "/mnt/hdfs/huangbaizhou/tmp/ckpt/debug_hbz2/LIMR_ID80_bon64/"





