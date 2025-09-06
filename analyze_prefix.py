import pickle
import glob
import os
from collections import defaultdict
import numpy as np
import json
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Levenshtein import distance as edit_distance
from transformers import AutoTokenizer

def compute_self_bleu_and_edit_distance_for_ids(responses: list[list[int]]):
    """
    计算一组由整数ID列表表示的文本的 Self-BLEU 和编辑距离均值。
    
    :param responses: list[list[int]], 包含多条由token ID组成的生成文本
    :return: (avg_self_bleu, avg_edit_distance)
    """
    bleu_scores = []
    edit_distances = []
    
    # 检查输入是否为空或只有一个序列，避免计算错误
    if len(responses) < 2:
        return 0.0, 0.0

    # 计算每对文本之间的 Self-BLEU 和编辑距离
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            # 直接获取已经 "分词" 好的序列
            candidate = responses[i]
            reference = responses[j]
            
            # 计算编辑距离
            # edit_distance 函数可以直接处理整数列表
            edit_dist = edit_distance(candidate, reference)
            edit_distances.append(edit_dist)
            
            # 计算 Self-BLEU
            # sentence_bleu 也直接使用整数ID列表
            # reference 需要被包裹在一个列表中, 因为一个candidate可以有多个references
            bleu_score = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
            bleu_scores.append(bleu_score)
            
    # 计算平均值
    avg_self_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_edit_distance = np.mean(edit_distances) if edit_distances else 0.0
    
    return avg_self_bleu, avg_edit_distance
def load_acc(dir, outname, bon=32):
    item2scores = defaultdict(list)
    for child_dir in glob.glob(f"{dir}/global_step*"):
        path = os.path.join(child_dir, "scores.pkl")
        with open(path, "rb") as f:
            scores = pickle.load(f)
        
        for score, item in scores:
            item2scores[item].append(score)
    assert all(len(scores) == bon for scores in item2scores.values())
    item2acc = {item: np.mean(scores) for item, scores in item2scores.items()}
    
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





