# ray_group_mean_cos.py
# pip install "transformers>=4.51.0" "sentence-transformers>=2.7.0" ray

import re
import os
import argparse
from typing import List, Tuple
import glob
import tempfile
import shutil
from tqdm import tqdm
import pickle
import numpy as np
import ray
import pandas as pd
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.metrics.distance import edit_distance
from transformers import AutoTokenizer


@ray.remote(num_gpus=1)
class EmbedWorker:
    """每个 GPU 常驻一个 SentenceTransformer 模型，计算单组文本的平均两两相似度。"""
    def __init__(
        self,
        model_name: str = "/mnt/hdfs/huangbaizhou/tmp/pretrain/Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 256,
        use_fa2: bool = True,
        pad_left: bool = True,
        output_dir: str = None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.cache_dir = f"{output_dir}/.cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        model_kwargs = {
            "torch_dtype": torch.bfloat16
        }
        if use_fa2:
            # 若未安装 flash-attn，SentenceTransformer/transformers 会回退或报错
            # 这里做个温和兜底
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                pass

        tokenizer_kwargs = {"padding_side": "left"} if pad_left else {}
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
    
    def compute_self_bleu_and_edit_distance_for_ids(self, idx_and_texts: Tuple[int, int, list[list[int]]]):
        """
        计算一组由整数ID列表表示的文本的 Self-BLEU 和编辑距离均值。
        
        :param responses: list[list[int]], 包含多条由token ID组成的生成文本
        :return: (avg_self_bleu, avg_edit_distance)
        """
        idx, length, responses = idx_and_texts

        N = len(responses)
        cnt = (N * (N - 1)) / 2

        cache_path = f"{self.cache_dir}/self_bleu_and_edit_distance-{idx}_{length}.pkl"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    bleu_scores, edit_distances = pickle.load(f)
                print(f"Loaded cached self bleu and edit distance for {idx}_{length}")
                return (idx, length, np.sum(bleu_scores) / cnt, np.sum(edit_distances) / cnt)
            except Exception:
                print(f"Failed to load cached self bleu and edit distance for {idx}_{length}")


        bleu_scores = np.zeros((len(responses), len(responses)))
        edit_distances = np.zeros((len(responses), len(responses)))
        
        # 计算每对文本之间的 Self-BLEU 和编辑距离
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                # 直接获取已经 "分词" 好的序列
                candidate = responses[i]
                reference = responses[j]
                
                # 计算编辑距离
                # edit_distance 函数可以直接处理整数列表
                edit_dist = edit_distance(candidate, reference)
                edit_distances[i, j] = edit_dist
                
                # 计算 Self-BLEU
                # sentence_bleu 也直接使用整数ID列表
                # reference 需要被包裹在一个列表中, 因为一个candidate可以有多个references
                bleu_score = sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
                bleu_scores[i, j] = bleu_score
                
        # 使用临时文件写入，然后移动到目标位置，避免程序突然退出时损坏缓存
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=os.path.dirname(cache_path)) as temp_f:
            pickle.dump((bleu_scores, edit_distances), temp_f)
            temp_path = temp_f.name
        
        # 原子性地移动到目标位置
        shutil.move(temp_path, cache_path)
        # 计算平均值
        avg_self_bleu = np.sum(bleu_scores) / cnt
        avg_edit_distance = np.sum(edit_distances) / cnt
        
        self_bleu_and_edit_distance = (idx, length, avg_self_bleu, avg_edit_distance)
        print(f"Finished computing self bleu and edit distance for {idx}_{length}")
        return self_bleu_and_edit_distance

    def compute_group_mean(self, idx_and_texts: Tuple[int, int, List[str]]) -> Tuple[int, float, int]:
        """返回 (组索引, 组内平均相似度, 组大小)。不返回相似度矩阵。"""
        import torch
        idx, length, texts = idx_and_texts # idx is int, length is int, texts is List[str]

        N = len(texts)
        cnt = (N * (N - 1)) / 2

        # 缓存
        cache_path = f"{self.cache_dir}/group_mean-{idx}_{length}.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                similarity = pickle.load(f)
                print(f"Loaded cached group mean for {idx}_{length}")
                return (idx, length, float(np.sum(similarity) / cnt))
        
        with torch.inference_mode():
            emb = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,        # torch.Tensor [N, D]
                normalize_embeddings=True,     # 归一化后 cosine = dot
            )
            # 计算相似度矩阵
            similarity = self.model.similarity(emb, emb)

            # mask out upper triangle
            similarity = torch.triu(similarity, diagonal=1)

            similarity = similarity.to(torch.float32).cpu().numpy()

        # 使用临时文件写入，然后移动到目标位置，避免程序突然退出时损坏缓存
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=os.path.dirname(cache_path)) as temp_f:
            pickle.dump(similarity, temp_f)
            temp_path = temp_f.name
        
        # 原子性地移动到目标位置
        shutil.move(temp_path, cache_path)
        mean_pairwise = np.sum(similarity) / cnt
        group_mean = (idx, length, float(mean_pairwise))
        del emb
        print(f"Finished computing group mean for {idx}_{length}")
        return group_mean


def ray_compute_group_means(
    output_dir: str,
    groups: List[Tuple[int, int, List[str]]],
    edit_groups: List[Tuple[int, int, List[List[int]]]],
    num_gpus: int = 16,
    batch_size: int = 256,
    use_fa2: bool = True,
    pad_left: bool = True,
    shards: int = None,
):
    """
    使用 Ray 在多 GPU 上并行计算每组文本的平均两两相似度。
    返回 List[Tuple[idx, mean, N]]，按 idx 排序。
    """
    if shards is None:
        shards = num_gpus

    # 启动 Ray（本地单机）
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 创建 Actor 池（每个占用 1 GPU）
    workers = [
        EmbedWorker.remote(
            output_dir=output_dir,
            batch_size=batch_size,
            use_fa2=use_fa2,
            pad_left=pad_left,
        )
        for _ in range(shards)
    ]

    # 轮转把任务分发给各个 Actor（64 条/组，负载很均衡，简单轮转即可）
    futures = []
    for i, group in enumerate(groups):
        w = workers[i % shards]
        futures.append(w.compute_group_mean.remote(group))
    
    edit_futures = []
    for i, edit_group in enumerate(edit_groups):
        w = workers[i % shards]
        edit_futures.append(w.compute_self_bleu_and_edit_distance_for_ids.remote(edit_group))


    results = ray.get(futures)
    edit_results = ray.get(edit_futures)

    return results, edit_results


def load_case(dir, args, output_dir, bon=64):
    prompts = []
    responses = []
    sorted_child_dirs = sorted(glob.glob(f"{dir}/global_step*"), key=lambda x: int(x.split("/")[-1].split("_")[-1]))
    for child_dir in sorted_child_dirs:
        if len(prompts) // bon > 1000:
            break
        path = os.path.join(child_dir, "prompts.pkl")
        with open(path, "rb") as f:
            prompts.extend(pickle.load(f))
        path = os.path.join(child_dir, "responses.pkl")
        with open(path, "rb") as f:
            responses.extend(pickle.load(f))
    print(f"Loaded {len(prompts)} prompts and {len(responses)} responses")
            
    # df = pd.DataFrame({"prompt": prompts, "response": responses, "item": items})

    # prompts = prompts[:640]
    # responses = responses[:640]
    tokenizer = AutoTokenizer.from_pretrained("/mnt/hdfs/huangbaizhou/tmp/pretrain/Qwen/Qwen3-4B-Base/" , use_fast=True)
    response_tokenized = tokenizer(responses, truncation=True, max_length=3000, add_special_tokens=False).input_ids

    idx = 0
    groups = []
    edit_groups = []
    while True:
        assert len(set(prompts[idx:idx+bon])) == 1
        # calculate response similarity
        cur_response_tokenized = response_tokenized[idx:idx+bon]
        for length in [50, 200, 1000, 2000, 4000]:
            trunc_response_tokenized = [i[:length] for i in cur_response_tokenized if len(i) > length]
            if len(trunc_response_tokenized) < 32:
                continue
            part_responses = tokenizer.batch_decode(trunc_response_tokenized, skip_special_tokens=True) # List[str]
            groups.append((idx, length, part_responses))
            edit_groups.append((idx, length, trunc_response_tokenized)) # Tuple[int, int, List[List[int]]]
        idx += bon
        if idx >= len(prompts):
            break
        if len(groups) > 10000:
            break
    print(f"All groups: {len(groups)}")


    results, edit_results = ray_compute_group_means(
        output_dir=output_dir,
        groups=groups,
        edit_groups=edit_groups,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        use_fa2=not args.no_fa2,
        pad_left=not args.no_pad_left,
    )

    with open(f"{output_dir}/case_emb_sim_mean.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(f"{output_dir}/case_edit_results.pkl", "wb") as f:
        pickle.dump(edit_results, f)


def mean_pairwise_similarity(S: np.ndarray, idx: np.ndarray) -> float:
    """
    计算子集 idx 的两两相似度均值（不含对角，按无序对计数）。
    """
    sub = S[np.ix_(idx, idx)]
    m = len(idx)
    return sub.sum() / (m * (m - 1) / 2)  # 等价于 sum_upper / C(m,2)

def sim_at_m_bootstrap(S: np.ndarray, m: int = 32, B: int = 5000,
                       replace: bool = False, ci: float = 0.95, seed: int | None = 0):
    """
    Bootstrap / Monte Carlo 估计 sim@m 的分布、点估计与置信区间。
    - replace=False: 无放回（更贴近“32 个不同变量”的定义）
    - replace=True : 有放回（严格的bootstrap重采样）
    """
    assert S.ndim == 2 and S.shape[0] == S.shape[1], "S 必须为 N×N 方阵"
    N = S.shape[0]
    rng = np.random.default_rng(seed)

    stats = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.choice(N, size=m, replace=replace)
        stats[b] = mean_pairwise_similarity(S, idx)

    # 经验分布点估计 + 置信区间（百分位法）
    est = stats.mean()
    # alpha = (1 - ci) / 2
    # lo, hi = np.quantile(stats, [alpha, 1 - alpha])

    # # 同时给出“整体均值”的解析量，便于 sanity check
    # off_mean = (S.sum() - np.trace(S)) / (N * (N - 1))  # 全局非对角平均
    # diag_mean = np.trace(S) / N
    # # 有放回时的期望（m 不出现；若 replace=False 则该值仅作参考）
    # exp_with_repl = (diag_mean / N) + (1 - 1 / N) * off_mean

    return est

    
def post_process(dir, args):
    with open(f"{dir}/case_emb_sim_mean.pkl", "rb") as f:
        results = pickle.load(f)
    with open(f"{dir}/case_edit_results.pkl", "rb") as f:
        edit_results = pickle.load(f)
    # from collections import defaultdict
    # idx2length2cossim = defaultdict(dict)
    # idx2length2editdist = defaultdict(dict)
    # idx2length2selfbleu = defaultdict(dict)
    # for idx, length, cossim in results:
    #     idx2length2cossim[idx][length] = cossim
    # for idx, length, selfbleu, editdist in edit_results:
    #     idx2length2editdist[idx][length] = editdist
    #     idx2length2selfbleu[idx][length] = selfbleu

    listofdict_cossim = []
    listofdict_editdist = []
    listofdict_selfbleu = []
    for idx, length, cossim in results:
        listofdict_cossim.append({"idx": idx, "length": length, "cossim": cossim})
    for idx, length, selfbleu, editdist in edit_results:
        listofdict_editdist.append({"idx": idx, "length": length, "editdist": editdist, "norm_editdist": editdist / length, "selfbleu": selfbleu})
        listofdict_editdist.append({"idx": idx, "length": length, "editdist": editdist, "norm_editdist": editdist / length, "selfbleu": selfbleu})
    
    cossim_df = pd.DataFrame(listofdict_cossim)
    editdist_df = pd.DataFrame(listofdict_editdist)
    # selfbleu_df = pd.DataFrame(listofdict_selfbleu)

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="whitegrid")
    # plt.figure(figsize=(10, 5))
    # sns.violinplot(x="length", y="cossim", data=cossim_df)
    # plt.savefig(f"case_emb_sim_mean.png")
    # plt.close()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    sns.violinplot(x="length", y="norm_editdist", data=editdist_df, ax=ax1)
    sns.violinplot(x="length", y="selfbleu", data=editdist_df, ax=ax2)
    plt.savefig(f"case_edit_dist_and_self_bleu_mean.png")
    plt.close()

    

    # sim_matrix_path_list = glob.glob(f"{dir}/.cache/group_mean-*")
    # for sim_matrix_path in sim_matrix_path_list:
    #     with open(sim_matrix_path, "rb") as f:
    #         sim_matrix = pickle.load(f)
    #     groups = re.match(r".cache/group_mean-(.*)_(\d+).pkl", sim_matrix_path).groups()
    #     idx, length = int(groups[0]), int(groups[1])
    edit_metrix_path_list = glob.glob(f"{dir}/.cache/self_bleu_and_edit_distance-*")
    all_list = []
    for edit_metrix_path in tqdm(edit_metrix_path_list):
        with open(edit_metrix_path, "rb") as f:
            selfbleu, editdist = pickle.load(f)
        groups = re.search(r".cache/self_bleu_and_edit_distance-(.*)_(\d+).pkl", edit_metrix_path).groups()
        idx, length = int(groups[0]), int(groups[1])
        # for i in range(len(rtn_tuple[0])):
        #     for j in range(i + 1, len(rtn_tuple[0])):
        #         all_list.append({"idx": idx, "length": length, "editdist": rtn_tuple[1][i, j], "norm_editdist": rtn_tuple[1][i, j] / length, "selfbleu": rtn_tuple[0][i, j]})
        selfbleu_sim_at_32 = sim_at_m_bootstrap(selfbleu, m=32)
        editdist_sim_at_32 = sim_at_m_bootstrap(editdist, m=32)
        all_list.append({"idx": idx, "length": length, "self-bleu": selfbleu_sim_at_32, "norm-editdist": editdist_sim_at_32 / length})
        
    editdist_df = pd.DataFrame(all_list)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

    sns.violinplot(x="length", y="norm-editdist", data=editdist_df, ax=ax1)
    sns.violinplot(x="length", y="self-bleu", data=editdist_df, ax=ax2)
    plt.savefig(f"case_edit_dist_and_self_bleu_mean_all.png")
    plt.close()

    
        



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--no_fa2", action="store_true")
    parser.add_argument("--no_pad_left", action="store_true")
    args = parser.parse_args()

    # Use id80 to calculate case similarity
    # ID80_LIMR = "/mnt/hdfs/huangbaizhou/tmp/ckpt/debug_hbz2/LIMR_ID80_bon64/"
    # load_case(ID80_LIMR, args, bon=64, output_dir=ID80_LIMR)
    # ID80_DAPO = "/mnt/hdfs/huangbaizhou/tmp/ckpt/debug_hbz2/DAPO_ID80_bon128/"
    # load_case(ID80_DAPO, args, bon=128, output_dir=ID80_DAPO)


    """Post Processing"""
    ID80_LIMR = "/mnt/hdfs/huangbaizhou/tmp/ckpt/debug_hbz2/LIMR_ID80_bon64/"
    post_process(ID80_LIMR, args)





