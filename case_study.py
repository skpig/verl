from collections import defaultdict
import json
import os
from typing import Tuple, List, Dict
from vllm.sequence import Logprob, SampleLogprobs
from vllm.outputs import CompletionOutput, RequestOutput
from transformers import AutoTokenizer
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path
from nltk.metrics.distance import edit_distance
import concurrent.futures
from tqdm import tqdm
import bisect
from Levenshtein import distance as levenshtein_distance



import datasets
import torch
import pickle
import multiprocessing

system_prompt0 = """When tackling complex reasoning tasks, you should first thinks about the reasoning process in the mind and then provides the answer. 

You should strictly follow the format below:

<think>
Your reasoning process step 1 here
</think>
<think>
Your reasoning process step 2 here
</think>
<think>
Your reasoning process step 3 here
</think>
...
<think>
Your reasoning process step N here
</think>
<answer>
Put your final answer within \\boxed{}.
</answer>
"""
system_prompt1 = """Your task is to solve the user's math problem. You should first thinks about the reasoning process in the mind and then provides the answer. 

Your reasoning must be broken down into 3~5 distinct steps, each enclosed in `<think>` tags. An ideal step represents the completion of **a clear sub-goal**. It should be a self-contained paragraph that explains how you achieved one milestone in the overall solution. Group related calculations and logic together.

You should strictly follow the format below:

<think>
Reasoning step 1 here
</think>
<think>
Reasoning step 2 here
</think>
...
<think>
Final reasoning step here
</think>
<answer>
Put your final answer within \\boxed{}.
</answer>
"""
system_prompt2 = """When tackling complex reasoning tasks, you should first thinks about the reasoning process step by step in the mind and then provides the answer. 
An ideal reasoning step represents the completion of **a clear sub-goal**. It should be a self-contained paragraph that explains how you achieved one milestone in the overall solution. Group related calculations and logic together.
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., 

<think> reasoning process here </think> <answer> answer here </answer>.
"""
system_prompt3 = """Please reason step by step, put your reasoning process within <think> </think> tags, and put your final answer within <answer> </answer> tags, respectively, i.e., 
<think> reasoning process here </think> <answer> answer here </answer>.
"""
all_prompts = [
    system_prompt0,
    system_prompt1,
    system_prompt2,
    # system_prompt3
]


def compute_ed(i, j, s1, s2):
    return i, j, edit_distance(s1, s2)


# 全局变量
GLOBAL_TOKENS = None

def init_worker(tokens):
    global GLOBAL_TOKENS
    GLOBAL_TOKENS = tokens

def process_block(args):
    i_start, i_end, j_start,j_end = args
    block = np.zeros((i_end - i_start, j_end - j_start), dtype=np.int16)
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            if j <= i:
                continue
            block[i - i_start, j - j_start] = levenshtein_distance(GLOBAL_TOKENS[i], GLOBAL_TOKENS[j])
    return (i_start, i_end, j_start, j_end, block)

def load_edit_distance_matrix(model_path, tokenizer, vocab_size, fill_value=99):
    # assert all ids of special tokens are greater than vocab_size
    for idx in tokenizer.added_tokens_decoder.keys():
        assert idx >= vocab_size
    vocab_size_w_special = len(tokenizer.vocab)
    # Precompute tokens for each id to avoid repeated conversion in each process
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]
    n = vocab_size
    block_size = 2048

    map_path = f".cache/split_steps/{os.path.basename(model_path)}_ed.dat"
    if os.path.exists(map_path):
        print(f"Loading existing edit distance matrix from {map_path}")
        D = np.memmap(map_path + ".new", mode="w+", dtype="int16", shape=(vocab_size_w_special, vocab_size_w_special))
        return D
    else:
        D = np.memmap(map_path, mode="w+", dtype="int16", shape=(vocab_size_w_special, vocab_size_w_special))
        with concurrent.futures.ProcessPoolExecutor(40, initializer=init_worker, initargs=(tokens,)) as executor:
            futures = []
            # 构建参数
            block_tasks = []
            # We only calculate the distance between non-special tokens
            for i in range(0, n, block_size):
                for j in range(0, n, block_size):
                    if j <= i:
                        continue
                    i_start, i_end, j_start, j_end = i, min(i + block_size, n), j, min(j + block_size, n)
                    block_tasks.append((i_start, i_end, j_start, j_end))
                    futures.append(executor.submit(process_block, block_tasks[-1]))
                    print(f"Submitting block: ({i_start}, {i_end}, {j_start}, {j_end})")

            idx = 0
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                i_start, i_end, j_start, j_end, block = future.result()
                D[i_start:i_end, j_start:j_end] = block
                D[j_start:j_end, i_start:i_end] = block.T  # Fill the symmetric part
                idx += 1
                with open("finished_tasks.tmp", 'a') as f:
                    f.write(f"{i_start},{i_end},{j_start},{j_end}\n")
                if idx % 10 == 0:
                    print(f"Processed {idx} blocks, flushing to disk...")
                    D.flush()
        D.flush()

        del D
        D = np.memmap(map_path, mode="r", dtype="int16", shape=(vocab_size_w_special, vocab_size_w_special))
    return D


def load_tree_structure(tokenizer, model_name, vocab_size):
    """For huggingface model"""
    if "Qwen2.5" in model_name:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    elif "Qwen3" in model_name:
        model_name = "Qwen/Qwen3-0.6B"
    merge_txt_path = os.path.join(MODEL_DIR,model_name, "merges.txt")
    tokenizer_json_path = os.path.join(MODEL_DIR,model_name, "tokenizer.json")
    if os.path.exists(merge_txt_path):
        with open(merge_txt_path) as f:
            merge_rules = [line.strip() for line in f]
    elif os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path) as f:
            merge_rules = json.load(f)["model"]["merges"]
    else:
        raise ValueError("No merge rules found")
    # process merge rules
    graph = np.zeros((vocab_size, vocab_size), dtype=np.int16)
    rows = []
    cols = []
    data = []
    ancestors = [[] for _ in range(vocab_size)]
    for i, line in enumerate(merge_rules):
        if line.startswith("#"):
            continue
        a, b = line.strip().split()
        cur_node = a + b
        cur_node_id, a_id, b_id = tokenizer.convert_tokens_to_ids([cur_node, a, b])

        ancestors[cur_node_id].extend([a_id, b_id])
        ancestors[cur_node_id].extend(ancestors[a_id])
        ancestors[cur_node_id].extend(ancestors[b_id])
        ancestors[cur_node_id] = list(set(ancestors[cur_node_id]))  # remove duplicates
        for anc in ancestors[cur_node_id]:
            ed = len(cur_node) - len(tokenizer.convert_ids_to_tokens(anc))
            graph[cur_node_id, anc] = ed 
            graph[anc, cur_node_id] = ed


        # Add edges from cur_node_id to a_id and b_id
        rows.extend([cur_node_id, cur_node_id])
        cols.extend([a_id, b_id])
        data.extend([1, 1])
SUM_NOT_REACH_TOPP = []
def topp_filter(logprob_lst: List[Tuple[str, Logprob]]):
    global SUM_NOT_REACH_TOPP
    logprob_lst = [x for x in logprob_lst if x[1].rank <= num_logprobs_in_vocab]
    logprobs = [token.logprob for _, token in logprob_lst]
    probs = np.exp(logprobs)
    prob_sum = np.sum(probs)
    SUM_NOT_REACH_TOPP.append(prob_sum)
    return logprob_lst
    sorted_indices = np.argsort(probs)[::-1]  # 从大到小排序
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff_index = np.searchsorted(cumulative_probs, topp)
    keep_indices = sorted_indices[:cutoff_index + 1]
    if cutoff_index == len(sorted_indices):
        SUM_NOT_REACH_TOPP += 1
        print(f"Warning: {SUM_NOT_REACH_TOPP} times not reach topp={topp} in the current batch.")
    return [(logprob_lst[i][0], logprob_lst[i][1]) for i in keep_indices]

def get_sum_prob(logprob_lst: List[Tuple[str, Logprob]]):
    logprobs = [token.logprob for _, token in logprob_lst]
    probs = np.exp(logprobs)
    return np.sum(probs)

def get_entropy_topk(logprob_lst: List[Tuple[str, Logprob]]):
    logprob_lst = topp_filter(logprob_lst)
    logprobs = [token.logprob for _, token in logprob_lst]
    probs = np.exp(logprobs)
    entropy = -np.sum(probs * logprobs)  # Add small value to avoid log(0)
    return -entropy

def get_entropy_w_tail(logprob_lst: List[Tuple[str, Logprob]]):
    logprob_lst = topp_filter(logprob_lst)
    logprobs = np.array([token.logprob for _, token in logprob_lst])
    probs = np.exp(logprobs)
    total_mass = np.sum(probs)
    tail_mass = max(1 - total_mass, 0.0)  # Ensure tail mass is non-negative

    # assert uniform tail
    tail_prob = tail_mass / (vocab_size - num_logprobs_in_vocab)
    entropy = -np.sum(probs * np.array(logprobs)) - tail_mass * np.log(tail_prob + 1e-10)  
    return -entropy



def get_entropy_normalized(logprob_lst: List[Tuple[str, Logprob]]):
    logprob_lst = topp_filter(logprob_lst)
    logprobs = np.array([token.logprob for _, token in logprob_lst])
    probs = np.exp(logprobs)
    total_mass = np.sum(probs)

    topknorm_probs = probs / total_mass
    entropy = -np.sum(topknorm_probs * np.log(topknorm_probs + 1e-10))  # Add small value to avoid log(0)
    return -entropy


def get_sample_logprob(logprob_lst: List[Tuple[str, Logprob]]):
    return logprob_lst[0][1].logprob

def get_entropy_rao(logprob_lst: List[Tuple[str, Logprob]]):
    logprob_lst = topp_filter(logprob_lst)
    ids, logprobs = zip(*[(int(idx), token.logprob) for idx, token in logprob_lst])
    probs = np.exp(logprobs)  # (topk,)
    distance_submatrix = distance_matrix[np.ix_(ids, ids)]  # (topk, topk)

    # Rao's entropy
    rao_entropy= np.matmul(np.matmul(probs[np.newaxis, :], distance_submatrix), probs[:, np.newaxis])

    return -rao_entropy[0, 0]  # Return the scalar value


def response_level(outputs):
    with open("logprobs.txt", "w") as f:
        for i, output in enumerate(outputs):
            all_steps = []
            for j in range(len(output.outputs)):
                sent_logprobs: List[List[Tuple[str, Logprob]]] = [list(x.items()) for x in output.outputs[j].logprobs]
                sent_ids = output.outputs[j].token_ids

                # Step 1: 对每个 token 计算得分（不含首token）
                score_list = [get_value(logprob) for logprob in sent_logprobs]

                # Step 2: 选出候选切点（排除开头和结尾）
                candidate_cutpoints = sorted(range(1, len(sent_ids)), key=lambda i: -score_list[i])

                # Step 3: 按分数优先尝试添加切点，确保不产生短片段
                cutpoints = []

                for cp in candidate_cutpoints:
                    i = bisect.bisect_right(cutpoints, cp)
                    left = 0 if i == 0 else cutpoints[i - 1]
                    right = len(sent_ids) if i == len(cutpoints) else cutpoints[i]
                    if (cp - left >= min_length) and (right - cp >= min_length):
                        bisect.insort(cutpoints, cp)

                # Step 4: 输出切分段落
                segments = [0] + cutpoints + [len(sent_ids)]
                num_steps = 0
                for seg_start, seg_end in zip(segments[:-1], segments[1:]):
                    if num_steps >= topk:
                        break
                    token_logprob = sent_logprobs[seg_start]
                    f.write(f"\n-------------------Value {get_value(token_logprob):.2f} ----ProbSum {get_sum_prob(token_logprob):.2f} ---- {[item.decoded_token for _, item in token_logprob]} \n")
                    token_ids = sent_ids[seg_start:seg_end]
                    decoded = tokenizer.decode(token_ids)
                    f.write(decoded)
                    num_steps += 1

                # 写入分隔线
                f.write("===================================\n")
                f.write("===================================\n")

                all_steps.append(num_steps)

            # 输出统计信息
            print(f"Mean of # steps: {np.mean(all_steps):.2f}, Std: {np.std(all_steps):.2f}")
            print(f"Min: {np.min(all_steps)}, Max: {np.max(all_steps)}")
            print(f"{all_steps=}")

    os.system("cp logprobs.txt " + output_path)






def prompt_level(outputs: List[RequestOutput]):
    with open("logprobs.txt", "w") as f:
        for i, output in enumerate(outputs):
            all_logprobs: List[SampleLogprobs] = []
            all_sents = []
            for j in range(len(output.outputs)):
                all_logprobs.append(output.outputs[j].logprobs)
                all_sents.append(output.outputs[j].token_ids)
            merged_logprobs = [list(token.items()) for sent in all_logprobs for token in sent]
            
            sorted_logprobs = sorted(merged_logprobs, key=get_value)
            # two_percentile = sorted_logprobs[int(len(sorted_logprobs) * 0.01)].logprob
            threshold = get_value(sorted_logprobs[16 * topk])



            all_steps = []
            for sent_prob, sent_ids in zip(all_logprobs, all_sents):
                current_ids = []
                num_steps = 0
                for token_logprob, token_id in zip(sent_prob, sent_ids):
                    _, logprobs = zip(*token_logprob.items())
                    candidates = [item.decoded_token for _, item in token_logprob.items()]
                    if get_value(logprobs) < threshold and len(current_ids) > min_length:
                        f.write(tokenizer.decode(current_ids) + "\n-------------------" + str(candidates) + "\n")
                        num_steps += 1
                        current_ids = [token_id]
                    else:
                        current_ids.append(token_id)
                f.write(tokenizer.decode(current_ids) + "\n\n")
                num_steps += 1
                all_steps.append(num_steps)
                f.write("===================================\n")
                f.write("===================================\n")
            print(f"Mean of # steps: {np.mean(all_steps)}, Std of # steps: {np.std(all_steps)}")
            print(f"Min of # steps: {np.min(all_steps)}, Max of # steps: {np.max(all_steps)}")
    
    # copy the logprobs.txt to the current directory
    os.system("cp logprobs.txt " + output_path)


def _add_system_prompt_to_doc(doc: dict):
    """
    Add system prompt to the document.
    """
    assert doc['prompt'][0]['role'] != "system", "The first message should not be a system message."
    doc['prompt'].insert(0, {"role": "system", "content": all_prompts[prompt_id]})

    return doc

if __name__ == "__main__":
    MODEL_DIR = "/data/pretrain"

    config = 'math500'  # change to 'dapo' to switch configurations
    model_path = "/data/pretrain/Qwen/Qwen2.5-1.5B-Instruct"
    # model_path = "/data/pretrain/Qwen/Qwen3-1.7B"
    # model_path = "/home/huangbz/verl/checkpoints"

    func_name = "prob"
    # func_name = "topkentropy"
    # func_name = "tailentropy"
    # func_name = "normentropy"
    # func_name = "edentropy"

    level = "response"
    # level = "prompt"

    min_length = 60 # the minimum step length
    topk = 10 # the maximum step num
    topp = 0.75 # qwen3 use 0.95
    prompt_id = 2
    num_logprobs_in_vocab = 20
    n = 16


    """AUTO CONFIG"""
    config_map = {
        'math500': {
            'cache_path': '.cache/split_steps/math500.',
            'dataset_path': "/data/datasets/MATH-500/test.parquet",
            'output_dir': ".cache/split_steps/math500_split_steps",
        },
        'dapo': {
            'cache_path': '.cache/split_steps/dapo.',
            'dataset_path': "/data/datasets/DAPO-Math-17k/test.parquet",
            'output_dir': ".cache/split_steps/dapo_split_steps",
        }
    }

    cache_path = config_map[config]['cache_path']
    dataset_path = config_map[config]['dataset_path']
    output_dir = config_map[config]['output_dir']



    name_to_func = {
        "prob": get_sample_logprob,
        "topkentropy": get_entropy_topk,
        "tailentropy": get_entropy_w_tail,
        "normentropy": get_entropy_normalized,
        "edentropy": get_entropy_rao,
    }
    get_value = name_to_func[func_name]



    cache_path += os.path.basename(model_path) + ".pkl"
    output_path = os.path.join(output_dir, os.path.basename(model_path), f"{func_name}_per_{level}.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    vocab_size = tokenizer.vocab_size
    # a = load_tree_structure(tokenizer, modek_path, vocab_size)


    if os.path.exists(cache_path):
    # if False:
        with open(cache_path, 'rb') as f:
            outputs = pickle.load(f)
        print("Loaded from cache.")
    else:
        print("Cache not found, loading dataset...")
        from vllm import LLM, SamplingParams
        # Load the DAPO-Math dataset
        dataset = datasets.load_dataset("parquet", data_files=dataset_path, split="train")

        dataset = dataset.map(
            _add_system_prompt_to_doc,
            desc="Adding system prompt",
        )[:50]

        prompts = dataset["prompt"]
        print("Sample prompt:")
        print(prompts[0])
        inputs = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            continue_final_message=True,
        )


        sampling_params = SamplingParams(
            temperature=1.1,
            top_p=1,
            top_k=-1,
            max_tokens=1024 * 4,
            logprobs=num_logprobs_in_vocab,
            n=n,
        )
        llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9)
        outputs = llm.generate(prompt_token_ids=inputs, sampling_params=sampling_params)

        with open(cache_path, 'wb') as f:
            pickle.dump(outputs, f)


        print("Saved to cache.")


    if func_name == "edentropy":
        distance_matrix = load_edit_distance_matrix(model_path, tokenizer, vocab_size)
    
    if level == "response":
        response_level(outputs)
    elif level == "prompt":
        prompt_level(outputs)
    
    import seaborn
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    seaborn.histplot(SUM_NOT_REACH_TOPP, bins=30, kde=True)
    plt.xlabel("Probability Sum")
    plt.ylabel("Frequency")
    plt.title("Distribution of SUM_NOT_REACH_TOPP")
    # plt.show()
    plt.savefig("tmp.png", dpi=300)

