import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np




import datasets
import torch
import pickle


def get_entropy_topk(logprob_lst):
    logprobs = [token.logprob for token in logprob_lst if token.rank <= num_logprobs_in_vocab]
    probs = np.exp(logprobs)
    entropy = -np.sum(probs * logprobs)  # Add small value to avoid log(0)
    return -entropy

def get_entropy_w_tail(logprob_lst):
    logprobs = np.array([token.logprob for token in logprob_lst if token.rank <= num_logprobs_in_vocab])
    probs = np.exp(logprobs)
    total_mass = np.sum(probs)
    tail_mass = max(1 - total_mass, 0.0)  # Ensure tail mass is non-negative

    # assert uniform tail
    tail_prob = tail_mass / (vocab_size - num_logprobs_in_vocab)
    entropy = -np.sum(probs * np.array(logprobs)) - tail_mass * np.log(tail_prob + 1e-10)  
    return -entropy



def get_entropy_normalized(logprob_lst):
    logprobs = np.array([token.logprob for token in logprob_lst if token.rank <= num_logprobs_in_vocab])
    probs = np.exp(logprobs)
    total_mass = np.sum(probs)

    topknorm_probs = probs / total_mass
    entropy = -np.sum(topknorm_probs * np.log(topknorm_probs + 1e-10))  # Add small value to avoid log(0)
    return -entropy


def get_sample_logprob(logprob_lst):
    return logprob_lst[0].logprob





if __name__ == "__main__":
    cache_path = '.cache/math500.pkl'
    # cache_path = '.cache/dapo.pkl'
    dataset_path = "/data/datasets/DAPO-Math-17k/test.parquet"
    # dataset_path = "/data/datasets/MATH-500/test.parquet"
    model_path = "/data/pretrain/Qwen/Qwen2.5-1.5B-Instruct"
    # model_path = "/home/huangbz/verl/checkpoints"
    # get_value = get_sample_logprob
    # get_value = get_entropy_topk
    get_value = get_entropy_w_tail



    tokenizer = AutoTokenizer.from_pretrained("/data/pretrain/Qwen/Qwen2.5-1.5B-Instruct")
    vocab_size = tokenizer.vocab_size
    num_logprobs_in_vocab = 20
    n = 16
    sampling_params = SamplingParams(
        temperature=1.1,
        top_p=1,
        top_k=-1,
        max_tokens=1024 * 4,
        logprobs=num_logprobs_in_vocab,
        n=n,
    )


    if os.path.exists(cache_path):
    # if False:
        with open(cache_path, 'rb') as f:
            outputs = pickle.load(f)
        print("Loaded from cache.")
    else:
        print("Cache not found, loading dataset...")
        # Load the DAPO-Math dataset
        dapo_dataset = datasets.load_dataset("parquet", data_files=dataset_path, split="train")[:100]

        prompts = dapo_dataset["prompt"]
        inputs = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            continue_final_message=True,
        )


        llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9)
        outputs = llm.generate(prompt_token_ids=inputs, sampling_params=sampling_params)

        with open(cache_path, 'wb') as f:
            pickle.dump(outputs, f)



    with open("logprobs.txt", "w") as f:
        for i, output in enumerate(outputs):
            all_logprobs = []
            all_sents = []
            for j in range(len(output.outputs)):
                all_logprobs.append(output.outputs[j].logprobs)
                all_sents.append(output.outputs[j].token_ids)
            merged_logprobs = [list(token.values()) for sent in all_logprobs for token in sent]
            
            sorted_logprobs = sorted(merged_logprobs, key=get_value)
            # two_percentile = sorted_logprobs[int(len(sorted_logprobs) * 0.01)].logprob
            two_percentile = get_value(sorted_logprobs[16 * 5])



            all_steps = []
            for sent_prob, sent_ids in zip(all_logprobs, all_sents):
                current_ids = []
                num_steps = 0
                for token_logprob, token_id in zip(sent_prob, sent_ids):
                    _, logprobs = zip(*token_logprob.items())
                    idx = token_id
                    if get_value(logprobs) < two_percentile:
                        f.write(tokenizer.decode(current_ids) + "\n-------------------\n")
                        num_steps += 1
                        current_ids = [idx]
                    else:
                        current_ids.append(idx)
                f.write(tokenizer.decode(current_ids) + "\n\n")
                num_steps += 1
                all_steps.append(num_steps)
                f.write("===================================\n")
                f.write("===================================\n")
            print(f"Mean of # steps: {np.mean(all_steps)}, Std of # steps: {np.std(all_steps)}")
            print(f"Min of # steps: {np.min(all_steps)}, Max of # steps: {np.max(all_steps)}")
