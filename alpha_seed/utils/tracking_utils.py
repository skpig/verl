import multiprocessing
import os
import time
import traceback

import wandb
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoTokenizer
from verl.protocol import DataProto
from hdfs_io import hcopy

child_tokenizer = None


def decode_worker_init(tokenizer_name_or_path, padding_side):
    # to avoid serializing/deserializing the tokenizer object from the main process
    # child process initializes it's own tokenizer
    global child_tokenizer
    child_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side=padding_side)


def clean_up_special_token(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # fixing the issue here: https://github.com/QwenLM/Qwen2.5/issues/834
    tokens = [t.translate(BYTE_TRANSLATE_MAP) if t else t for t in tokens]
    return tokens


def decode_response(prompt, response):
    prompt = prompt[prompt > 0]
    decoded_prompt = child_tokenizer.decode(prompt, skip_special_tokens=True)
    decoded_response = child_tokenizer.decode(response, skip_special_tokens=True)
    decoded_response_clean = clean_up_special_token(child_tokenizer, response)
    return decoded_prompt, decoded_response, decoded_response_clean


def log_samples_to_wandb(batch, tokenizer, global_step):
    responses = batch.batch["responses"]
    batch_size, response_length = responses.shape
    print(time.ctime(), "sample shape", responses.shape)

    select_keys = [
        "rollout_behavior_log_probs", "old_log_probs", "old_entropy", "raw_scores", "returns", "values",
        "origin_advantages", "token_level_rewards", "token_level_scores", "upgo_advantages"
    ]
    real_response_lens = batch.batch['attention_mask'][:, -response_length:].numpy().sum(-1).tolist()
    raw_scores = batch.batch["raw_scores"].numpy().sum(-1).tolist()
    print(time.ctime(), "sample tolist done")
    samples = [None for i in range(batch_size)]

    max_workers = max(32, multiprocessing.cpu_count() // 2)
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=decode_worker_init,
                             initargs=(tokenizer.name_or_path, tokenizer.padding_side)) as executor:
        for i in range(batch_size):
            future = executor.submit(decode_response, batch.batch["prompts"][i], responses[i])

            per_token_info = {}
            for k in select_keys:
                if k in batch.batch:
                    v = batch.batch[k][i].tolist()
                    assert len(v) == response_length, f"Metrics[{k}] must match response_length"
                    per_token_info[k] = v

            sample_info = {
                "raw_score": raw_scores[i],
                "response_length": real_response_lens[i],
            }
            samples[i] = [future, per_token_info, sample_info]

    rl_samples = [None for i in range(batch_size)]
    for i, item in enumerate(samples):
        decoded_prompt, decoded_response, decoded_response_clean = item[0].result()
        sample = wandb.RlSample(decoded_prompt, decoded_response, decoded_response_clean, *item[1:])
        rl_samples[i] = sample

    print(time.ctime(), "sample to RlSample done")
    wandb.log({"train_samples": rl_samples}, step=global_step)
    print(time.ctime(), "sample wandb.log done")


def make_bytes_char():
    bs = []
    # Add characters from '!' to '~' (ASCII 33 to 126)
    bs.extend(range(ord('!'), ord('~') + 1))
    # Add characters from '\xA1' to '\xAC' (ASCII 161 to 172)
    bs.extend(range(0xA1, 0xAD))
    # Add characters from '\xAE' to '\xFF' (ASCII 174 to 255)
    bs.extend(range(0xAE, 0x100))
    # Create a list of Unicode values (UTF-32)
    cs = [b for b in bs]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # Create a dictionary mapping Unicode characters to bytes
    char_to_byte = {}
    for i in range(len(bs)):
        l = chr(cs[i])
        r = chr(bs[i])
        if l != r:
            char_to_byte[l] = r

    byte_translate_map = str.maketrans(char_to_byte)
    return byte_translate_map


def async_process_batch_samples_to_wandb(fname, hdfs_dir_name, tokenizer, step):
    print(f"[{time.ctime()}]async hcopy {fname} to {hdfs_dir_name}")
    hcopy(fname, hdfs_dir_name)
    # load from dist to prevent IPC
    print(f"[{time.ctime()}]logging samples from {fname} to wandb")
    wandb_batch = DataProto.load_from_disk(fname)
    try:
        log_samples_to_wandb(wandb_batch, tokenizer, step)
    except Exception as e:
        print('wandb internal exception, ignore this time')
        traceback.print_exc()
    print(f"[{time.ctime()}]removing {fname}")
    os.remove(fname)


BYTE_TRANSLATE_MAP = make_bytes_char()

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    tokenizer.padding_side = "left"

    from verl.protocol import DataProto
    batch = DataProto.load_from_disk("global_step_1_batch.pickle")

    wandb.init(project="test_async")
    step = 0
    import time
    s = time.time()
    log_samples_to_wandb(batch, tokenizer, step)
    print(time.time() - s)
