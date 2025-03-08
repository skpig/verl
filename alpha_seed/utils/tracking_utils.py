import wandb
from concurrent.futures import ProcessPoolExecutor


def clean_up_special_token(ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # fixing the issue here: https://github.com/QwenLM/Qwen2.5/issues/834
    tokens = [t.translate(BYTE_TRANSLATE_MAP) if t else t for t in tokens]
    return tokens


def log_samples_to_wandb(batch, tokenizer, global_step):
    responses = batch.batch["responses"]
    batch_size, response_length = responses.shape
    print(responses.shape)

    select_keys = [
        "rollout_log_probs", "old_log_probs", "old_entropy", "raw_scores", "returns", "values", "origin_advantages",
        "token_level_rewards", "token_level_scores", "upgo_advantages"
    ]
    real_response_lens = batch.batch['attention_mask'][:, -response_length:].sum(-1).tolist()
    raw_scores = batch.batch["raw_scores"].sum(-1).tolist()

    samples = []
    import time
    s = time.time()
    with ProcessPoolExecutor(max_workers=32) as executor:
        for i in range(batch_size):
            prompt = tokenizer.decode(batch.batch["prompts"][i], skip_special_tokens=True)
            response = tokenizer.decode(responses[i], skip_special_tokens=True)
            future = executor.submit(clean_up_special_token, responses[i])

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
            samples.append([prompt, response, future, per_token_info, sample_info])

    print(time.time() - s, 1)
    rl_samples = []
    for item in samples:
        item[2] = item[2].result()
        sample = wandb.RlSample(*item)
        rl_samples.append(sample)
    print(time.time() - s, 2)

    wandb.log({"train_samples": rl_samples}, step=global_step)
    return rl_samples


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


def transform(mapped_str, char_map):
    for encoded_char, decoded_char in char_map.items():
        mapped_str = mapped_str.replace(encoded_char, decoded_char)
    return mapped_str


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

    step += 1
    s = time.time()
    log_samples_to_wandb(batch.repeat(2), tokenizer, step)
    print(time.time() - s)

    s = time.time()
    step += 1
    log_samples_to_wandb(batch.repeat(4), tokenizer, step)
    print(time.time() - s)
