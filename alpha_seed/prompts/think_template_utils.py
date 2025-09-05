import os


def get_special_tokens_dict_or_name(name=None, version=None):
    if version is None:
        think_template = os.getenv("THINK_TEMPLATE")
    else:
        think_template = version
    assert name in ["bos", "eos", "think_start_token", "think_end_token", "soi", "eoi",
                    None], f"unknown name: {name}, please check"

    if think_template == 'v1':
        special_tokens_dict = {
            "bos": "<[BOS]>",
            "eos": "<[EOS]>",
            "think_start_token": "<|begin_of_thought|>",
            "think_end_token": "<|end_of_thought|>",
            "soi": "[SOI]",
            "eoi": "[EOI]",
        }
    elif think_template == 'v2':
        special_tokens_dict = {
            "bos": "<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
            "eos": "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
            "think_start_token": "<think>",
            "think_end_token": "</think>",
            "soi": "[SOI]",
            "eoi": "[EOI]",
        }
    elif think_template == 'v3':
        special_tokens_dict = {
            "bos": "<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
            "eos": "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
            "think_start_token": "<think_never_used_51bce0c785ca2f68081bfa7d91973934>",
            "think_end_token": "</think_never_used_51bce0c785ca2f68081bfa7d91973934>",
            "soi": "[SOI]",
            "eoi": "[EOI]",
        }
    else:
        raise NotImplementedError

    if name is None:
        return special_tokens_dict
    else:
        return special_tokens_dict[name]


def align_special_tokens(text):
    think_template = os.getenv("THINK_TEMPLATE", "v3")
    if think_template not in ["v2", "v3"]:  ## only convert between v2 and v3
        return text

    if think_template == "v3":
        special_tokens_dict_target = get_special_tokens_dict_or_name(version="v3")
        special_tokens_dict_input = get_special_tokens_dict_or_name(version="v2")
    elif think_template == "v2":
        special_tokens_dict_target = get_special_tokens_dict_or_name(version="v2")
        special_tokens_dict_input = get_special_tokens_dict_or_name(version="v3")
    else:
        raise NotImplementedError

    for key in special_tokens_dict_target:
        if key in special_tokens_dict_input and special_tokens_dict_target[key] != special_tokens_dict_input[key]:
            if special_tokens_dict_input[key] in text:
                text = text.replace(special_tokens_dict_input[key], special_tokens_dict_target[key])
                #print(f"[WARNING]: use think_template version {think_template} for {key}, but detected {special_tokens_dict_input[key]}")
    return text


def check_tokenizer_with_template(tokenizer):
    think_template = os.getenv("THINK_TEMPLATE", "v3")
    if think_template == "v3":
        think_token = tokenizer.encode("<think_never_used_51bce0c785ca2f68081bfa7d91973934>")
        assert len(
            think_token
        ) == 1, f"use v3 template, but tokenizer encode <think_never_used_51bce0c785ca2f68081bfa7d91973934> into {think_token}"
    if think_template != "v3":
        think_token = tokenizer.encode("<think_never_used_51bce0c785ca2f68081bfa7d91973934>")
        assert len(
            think_token
        ) != 1, f"use {think_template} template, but tokenizer encode <think_never_used_51bce0c785ca2f68081bfa7d91973934> into {think_token}"
