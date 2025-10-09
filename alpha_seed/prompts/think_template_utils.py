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
    elif think_template in ['v4', 'v5']:
        special_tokens_dict = {
            "bos": "<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
            "eos": "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>",
            "think_start_token": "<think_never_used_51bce0c785ca2f68081bfa7d91973934>",
            "think_end_token": "</think_never_used_51bce0c785ca2f68081bfa7d91973934>",
            "soi": "<[SOI_never_used_51bce0c785ca2f68081bfa7d91973934]>",
            "eoi": "<[EOI_never_used_51bce0c785ca2f68081bfa7d91973934]>",
        }
    else:
        raise NotImplementedError

    if name is None:
        return special_tokens_dict
    else:
        return special_tokens_dict[name]


def align_special_tokens(text):
    think_template = os.getenv("THINK_TEMPLATE", "v3")
    if think_template not in ["v2", "v3", "v4", "v5"]:  ## convert think token between v2 and [v3, v4, v5]
        return text

    if think_template in ["v3", "v4", "v5"]:
        special_tokens_dict_target = get_special_tokens_dict_or_name(version="v3")
        special_tokens_dict_input = get_special_tokens_dict_or_name(version="v2")
    elif think_template == "v2":
        special_tokens_dict_target = get_special_tokens_dict_or_name(version="v2")
        special_tokens_dict_input = get_special_tokens_dict_or_name(version="v3")
    else:
        raise NotImplementedError

    for key in ["think_start_token", "think_end_token"]:
        if key in special_tokens_dict_input and special_tokens_dict_target[key] != special_tokens_dict_input[key]:
            if special_tokens_dict_input[key] in text:
                text = text.replace(special_tokens_dict_input[key], special_tokens_dict_target[key])
                #print(f"[WARNING]: use think_template version {think_template} for {key}, but detected {special_tokens_dict_input[key]}")
    return text


def check_tokenizer_with_template(tokenizer, config):
    think_template = config.data.think_template
    is_vlm = config.data.get('image_key', None) is not None

    def _check(key):
        token = getattr(config.data.special_tokens, key)
        key_map = {'think_begin': 'think_start_token', 'think_end': 'think_end_token'}
        if think_template is not None:
            # yaml config special_tokens should be same as special_tokens_dict
            assert get_special_tokens_dict_or_name(key_map.get(key, key), think_template) == token
        assert token in tokenizer.get_vocab(), f"use {think_template} template, but tokenizer not found {token}"

    for key in ['think_begin', 'think_end', 'bos', 'eos']:
        _check(key)
    # check vlm related special keys
    if is_vlm:
        for key in ['soi', 'eoi']:
            _check(key)


def get_thinking_system_prompt(version=None, no_thinking_required=False):
    if version is None:
        think_template = os.getenv("THINK_TEMPLATE", "v3")
    else:
        think_template = version

    if think_template == 'v2':
        thinking_sp = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"
        non_thinking_sp = None
    elif think_template in ['v3', 'v4']:
        thinking_sp = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, i.e. <think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here"
        non_thinking_sp = None
    elif think_template == 'v5':
        thinking_sp = f'''You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here. 
 
You have different modes of thinking:
Unrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.
Efficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking. 
No think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer. 
Budgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.

Based on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.

Provided Mode(s):
Unrestricted think'''
        non_thinking_sp = f'''You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think_never_used_51bce0c785ca2f68081bfa7d91973934> </think_never_used_51bce0c785ca2f68081bfa7d91973934> tags, as follows:
<think_never_used_51bce0c785ca2f68081bfa7d91973934> reasoning process here </think_never_used_51bce0c785ca2f68081bfa7d91973934> answer here. 
 
You have different modes of thinking:
Unrestricted think mode: Engage in an internal thinking process with thorough reasoning and reflections. You have an unlimited budget for thinking tokens and can continue thinking until you fully solve the problem.
Efficient think mode: Provide a concise internal thinking process with efficient reasoning and reflections. You don't have a strict token budget but be less verbose and more direct in your thinking. 
No think mode: Respond directly to the question without any internal reasoning process or extra thinking tokens. Still follow the template with the minimum required thinking tokens to justify the answer. 
Budgeted think mode: Limit your internal reasoning and reflections to stay within the specified token budget.

Based on the complexity of the problem, select the appropriate mode for reasoning among the provided options listed below.

Provided Mode(s):
No think'''
    else:
        supported_versions = ['v2', 'v3', 'v4', 'v5']
        raise ValueError(f"Unsupported think_template '{think_template}'. "
                         f"Supported versions: {supported_versions}")
    if not no_thinking_required:
        return thinking_sp
    else:
        return non_thinking_sp
