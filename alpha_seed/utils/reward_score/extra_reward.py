import re
import torch


def add_length_reward(thinking_len, correct_reward, config, current_mean_len=None):
    """
    ratio: 从多长开始奖励
    enhance: 最多多奖励多少
    TODO: just stepv2 and overlong is justified. others not
    """
    assert isinstance(thinking_len, int), thinking_len
    # length reward
    length_reward = 0
    if config.algorithm.inference_scaling == "v1":  # 从 0.1 * max_length 以上开始奖励
        ratio = 0.1
        enhance = 2.0 if correct_reward > 0 else 0.0
        length_benefit = (thinking_len - ratio * config.data.max_response_length) / (
            (1 - ratio) * config.data.max_response_length)
        correct_reward = correct_reward * max(1 + length_benefit * enhance, 1)
    elif config.algorithm.inference_scaling == "v2":  # 从 0 开始奖励，0 = -1，Max = 5
        ratio = 0.0
        min_reward = -1
        max_reward = 5
        if correct_reward > 0:
            correct_reward = (max_reward -
                              min_reward) * thinking_len / config.data.max_response_length + min_reward  # linear reward
    elif config.algorithm.inference_scaling == "v3":  # 从 0 开始奖励，0 = -1，Max = 3
        min_reward = -1
        max_reward = 3
        if correct_reward > 0:
            correct_reward = (max_reward -
                              min_reward) * thinking_len / config.data.max_response_length + min_reward  # linear reward
    elif config.algorithm.inference_scaling == "v4":  # 从 0 开始奖励，0 = 0，Max = 3
        min_reward = 0
        max_reward = 3
        if correct_reward > 0:
            correct_reward = (max_reward -
                              min_reward) * thinking_len / config.data.max_response_length + min_reward  # linear reward
    elif config.algorithm.inference_scaling == "v5":  # 从 0 开始奖励，0 = 0，Max = 2
        min_reward = 0
        max_reward = 2
        if correct_reward > 0:
            correct_reward = (max_reward -
                              min_reward) * thinking_len / config.data.max_response_length + min_reward  # linear reward
    elif config.algorithm.inference_scaling == 'v6':
        min_reward = 0
        max_reward = 1
        if correct_reward > 0:
            correct_reward = (max_reward -
                              min_reward) * thinking_len / config.data.max_response_length + min_reward  # linear reward
    elif config.algorithm.inference_scaling == 'meanv0':
        min_reward = 0.0
        max_reward = 1.0
        interval = 1024.0
        if correct_reward > 0:
            correct_reward = torch.sigmoid(torch.tensor(
                (thinking_len - current_mean_len) / interval)) * (max_reward - min_reward) + min_reward
            correct_reward = correct_reward.item()
    elif config.algorithm.inference_scaling == 'meanv1':
        min_reward = 0.0
        max_reward = 2.0
        interval = 2048.0
        if correct_reward > 0:
            correct_reward = (max(min((thinking_len - current_mean_len) / interval, 1.0), -1.0) * 0.5 +
                              0.5) * (max_reward - min_reward) + min_reward
    elif 'stepv2' in config.algorithm.inference_scaling:
        if config.algorithm.no_length_reward == 'v1':
            if thinking_len > 12288:
                return correct_reward

        length_reward_slope = config.algorithm.get("inference_scaling_slope", 0.1)
        length_reward_steps = config.algorithm.get("length_reward_steps", 1)
        # length_reward_step_interval = config.algorithm.get("length_reward_step_interval", 512)
        if thinking_len > current_mean_len:
            length_reward = length_reward_steps
        elif thinking_len < current_mean_len:
            length_reward = -length_reward_steps
        else:
            length_reward = 0
        length_reward = length_reward * length_reward_slope
    elif 'step' in config.algorithm.inference_scaling and 'stepv1' not in config.algorithm.inference_scaling:
        length_reward_slope = config.algorithm.get("inference_scaling_slope", 0.1)
        length_reward_steps = config.algorithm.get("length_reward_steps", 1)
        length_reward_step_interval = config.algorithm.get("length_reward_step_interval", 512)
        length_reward = max(min((thinking_len - current_mean_len) // length_reward_step_interval, length_reward_steps),
                            -length_reward_steps) * length_reward_slope
        if config.algorithm.inference_scaling == "step_all":
            correct_reward = length_reward + correct_reward
        elif config.algorithm.inference_scaling == "step_correct":
            if correct_reward > 0:
                correct_reward = length_reward + correct_reward
        elif config.algorithm.inference_scaling == "step_wrong":
            if correct_reward < 0:
                correct_reward = length_reward + correct_reward
        else:
            raise NotImplementedError
    elif 'stepv1' in config.algorithm.inference_scaling:
        length_reward_slope = config.algorithm.get("inference_scaling_slope", 0.1)
        punish_overlong_interval = config.algorithm.get("punish_overlong_interval",
                                                        min(int(config.data.max_response_length / 16), 1024))

        if current_mean_len > config.data.max_response_length - punish_overlong_interval:  #均值已经过长，越短越好
            length_reward = -(1 if thinking_len - current_mean_len > 0 else -1) * length_reward_slope
        else:  # 均值还不够长，越长越好
            length_reward = (1 if thinking_len - current_mean_len > 0 else -1) * length_reward_slope

        # 不管够不够长，太长的集体集体打压， * 4 这样14k的反而更高分，
        if thinking_len > config.data.max_response_length - punish_overlong_interval:
            length_reward -= length_reward_slope * 4

        if config.algorithm.inference_scaling == "stepv1_all":
            correct_reward = length_reward + correct_reward
        elif config.algorithm.inference_scaling == "stepv1_correct":
            if correct_reward > 0:
                correct_reward = length_reward + correct_reward
        elif config.algorithm.inference_scaling == "stepv1_wrong":
            if correct_reward < 0:
                correct_reward = length_reward + correct_reward
        else:
            raise NotImplementedError
    elif config.algorithm.inference_scaling == "v0":
        length_reward = 0
    else:
        raise NotImplementedError(config.algorithm.inference_scaling)

    if "_all" in config.algorithm.inference_scaling:
        length_reward = length_reward
    elif "_correct" in config.algorithm.inference_scaling:
        if correct_reward < 0:
            length_reward = 0
    elif "_wrong" in config.algorithm.inference_scaling:
        if correct_reward > 0:
            length_reward = 0

    # overlong punishment
    overlong_reward = 0
    if config.algorithm.overlong_punish == 'v1':
        overlong_length = config.data.max_response_length - config.algorithm.overlong_punish_cache
        # overlong_length -> config.data.max_response_length, 0 -> -1
        if thinking_len > overlong_length:
            overlong_reward = -(thinking_len - overlong_length) / (config.data.max_response_length - overlong_length)
    if overlong_reward != 0:  # 如果有overlong reward，直接返回correct_reward + overlong reward，不要length reward
        return 0, overlong_reward

    return length_reward, overlong_reward


# # copied from shengding
# def add_length_reward(thinking_len, correct_reward, config):
#     if config.reward_model.length_reward == "v1":
#         ratio = 0
#         enhance = 2.0 if correct_reward > 0 else 1.0
#         length_benefit = (thinking_len - ratio * config.data.max_response_length)/(1-ratio)/config.data.max_response_length
#         correct_reward = correct_reward * max(1 + length_benefit*enhance, 1)
#         return correct_reward
#     elif config.reward_model.length_reward == "v2":
#         ratio = 0.5
#         enhance = 2.0 if correct_reward > 0 else 0.0
#         length_benefit = (thinking_len - ratio * config.data.max_response_length)/(1-ratio)/config.data.max_response_length
#         correct_reward = correct_reward * max(1 + length_benefit*enhance, 1)
#         return correct_reward
#     elif config.reward_model.length_reward == "v3":
#         ratio = 0.0
#         enhance = 2.0 if correct_reward > 0 else 0.0
#         length_benefit = (thinking_len - ratio * config.data.max_response_length)/(1-ratio)/config.data.max_response_length
#         correct_reward = correct_reward * max(1 + length_benefit*enhance, 1)
#         return correct_reward
#     elif config.reward_model.length_reward == "none":
#         return correct_reward
#     else:
#         raise NotImplementedError

# def punish_format(solution_ids, final_reward, format_tokens, config):
#     punish_not_think_response = config.reward_model.punish_not_think_response
#     punish_response_over_long = config.reward_model.punish_response_over_long
#     response_interval_max_len = config.reward_model.response_interval_max_len
#     eot = torch.where(solution_ids == format_tokens[1])[0]
#     bor = torch.where(solution_ids == format_tokens[2])[0]
#     eor = torch.where(solution_ids == format_tokens[3])[0]

#     if punish_not_think_response < 0:
#         correct1 = solution_ids[0] == format_tokens[0] and \
#                   solution_ids[-1] == format_tokens[3] and \
#                   len(bor) > 0 and \
#                   len(eot) > 0 and \
#                   bor - eot == 1

#         if not correct1:
#             final_reward += punish_not_think_response

#     if punish_response_over_long < 0:
#         correct2 = len(eor) > 0 and len(bor) > 0 and eor - bor < response_interval_max_len
#         if not correct2:
#             final_reward += punish_response_over_long
#     return final_reward


def punish_format_return_positions(text, config):
    pattern = re.compile(
        r'^(?P<leading>\n{0,2})'  # optional leading whitespace
        r'(?P<thinking_open><(thinking|think|\|object_ref_start\|)>)'
        # Capture the thinking body, but fail if we see
        # <thinking>, </thinking>, <answer>, or </answer> inside it again:
        r'(?P<thinking_body>(?:(?!<thinking>|</thinking>|<answer>|</answer>).)*)'
        r'(?P<thinking_close><(/thinking|/think|\|object_ref_end\|)>\n)'
        r'(?P<answer_open><(answer|\|box_start\|)>)'
        # Capture the answer body, with the same restriction:
        r'(?P<answer_body>(?:(?!<thinking>|</thinking>|<answer>|</answer>).)*)'
        r'(?P<answer_close><(/answer|\|box_end\|)>\n{0,2})'
        r'(<\|endoftext\|>|<\|im_end\|>)'
        r'$',
        re.DOTALL)

    match = pattern.match(text)
    if not match:
        return config.reward_model.format_punish_score, None  # Does not match the required format

    # Get positions of each captured group
    positions = {
        'thinking_open': (match.start('thinking_open'), match.end('thinking_open')),
        'thinking_close': (match.start('thinking_close'), match.end('thinking_close')),
        'answer_open': (match.start('answer_open'), match.end('answer_open')),
        'answer_close': (match.start('answer_close'), match.end('answer_close')),
    }

    return 0, [
        positions['thinking_open'][0], positions['thinking_close'][0], positions['answer_open'][0],
        positions['answer_close'][0]
    ]


# def punish_format(generation, config):
#     """
#     return: reward, [bot, eot, bor, eor]
#     """

#     pause_tokens = {'thinking': ['<thinking>', '</thinking>'], 'response': ['<answer>', '</answer>']}
#     pause_tokens_all = pause_tokens['thinking'] + pause_tokens['response']

#     bot = [m.start() for m in re.finditer(pause_tokens['thinking'][0], generation)]
#     eot = [m.start() for m in re.finditer(pause_tokens['thinking'][1], generation)]
#     bor = [m.start() for m in re.finditer(pause_tokens['response'][0], generation)]
#     eor = [m.start() for m in re.finditer(pause_tokens['response'][1], generation)]

#     if len(bot) != 1 or len(eot) != 1 or len(bor) != 1 or len(eor) != 1:  # 如果特殊token不在里面或者出现大于1次就惩罚
#         return config.reward_model.format_punish_score, None

#     bot, eot, bor, eor = bot[0], eot[0], bor[0], eor[0]

#     if (eot - bot) < (eor - bor):  # 思考比答案短，罚
#         return config.reward_model.format_punish_score, None

#     if len(generation) - eor > 20:  # </answer>不在最后，罚
#         return -1, None

#     if bot > 5 or bor - eot > 20:
#         return config.reward_model.format_punish_score, None
#     return 0, [bot, eot, bor, eor]

#     if punish_not_think_response < 0:
#         correct1 = solution_ids[0] == format_tokens[0] and \
#                   solution_ids[-1] == format_tokens[3] and \
#                   len(bor) > 0 and \
#                   len(eot) > 0 and \
#                   bor - eot == 1

#         if not correct1:
#             final_reward += punish_not_think_response

#     if punish_response_over_long < 0:
#         correct2 = len(eor) > 0 and len(bor) > 0 and eor - bor < response_interval_max_len
#         if not correct2:
#             final_reward += punish_response_over_long
#     return final_reward
