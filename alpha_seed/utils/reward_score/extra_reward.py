import re


def add_length_reward(thinking_len, correct_reward, config):
    """
    ratio: 从多长开始奖励
    enhance: 最多多奖励多少
    """
    assert isinstance(thinking_len, int), thinking_len
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
    elif config.algorithm.inference_scaling == "v0":
        return correct_reward
    else:
        raise NotImplementedError(config.algorithm.inference_scaling)

    return correct_reward


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


def punish_format(generation, config):
    """
    return: reward, [bot, eot, bor, eor]
    """

    pause_tokens = {'thinking': ['<思考>', '</思考>'], 'response': ['<回复>', '</回复>']}
    pause_tokens_all = pause_tokens['thinking'] + pause_tokens['response']

    bot = [m.start() for m in re.finditer(pause_tokens['thinking'][0], generation)]
    eot = [m.start() for m in re.finditer(pause_tokens['thinking'][1], generation)]
    bor = [m.start() for m in re.finditer(pause_tokens['response'][0], generation)]
    eor = [m.start() for m in re.finditer(pause_tokens['response'][1], generation)]

    if len(bot) != 1 or len(eot) != 1 or len(bor) != 1 or len(eor) != 1:  # 如果特殊token不在里面或者出现大于1次就惩罚
        return config.reward_model.format_punish_score, None

    bot, eot, bor, eor = bot[0], eot[0], bor[0], eor[0]

    if (eot - bot) < (eor - bor):  # 思考比答案短，罚
        return config.reward_model.format_punish_score, None

    return 0, [bot, eot, bor, eor]

    if punish_not_think_response < 0:
        correct1 = solution_ids[0] == format_tokens[0] and \
                  solution_ids[-1] == format_tokens[3] and \
                  len(bor) > 0 and \
                  len(eot) > 0 and \
                  bor - eot == 1

        if not correct1:
            final_reward += punish_not_think_response

    if punish_response_over_long < 0:
        correct2 = len(eor) > 0 and len(bor) > 0 and eor - bor < response_interval_max_len
        if not correct2:
            final_reward += punish_response_over_long
    return final_reward
