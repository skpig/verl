import base64
import io
import json
import re

import numpy as np
from PIL import Image
from skimage.draw import line

from alpha_seed.prompts.think_template_utils import get_special_tokens_dict_or_name
from alpha_seed.utils.reward_score.vlm_verifiers.extra_reward import match_visual_cot_format
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, ExtractAnswerFailed
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import VerifierFailed
from alpha_seed.utils.reward_score.vlm_verifiers.utils import get_valid_visual_tool_calls


def get_maze_bounds(gray_array, white_threshold=250):
    mask = gray_array < white_threshold  # 非白区域
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise VerifierFailed(message="未检测到迷宫区域")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, x_max, y_min, y_max


def is_valid_step(gray_array, point1, point2, threshold=128):
    valid = True
    reason = "合法"

    height, width = gray_array.shape

    x1, y1 = int(point1[0] * width / 1000), int(point1[1] * height / 1000)
    x2, y2 = int(point2[0] * width / 1000), int(point2[1] * height / 1000)

    # 获取迷宫有效区域边界
    x_min, x_max, y_min, y_max = get_maze_bounds(gray_array)

    # 路径像素点判断
    rr, cc = line(y1, x1, y2, x2)

    rr = np.clip(rr, 0, height - 1)
    cc = np.clip(cc, 0, width - 1)

    # 越界
    if not (0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height):
        return False, "起点终点不在图像范围内"

    # 起点终点在边界外（外部白边区域）
    for (x, y) in [(x1, y1), (x2, y2)]:
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return False, "起点终点在外围白边区域"
        if gray_array[y, x] < threshold:
            return False, "起点终点在墙上"

    for r, c in zip(rr, cc):
        if not (x_min <= c <= x_max and y_min <= r <= y_max):
            valid = False
            reason = "路径穿过外围白边区域"
            break
        if gray_array[r, c] < threshold:
            valid = False
            reason = "路径穿过墙"
            break

    return valid, reason


def extract_points(point_sequence):
    # Pattern to match <point>x y</point> format
    pattern = r'<point>(.*?)</point>'

    # Find all matches
    matches = re.findall(pattern, point_sequence)

    # Convert each match to [x, y] coordinates
    points = []
    for match in matches:
        # Split by whitespace and convert to float
        coords = match.strip().split()
        if len(coords) == 2:
            x, y = int(coords[0]), int(coords[1])
            points.append([x, y])
        else:
            print('[XYDEUG.INVALID POINT STR] match: {}, point_sequence: {}'.format(match, point_sequence))

    return points


def get_valid_point_seqs(response):
    last_call_valid, last_call_str, valid_tool_calls = get_valid_visual_tool_calls(response, num_input_images=1)
    if not last_call_valid:
        return []
    cur_tool = valid_tool_calls[-1]
    if cur_tool['name'] != 'POINT':
        return []

    final_points_str = []
    recur_cnt = 0
    # gather ploted point sequences
    while cur_tool:
        recur_cnt += 1
        if recur_cnt >= 20:
            print('[XYDEUG.MAZE_VERIFER_FAIL] recur_cnt: {}, response:{}'.format(recur_cnt, response))
            break

        if cur_tool['name'] != 'POINT':
            cur_tool = valid_tool_calls[cur_tool['parameters']['imgidx']]
            continue

        cur_imgidx = cur_tool['parameters']['imgidx']

        if ('imgidx' not in cur_tool['parameters']) or (cur_imgidx < 0) or (cur_imgidx >= len(valid_tool_calls)):
            raise ExtractAnswerFailed("invalid fc params")

        # only record POINT FC
        final_points_str.append(cur_tool['parameters']['points'])
        prev_imgidx = cur_tool['parameters']['imgidx']
        cur_tool = valid_tool_calls[prev_imgidx]

    final_points_str = final_points_str[::-1]

    # remove repeat adjacent points
    point_list = []
    for point_str in final_points_str:
        point_seq_ = extract_points(point_str)
        for i in range(len(point_seq_)):
            if not point_list or point_list[-1] != point_seq_[i]:
                point_list.append(point_seq_[i])

    return point_list


def wall_judge(response, image_base64, answer, maze_size="5"):

    results_gt = list(map(int, re.findall(r"\d+", answer)))
    results_gt_new = []
    for i in range(0, len(results_gt), 2):
        results_gt_new.append([results_gt[i], results_gt[i + 1]])

    point_list_new = get_valid_point_seqs(response)
    if len(point_list_new) < 1:
        return 0

    # load wall info
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    gray_array = np.array(image.convert('L'))

    # judge if crossing wall
    last_point = point_list_new[0]
    for i in range(1, len(point_list_new)):
        valid, reason = is_valid_step(gray_array, point_list_new[i - 1], point_list_new[i])
        if not valid:
            # last_point = point_list_new[i - 1]
            # The reward for crossing a wall is 0.
            return 0

        if i == len(point_list_new) - 1:
            last_point = point_list_new[i]

    # find the closest steps in gt
    index = 0

    if maze_size == "5":
        min_delta = 100
    elif maze_size == "3":
        min_delta = 167
    elif maze_size == "7":
        min_delta = 71
    elif maze_size == "9":
        min_delta = 56
    else:
        raise NotImplementedError

    for i, (px, py) in enumerate(results_gt_new):
        dist = np.sqrt((px - last_point[0])**2 + (py - last_point[1])**2 + 0.01)
        if dist < min_delta:
            min_delta = dist
            index = i

    return index / (len(results_gt_new) - 1)


class VisualCoTVerifier_Maze(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        # 检查FC格式错误
        if not match_visual_cot_format(response, verifier_feature=verifier_feature_dict):
            raise ExtractAnswerFailed

        # query = verifier_feature_dict['query']
        answer = verifier_feature_dict['answer']

        if 'maze_size' in verifier_feature_dict:
            maze_size = verifier_feature_dict['maze_size']
        else:
            maze_size = json.loads(verifier_feature_dict['extra_info'])['maze_size']

        image_base64 = verifier_feature_dict['image_base64']

        eot_token = get_special_tokens_dict_or_name("think_end_token")
        eos_token = get_special_tokens_dict_or_name("eos")
        final_answer: str = response.split(eot_token)[-1].split(eos_token)[0]

        if not final_answer.strip():
            raise ExtractAnswerFailed

        score_wall = wall_judge(response, image_base64, answer, maze_size)
        # TODO: add supervision on final output?

        return VerifyResult(score=score_wall, extracted_answer=response)
