import re

from word2number import w2n


def calculate_iou(box1, box2):
    '''计算两个框 box1 和 box2 的 IoU。每一个框的格式是 [x1, y1, x2, y2]。'''
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / (box1_area + box2_area - intersection)
    return iou


def is_truncated(input_string):
    truncate_cases = ['<', '<p', '<po', '<poi', '<poin', '<point', '<b', '<bb', '<bbo', '<bbox', '<bbox']
    if "<point>" in input_string or '<bbox>' in input_string:
        return True
    for truncate_case in truncate_cases:
        if input_string.endswith(truncate_case):
            return True
    return False


def extract_and_convert_number(input_string):
    number_of_points = 0
    if "<point>" in input_string:
        point_patterns = re.findall('<point>.*?\<\/point>', input_string)
        # 获取输出的点的数量，防止超长的问题出现
        number_of_points = len(point_patterns)

        # 定义一个包含数字词的元组
        pattern = re.compile(r'<point>.*?</point>', re.IGNORECASE | re.DOTALL)
        # 使用空字符串替换匹配的模式
        input_string = re.sub(pattern, '', input_string)
    elif "<bbox>" in input_string:
        point_patterns = re.findall('<bbox>.*?\<\/bbox>', input_string)
        # 获取输出的点的数量，防止超长的问题出现
        number_of_points = len(point_patterns)

        # 定义一个包含数字词的元组
        pattern = re.compile(r'<bbox>.*?</bbox>', re.IGNORECASE | re.DOTALL)
        # 使用空字符串替换匹配的模式
        input_string = re.sub(pattern, '', input_string)

    # 确定是否有截断问题
    truncated = is_truncated(input_string)
    if truncated:
        return number_of_points, True

    number_words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
                    "million", "billion", "trillion")

    # 构建正则表达式模式
    pattern = re.compile(r'\b(?:' + '|'.join(number_words) + r'|\d+)\b', re.IGNORECASE)
    digit_pattern = re.compile(r'\d+')
    matches = digit_pattern.findall(input_string)
    matches.extend(pattern.findall(input_string))
    for match in matches:
        try:
            # 尝试将英文形式的数字转为阿拉伯数字
            arabic_number = w2n.word_to_num(match.lower())
            return arabic_number, False
        except ValueError:
            try:
                # 如果是阿拉伯数字则直接转换
                return int(match), False
            except ValueError:
                continue

    # 如果没有匹配到数字，则返回0
    return number_of_points, True


def remove_bold(text):
    # 去除加粗 **text** 或 __text__
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    return text


def get_valid_options(options):
    valid_options = []
    if options is not None:
        for i in range(len(options)):
            valid_options.append(chr(ord('A') + i))
        return valid_options
    else:
        return ['A', 'B', 'C', 'D', 'E', 'F']


def extract_option(text, options=['A', 'B', 'C', 'D', 'E', 'F']):
    text = remove_bold(text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text in options:
        return text
    options_str = "|".join(options)

    # start with option, e.g. '(A) xxxx' or '(A)'
    pattern = re.compile(f"^\(?({options_str})\)?\.?(\ |$)")
    answer = re.findall(pattern, text)
    if len(answer):
        return answer[0][0]

    # have option in \boxed{}. e.g. '\boxed{A. dasdasd}'
    if '\\boxed' in text:
        content = re.findall(r'\\boxed\{(.+?)\}', text)
        if content:
            pattern = re.compile(f"^\(?({options_str})\)?\.?(\ |$)")
            answer = re.findall(pattern, content[0].strip())
            if len(answer):
                return answer[0][0]

    # have option with valid English prefix, e.g. 'the best answer is A xxxx'
    pattern = re.compile(f"([Aa]nswer|[Oo]ption|[Cc]hoice)(\ is|:|\ is:)\ ?\(?({options_str})\)?")
    answer = re.findall(pattern, text)
    if len(answer):
        return answer[0][-1]

    # have option with valid Chinese prefix, e.g. '最佳选项是：(A)'
    pattern = re.compile(f"(回答|选择|选项|答案)(是|：|是：)\ ?\(?({options_str})\)?")
    answer = re.findall(pattern, text)
    if len(answer):
        return answer[0][-1]
    return ""


def extract_boxed_number(input_string):
    """
    从字符串中提取 \boxed{} 中的数字

    参数:
        input_string (str): 输入字符串

    返回:
        str: 花括号中提取的数字，如果没有匹配项，则返回 None
    """
    # 定义匹配 \boxed{} 中数字的正则表达式模式
    pattern = r'\\boxed\{(\d+)\}'

    # 搜索匹配项
    match = re.search(pattern, input_string)

    # 如果找到匹配项，返回匹配的数字
    if match:
        return match.group(1)
    else:
        return None


def union_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return list()
    intervals.sort(key=lambda x: x[0])
    merged = []
    current_start, current_end = intervals[0]
    for i in range(1, len(intervals)):
        next_start, next_end = intervals[i]
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))
    return merged


def intersect_intervals(
    list_a: list[tuple[float, float]],
    list_b: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    intersection = list()
    i = j = 0
    while i < len(list_a) and j < len(list_b):
        a_start, a_end = list_a[i]
        b_start, b_end = list_b[j]
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        if overlap_start < overlap_end:
            intersection.append((overlap_start, overlap_end))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return intersection


def get_total_length(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def get_base_precision_recall(gt: list[tuple[float, float]], pred: list[tuple[float, float]]) -> dict:
    gt_union = union_intervals(gt)
    gt_total_time = get_total_length(gt_union)

    pred_total_cost = sum(end - start for start, end in pred)

    if gt_total_time == 0 and pred_total_cost == 0:
        return {'precision': 1.0, 'recall': 1.0}
    if gt_total_time == 0 or pred_total_cost == 0:
        return {'precision': 0.0, 'recall': 0.0}

    pred_union = union_intervals(pred)
    intersection = intersect_intervals(gt_union, pred_union)
    tp_time = get_total_length(intersection)

    recall = tp_time / gt_total_time
    precision = tp_time / pred_total_cost

    return {'precision': precision, 'recall': recall}
