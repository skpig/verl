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
