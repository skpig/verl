import re
import json


def _rm(text, key_pair):
    result_text = []  # 存储去除标签后的文本
    think_contents = []  # 存储标签内的内容
    start = 0
    text_length = len(text)
    while start < text_length:
        think_start = text.find(key_pair[0], start)
        if think_start == -1:
            result_text.append(text[start:])
            break

        think_end = text.find(key_pair[1], think_start)
        if think_end == -1:
            think_contents.append(text[start:])
            result_text.append("思考过程过长，被截断")
            break

        # 添加标签前的文本
        result_text.append(text[start:think_start])

        # 提取并存储标签内的内容 (去除<think>和</think>标签)
        content_start = think_start + len(key_pair[0])  # <think> 的长度是7
        think_contents.append(text[content_start:think_end])

        start = think_end + len(key_pair[1])  # </think> 的长度是8

    return ''.join(result_text), think_contents


def compute_score(solution_str, ground_truth, **argv) -> float:

    ####先删除思考内容，再进行答案判断

    result_text, _ = _rm(solution_str, key_pair=('<think>', '</think>'))
    result_text, _ = _rm(result_text, key_pair=('<doubaothinking>', '</doubaothinking>'))

    try:

        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)
        gt_files = ground_truth
        fp = 0
        fn = 0
        tp = 0
        used = set()

        pattern = re.compile(r'```(.*?)```', re.DOTALL)
        match = pattern.search(result_text)
        if match:
            response = match.group(1).split('\n')
        else:
            response = result_text.replace('```', '').split('\n')
        for line in response:
            valid = False
            if line.strip():
                for gt_file in gt_files:
                    if gt_file in line and gt_file not in used:
                        tp += 1
                        valid = True
                        used.add(gt_file)
                        break
                if not valid:
                    fp += 1
        for gt_file in gt_files:
            if gt_file not in used:
                fn += 1

        report = {
            'precision/simplicity': tp / (tp + fp),
            'recall/coverage': tp / len(gt_files),
        }
        report['score'] = 2 * report['precision/simplicity'] * report['recall/coverage'] / (
            report['precision/simplicity'] +
            report['recall/coverage']) if report['precision/simplicity'] + report['recall/coverage'] > 0 else 0
        return report['score']
    except:
        return -1


def test_compute_score():
    solution_str = """```
django/db/backends/postgresql/client.py
sympy/functions/elementary/trigonometric.py
sympy/core/evaluate.py
sympy/core/sympify.py
sympy/functions/special/benchmarks/bench_special.py
```
"""
    gold_str = json.dumps(["django/db/backends/postgresql/client.py"])
    print(compute_score(solution_str, gold_str))


if __name__ == '__main__':
    test_compute_score()
