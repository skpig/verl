import ast
import re

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult


class GUIVerifier(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        gt_content = verifier_feature_dict['answer']
        item_value_correct = rule_for_action(gt_content, response)

        if item_value_correct:
            return VerifyResult(score=1.0, extracted_answer=response)
        else:
            return VerifyResult(score=0.0, extracted_answer=response)


# 正则表达式预编译可加速
re_zh = re.compile(r'[\u4e00-\u9fff]')
re_en = re.compile(r'[A-Za-z]')


def detect_lang(text: str, thresh: float = 0.7) -> str:
    """
    根据字符占比判断文本主要语言（中文 zh、英文 en、混合 mixed、未知 unknown）
    thresh: 判定“主要语言”的比例阈值；0.7 表示 ≥70% 即视为主导语言
    """
    zh_cnt = len(re_zh.findall(text))
    en_cnt = len(re_en.findall(text))
    total = zh_cnt + en_cnt
    if total == 0:
        return 'unknown'

    zh_ratio = zh_cnt / total
    en_ratio = en_cnt / total

    if en_ratio >= thresh:
        return 'en'
    else:
        return 'zh'


def get_pure_text(text):
    # 去除加粗 **text** 或 __text__
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    return text


def get_truth_action_type_value(content):
    content = get_pure_text(content)
    pattern = r"Action: (\w+)\((.*)\)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        action_type = match.group(1)  # 提取 action type (click)
        action_value = match.group(2)
        if action_type == 'type':
            # 抽取文本答案
            content_pattern = r"content='(.*?)'"
            content_match = re.search(content_pattern, action_value, re.DOTALL)
            line_content = content_match.group(1) if content_match else None
            # 抽取bbox
            bbox = []
            value_pattern = r"point='<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                numbers = re.findall("\d+", value_match.group(1))
                if len(numbers) == 2:
                    bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000
                    ]
                elif len(numbers) == 4:
                    bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[2]) / 1000,
                        float(numbers[3]) / 1000
                    ]
            return action_type, (line_content, bbox)

        elif action_type == 'drag':
            # 抽取start_bbox
            start_bbox = []
            value_pattern = r"start[_ ]point='<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                numbers = re.findall("\d+", value_match.group(1))
                if len(numbers) == 2:
                    start_bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000
                    ]
                elif len(numbers) == 4:
                    start_bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[2]) / 1000,
                        float(numbers[3]) / 1000
                    ]
            # 抽取end_bbox
            end_bbox = []
            value_pattern = r"end[_ ]point='<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                numbers = re.findall("\d+", value_match.group(1))
                if len(numbers) == 2:
                    end_bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000
                    ]
                elif len(numbers) == 4:
                    end_bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[2]) / 1000,
                        float(numbers[3]) / 1000
                    ]
            return action_type, (start_bbox, end_bbox)

        elif action_type == 'scroll':
            # 确定GUI的方向
            direction_pattern = r"direction='(.*?)'"
            direction_match = re.search(direction_pattern, action_value, re.DOTALL)
            direction = direction_match.group(1) if direction_match else None
            # 确定bbox
            bbox = []
            value_pattern = r"point='<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                numbers = re.findall("\d+", value_match.group(1))
                if len(numbers) == 2:
                    bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000
                    ]
                elif len(numbers) == 4:
                    bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[2]) / 1000,
                        float(numbers[3]) / 1000
                    ]
            return action_type, (direction, bbox)
        else:
            value_pattern = r"point='<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                numbers = re.findall("\d+", value_match.group(1))
                if len(numbers) == 2:
                    bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000
                    ]
                    return action_type, bbox
                elif len(numbers) == 4:
                    bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[2]) / 1000,
                        float(numbers[3]) / 1000
                    ]
                    return action_type, bbox

            value_pattern = r"'<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                quoted_value = value_match.group(1)  # 提取引号中的内容
                return action_type, quoted_value
            else:
                # print(f"truth doesn't found match value:{content}")  #finished wait
                return action_type, action_value
    else:
        try:
            final_value = ast.literal_eval(content)
            final_value = [final_value[0], final_value[1], final_value[0], final_value[1]]
            return "click", final_value
        except Exception as e:
            print("[ERROR]", content)


def get_pred_action_type_value(content):
    content = get_pure_text(content)
    pattern = r"Action: (\w+)\((.*)\)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        action_type = match.group(1)  # 提取 action type (click)
        action_value = match.group(2)
        if action_type == 'type':
            # 抽取文本答案
            content_pattern = r"content='(.*?)'"
            content_match = re.search(content_pattern, action_value, re.DOTALL)
            line_content = content_match.group(1) if content_match else None
            # 抽取bbox
            bbox = []
            value_pattern = r"point='<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                numbers = re.findall("\d+", value_match.group(1))
                if len(numbers) >= 2:
                    bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000
                    ]
            return action_type, (line_content, bbox)
        elif action_type == 'drag':
            # 抽取start_bbox
            start_bbox = []
            value_pattern = r"start[_ ]point='<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                numbers = re.findall("\d+", value_match.group(1))
                if len(numbers) >= 2:
                    start_bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000
                    ]
            # 抽取end_bbox
            end_bbox = []
            value_pattern = r"end[_ ]point='<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                numbers = re.findall("\d+", value_match.group(1))
                if len(numbers) >= 2:
                    end_bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000
                    ]
            return action_type, (start_bbox, end_bbox)
        elif action_type == 'scroll':
            # 确定滑动的方向
            direction_pattern = r"direction='(.*?)'"
            direction_match = re.search(direction_pattern, action_value, re.DOTALL)
            direction = direction_match.group(1) if direction_match else None
            # 确定bbox
            bbox = []
            value_pattern = r"point='<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                numbers = re.findall("\d+", value_match.group(1))
                if len(numbers) >= 2:
                    bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000
                    ]
            return action_type, (direction, bbox)
        else:
            value_pattern = r"point='<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                numbers = re.findall("\d+", value_match.group(1))
                if len(numbers) >= 2:
                    bbox = [
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000,
                        float(numbers[0]) / 1000,
                        float(numbers[1]) / 1000
                    ]
                    return action_type, bbox

            value_pattern = r"'<point>(.*?)</point>'"
            value_match = re.search(value_pattern, action_value)
            if value_match:
                quoted_value = value_match.group(1)  # 提取引号中的内容
                return action_type, quoted_value
            else:
                # print(f"truth doesn't found match value:{content}")  #finished wait
                return action_type, action_value
    else:
        # print(f"pred doesn't have match action:{content}")
        return None, None


def cal_distance(golden_cord, pred_cord):
    golden_point = [(golden_cord[0] + golden_cord[2]) / 2, (golden_cord[1] + golden_cord[3]) / 2]
    pred_point = [(pred_cord[0] + pred_cord[2]) / 2, (pred_cord[1] + pred_cord[3]) / 2]
    distance = ((golden_point[0] - pred_point[0])**2 + (golden_point[1] - pred_point[1])**2)**0.5
    return distance


def metric_distance(gt_bbox, pred_bbox):
    try:
        dis = cal_distance(gt_bbox, pred_bbox)
        dis_value_error = False
    except Exception as e:
        # print(f"cal dis error:{e}, gt_value:{gt_bbox}, pred_value:{pred_bbox}, pred_ori:{pred_line}")
        dis_value_error = True
        dis = 100
    return dis, dis_value_error


def rule_for_action(gt_content, pred_content):
    from rouge_chinese import Rouge
    scorer = Rouge()
    dis_threshold = 0.05
    item_rouge_score = None
    item_dis = None
    item_type_correct = False
    item_value_correct = False
    if 'Thought: ' in gt_content and 'Thought: ' not in pred_content or 'Thought: ' in gt_content and 'Action:' in pred_content and detect_lang(
            pred_content.split('Action:')[0]) != "zh":
        print('thought lang error')
        item_value_correct = False
        return item_value_correct

    gt_result = get_truth_action_type_value(gt_content)
    if gt_result is not None:
        gt_type, gt_value = gt_result
    else:
        return False

    pred_type, pred_value = get_pred_action_type_value(pred_content)
    # print(gt_type, gt_value, pred_type, pred_value)
    if pred_type == "left_single":
        pred_type = "click"
    if gt_type == pred_type and gt_type is not None:
        item_type_correct = True
        if gt_type.lower() in ['click', 'select', 'hover', 'right_single', 'left_double', 'left_single']:
            gt_bbox = gt_value
            pred_bbox = pred_value
            item_dis, dis_value_error = metric_distance(gt_bbox, pred_bbox)
            if item_dis < dis_threshold:
                item_value_correct = True
        elif gt_type.lower() == 'scroll':
            gt_direction = gt_value[0]
            pred_direction = pred_value[0]
            gt_bbox = gt_value[1]
            pred_bbox = pred_value[1]
            if gt_bbox != []:
                item_dis, dis_value_error = metric_distance(gt_bbox, pred_bbox)
                if item_dis < dis_threshold and gt_direction == pred_direction:
                    item_value_correct = True
            else:
                if gt_direction == pred_direction:
                    item_value_correct = True

        elif gt_type.lower() == 'drag':
            gt_start_bbox = gt_value[0]
            pred_start_bbox = pred_value[0]
            gt_end_bbox = gt_value[1]
            pred_end_bbox = pred_value[1]
            gt_bbox = [gt_start_bbox, gt_end_bbox]
            pred_bbox = [pred_start_bbox, pred_end_bbox]
            dis_start, dis_start_value_error = metric_distance(gt_start_bbox, pred_start_bbox)
            dis_end, dis_end_value_error = metric_distance(gt_end_bbox, pred_end_bbox)
            item_dis = (dis_start + dis_end) / 2
            dis_value_error = dis_start_value_error or dis_end_value_error
            if dis_start < dis_threshold and dis_end < dis_threshold:
                item_value_correct = True
        elif gt_type.lower() in ['navigate_back', 'navigate_home', 'enter', 'wait', 'finished']:
            item_value_correct = True
        elif gt_type.lower() == 'type':
            import jieba
            gt_content = gt_value[0]
            gt_bbox = gt_value[1]
            pred_content = pred_value[0]
            pred_bbox = pred_value[1]
            gt_content_seg = ' '.join(jieba.cut(str(gt_content)))
            pred_content_seg = ' '.join(jieba.cut(str(pred_content)))
            if gt_content_seg == '\n':
                gt_content_seg = '\\n'
            if pred_content_seg == '\n':
                pred_content_seg = '\\n'
            try:
                scores = scorer.get_scores(gt_content_seg, pred_content_seg)
                item_rouge_score = scores[0]['rouge-l']['f']
            except:
                item_rouge_score = 0
            if item_rouge_score >= 0.5:
                item_value_correct = True
            else:
                item_value_correct = False
        else:
            if gt_value == pred_value:
                item_value_correct = True
    return item_value_correct


if __name__ == "__main__":
    # import pandas as pd
    # import json
    print('##################Starting##################')

    # input_file = "/mnt/bn/fangjunjiedev/m8_20b_trainingData/datasets/sft_interactive_Agent_Instruction_Follow_human_ch_RLFormatSample2k_MaxLen7k_history1_nothink_v3_0607.parquet"
    # input_file = "/mnt/bn/fangjunjiedev/m8_20b_trainingData/datasets/processed_parquet_chunks_0620_RLFormatSample50_MaxLen7k_history1_longcot_v3_2025-06-23_fixedAnswerType.parquet"
    # thought_gt = "Thought: 首先，我需要确定当前处于PDF阅读器界面，而任务是找论文作者的个人网页。但是我还不知道作者是谁，因此我第一步需要做的是向上滚动PDF直至出现作者名字。\nAction: scroll(direction='up', point='<point>824 513</point>')"
    # think_gt = '<think>Looking at this document, I can see that I\'ve successfully created a line break after the first sentence "The tutorial will constitute 30% of your total grade." But now I notice there\'s something problematic - there appears to be a bullet point or formatting element that\'s been carried over to the new line. I can see there\'s a small square or bullet point symbol at the beginning of the empty line I just created.\n\nThis isn\'t what I want. The user asked me to create clean separations between sentences to improve readability, not to introduce additional formatting elements that could make the document look messy or inconsistent. I need to remove this unwanted bullet point or formatting symbol.\n\nLet me think about what happened here. When I pressed Enter after the first sentence, the word processor probably automatically continued the bullet point formatting from the previous line, which is a common behavior in document editors like LibreOffice Writer. This created an empty bullet point on the new line.\n\nTo fix this, I should use the backspace key to remove this unwanted formatting element. This will give me a clean empty line between the sentences, which is exactly what the user requested for better readability.\n\nAfter I remove this formatting issue, I can continue with the rest of the assessment section, making sure to place the cursor at the end of each sentence and create proper line breaks without introducing unwanted formatting elements.</think>Action: hotkey(key=\'backspace\')'
    # data = pd.read_parquet(input_file).to_dict("records")

    # gui_verifier = GUIVerifier()
    # for i, d in enumerate(data):
    #     verifier_feature = json.loads(d['session']['verifier_feature'])
    #     gt = verifier_feature['answer']
    #     #用老数据时的过渡使用
    #     gt = gt.replace("<bbox>","<point>").replace("</bbox>","</point>")
    #     if "drag" in gt:
    #         gt = gt.replace("start_box", "start_point").replace("end_box", "end_point")
    #     else:
    #         gt = gt.replace("start_box", "point")
    #     pred = thought_gt
    #     verifier_feature['answer'] = gt
    #     res = gui_verifier.verify(pred, verifier_feature_dict=verifier_feature)
    #     print(i, ' - ', res.score)
    #     status = json.dumps({'tag': 'verified', 'pred': res.extracted_answer, 'answer': gt, 'score': res.score},
    #                         ensure_ascii=False)
    # if res.score == 0:
    #     print(f"----{i}----")
    # print(f'[VERIFIER INFO] {status}', flush=True)
