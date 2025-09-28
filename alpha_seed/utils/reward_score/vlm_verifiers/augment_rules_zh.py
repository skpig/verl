import re
import json
import string
import random
import emoji
from collections import Counter
import locale
from zhon.hanzi import punctuation
import hanzidentifier
#pip install hanzidentifier
import pkuseg

pku_seg = pkuseg.pkuseg()

################################################
# DEFINE RULE TYPES
################################################

ZH_RULE_TYPES = [
    #语言类
    "LANGUAGE_ONLY",
    "LANGUAGE_CONTAIN",
    "LANGUAGE_NUM",
    "LANGUAGE_LOC",
    # "LANGUAGE_NUM_LOC", #表意不清晰，暂时舍去
    "LANGUAGE_NEGATE",
    #中文文本类
    "TEXT_ONLY",
    "TEXT_CONTAIN",
    "TEXT_NEGATE",
    "TEXT_CONTAIN_LOC",
    "TEXT_CONTAIN_NUM",
    "TEXT_START_END",
    "TEXT_START_END_NEGATE",
    "TEXT_START_LOC",
    "TEXT_END_LOC",
    "TEXT_END_LOC_NEGATE",
    "TEXT_REPLACE",
    #英文字母类
    "LETTER_CONTAIN",
    "LETTER_NUM",
    "LETTER_CASE_ONLY",
    "LETTER_CASE_CONTAIN",
    "LETTER_CASE_LOC",
    "LETTER_CASE_NEGATE",
    #重复类
    "REPEAT_TEXT_NEGATE",
    "REPEAT_LOC_NEGATE",
    "REPEAT_TEXT_LOC_NEGATE",
    #emoji类
    "EMOJI_NUM",
    "EMOJI_ONLY",
    "EMOJI_CONTAIN",
    "EMOJI_NEGATE",
    "EMOJI_START_END",
    "EMOJI_NUM_START_END",
    "EMOJI_NUM_LOC",
    #字数类
    "ZH_NUM",
    "ZH_NUM_LOC",
    #标点类
    "PUNC_NUM",
    "PUNC_NUM_LOC",
    "PUNC_REPLACE",
    "PUNC_NEGATE",
    "PUNC_CONTAIN",
    "PUNC_END",
    "PUNC_END_NEGATE",
    "PUNC_END_LOC",
    #数字类
    "NUMBER_NEGATE",
    "NUMBER_CONTAIN",
    "NUMBER_START_END",
    "NUMBER_TYPE",
    "DECIMAL_TYPE",
    #句子类型
    "SENT_TYPE_ONLY",
    "SENT_TYPE_CONTAIN",
    "SENT_TYPE_NUM",
    "SENT_TYPE_START_END",
    "SENT_TYPE_START_END_NEGATE",
    "SENT_TYPE_NEGATE",
    "SENT_TYPE_REPLACE",
    #句子数量
    "SENT_NUM",
    "SENT_NUM_LOC",
    #段落数量
    "PARA_NUM",
    #格式类
    "FORMAT_JSON",
    "FORMAT_SEP",
    "FORMAT_TAG_NEGATE",
    "FORMAT_SEP_PIECE",
    "FORMAT_SEP_NEWLINE",
    "FORMAT_WRAP",
    #加粗类
    "BOLD_PARA",
    "BOLD_SENT",
    "BOLD_WORD",
    # "BOLD_HEAD", #先去掉，标题不完全是markdown格式
    "BOLD_PARA_FIRST",
    "BOLD_NEGATE",
    "BOLD_DIGIT",
    # "BOLD_LETTER", #括字母还是括整个句子不确定
    "LANGUAGE_MIX",
    "SENT_NUM_EXPAND",
    "SENT_TYPE_NUM_EXPAND",
    "FORMAT_INDEX",
    "FORMAT_TITLE",
    "ZH_NUM_TOTAL_AROUND",
]
# print(len(ZH_RULE_TYPES)) 70rules

OVERALL_INSTRUCTION_LIST = [
    """{meta_instruction}\n{instruction_ori}""",
    """{instruction_ori}\n{meta_instruction}""",
    """{meta_instruction}\n\n{instruction_ori}""",
    """{instruction_ori}\n\n{meta_instruction}""",
]


def remove_punctuation(s):
    en_punctuation_set = set(string.punctuation)
    zh_punctuation_set = punctuation
    return ''.join(char for char in s if char not in en_punctuation_set and char not in zh_punctuation_set)


def arabic_to_chinese(num):
    # 基本数字映射
    num_map = "零一二三四五六七八九"
    # 单位映射
    unit_map = ["", "十", "百", "千"]
    # 大单位映射
    big_unit_map = ["", "万", "亿"]

    if num == 0:
        return "零"

    result = ""
    str_num = str(num)
    length = len(str_num)

    is_zero = False  # 跟踪是否前面一个数字是零
    is_first_zero = False  # 跟踪是否第一个有效的零已经输出

    for i in range(length):
        digit = int(str_num[i])
        unit_index = (length - 1 - i) % 4
        big_unit_index = (length - 1 - i) // 4

        if digit == 0:
            is_zero = True
            if not is_first_zero and (len(result) == 0 or not result.endswith("零")):
                result += "零"
                is_first_zero = True
        else:
            if is_zero and result[-1] == "零":
                result = result[:-1]  # 去掉多余的 "零"
            result += num_map[digit] + unit_map[unit_index]
            is_zero = False
            is_first_zero = False

        if unit_index == 0 and i != length - 1:
            result += big_unit_map[big_unit_index]

    # 去掉末尾的 "零"
    if result[-1] == "零":
        result = result[:-1]

    # 特殊处理：去掉前面多余的 "一十"
    if result.startswith("一十") and num >= 10 and num < 20:
        result = result[1:]

    return result


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sents = para.split("\n")
    sents = [sent for sent in sents if sent != '']
    return sents


def count_emoji(text):
    count = 0
    for char in text:
        if char in emoji.EMOJI_DATA:
            count += 1
    return count


def count_punc(text):
    count = 0
    for char in text:
        if char in punctuation or char in string.punctuation:
            count += 1
    return count


def extract_chinese_characters(s):
    chinese_regex = re.compile(r'[\u4E00-\u9FFF]')
    chinese_characters = chinese_regex.findall(s)
    return ''.join(chinese_characters)


class Rule:

    def __init__(self, rule_type, meta_instr):
        self.rule_type = rule_type
        self.meta_instr = meta_instr

    def check():
        raise NotImplementedError

    def do():
        raise NotImplementedError


################################################
# 语言类
################################################


class Rule_LANGUAGE_ONLY(Rule):

    def __init__(self):
        self.rule_type = "LANGUAGE_ONLY"
        #TODO：添加只包含的表述
        self.meta_instr = [
            '用{language}回答',
            '用{language}作答',
            '请使用{language}作答',
            '请用{language}来回答',
            '用{language}进行回答',
            '始终用{language}回答',
            '采用{language}回答',
            '以{language}回答',
            '以{language}回答问题',
            '以{language}进行回答',
            '请以{language}作答',
            '用{language}作答',
            '回答时使用{language}',
            '请用{language}回答问题',
            '使用{language}进行回复',
            '只能用{language}回复',
            '用{language}回复',
            '以{language}回复',
            '语言为{language}',
            '语种为{language}',
            '语种：{language}',
            '要求用{language}说话',
            '用{language}对话',
            '用{language}输出',
            '以{language}输出',
            '{language}输出',
            '翻译成{language}',
            '翻译为{language}',
            '译成{language}',
            '译为{language}',
            '用{language}翻译',
            '用{language}写',
            '用{language}书写',
            '只能使用{language}',
            '需为{language}',
            '用{language}交流',
            '使用{language}交流',
            '使用{language}回答',
            '只包含{language}表述',
            '用{language}表述',
            '答案使用{language}',
            '回复使用{language}',
            '转化成{language}',
            '转换成{language}',
            '用{language}展示',
            '用{language}完成',
            '用{language}回应',
            '要求说{language}',
            '必须是{language}',
            '要用{language}',
            '用{language}说',
            '使用{language}',
            '{language}回答',
            '回复{language}',
            '用{language}',
            '只包含{language}',
        ]
        self.language_choice = [
            "英文", "纯英文", "英语", "中文", "纯中文", "汉语", "普通话", "繁体字", "简体中文", "西班牙语", "西班牙文", "葡萄牙语", "葡萄牙文", "法语", "法文",
            "俄语", "俄文", "德语", "德文", "日语", "日文", "韩语", "韩文", "阿拉伯语", "印地语"
        ]

    def check(self, response, slots):
        language = slots["language"]
        if language in ["英文", "纯英文", "英语"]:
            pattern = re.compile(r'^[0-9a-zA-Z\s’‘“”\'\",.?!;:()\[\]{}\-&]+$')
            return bool(pattern.match(response))
        elif language in ["中文", "纯中文", "汉语"]:
            pattern = re.compile(r'^[0-9\u4e00-\u9fa5\s，。！？；：“”‘’、（）：《》〈〉【】…—－—·]+$')
            return bool(pattern.match(response))
        elif language in ["普通话", "简体中文"]:
            chinese_punctuations = set("，。！？；：“”‘’、（）：《》〈〉【】…—－—· \n")
            for char in response:
                if not (hanzidentifier.is_simplified(char) or char in chinese_punctuations or char.isdigit()):
                    return False
            return True
        elif language in ["繁体字"]:
            chinese_punctuations = set("，。！？；：“”‘’、（）：《》〈〉【】…—－—· \n")
            for char in response:
                if not (hanzidentifier.is_traditional(char) or char in chinese_punctuations or char.isdigit()):
                    return False
            return True
        elif language in ["日语", "日文"]:
            # 平假名：\u3040-\u309F
            # 片假名：\u30A0-\u30FF
            # 拡張片假名：\u31F0-\u31FF
            # 全角标点符号：\u3000-\u303F（其中包括一些非日语字符，但能涵盖常用标点）
            # 汉字：分布范围较广，常用的是 \u4E00-\u9FAF（与中文共享）
            japanese_punctuations = set("。、，・？！ー（）{}「」『』【】《》〈〉［］…—・ー「」『』 \n")
            for char in response:
                # 检查字符是否属于以下任意一个范围
                if not (('\u3040' <= char and char <= '\u309F') or ('\u30A0' <= char and char <= '\u30FF') or
                        ('\u31F0' <= char and char <= '\u31FF') or ('\u4E00' <= char and char <= '\u9FAF') or
                        ('\u3000' <= char and char <= '\u303F') or (char in japanese_punctuations) or char.isdigit()):
                    return False
            return True
        elif language in ["韩语", "韩文"]:
            # 韩文字母（Hangul Syllables）：\uAC00-\uD7AF
            # 韩文字母（Hangul Jamo）：\u1100-\u11FF 和 \u3130-\u318F
            # Hangul Compatibility Jamo：\FFA0-\uFFDC
            korean_punctuations = set("。，、？！「」『』（）【】《》〈〉-· \n")
            for char in response:
                # 检查字符是否属于以下任意一个范围
                if not (('\uAC00' <= char and char <= '\uD7AF') or ('\u1100' <= char and char <= '\u11FF') or
                        ('\u3130' <= char and char <= '\u318F') or ('\uFFA0' <= char and char <= '\uFFDC') or
                        (char in korean_punctuations) or char.isdigit()):
                    return False
            return True
        elif language in ["西班牙语", "西班牙文"]:
            # 西班牙语字母：a-z, A-Z
            # 西班牙语重音字母：á, é, í, ó, ú, Á, É, Í, Ó, Ú,
            # 其他特殊字符：ü, Ü, ñ, Ñ
            spanish_characters = set("abcdefghijklmnopqrstuvwxyz"
                                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                     "áéíóúÁÉÍÓÚ"
                                     "üÜñÑ"
                                     " .,;:¿?¡!\"'()[]{}<>«»\n"
                                     "0123456789")
            for char in response:
                if char not in spanish_characters:
                    return False
            return True
        elif language in ["葡萄牙语", "葡萄牙文"]:
            # 葡萄牙语字母：a-z, A-Z
            # 葡萄牙语重音字母：á, é, í, ó, ú, à, è, ì, ò, ù, â, ê, î, ô, û, ã, õ, ç, Á, É, Í, Ó, Ú, À, È, Ì, Ò, Ù, Â, Ê, Î, Ô, Û, Ã, Õ, Ç
            portuguese_characters = set("abcdefghijklmnopqrstuvwxyz"
                                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "áéíóúàèìòùâêîôûãõç"
                                        "ÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇ"
                                        " .,;:!?\"'()[]{}<>«»\n"
                                        "0123456789")
            for char in response:
                if char not in portuguese_characters:
                    return False
            return True
        elif language in ["法语", "法文"]:
            # 法语字母：a-z, A-Z
            # 法语重音字母：à, â, ä, é, è, ê, ë, î, ï, ô, ö, ù, û, ü, ÿ, ç, À, Â, Ä, É, È, Ê, Ë, Î, Ï, Ô, Ö, Ù, Û, Ü, Ÿ, Ç
            french_characters = set("abcdefghijklmnopqrstuvwxyz"
                                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                    "àâäéèêëîïôöùûüÿç"
                                    "ÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ"
                                    " .,;:!?\"'()[]{}<>«»\n"
                                    "0123456789")
            for char in response:
                if char not in french_characters:
                    return False
            return True
        elif language in ["俄语", "俄文"]:
            # 基本西里尔字母：\u0400-\u04FF
            # 西里尔补充字符：\u0500-\u052F
            # 西里尔扩展字符：\u2DE0-\u2DFF、\uA640-\uA69F
            # 其他特殊字符：Ёё
            russian_punctuations = set(".,;:!?\"'()[]{}<>«»–-— \n")
            for char in response:
                # 检查字符是否属于西里尔字符范围或在允许的标点符号集合中
                if not (('\u0400' <= char and char <= '\u04FF') or ('\u0500' <= char and char <= '\u052F') or
                        ('\u2DE0' <= char and char <= '\u2DFF') or ('\uA640' <= char and char <= '\uA69F') or
                        (char in 'Ёё') or (char in russian_punctuations) or char.isdigit()):
                    return False
            return True
        elif language in ["德语", "德文"]:
            # 德语字母：a-z, A-Z
            # 德语重音字母：ä, ö, ü, Ä, Ö, Ü, ß
            german_characters = set("abcdefghijklmnopqrstuvwxyz"
                                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                    "äöüßÄÖÜ"
                                    " .,;:!?\"'()[]{}<>«»–-—\n"
                                    "0123456789")
            for char in response:
                if char not in german_characters:
                    return False
            return True
        elif language in ["印地语"]:
            # Devanagari 字母：\u0900-\u097F（包括元音、辅音、数字和标志）
            hindi_punctuations = set("।,।!?-—“”‘’()[]{}<>«»\"\n ")
            for char in response:
                if not ('\u0900' <= char <= '\u097F' or char in hindi_punctuations or char.isdigit()):
                    return False
            return True
        elif language in ["阿拉伯语"]:
            # 基本阿拉伯字母：\u0600-\u06FF
            # 扩展字母：\u0750-\u077F, \u08A0-\u08FF, \uFB50-\uFDFF, \uFE70-\uFEFF
            arabic_punctuations = set("،؛؟!«»“”'()[]{}<>٪-' \n")
            for char in response:
                if not (('\u0600' <= char and char <= '\u06FF') or ('\u0750' <= char and char <= '\u077F') or
                        ('\u08A0' <= char and char <= '\u08FF') or ('\uFB50' <= char and char <= '\uFDFF') or
                        ('\uFE70' <= char and char <= '\uFEFF') or (char in arabic_punctuations) or char.isdigit()):
                    return False
            return True
        else:
            raise ValueError("Invalid language.")
        #TODO：主观pe接口检测其他语言

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        # 先包含世界十大语种：
        language = random.choice(self.language_choice)
        # language = random.choice(["东北话","广东话","山东话","四川话","重庆话","广东话","上海话","藏语","维语","粤语", # 方言
        #                           "脏话","文言文","古文"]) #还不可检测

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(language=language)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "language": language
            }
        }


class Rule_LANGUAGE_CONTAIN(Rule):

    def __init__(self):
        self.rule_type = "LANGUAGE_CONTAIN"
        #TODO：扩充包含的表述
        self.meta_instr = [
            '回答中包含{language}',
            '回答中必须包含{language}',
            '回答中含{language}',
            '包含{language}',
            '必须包含{language}',
            '含{language}',
            '回复包含{language}',
            '回复必须包含{language}',
            '回复中含{language}',
            '答案包含{language}',
            '答案必须包含{language}',
            '答案中含{language}',
        ]
        self.language_choice = [
            "英文", "英语", "中文", "汉语", "普通话", "繁体字", "简体中文", "西班牙语", "西班牙文", "葡萄牙语", "葡萄牙文", "法语", "法文", "俄语", "俄文", "德语",
            "德文", "日语", "日文", "韩语", "韩文", "阿拉伯语", "印地语"
        ]

    def check(self, response, slots):
        language = slots["language"]
        if language in ["英文", "英语"]:
            english_regex = re.compile(r'[a-zA-Z]')
            return bool(english_regex.search(response))
        elif language in ["中文", "汉语"]:
            chinese_regex = re.compile(r'[\u4E00-\u9FAF]')
            return bool(chinese_regex.search(response))
        elif language in ["普通话", "简体中文"]:
            for char in response:
                if hanzidentifier.is_simplified(char):
                    return True
            return False
        elif language in ["繁体字"]:
            for char in response:
                if hanzidentifier.is_traditional(char):
                    return True
            return False
        elif language in ["日语", "日文"]:
            # 平假名：\u3040-\u309F
            # 片假名：\u30A0-\u30FF
            # 拡張片假名：\u31F0-\u31FF
            # 全角标点符号：\u3000-\u303F（其中包括一些非日语字符，但能涵盖常用标点）
            # 日语汉字（CJK Unified Ideographs）：\u4E00-\u9FAF
            japanese_regex = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\u4E00-\u9FAF]')
            return bool(japanese_regex.search(response))
        elif language in ["韩语", "韩文"]:
            # 韩文字母（Hangul Syllables）：\uAC00-\uD7AF
            # 韩文字母（Hangul Jamo）：\u1100-\u11FF 和 \u3130-\u318F
            # Hangul Compatibility Jamo：\FFA0-\uFFDC
            korean_regex = re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uFFA0-\uFFDC]')
            return bool(korean_regex.search(response))
        elif language in ["西班牙语", "西班牙文"]:
            # 西班牙语字母：a-z, A-Z
            # 西班牙语重音字母：á, é, í, ó, ú, Á, É, Í, Ó, Ú,
            # 其他特殊字符：ü, Ü, ñ, Ñ
            spanish_regex = re.compile(r'[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]')
            return bool(spanish_regex.search(response))
        elif language in ["葡萄牙语", "葡萄牙文"]:
            # 葡萄牙语字母：a-z, A-Z
            # 葡萄牙语重音字母：á, é, í, ó, ú, à, è, ì, ò, ù, â, ê, î, ô, û, ã, õ, ç, Á, É, Í, Ó, Ú, À, È, Ì, Ò, Ù, Â, Ê, Î, Ô, Û, Ã, Õ, Ç
            portuguese_regex = re.compile(r'[a-zA-ZáéíóúàâãçÁÉÍÓÚÀÂÃÇ]')
            return bool(portuguese_regex.search(response))
        elif language in ["法语", "法文"]:
            # 法语字母：a-z, A-Z
            # 法语重音字母：à, â, ä, é, è, ê, ë, î, ï, ô, ö, ù, û, ü, ÿ, ç, À, Â, Ä, É, È, Ê, Ë, Î, Ï, Ô, Ö, Ù, Û, Ü, Ÿ, Ç
            french_regex = re.compile(r'[a-zA-ZàâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ]')
            return bool(french_regex.search(response))
        elif language in ["俄语", "俄文"]:
            # 基本西里尔字母：\u0400-\u04FF
            # 西里尔补充字符：\u0500-\u052F
            # 西里尔扩展字符：\u2DE0-\u2DFF、\uA640-\uA69F
            # 其他特殊字符：Ёё
            russian_regex = re.compile(r'[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69FЁё]')
            return bool(russian_regex.search(response))
        elif language in ["德语", "德文"]:
            # 德语字母：a-z, A-Z
            # 德语重音字母：ä, ö, ü, Ä, Ö, Ü, ß
            german_regex = re.compile(r'[abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüßÄÖÜ]')
            return bool(german_regex.search(response))
        elif language in ["印地语"]:
            # Devanagari 字母：\u0900-\u097F（包括元音、辅音、数字和标志）
            hindi_regex = re.compile(r'[\u0900-\u097F]')
            # 搜索字符串中是否存在符合印地语字符的部分
            return bool(hindi_regex.search(response))
        elif language in ["阿拉伯语"]:
            # 基本阿拉伯字母：\u0600-\u06FF
            # 扩展字母：\u0750-\u077F, \u08A0-\u08FF, \uFB50-\uFDFF, \uFE70-\uFEFF
            arabic_regex = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
            return bool(arabic_regex.search(response))
        #TODO：主观pe接口检测其他语言

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        # 先包含世界十大语种：
        language = random.choice(self.language_choice)
        # language = random.choice(["东北话","广东话","山东话","四川话","重庆话","广东话","上海话","藏语","维语","粤语", # 方言
        #                           "脏话","文言文","古文"]) #还不可检测

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(language=language)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "language": language
            }
        }


class Rule_LANGUAGE_NUM(Rule):

    def __init__(self):
        self.rule_type = "LANGUAGE_NUM"
        self.meta_instr = [
            '无论输入何种语种的特定指示语，都以“中、英”两种语言输出',
            '以“中、英”两种语言输出',
            '中英文对照',
            '用中英双语回答',
        ]
        #TODO:除了中英双语还需不需要其他混合语言

    def check(self, response, slots):
        #pattern = re.compile(r'^[0-9a-zA-Z\u4e00-\u9fa5\s’‘“”\'\",.?!;:()\[\]{}\-&，。！？；：“”‘’、（）：《》〈〉【】…—－—·]+$')
        #return bool(pattern.match(response))
        english_regex = re.compile(r'[a-zA-Z]')
        chinese_regex = re.compile(r'[\u4E00-\u9FAF]')
        return bool(chinese_regex.search(response)) and bool(english_regex.search(response))

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_LANGUAGE_LOC(Rule):

    def __init__(self):
        self.rule_type = "LANGUAGE_LOC"
        self.meta_instr_para = [
            '第{which}段请使用{language}',
            '第{which}个段落请用{language}',
            '第{which}段使用{language}',
            '第{which}个段落只能用{language}',
            '第{which}段语种：{language}',
            '第{which}个段落要求用{language}说话',
            '第{which}段以{language}输出',
            '第{which}个段落{language}输出',
            '第{which}段译成{language}',
            '第{which}个段落译为{language}',
            '第{which}段用{language}书写',
            '第{which}个段落只能使用{language}',
            '第{which}段用{language}表述',
            '第{which}个段落为{language}表述',
            '第{which}段用{language}展示',
            '第{which}个段落要求说{language}',
            '第{which}段用{language}说',
            '第{which}个段落只包含{language}',
        ]
        self.meta_instr_sent = [
            '第{which}句用{language}',
            '第{which}个句子用{language}',
            '第{which}句采用{language}',
            '第{which}个句子以{language}进行回答',
            '第{which}句语言为{language}',
            '第{which}个句子语种为{language}',
            '第{which}句用{language}对话',
            '第{which}个句子用{language}输出',
            '第{which}句翻译成{language}',
            '第{which}个句子翻译为{language}',
            '第{which}句用{language}翻译',
            '第{which}个句子用{language}写',
            '第{which}句需为{language}',
            '第{which}个句子只包含{language}表述',
            '第{which}句转化成{language}',
            '第{which}个句子转换成{language}',
            '第{which}句必须是{language}',
            '第{which}个句子要用{language}',
        ]
        self.language_choice = [
            "英文", "纯英文", "英语", "中文", "纯中文", "汉语", "普通话", "繁体字", "简体中文", "西班牙语", "西班牙文", "葡萄牙语", "葡萄牙文", "法语", "法文",
            "俄语", "俄文", "德语", "德文", "日语", "日文", "韩语", "韩文", "阿拉伯语", "印地语"
        ]

    def check(self, response, slots):
        loc = slots["loc"]
        which = slots["which"]
        language = slots["language"]
        eval_class = Rule_LANGUAGE_ONLY()
        if loc == 'para':
            paragraphs = re.split(r'\n\s*\n*', response.strip())
            for i, paragraph in enumerate(paragraphs):
                if i + 1 == which and eval_class.check(paragraph, slots):
                    return True
            return False
        elif loc == 'sent':
            sents = cut_sent(response)
            sent_count = len(sents)
            for i, sent in enumerate(sents):
                if i + 1 == which and eval_class.check(sent, slots):
                    return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        language = random.choice(self.language_choice)
        loc_choice = ['para', 'sent']
        loc = random.choice(loc_choice)
        self.meta_instr = self.meta_instr_para if loc == 'para' else self.meta_instr_sent
        if loc == 'para':
            paragraphs = re.split(r'\n\s*\n*', response.strip())
            paragraph_count = len(paragraphs)
            which = random.randint(1, paragraph_count)
        elif loc == 'sent':
            sents = cut_sent(response)
            sent_count = len(sents)
            which = random.randint(1, sent_count)
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        if random.random() < 0.2:
            zh_num = arabic_to_chinese(which)
            instruction_meta = self.meta_instr[indicator_meta].format(language=language, which=zh_num)
        else:
            instruction_meta = self.meta_instr[indicator_meta].format(language=language, which=which)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "language": language,
                "which": which,
                "loc": loc
            }
        }


# 表意不清晰，暂时舍去
class Rule_LANGUAGE_NUM_LOC(Rule):

    def __init__(self):
        self.rule_type = "LANGUAGE_NUM_LOC"
        self.meta_instr_zh_first = [
            '中文在前，英语在后',
            '先用中文回答，再用英文回答',
            '前面是中文，后面是英语',
            '开头先说中文，后面再说英文',
            '先写中文，再写英语',
        ]
        self.meta_instr_en_first = [
            '英文在前，汉语在后',
            '先用英语回答，再用中文回答',
            '前面是英文，后面是中文',
            '开头先说英语，后面再说汉语',
            '先写英文，再写中文',
        ]
        #TODO:除了中英双语还需不需要其他混合语言

    def check(self, response, slots):
        first_lang = slots["first_lang"]
        if first_lang == "zh_first":
            chinese_regex = re.compile(r'^[\u4E00-\u9FFF]+')
            english_regex = re.compile(r'[a-zA-Z]+$')
            return bool(chinese_regex.search(response)) and bool(english_regex.search(response))
        else:
            chinese_regex = re.compile(r'[\u4E00-\u9FFF]+')
            english_regex = re.compile(r'^[a-zA-Z]+$')
            return bool(chinese_regex.search(response)) and bool(english_regex.search(response))

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        first_lang = random.choice(["zh_first", "en_first"])
        self.meta_instr = self.meta_instr_zh_first if first_lang == "zh_first" else self.meta_instr_en_first
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "first_lang": first_lang
            }
        }


class Rule_LANGUAGE_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "LANGUAGE_NEGATE"
        #TODO：扩充禁止的表述
        self.meta_instr = [
            '回答中禁止包含{language}',
            '回答中不能包含{language}',
            '回答中不含{language}',
            '不可以使用{language}',
            '不能出现{language}',
            '不能使用{language}',
            '回复不能有{language}',
            '回复不要用{language}',
            '回复中不能使用{language}',
            '答案不可以包括{language}',
            '答案不要使用{language}',
            '答案中禁止出现{language}',
            '不可以使用{language}回复',
            '你在回复时不能使用{language}',
        ]

    def check(self, response, slots):
        language = slots["language"]
        if language in ["英文", "英语"]:
            english_regex = re.compile(r'[a-zA-Z]')
            return not bool(english_regex.search(response))
        elif language in ["中文", "汉语"]:
            chinese_regex = re.compile(r'[\u4E00-\u9FAF]')
            return not bool(chinese_regex.search(response))
        elif language in ["普通话", "简体中文"]:
            for char in response:
                if hanzidentifier.is_simplified(char):
                    return False
            return True
        elif language in ["繁体字"]:
            for char in response:
                if hanzidentifier.is_traditional(char):
                    return False
            return True
        elif language in ["日语", "日文"]:
            # 平假名：\u3040-\u309F
            # 片假名：\u30A0-\u30FF
            # 拡張片假名：\u31F0-\u31FF
            # 全角标点符号：\u3000-\u303F（其中包括一些非日语字符，但能涵盖常用标点）
            japanese_regex = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]')
            return not bool(japanese_regex.search(response))
        elif language in ["韩语", "韩文"]:
            # 韩文字母（Hangul Syllables）：\uAC00-\uD7AF
            # 韩文字母（Hangul Jamo）：\u1100-\u11FF 和 \u3130-\u318F
            # Hangul Compatibility Jamo：\FFA0-\uFFDC
            korean_regex = re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uFFA0-\uFFDC]')
            return not bool(korean_regex.search(response))
        elif language in ["俄语", "俄文"]:
            # 基本西里尔字母：\u0400-\u04FF
            # 西里尔补充字符：\u0500-\u052F
            # 西里尔扩展字符：\u2DE0-\u2DFF、\uA640-\uA69F
            # 其他特殊字符：Ёё
            russian_regex = re.compile(r'[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69FЁё]')
            return not bool(russian_regex.search(response))
        elif language in ["印地语"]:
            # Devanagari 字母：\u0900-\u097F（包括元音、辅音、数字和标志）
            hindi_regex = re.compile(r'[\u0900-\u097F]')
            return not bool(hindi_regex.search(response))
        elif language in ["阿拉伯语"]:
            # 基本阿拉伯字母：\u0600-\u06FF
            # 扩展字母：\u0750-\u077F, \u08A0-\u08FF, \uFB50-\uFDFF, \uFE70-\uFEFF
            arabic_regex = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
            return not bool(arabic_regex.search(response))
        #TODO：主观pe接口检测其他语言

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        # 先包含世界十大语种：
        language = random.choice(
            ["英文", "英语", "中文", "汉语", "普通话", "繁体字", "简体中文", "俄语", "俄文", "日语", "日文", "韩语", "韩文", "阿拉伯语", "印地语"])
        # language = random.choice(["东北话","广东话","山东话","四川话","重庆话","广东话","上海话","藏语","维语","粤语", # 方言
        #                           "脏话","文言文","古文"]) #还不可检测

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(language=language)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "language": language
            }
        }


################################################
# 中文文本类
################################################


class Rule_TEXT_ONLY(Rule):

    def __init__(self):
        self.rule_type = "TEXT_ONLY"
        self.meta_instr = [
            '只回答“{text}”',
            '直接回复“{text}”',
            '只输出“{text}”',
            '只需要回复“{text}”',
            '只回复“{text}”',
            '只能用“{text}”作答',
            '仅输出“{text}”',
            '只输出“{text}”即可',
            '注意只需要输出“{text}”',
            '请只回答“{text}”',
            '要求仅输出“{text}”',
            '请直接回复“{text}”',
        ]
        self.suffix = [
            '不要输出其他内容',
            '不做解释',
            '不要给出多余的内容',
            '不需要解释原因',
            '不需要输出解释',
            '不需要额外的解释',
            '无需其他解释',
            '不需要输出额外的解释或说明',
            '不需要其他额外内容',
            '不要回答其他内容',
            '不需要解释或说明理由',
            '不要输出其他额外内容',
            '不需要输出其他信息',
            '不需要回复分析',
            '不需要解释',
            '不需要说明理由',
            '不需要输出其他内容',
            '不需要输出理由或解释等其他信息',
            '不需要输出分析、解释等其他内容或标点符号',
        ]

    def check(self, response, slots):
        text = slots["text"]
        if response.strip() == text:
            return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        piece_len = 10
        text_seg = pku_seg.cut(response)
        #text = response[:random.sample(range(piece_len), 1)[0]]
        text = ''.join(text_seg[:random.sample(range(piece_len), 1)[0]])
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        suffix_meta = random.randint(0, len(self.suffix) - 1)
        if random.random() < 0.5:
            instruction_meta = self.meta_instr[indicator_meta].format(text=text) + "，" + self.suffix[suffix_meta]
        else:
            instruction_meta = self.meta_instr[indicator_meta].format(text=text)
        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "text": text
            }
        }


class Rule_TEXT_CONTAIN(Rule):

    def __init__(self):
        self.rule_type = "TEXT_CONTAIN"
        self.meta_instr_or = [
            '回答必须包括“{piece1}”或“{piece2}”',
            '答案中必须有“{piece1}”或“{piece2}”',
            '必须带有“{piece1}”或“{piece2}”',
            '答复时应包含“{piece1}”或“{piece2}”',
        ]
        self.meta_instr_and = [
            '回答必须包括“{piece1}”和“{piece2}”',
            '必须在回答中提到“{piece1}”和“{piece2}”',
            '必须在回复里包含“{piece1}”和“{piece2}”',
            '需有“{piece1}”和“{piece2}”出现',
        ]
        self.meta_instr_normal = [
            '回答必须包含“{piece1}”', '每次回复必须说“{piece1}”', '回复中要有“{piece1}”', '答复必须带有“{piece1}”', '必须在回答中包含“{piece1}”',
            '你的答复需要包含“{piece1}”', '回答需包括“{piece1}”', '要求包含“{piece1}”'
        ]

    def check(self, response, slots):
        relation_indicator = slots["relation_indicator"]
        piece1 = slots["piece1"]
        piece2 = slots["piece2"]
        if relation_indicator == '或':
            if piece1 in response or piece2 in response:
                return True
            return False
        elif relation_indicator == '和':
            if piece1 in response and piece2 in response:
                return True
            return False
        else:
            if piece1 in response:
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        # piece_len = 10
        text_seg = pku_seg.cut(response)
        relation_indicator = random.choice(['或', '和', ''])
        if relation_indicator == '或':
            # sample_idx = random.sample(range(len(response)),2)
            # sample_idx.sort()
            # piece1 = response[sample_idx[0]:sample_idx[0]+random.sample(range(piece_len), 1)[0]]
            # piece2 = response[sample_idx[1]:sample_idx[1]+random.sample(range(piece_len), 1)[0]]
            sample_idx = random.sample(range(len(text_seg)), 2)
            sample_idx.sort()
            piece1 = text_seg[sample_idx[0]]
            piece2 = text_seg[sample_idx[1]]
            indicator_meta = random.randint(0, len(self.meta_instr_or) - 1)
            instruction_meta = self.meta_instr_or[indicator_meta].format(piece1=piece1, piece2=piece2)
        elif relation_indicator == '和':
            # sample_idx = random.sample(range(len(response)),2)
            # sample_idx.sort()
            # piece1 = response[sample_idx[0]:sample_idx[0]+random.sample(range(piece_len), 1)[0]]
            # piece2 = response[sample_idx[1]:sample_idx[1]+random.sample(range(piece_len), 1)[0]]
            sample_idx = random.sample(range(len(text_seg)), 2)
            sample_idx.sort()
            piece1 = text_seg[sample_idx[0]]
            piece2 = text_seg[sample_idx[1]]
            indicator_meta = random.randint(0, len(self.meta_instr_and) - 1)
            instruction_meta = self.meta_instr_and[indicator_meta].format(piece1=piece1, piece2=piece2)
        else:
            # sample_idx = random.sample(range(len(response)),1)
            # sample_idx.sort()
            # piece1 = response[sample_idx[0]:sample_idx[0]+random.sample(range(piece_len), 1)[0]]
            # piece2 = ''
            sample_idx = random.sample(range(len(text_seg)), 1)
            piece1 = text_seg[sample_idx[0]]
            piece2 = ''
            indicator_meta = random.randint(0, len(self.meta_instr_normal) - 1)
            instruction_meta = self.meta_instr_normal[indicator_meta].format(piece1=piece1)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "relation_indicator": relation_indicator,
                "piece1": piece1,
                "piece2": piece2
            }
        }


################################################
# 中文字数check
################################################

IGNORE_TOKEN_SET = set([
    "！", "？", "｡", "。", "＂", "＃", "＄", "％", "＆", "＇", "（", "）", "＊", "＋", "，", "－", "／", "：", "；", "＜", "＝", "＞", "＠",
    "［", "＼", "］", "＾", "＿", "｀", "｛", "｜", "｝", "～", "｟", "｠", "｢", "｣", "､", "、", "〃", "》", "「", "」", "『", "』", "【",
    "】", "〔", "〕", "〖", "〗", "〘", "〙", "〚", "〛", "〜", "〝", "〞", "〟", "〰", "〾", "〿", "–", "—", "‘", "’", "‛", "“", "”",
    "„", "‟", "…", "‧", "﹏", ".", "!", "\"", "#", "$", "%", "&", "'", "\\", "(", ")", "*", "+", ",", "-", ".", "/", ":",
    ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", " ", "《", "》"
])


def count_chinese_chars(text):
    chinese_characters = re.split(r'(\s|[\u4e00-\u9fff。、《》【】「」])', text)
    chinese_characters = [i for i in chinese_characters if i]
    char_count = 0
    for n, i in enumerate(chinese_characters):
        if re.search(r'\s|——', i):
            continue

        if i in IGNORE_TOKEN_SET:
            continue

        char_count += 1
    return char_count


class Rule_ZH_WORD_COUNT(Rule):

    def __init__(self):
        self.rule_type = "ZH_WORD_COUNT"

    def check(self, response, slots):
        num = slots["num"]
        op = slots["op"]
        #matches = re.findall(r'[\u4e00-\u9fa5]', response)
        num_char = count_chinese_chars(response)
        if op == 'ABOUT':
            if abs(num_char - num) / num <= 0.25:
                return True
            return False
        if op == 'EQUAL':
            #if num_char == num:
            if abs(num_char - num) / num <= 0.1:
                return True
            return False
        elif op == 'MORE':
            if num_char >= num:
                return True
            return False
        elif op == 'LESS':
            if num_char <= num and num_char >= 1:
                return True
            return False
        return False


class Rule_TEXT_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "TEXT_NEGATE"
        self.meta_instr = [
            '不要使用以下违规词：“{piece}”',
            '避免提到“{piece}”',
            '回答中不能出现“{piece}”',
            '不能提及“{piece}”',
            '不要出现“{piece}”',
            '禁止出现“{piece}”',
            '不能使用【{piece}】',
            '不能出现“{piece}”这几个字',
            '回答时不要使用“{piece}”',
            '不要提到“{piece}”',
            '避免出现“{piece}”字眼',
        ]

    def check(self, response, slots):
        piece = slots['piece']
        if piece in response:
            return False
        else:
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        # piece_len = 10
        # sample_idx = random.sample(range(len(response)),1)
        # sample_idx.sort()
        # piece = response[sample_idx[0]:sample_idx[0]+random.sample(range(piece_len),1)[0]]
        text_seg = pku_seg.cut(response)
        sample_idx = random.sample(range(len(text_seg)), 1)
        piece = text_seg[sample_idx[0]]
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(piece=piece)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "piece": piece
            }
        }


class Rule_TEXT_CONTAIN_LOC(Rule):

    def __init__(self):
        self.rule_type = "TEXT_CONTAIN_LOC"
        self.meta_instr_which = [
            '“{piece}”输出在第{which}段',
            '回复的第{which}段需回复“{piece}”',
            '第{which}段必须包含“{piece}”',
            '第{which}段必须说“{piece}”',
            '第{which}段中要有“{piece}”',
            '第{which}段必须带有“{piece}”',
            '必须在第{which}段包含“{piece}”',
            '你的答复第{which}段需要包含“{piece}”',
            '第{which}段需包括“{piece}”',
            '第{which}段要求包含“{piece}”',
        ]
        self.meta_instr_first = [
            '“{piece}”输出在最开头一段',
            '回答的最开头一段需回复“{piece}”',
            '首段必须包含“{piece}”',
            '第一段必须说“{piece}”',
            '最开头一段中要有“{piece}”',
            '首段必须带有“{piece}”',
            '必须在第一段包含“{piece}”',
            '你的答复最开头一段需要包含“{piece}”',
            '首段需包括“{piece}”',
            '第一段要求包含“{piece}”',
        ]
        self.meta_instr_last = [
            '“{piece}”输出在最后一段',
            '每次回复的最后一段需回复“{piece}”',
            '末段必须包含“{piece}”',
            '末段必须说“{piece}”',
            '最后一段中要有“{piece}”',
            '最后一段必须带有“{piece}”',
            '必须在末段包含“{piece}”',
            '你的答复末段需要包含“{piece}”',
            '最后一段需包括“{piece}”',
            '最后一段要求包含“{piece}”',
        ]

    def check(self, response, slots):
        which = slots["which"]
        piece = slots["piece"]
        paragraphs = re.split(r'\n\s*\n*', response.strip())
        if which == 0:
            if piece in paragraphs[0]:
                return True
            return False
        elif which == -1:
            if piece in paragraphs[-1]:
                return True
            return False
        else:
            for i, paragraph in enumerate(paragraphs):
                if i + 1 == which and piece in paragraph:
                    return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        # piece_len = 10
        paragraphs = re.split(r'\n\s*\n*', response.strip())
        paragraph_count = len(paragraphs)
        # sample_idx = random.sample(range(len(response)),1)
        # sample_idx.sort()
        # piece = response[sample_idx[0]:sample_idx[0]+random.sample(range(piece_len),1)[0]]
        text_seg = pku_seg.cut(response)
        sample_idx = random.sample(range(len(text_seg)), 1)
        piece = text_seg[sample_idx[0]]

        para_type = random.choice(['which', 'first', 'last'])
        if para_type == 'which':
            indicator_meta = random.randint(0, len(self.meta_instr_which) - 1)
            random_int = random.randint(1, min(paragraph_count, 10))
            if random.random() < 0.2:
                zh_num = arabic_to_chinese(random_int)
                instruction_meta = self.meta_instr_which[indicator_meta].format(which=zh_num, piece=piece)
            else:
                instruction_meta = self.meta_instr_which[indicator_meta].format(which=random_int, piece=piece)
        elif para_type == 'first':
            random_int = 0
            indicator_meta = random.randint(0, len(self.meta_instr_first) - 1)
            instruction_meta = self.meta_instr_first[indicator_meta].format(which=random_int, piece=piece)
        elif para_type == 'last':
            random_int = -1
            indicator_meta = random.randint(0, len(self.meta_instr_last) - 1)
            instruction_meta = self.meta_instr_last[indicator_meta].format(which=random_int, piece=piece)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "which": random_int,
                "piece": piece
            }
        }


class Rule_TEXT_CONTAIN_NUM(Rule):

    def __init__(self):
        self.rule_type = "TEXT_CONTAIN_NUM"
        self.meta_instr_equal = [
            '回复中要有“{piece}”且要重复{num}次',
            '回复中“{piece}”出现{num}次',
            '确保“{piece}”出现{num}次',
        ]
        self.meta_instr_equal_negate = [
            '',
            '',
            '',
        ]
        self.meta_instr_more = [
            '回复中要有“{piece}”且至少要重复{num}次',
            '回复中“{piece}”至少出现{num}次',
            '确保“{piece}”至少出现{num}次',
            '回复中要有“{piece}”且最少重复{num}次',
            '回复中“{piece}”最少出现{num}次',
            '确保“{piece}”最少出现{num}次',
            '回复中要有“{piece}”且重复不少于{num}次',
            '回复中“{piece}”出现不少于{num}次',
            '确保“{piece}”出现不少于{num}次',
            '回复中要有“{piece}”且重复次数大于等于{num}',
            '回复中“{piece}”出现次数大于等于{num}',
            '确保“{piece}”出现次数大于等于{num}',
        ]
        self.meta_instr_more_negate = [
            '回复中“{piece}”重复要少于{num}次',
            '回复中“{piece}”出现少于{num}次',
            '确保“{piece}”出现少于{num}次',
            '回复中“{piece}”重复少于{num}次',
            '回复中“{piece}”出现一定要少于{num}次',
            '确保“{piece}”出现少于{num}次',
            '回复中的“{piece}”重复少于{num}次',
            '回复中“{piece}”出现少于{num}次',
            '确保“{piece}”出现少于{num}次',
            '回复中的“{piece}”重复次数小于{num}',
            '回复中“{piece}”出现次数小于{num}',
            '确保“{piece}”出现次数小于{num}',
        ]
        self.meta_instr_less = [
            '回复中要有“{piece}”且至多重复{num}次',
            '确保“{piece}”出现，但至多出现{num}次',
            '回复中要有“{piece}”且最多重复{num}次',
            '确保“{piece}”出现，但最多出现{num}次',
            '回复中要有“{piece}”且重复不多于{num}次',
            '确保“{piece}”出现，但出现不多于{num}次',
            '回复中要有“{piece}”且重复次数小于等于{num}',
            '确保“{piece}”出现，但出现次数小于等于{num}',
        ]
        self.meta_instr_less_negate = [
            '回复中要有“{piece}”且重复多于{num}次',
            '确保“{piece}”出现，且多于{num}次',
            '回复中要有“{piece}”且重复超过{num}次',
            '确保“{piece}”出现，并且要出现多于{num}次',
            '回复中要有“{piece}”且重复多于{num}次',
            '确保“{piece}”出现多于{num}次',
            '回复中要有“{piece}”且重复次数大于{num}',
            '确保“{piece}”出现，并保证出现次数大于{num}',
        ]

    def check(self, response, slots):
        piece = slots["piece"]
        num = slots["num"]
        indicator_count_version = slots["indicator_count_version"]
        piece_num = response.count(piece)
        if indicator_count_version == 'EQUAL':
            if piece_num == num:
                return True
            return False
        elif indicator_count_version == 'MORE':
            if piece_num >= num:
                return True
            return False
        elif indicator_count_version == 'LESS':
            if piece_num <= num and piece_num >= 1:
                return True
            return False
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_count_version = random.choice(['EQUAL', 'LESS', 'MORE'])
        num = random.randint(1, 5)
        # piece_len = 5
        # sample_idx = random.sample(range(len(response)),1)
        # sample_idx.sort()
        # piece = response[sample_idx[0]:sample_idx[0]+random.sample(range(piece_len),1)[0]]
        text_seg = pku_seg.cut(response)
        sample_idx = random.sample(range(len(text_seg)), 1)
        piece = text_seg[sample_idx[0]]

        if random.random() < 0.5:
            zh_num = arabic_to_chinese(num)
            if indicator_count_version == 'EQUAL':
                indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
                instruction_meta = self.meta_instr_equal[indicator_meta].format(piece=piece, num=zh_num)
            elif indicator_count_version == 'LESS':
                indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
                instruction_meta = self.meta_instr_less[indicator_meta].format(piece=piece, num=zh_num)
            elif indicator_count_version == 'MORE':
                indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
                instruction_meta = self.meta_instr_more[indicator_meta].format(piece=piece, num=zh_num)
        else:
            if indicator_count_version == 'EQUAL':
                indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
                instruction_meta = self.meta_instr_equal[indicator_meta].format(piece=piece, num=num)
            elif indicator_count_version == 'LESS':
                indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
                instruction_meta = self.meta_instr_less[indicator_meta].format(piece=piece, num=num)
            elif indicator_count_version == 'MORE':
                indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
                instruction_meta = self.meta_instr_more[indicator_meta].format(piece=piece, num=num)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "indicator_count_version": indicator_count_version,
                "piece": piece,
                "num": num
            }
        }


class Rule_TEXT_START_END(Rule):

    def __init__(self):
        self.rule_type = "TEXT_START_END"
        self.meta_instr_start = [
            '以“{text}”作为开头',
            '回复用户指令的话术用“{text}”开头',
            '每次回复的开头都加上“{text}”',
            '以“{text}”开头',
            '开头必须为“{text}”',
            '每次回复以“{text}”开头',
            '你的回答必须以“{text}”开头',
            '每次回答开始时，先输出“{text}”',
        ]
        self.meta_instr_end = [
            '以“{text}”作为结尾',
            '回复用户指令的话术用“{text}”结尾',
            '每次回复的最后都加上“{text}”',
            '以“{text}”结束',
            '结尾必须为“{text}”',
            '每次回复以“{text}”结束',
            '你的回答必须以“{text}”结尾',
            '每次回答结束时，输出“{text}”',
        ]

    def check(self, response, slots):
        text = slots["text"]
        loc_type = slots["loc_type"]
        if loc_type == 'start':
            if response.startswith(text):
                return True
            return False
        else:
            if response.endswith(text):
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        piece_len = 10
        random_int_start = random.randint(1, piece_len)
        # st = response[:random_int_start]
        #random_int_end = random.randint(2, piece_len)
        # et = response[-random_int_end:]

        text_seg = pku_seg.cut(response)
        st = ''.join(text_seg[:random_int_start])
        random_int_end = random.randint(1, piece_len)
        et = ''.join(text_seg[-random_int_end:])

        loc_type = random.choice(['start', 'end'])
        text = st if loc_type == 'start' else et
        self.meta_instr = self.meta_instr_start if loc_type == 'start' else self.meta_instr_end
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(text=text)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "loc_type": loc_type,
                "text": text
            }
        }


class Rule_TEXT_START_END_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "TEXT_START_END_NEGATE"
        self.meta_instr_start = [
            '不要以“{text}”作为开头',
            '回复用户指令的话术不用“{text}”开头',
            '每次回复的开头都不能加上“{text}”',
            '禁止以“{text}”开头',
            '开头不可以为“{text}”',
            '每次回复不以“{text}”开头',
            '你的回答禁止以“{text}”开头',
            '每次回答开始时，不要输出“{text}”',
        ]
        self.meta_instr_end = [
            '不要以“{text}”作为结尾',
            '回复用户指令的话术不用“{text}”结尾',
            '每次回复的最后都不能加上“{text}”',
            '禁止以“{text}”结束',
            '结尾不可以为“{text}”',
            '每次回复不以“{text}”结束',
            '你的回答禁止以“{text}”结尾',
            '每次回答结束时，不要输出“{text}”',
        ]

    def check(self, response, slots):
        text = slots["text"]
        loc_type = slots["loc_type"]
        if loc_type == 'start':
            if not response.startswith(text):
                return True
            return False
        else:
            if not response.endswith(text):
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        piece_len = 10
        random_int_start = random.randint(1, piece_len)
        # st = response[:random_int_start]
        # random_int_end = random.randint(2, piece_len)
        # et = response[-random_int_end:]
        text_seg = pku_seg.cut(response)
        st = ''.join(text_seg[:random_int_start])
        random_int_end = random.randint(1, piece_len)
        et = ''.join(text_seg[-random_int_end:])

        loc_type = random.choice(['start', 'end'])
        text = st if loc_type == 'start' else et
        self.meta_instr = self.meta_instr_start if loc_type == 'start' else self.meta_instr_end
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(text=text)
        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "loc_type": loc_type,
                "text": text
            }
        }


class Rule_TEXT_START_LOC(Rule):

    def __init__(self):
        self.rule_type = "TEXT_START_LOC"
        self.meta_instr = [
            '每句话的开头字符必须是一样的',
        ]

    def check(self, response, slots):
        sents = cut_sent(response)
        sent_count = len(sents)
        start_text = ''
        for i, sent in enumerate(sents):
            if i == 0:
                start_text = sent[0]
            else:
                if not sent.startswith(start_text):
                    return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_TEXT_END_LOC(Rule):

    def __init__(self):
        self.rule_type = "TEXT_END_LOC"
        self.meta_instr_sent = [
            '每句话结尾加上“{text}”字',
            '每个句子最后加一个“{text}”',
            '每句话以“{text}”作为结尾',
            '每个句子以“{text}”结尾',
            '每句话以“{text}”结束',
            '每个句子结尾为“{text}”',
            '每句话的最后，输出{text}',
        ]
        self.meta_instr_para = [
            '每段话结尾加上“{text}”字',
            '每段话最后加一个“{text}”',
            '每个段落以“{text}”作为结尾',
            '每段话以“{text}”结尾',
            '每个段落以“{text}”结束',
            '每段话结尾为“{text}”',
            '每个段落的最后，输出{text}',
        ]

    def check(self, response, slots):
        loc = slots["loc"]
        text = slots["text"]
        if loc == 'para':
            paragraphs = re.split(r'\n\s*\n*', response.strip())
            for i, paragraph in enumerate(paragraphs):
                paragraph = remove_punctuation(paragraph.strip())
                if not paragraph.endswith(text):
                    return False
            return True
        elif loc == 'sent':
            sents = cut_sent(response)
            sent_count = len(sents)
            for i, sent in enumerate(sents):
                sent = remove_punctuation(sent.strip())
                if not sent.endswith(text):
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        text = random.choice(['呐', '哦', '呀', '嘿', '啦', '呢', '哈', '喵'])
        loc = random.choice(['para', 'sent'])
        self.meta_instr = self.meta_instr_para if loc == 'para' else self.meta_instr_sent
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(text=text)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "loc": loc,
                "text": text
            }
        }


class Rule_TEXT_END_LOC_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "TEXT_END_LOC_NEGATE"
        self.meta_instr_sent = [
            '每句话结尾不要加上“{text}”字',
            '每个句子最后不可以加“{text}”',
            '每句话禁止以“{text}”作为结尾',
            '每个句子不要以“{text}”结尾',
            '每句话不能以“{text}”结束',
            '每个句子结尾不可以为“{text}”',
            '每句话的最后，不要输出{text}',
        ]
        self.meta_instr_para = [
            '每段话结尾不要加上“{text}”字',
            '每段话最后不可以加“{text}”',
            '每个段落禁止以“{text}”作为结尾',
            '每段话不要以“{text}”结尾',
            '每个段落不能以“{text}”结束',
            '每段话结尾不可以为“{text}”',
            '每个段落的最后，禁止输出{text}',
        ]

    def check(self, response, slots):
        loc = slots["loc"]
        text = slots["text"]
        if loc == 'para':
            paragraphs = re.split(r'\n\s*\n*', response.strip())
            for i, paragraph in enumerate(paragraphs):
                paragraph = remove_punctuation(paragraph.strip())
                if paragraph.endswith(text):
                    return False
            return True
        elif loc == 'sent':
            sents = cut_sent(response)
            sent_count = len(sents)
            for i, sent in enumerate(sents):
                sent = remove_punctuation(sent.strip())
                if sent.endswith(text):
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        text = random.choice(['呐', '哦', '呀', '嘿', '啦', '呢', '哈', '喵'])
        loc = random.choice(['para', 'sent'])
        self.meta_instr = self.meta_instr_para if loc == 'para' else self.meta_instr_sent
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(text=text)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "loc": loc,
                "text": text
            }
        }


class Rule_TEXT_REPLACE(Rule):

    def __init__(self):
        self.rule_type = "TEXT_REPLACE"
        self.meta_instr = [
            '如果文本中有“{text_ori}”，要替换成“{text_new}”',
            '在回复中，用“{text_new}”替换“{text_ori}”',
            '在回复中，把“{text_ori}”换成“{text_new}”',
            '用“{text_new}”替换“{text_ori}”',
            '把“{text_ori}”换成“{text_new}”',
            '所有“{text_ori}”都用“{text_new}”换掉',
            '把所有“{text_ori}”换成“{text_new}”',
            '文本中的“{text_ori}”都换成“{text_new}”',
        ]

    def check(self, response, slots):
        text_ori = slots['text_ori']
        text_new = slots['text_new']
        pattern = f"[{text_ori}{text_new}]"
        matches = re.findall(pattern, response)
        flag = 1
        for match in matches:
            if match != text_new:
                flag = 0
        if flag:
            return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        cand_text_ori = list(set(remove_punctuation(response.strip())))
        digits_set = set(string.digits)
        letters_set = set(string.ascii_letters)
        alphanumeric_set = list(digits_set | letters_set)
        text_ori = random.choice(cand_text_ori)
        text_new = random.choice([p for p in alphanumeric_set if p != text_ori])

        response = response.replace(text_ori, text_new)
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(text_new=text_new, text_ori=text_ori)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "text_ori": text_ori,
                "text_new": text_new
            }
        }


################################################
# 英文字母类
################################################


class Rule_LETTER_CONTAIN(Rule):

    def __init__(self):
        self.rule_type = "LETTER_CONTAIN"
        self.meta_instr = [
            '回答必须包含英文字母“{letter}”', '回复中要有字母“{letter}”', '答复必须带有英文字母“{letter}”', '必须在回答中包含字母“{letter}”',
            '你的答复需要包含英文字母“{letter}”', '回答需包括字母“{letter}”', '要求包含英文字母“{letter}”'
        ]

    def check(self, response, slots):
        letter = slots["letter"]
        if not (letter in response):
            return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        letters_set = set(string.ascii_letters)
        letter = random.choice(list(letters_set))
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(letter=letter)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "letter": letter
            }
        }


class Rule_LETTER_NUM(Rule):

    def __init__(self):
        self.rule_type = "LETTER_NUM"
        self.meta_instr_specific_equal = [
            '回复中要有{num}个英文字母{letter}',
            '包含{num}个字母{letter}',
            '回复中“{letter}”出现{num}次',
            '确保字母“{letter}”出现{num}次',
        ]
        self.meta_instr_specific_more = [
            '回复中英文字母{letter}至少有{num}个',
            '确保英文字母“{letter}”至少出现{num}个',
            '回复中要有字母{letter}且最少{num}个',
            '回复中字母“{letter}”最少出现{num}个',
            '确保{letter}最少出现{num}个',
            '回复中要有“{letter}”且不少于{num}个',
            '回复中英文字母{letter}出现不少于{num}个',
            '确保英文字母“{letter}”出现不少于{num}个',
            '回复中要有字母{letter}且个数大于等于{num}',
            '回复中字母“{letter}”个数大于等于{num}',
            '确保“{letter}”个数大于等于{num}',
        ]
        self.meta_instr_specific_less = [
            '回复中要有英文字母{letter}且至多有{num}个',
            '确保英文字母“{letter}”出现，但至多出现{num}个',
            '回复中要有字母{letter}且最多出现{num}个',
            '确保字母“{letter}”出现，但最多出现{num}个',
            '回复中要有{letter}且出现不多于{num}个',
            '确保“{letter}”出现，但出现不多于{num}个',
            '回复中要有字母“{letter}”且个数小于等于{num}',
            '确保字母{letter}出现，但个数小于等于{num}',
        ]
        self.meta_instr_general_equal = [
            '回复中要有{num}个英文字母',
            '包含{num}个英文字母',
            '回复中英文字母出现{num}次',
            '确保英文字母出现{num}次',
        ]
        self.meta_instr_general_more = [
            '回复中英文字母至少有{num}个',
            '确保英文字母至少出现{num}个',
            '回复中要有英文字母且最少{num}个',
            '回复中英文字母最少出现{num}个',
            '确保英文字母最少出现{num}个',
            '回复中要有英文字母且不少于{num}个',
            '回复中英文字母出现不少于{num}个',
            '确保英文字母出现不少于{num}个',
            '回复中要有英文字母且个数大于等于{num}',
            '回复中英文字母个数大于等于{num}',
            '确保英文字母个数大于等于{num}',
        ]
        self.meta_instr_general_less = [
            '回复中要有英文字母且至多有{num}个',
            '确保英文字母出现，但至多出现{num}个',
            '回复中要有英文字母且最多出现{num}个',
            '确保英文字母出现，但最多出现{num}个',
            '回复中要有英文字母且出现不多于{num}个',
            '确保英文字母出现，但出现不多于{num}个',
            '回复中要有字母英文字母且个数小于等于{num}',
            '确保英文字母出现，但个数小于等于{num}',
        ]

    def check(self, response, slots):
        letter = slots["letter"]
        num = slots["num"]
        indicator_count_version = slots["indicator_count_version"]
        if letter == "general":
            matches = re.findall(r'[a-zA-Z]', response)
            if indicator_count_version == 'EQUAL':
                if len(matches) == num:
                    return True
                return False
            elif indicator_count_version == 'MORE':
                if len(matches) >= num:
                    return True
                return False
            elif indicator_count_version == 'LESS':
                if len(matches) <= num and len(matches) >= 1:
                    return True
                return False
            return False
        else:
            letter_num = response.count(letter)
            if indicator_count_version == 'EQUAL':
                if letter_num == num:
                    return True
                return False
            elif indicator_count_version == 'MORE':
                if letter_num >= num:
                    return True
                return False
            elif indicator_count_version == 'LESS':
                if letter_num <= num and letter_num >= 1:
                    return True
                return False
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_count_version = random.choice(['EQUAL', 'LESS', 'MORE'])
        letter_type = random.choice(['general', 'specific'])
        num = random.randint(1, 10)
        if letter_type == 'general':
            letter = 'general'
            if indicator_count_version == 'EQUAL':
                indicator_meta = random.randint(0, len(self.meta_instr_general_equal) - 1)
                instruction_meta = self.meta_instr_general_equal[indicator_meta].format(num=num)
            elif indicator_count_version == 'LESS':
                indicator_meta = random.randint(0, len(self.meta_instr_general_less) - 1)
                instruction_meta = self.meta_instr_general_less[indicator_meta].format(num=num)
            elif indicator_count_version == 'MORE':
                indicator_meta = random.randint(0, len(self.meta_instr_general_more) - 1)
                instruction_meta = self.meta_instr_general_more[indicator_meta].format(num=num)
        else:
            letters_set = set(string.ascii_letters)
            letter = random.choice(list(letters_set))
            if indicator_count_version == 'EQUAL':
                indicator_meta = random.randint(0, len(self.meta_instr_specific_equal) - 1)
                instruction_meta = self.meta_instr_specific_equal[indicator_meta].format(letter=letter, num=num)
            elif indicator_count_version == 'LESS':
                indicator_meta = random.randint(0, len(self.meta_instr_specific_less) - 1)
                instruction_meta = self.meta_instr_specific_less[indicator_meta].format(letter=letter, num=num)
            elif indicator_count_version == 'MORE':
                indicator_meta = random.randint(0, len(self.meta_instr_specific_more) - 1)
                instruction_meta = self.meta_instr_specific_more[indicator_meta].format(letter=letter, num=num)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "indicator_count_version": indicator_count_version,
                "letter": letter,
                "num": num
            }
        }


class Rule_LETTER_CASE_ONLY(Rule):

    def __init__(self):
        self.rule_type = "LETTER_CASE_ONLY"
        self.meta_instr_low = [
            '所有出现的英文单词都要小写',
            '英文单词需要小写',
            '英文字母全部小写',
            '英文以小写形式出现',
            '所有英文字母都小写',
            '英文全转化成小写',
            '所有出现的英文都要小写',
            '确保所有字母都是小写形式的',
        ]
        self.meta_instr_up = [
            '所有出现的英文单词都要大写',
            '英文单词需要大写',
            '英文字母全部大写',
            '英文以大写形式出现',
            '所有英文字母都大写',
            '英文全转化成大写',
            '所有出现的英文都要大写',
            '确保所有字母都是大写形式的',
        ]

    def check(self, response, slots):
        case = slots["case"]
        letter_matches = re.findall(r'[a-zA-Z]', response)
        if case == 'low':
            for char in letter_matches:
                if char.isalpha() and not char.islower():
                    return False
            return True
        else:
            for char in letter_matches:
                if char.isalpha() and not char.isupper():
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        case = random.choice(['up', 'low'])
        if case == 'low':
            indicator_meta = random.randint(0, len(self.meta_instr_low) - 1)
            instruction_meta = self.meta_instr_low[indicator_meta]
        else:
            indicator_meta = random.randint(0, len(self.meta_instr_up) - 1)
            instruction_meta = self.meta_instr_up[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "case": case
            }
        }


class Rule_LETTER_CASE_CONTAIN(Rule):

    def __init__(self):
        self.rule_type = "LETTER_CASE_CONTAIN"
        self.meta_instr_low = [
            '要有小写英文字母',
            '有以小写形式出现的字母',
            '需要有小写形式的英文字母',
            '小写字母要出现',
            '回答中需要包含小写字母',
        ]
        self.meta_instr_up = [
            '要有大写英文字母',
            '有以大写形式出现的字母',
            '需要有大写形式的英文字母',
            '大写字母要出现',
            '回答中需要包含大写字母',
        ]

    def check(self, response, slots):
        case = slots["case"]
        letter_matches = re.findall(r'[a-zA-Z]', response)
        if case == 'low':
            for char in letter_matches:
                if char.isalpha() and char.islower():
                    return True
            return False
        else:
            for char in letter_matches:
                if char.isalpha() and char.isupper():
                    return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        case = random.choice(['up', 'low'])
        if case == 'low':
            indicator_meta = random.randint(0, len(self.meta_instr_low) - 1)
            instruction_meta = self.meta_instr_low[indicator_meta]
        else:
            indicator_meta = random.randint(0, len(self.meta_instr_up) - 1)
            instruction_meta = self.meta_instr_up[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'case': case
            }
        }


class Rule_LETTER_CASE_LOC(Rule):

    def __init__(self):
        self.rule_type = "LETTER_CASE_LOC"
        self.meta_instr_up_first = [
            '所有英文单词首字母大写',
            '每个单词首字母改成大写',
            '所有单词第一个字母大写',
            '每个单词首字母大写',
            '每个单词第一个字母大写',
        ]
        self.meta_instr_up_last = [
            '所有英文单词末尾字母大写',
            '每个单词末尾字母改成大写',
            '所有单词最后一个字母大写',
            '每个单词末尾字母大写',
            '每个单词最后一个字母大写',
        ]
        self.meta_instr_low_first = [
            '所有英文单词首字母小写',
            '每个单词首字母改成小写',
            '所有单词第一个字母小写',
            '每个单词首字母小写',
            '每个单词第一个字母小写',
        ]
        self.meta_instr_low_last = [
            '所有英文单词末尾字母小写',
            '每个单词末尾字母改成小写',
            '所有单词最后一个字母小写',
            '每个单词末尾字母小写',
            '每个单词最后一个字母小写',
        ]

    def check(self, response, slots):
        case = slots["case"]
        loc = slots["loc"]
        words = re.findall(r'\b[a-zA-Z]+\b', response)  ###TODO：Let's 有漏洞
        if case == 'low' and loc == 'first':
            for word in words:
                if not word[0].islower():
                    return False
            return True
        elif case == 'up' and loc == 'first':
            for word in words:
                if not word[0].isupper():
                    return False
            return True
        elif case == 'low' and loc == 'last':
            for word in words:
                if not word[-1].islower():
                    return False
            return True
        elif case == 'up' and loc == 'last':
            for word in words:
                if not word[-1].isupper():
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        case = random.choice(['up', 'low'])
        loc = random.choice(['first', 'last'])
        if case == 'low' and loc == 'first':
            indicator_meta = random.randint(0, len(self.meta_instr_low_first) - 1)
            instruction_meta = self.meta_instr_low_first[indicator_meta]
        elif case == 'low' and loc == 'last':
            indicator_meta = random.randint(0, len(self.meta_instr_low_last) - 1)
            instruction_meta = self.meta_instr_low_last[indicator_meta]
        elif case == 'up' and loc == 'first':
            indicator_meta = random.randint(0, len(self.meta_instr_up_first) - 1)
            instruction_meta = self.meta_instr_up_first[indicator_meta]
        elif case == 'up' and loc == 'last':
            indicator_meta = random.randint(0, len(self.meta_instr_up_last) - 1)
            instruction_meta = self.meta_instr_up_last[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'case': case,
                'loc': loc
            }
        }


class Rule_LETTER_CASE_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "LETTER_CASE_NEGATE"
        self.meta_instr_low = [
            '英文字母不要有小写',
            '没有以小写形式出现的字母',
            '字母不能有小写形式',
            '小写字母不要出现',
            '回答中禁止包含小写字母',
            '英文单词不能小写',
            '禁止小写英文',
        ]
        self.meta_instr_up = [
            '英文字母不要有大写',
            '没有以大写形式出现的字母',
            '字母不能有大写形式',
            '大写字母不要出现',
            '回答中禁止包含大写字母',
            '英文单词不能大写',
            '禁止大写英文',
        ]

    def check(self, response, slots):
        case = slots["case"]
        letter_matches = re.findall(r'[a-zA-Z]', response)
        if case == 'low':
            for char in letter_matches:
                if char.isalpha() and char.islower():
                    return False
            return True
        else:
            for char in letter_matches:
                if char.isalpha() and char.isupper():
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        case = random.choice(['up', 'low'])
        if case == 'low':
            indicator_meta = random.randint(0, len(self.meta_instr_low) - 1)
            instruction_meta = self.meta_instr_low[indicator_meta]
        else:
            indicator_meta = random.randint(0, len(self.meta_instr_up) - 1)
            instruction_meta = self.meta_instr_up[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'case': case
            }
        }


################################################
# 重复类
################################################


class Rule_REPEAT_TEXT_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "REPEAT_TEXT_NEGATE"
        self.meta_instr = [
            '回答内不要出现重复的字符',
            '请确保回答没有任何重复字符',
            '答复里不要有重复的字符',
            '请避免在回答中使用重复的字符',
            '回答时不要使用重复字符',
            '回复中请确保没有重复字符',
            '确保回答内不包含重复的字符',
            '你的回答中不能有重复的字符',
            '请在回答中避免出现重复字符',
            '在回答中确保所有字符不重复',
            '去掉重复的字符',
            '字符不能重复',
            '每个字符只能出现一次',
        ]

    def check(self, response, slots):
        char_dict = {}
        for char in response:
            if char not in char_dict:
                char_dict[char] = 0
            char_dict[char] += 1
            if char_dict[char] > 1:
                return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_REPEAT_LOC_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "REPEAT_LOC_NEGATE"
        self.meta_instr = [
            '回答内不要出现和问题中重复的句子',
            '请确保回答没有任何和问题中一样的句子',
            '请避免在回答中使用与问题中重复的句子',
            '回答时不要使用问题中出现过的句子',
            '去掉和问题中重复的句子',
            '回答不能和问题有任何一句话重复',
            '回答中不能有与问题完全一致的句子',
        ]

    def check(self, response, slots):
        instruction = slots["instruction"]
        sents_prompt = set(cut_sent(instruction))
        sents_response = cut_sent(response)
        for sent in sents_response:
            if sent in sents_prompt or sent.strip() in sents_prompt:
                return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "instruction": instruction_overall
            }
        }


class Rule_REPEAT_TEXT_LOC_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "REPEAT_TEXT_LOC_NEGATE"
        self.meta_instr = [
            '每句的第一个字不可重复',
            '每句话的首字符不能重复',
            '不要出现第一个字重复的句子',
            '去掉第一个字重复的句子',
            '每句话的第一个字禁止重复',
        ]

    def check(self, response, slots):
        sents_response = cut_sent(response)
        word_dict = dict()
        for sent in sents_response:
            for word in sent:
                if word in word_dict:
                    return False
                word_dict[word] = 1
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


################################################
# EMOJI类
################################################


class Rule_EMOJI_NUM(Rule):

    def __init__(self):
        self.rule_type = "EMOJI_NUM"
        self.meta_instr_equal = [
            '在回答内加上{num1}个合适的emoji表情',
            '请在回答中加入{num1}个表情符号',
            '回答中需要包含{num1}个emoji表情',
            '在你的回答里添加{num1}个适当的emoji表情',
            '请在答复中加上{num1}个emoji',
            '回答时请包含{num1}个适用的表情符号',
            '在回复中插入{num1}个表情符号',
            '你的回答应带有{num1}个emoji表情',
            '请将{num1}个emoji添加到回答中',
            '回答需要嵌入{num1}个表情符号',
            '在你的答复中使用{num1}个emoji表情',
            '回答中的emoji出现{num1}次',
        ]
        self.meta_instr_more = [
            '在回答内加上不少于{num1}个合适的emoji表情',
            '请在回答中加入大于等于{num1}个表情符号',
            '回答中需要包含不少于{num1}个emoji表情',
            '在你的回答里添加大于等于{num1}个适当的emoji表情',
            '请在答复中加上不少于{num1}个emoji',
            '回答时请包含大于等于{num1}个适用的表情符号',
            '在回复中插入不少于{num1}个表情符号',
            '你的回答应带有大于等于{num1}个emoji表情',
            '请将不少于{num1}个emoji添加到回答中',
            '回答需要嵌入大于等于{num1}个表情符号',
            '在你的答复中使用不少于{num1}个emoji表情',
            '回答中的emoji不少于{num1}',
        ]
        self.meta_instr_less = [
            '在回答内加上不超过{num1}个合适的emoji表情',
            '请在回答中加入小于等于{num1}个表情符号',
            '回答中需要包含不超过{num1}个emoji表情',
            '在你的回答里添加小于等于{num1}个适当的emoji表情',
            '请在答复中加上不超过{num1}个emoji',
            '回答时请包含小于等于{num1}个适用的表情符号',
            '在回复中插入不超过{num1}个表情符号',
            '你的回答应带有小于等于{num1}个emoji表情',
            '请将不超过{num1}个emoji添加到回答中',
            '回答需要嵌入小于等于{num1}个表情符号',
            '在你的答复中使用不超过{num1}个emoji表情',
            '回答中的emoji不超过{num1}',
        ]
        self.meta_instr_interval = [
            'emoji数量控制在{num1}到{num2}之间',
            '回答中的emoji不超过{num2}个，不少于{num1}个',
            'emoji个数在{num1}到{num2}之间',
            '表情个数在{num1}到{num2}',
            'emoji表情在{num1}到{num2}个',
        ]

    def check(self, response, slots):
        num1 = slots["num1"]
        num2 = slots["num2"]
        interval_type = slots["interval_type"]
        if interval_type == "EQUAL":
            if count_emoji(response) == num1:
                return True
            return False
        elif interval_type == "MORE":
            if count_emoji(response) >= num1:
                return True
            return False
        elif interval_type == "LESS":
            if count_emoji(response) <= num1:
                return True
            return False
        elif interval_type == "INTERVAL":
            if count_emoji(response) <= num2 and count_emoji(response) >= num1:
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        interval_type = random.choice(["EQUAL", "MORE", "LESS", "INTERVAL"])
        random_nums = random.sample(range(1, 10), 2)
        random_num1, random_num2 = min(random_nums), max(random_nums)

        if interval_type == "EQUAL":
            indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
            instruction_meta = self.meta_instr_equal[indicator_meta].format(num1=random_num1)
            num1 = random_num1
            num2 = ''
        elif interval_type == "MORE":
            indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
            instruction_meta = self.meta_instr_more[indicator_meta].format(num1=random_num1)
            num1 = random_num1
            num2 = ''
        elif interval_type == "LESS":
            indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
            instruction_meta = self.meta_instr_less[indicator_meta].format(num1=random_num1)
            num1 = random_num1
            num2 = ''
        elif interval_type == "INTERVAL":
            indicator_meta = random.randint(0, len(self.meta_instr_interval) - 1)
            instruction_meta = self.meta_instr_interval[indicator_meta].format(num1=random_num1, num2=random_num2)
            num1 = random_num1
            num2 = random_num2

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "interval_type": interval_type,
                "num1": num1,
                "num2": num2
            }
        }


class Rule_EMOJI_ONLY(Rule):

    def __init__(self):
        self.rule_type = "EMOJI_ONLY"
        self.meta_instr = [
            '把回答翻译成emoji',
            '只用emoji回答',
            '回复只用表情符号',
            '请用emoji作答',
            '回答时仅使用emoji表情',
            '答案只包含emoji',
            '回复中不要使用文字，只用emoji',
            '只用表情符号来回答',
            '用emoji来回复',
            '仅用emoji表情符号回答问题',
            '回答只使用emoji表情符号',
        ]

    def check(self, response, slots):
        for char in response:
            if char not in emoji.EMOJI_DATA:
                return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_EMOJI_CONTAIN(Rule):

    def __init__(self):
        self.rule_type = "EMOJI_CONTAIN"
        self.meta_instr = [
            '加些emoji表情',
            '加入一些emoji',
            '需要包含emoji',
            '回复中要包含表情符号',
            '回答时加入emoji表情',
            '答案需包含emoji',
            '回复中加上emoji',
            '加入表情符号来回答',
            '回答中使用一些emoji表情符号',
        ]

    def check(self, response, slots):
        for char in response:
            if char in emoji.EMOJI_DATA:
                return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_EMOJI_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "EMOJI_NEGATE"
        self.meta_instr = [
            '不要出现emoji',
            '不要出现表情',
            '不带emoji',
            '不带表情',
            '输出中没有emoji',
            '不要使用表情',
            '避免emoji',
            '回答中不能出现表情',
            '禁止出现emoji',
            '忽略所有表情',
            '省略所有emoji',
            '不要输出表情',
            '去除所有emoji',
            '省略回复中的表情',
            '去掉所有emoji',
            '不能输出表情',
            '输出结果不能包含emoji',
            '输出不要带表情',
            '不要表情',
            '不使用表情符号',
            '禁止使用表情符号',
        ]

    def check(self, response, slots):
        for char in response:
            if char in emoji.EMOJI_DATA:
                return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_EMOJI_START_END(Rule):

    def __init__(self):
        self.rule_type = "EMOJI_START_END"
        self.meta_instr_start = [
            '回答开头出现emoji',
            '在回复的开头处使用emoji',
            '开头加些emoji表情',
            '开头加入一些emoji',
            '开头需要包含emoji',
            '回复开头要包含表情符号',
            '回答开头时加入emoji表情',
            '答案开头需包含emoji',
            '回复开头加上emoji',
            '开头加入表情符号来回答',
            '回答开头使用一些emoji表情符号',
            '每次回复用户都要在开头加emoji',
        ]
        self.meta_instr_end = [
            '回答结尾出现emoji',
            '在回复的末尾使用emoji',
            '结尾加些emoji表情',
            '末尾加入一些emoji',
            '结尾需要包含emoji',
            '回复末尾要包含表情符号',
            '回答结尾时加入emoji表情',
            '答案末尾需包含emoji',
            '回复结尾加上emoji',
            '末尾加入表情符号来回答',
            '回答结尾使用一些emoji表情符号',
            '每次回复用户都要在结尾加emoji',
        ]

    def check(self, response, slots):
        loc = slots["loc"]
        if loc == 'start':
            if response[0] in emoji.EMOJI_DATA:
                return True
            return False
        else:
            if response[-1] in emoji.EMOJI_DATA:
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        loc = random.choice(['start', 'end'])
        self.meta_instr = self.meta_instr_start if loc == 'start' else self.meta_instr_end
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "loc": loc
            }
        }


class Rule_EMOJI_NUM_START_END(Rule):

    def __init__(self):
        self.rule_type = "EMOJI_NUM_START_END"
        self.meta_instr_start = [
            '回答开头出现{num}个emoji',
            '在回复的开头处使用{num}个emoji',
            '开头加{num}个emoji表情',
            '开头加入{num}个emoji',
            '开头需要包含{num}个emoji',
            '回复开头要包含{num}个表情符号',
            '回答开头时加入{num}个emoji表情',
            '答案开头需包含{num}个emoji',
            '回复开头加上{num}个emoji',
            '开头加入{num}个表情符号来回答',
            '回答开头使用{num}个emoji表情符号',
            '每次回复用户都要在开头加{num}个emoji',
        ]
        self.meta_instr_end = [
            '回答结尾出现{num}个emoji',
            '在回复的末尾使用{num}个emoji',
            '结尾加{num}个emoji表情',
            '末尾加入{num}个emoji',
            '结尾需要包含{num}个emoji',
            '回复末尾要包含{num}个表情符号',
            '回答结尾时加入{num}个emoji表情',
            '答案末尾需包含{num}个emoji',
            '回复结尾加上{num}个emoji',
            '末尾加入{num}个表情符号来回答',
            '回答结尾使用{num}个emoji表情符号',
            '每次回复用户都要在结尾加{num}个emoji',
        ]

    def check(self, response, slots):
        loc = slots["loc"]
        num = slots["num"]
        if loc == 'start':
            for char in response[:num]:
                if char not in emoji.EMOJI_DATA:
                    return False
            return True
        else:
            for char in response[-num:]:
                if char not in emoji.EMOJI_DATA:
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        loc = random.choice(['start', 'end'])
        num = random.randint(1, 5)
        self.meta_instr = self.meta_instr_start if loc == 'start' else self.meta_instr_end
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        if random.random() < 0.5:
            zh_num = arabic_to_chinese(num)
            instruction_meta = self.meta_instr[indicator_meta].format(num=zh_num)
        else:
            instruction_meta = self.meta_instr[indicator_meta].format(num=num)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "loc": loc,
                "num": num
            }
        }


class Rule_EMOJI_NUM_LOC(Rule):

    def __init__(self):
        self.rule_type = "EMOJI_NUM_LOC"
        self.meta_instr_equal = [
            '第{which}段加上{num1}个合适的emoji表情',
            '请在第{which}段加入{num1}个合适的表情符号',
            '第{which}段需要包含{num1}个合适的emoji表情',
            '在第{which}段里添加{num1}个适当的emoji表情',
            '请在第{which}段加上{num1}个合适的emoji',
            '第{which}段请包含{num1}个适用的表情符号',
            '在第{which}段插入{num1}个合适的表情符号',
            '第{which}段应带有{num1}个适当的emoji表情',
            '请将{num1}个合适的emoji添加到第{which}段',
            '第{which}段需要嵌入{num1}个适当的表情符号',
            '在第{which}段中使用{num1}个合适的emoji表情',
            '第{which}段中的emoji出现{num1}次',
        ]
        self.meta_instr_more = [
            '第{which}段加上不少于{num1}个合适的emoji表情',
            '请在第{which}段加入大于等于{num1}个合适的表情符号',
            '第{which}段需要包含不少于{num1}个合适的emoji表情',
            '在第{which}段里添加大于等于{num1}个适当的emoji表情',
            '请在第{which}段加上不少于{num1}个合适的emoji',
            '第{which}段请包含大于等于{num1}个适用的表情符号',
            '第{which}段中插入不少于{num1}个合适的表情符号',
            '第{which}段应带有大于等于{num1}个适当的emoji表情',
            '请将不少于{num1}个合适的emoji添加到第{which}段',
            '第{which}段需要嵌入大于等于{num1}个适当的表情符号',
            '在第{which}段使用不少于{num1}个合适的emoji表情',
            '第{which}段的emoji不少于{num1}',
        ]
        self.meta_instr_less = [
            '在第{which}段加上不超过{num1}个合适的emoji表情',
            '请在第{which}段加入小于等于{num1}个合适的表情符号',
            '第{which}段中需要包含不超过{num1}个合适的emoji表情',
            '在第{which}段里添加小于等于{num1}个适当的emoji表情',
            '请在第{which}段加上不超过{num1}个合适的emoji',
            '第{which}段请包含小于等于{num1}个适用的表情符号',
            '第{which}段插入不超过{num1}个合适的表情符号',
            '第{which}段应带有小于等于{num1}个适当的emoji表情',
            '请将不超过{num1}个合适的emoji添加到第{which}段',
            '第{which}段需要嵌入小于等于{num1}个适当的表情符号',
            '在第{which}段中使用不超过{num1}个合适的emoji表情',
            '第{which}段中的emoji不超过{num1}',
        ]
        self.meta_instr_interval = [
            '第{which}段emoji数量控制在{num1}到{num2}之间',
            '第{which}段中的表情符号不超过{num2}个，不少于{num1}个',
            '第{which}段emoji个数在{num1}到{num2}之间',
            '第{which}段emoji表情个数在{num1}到{num2}',
            '第{which}段emoji符号在{num1}到{num2}个',
        ]

    def check(self, response, slots):
        interval_type = slots["interval_type"]
        num1 = slots["num1"]
        num2 = slots["num2"]
        which = slots["which"]
        paragraphs = re.split(r'\n\s*\n*', response.strip())

        if interval_type == "EQUAL":
            for i, paragraph in enumerate(paragraphs):
                if i + 1 == which and count_emoji(paragraph) == num1:
                    return True
            return False
        elif interval_type == "MORE":
            for i, paragraph in enumerate(paragraphs):
                if i + 1 == which and count_emoji(paragraph) >= num1:
                    return True
            return False
        elif interval_type == "LESS":
            for i, paragraph in enumerate(paragraphs):
                if i + 1 == which and count_emoji(paragraph) <= num1:
                    return True
            return False
        elif interval_type == "INTERVAL":
            for i, paragraph in enumerate(paragraphs):
                if i + 1 == which and count_emoji(paragraph) <= num2 and count_emoji(paragraph) >= num1:
                    return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        interval_type = random.choice(["EQUAL", "MORE", "LESS", "INTERVAL"])
        random_nums = random.sample(range(1, 10), 2)
        random_num1, random_num2 = min(random_nums), max(random_nums)
        paragraphs = re.split(r'\n\s*\n*', response.strip())
        paragraph_count = len(paragraphs)
        random_which = random.randint(1, min(paragraph_count, 10))

        if interval_type == "EQUAL":
            indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
            instruction_meta = self.meta_instr_equal[indicator_meta].format(which=random_which, num1=random_num1)
            which = random_which
            num1 = random_num1
            num2 = ''
        elif interval_type == "MORE":
            indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
            instruction_meta = self.meta_instr_more[indicator_meta].format(which=random_which, num1=random_num1)
            which = random_which
            num1 = random_num1
            num2 = ''
        elif interval_type == "LESS":
            indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
            instruction_meta = self.meta_instr_less[indicator_meta].format(which=random_which, num1=random_num1)
            which = random_which
            num1 = random_num1
            num2 = ''
        elif interval_type == "INTERVAL":
            indicator_meta = random.randint(0, len(self.meta_instr_interval) - 1)
            instruction_meta = self.meta_instr_interval[indicator_meta].format(which=random_which,
                                                                               num1=random_num1,
                                                                               num2=random_num2)
            which = random_which
            num1 = random_num1
            num2 = random_num2

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "interval_type": interval_type,
                "num1": num1,
                "num2": num2,
                "which": which
            }
        }


################################################
# 字数类
################################################


class Rule_ZH_NUM(Rule):

    def __init__(self):
        self.rule_type = "ZH_NUM"
        self.meta_instr_specific_equal = [
            '回复中要有{num}个{zh_word}字',
            '包含{num}个{zh_word}字',
            '回复中“{zh_word}”出现{num}次',
            '确保“{zh_word}”出现{num}次',
        ]
        self.meta_instr_specific_more = [
            '回复中{zh_word}字至少有{num}个',
            '确保“{zh_word}”至少出现{num}个',
            '回复中要有{zh_word}字且最少{num}个',
            '回复中“{zh_word}”最少出现{num}个',
            '回复中要有“{zh_word}”且不少于{num}个',
            '确保“{zh_word}”字出现不少于{num}个',
            '回复中“{zh_word}”字个数大于等于{num}',
            '确保“{zh_word}”个数大于等于{num}',
            '包含至少{num}个“{zh_word}”字',
        ]
        self.meta_instr_specific_less = [
            '回复中要有{zh_word}字且至多有{num}个',
            '确保“{zh_word}”出现，但至多出现{num}个',
            '回复中要有“{zh_word}”字且最多出现{num}个',
            '确保“{zh_word}”出现，但最多出现{num}个',
            '回复中要有“{zh_word}”且出现不多于{num}个',
            '确保“{zh_word}”字出现，但出现不多于{num}个',
            '回复中要有“{zh_word}”字且个数小于等于{num}',
            '确保“{zh_word}”出现，但个数小于等于{num}',
        ]
        self.meta_instr_general_equal = [
            '回复中要有{num}个字',
            '包含{num}个字',
            '用{num}个字概括',
            '用{num}字回答',
            '用{num}个字表达',
            '用{num}个字概括回答',
            '答案为{num}个字',
            '{num}字',
            '必须是{num}个字',
        ]
        self.meta_instr_general_more = [
            '至少{num}个字'
            '{num}个字以上',
            '不少于{num}字',
        ]
        self.meta_instr_general_less = [
            '{num}个字以内',
            '要求{num}个字以内',
            '回答不超过{num}个字',
            '字数小于等于{num}个字',
            '不超过{num}个字',
        ]
        self.meta_instr_general_interval = [
            '{num1}-{num2}字左右',
            '{num1}至{num2}字',
            '{num1}字到{num2}字之间',
            '请用{num1}-{num2}字回答',
            '{num1}到{num2}字',
            '字数{num1}-{num2}字',
        ]

    def check(self, response, slots):
        zh_word = slots["zh_word"]
        num = slots["num"]
        indicator_count_version = slots["indicator_count_version"]
        if zh_word == "general":
            #matches = re.findall(r'[\u4e00-\u9fa5]', response)
            num_char = count_chinese_chars(response)
            if indicator_count_version == 'EQUAL':
                if num_char == num:
                    return True
                return False
            elif indicator_count_version == 'MORE':
                if num_char >= num:
                    return True
                return False
            elif indicator_count_version == 'LESS':
                if num_char <= num and num_char >= 1:
                    return True
                return False
            return False
        else:
            zh_num = response.count(zh_word)
            if indicator_count_version == 'EQUAL':
                if zh_num == num:
                    return True
                return False
            elif indicator_count_version == 'MORE':
                if zh_num >= num:
                    return True
                return False
            elif indicator_count_version == 'LESS':
                if zh_num <= num and zh_num >= 1:
                    return True
                return False
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_count_version = random.choice(['EQUAL', 'LESS', 'MORE'])
        letter_type = random.choice(['general', 'specific'])

        if letter_type == 'general':
            zh_word = 'general'
            num = random.choice([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800,
                900, 1000, 2000, 3000, 4000
            ])
            if indicator_count_version == 'EQUAL':
                indicator_meta = random.randint(0, len(self.meta_instr_general_equal) - 1)
                if random.random() < 0.5:
                    zh_num = arabic_to_chinese(num)
                    instruction_meta = self.meta_instr_general_equal[indicator_meta].format(num=zh_num)
                else:
                    instruction_meta = self.meta_instr_general_equal[indicator_meta].format(num=num)
            elif indicator_count_version == 'LESS':
                indicator_meta = random.randint(0, len(self.meta_instr_general_less) - 1)
                if random.random() < 0.5:
                    zh_num = arabic_to_chinese(num)
                    instruction_meta = self.meta_instr_general_less[indicator_meta].format(num=zh_num)
                else:
                    instruction_meta = self.meta_instr_general_less[indicator_meta].format(num=num)
            elif indicator_count_version == 'MORE':
                indicator_meta = random.randint(0, len(self.meta_instr_general_more) - 1)
                if random.random() < 0.5:
                    zh_num = arabic_to_chinese(num)
                    instruction_meta = self.meta_instr_general_more[indicator_meta].format(num=zh_num)
                else:
                    instruction_meta = self.meta_instr_general_more[indicator_meta].format(num=num)
        else:
            num = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            zh_word_set = set(extract_chinese_characters(response))
            if len(zh_word_set) == 0:
                return {'instruction': instruction, 'rule': '', 'output': response, 'type': None, 'slots': {}}
            zh_word = random.choice(list(zh_word_set))
            if indicator_count_version == 'EQUAL':
                indicator_meta = random.randint(0, len(self.meta_instr_specific_equal) - 1)
                if random.random() < 0.5:
                    zh_num = arabic_to_chinese(num)
                    instruction_meta = self.meta_instr_specific_equal[indicator_meta].format(zh_word=zh_word,
                                                                                             num=zh_num)
                else:
                    instruction_meta = self.meta_instr_specific_equal[indicator_meta].format(zh_word=zh_word, num=num)
            elif indicator_count_version == 'LESS':
                indicator_meta = random.randint(0, len(self.meta_instr_specific_less) - 1)
                if random.random() < 0.5:
                    zh_num = arabic_to_chinese(num)
                    instruction_meta = self.meta_instr_specific_less[indicator_meta].format(zh_word=zh_word, num=zh_num)
                else:
                    instruction_meta = self.meta_instr_specific_less[indicator_meta].format(zh_word=zh_word, num=num)
            elif indicator_count_version == 'MORE':
                indicator_meta = random.randint(0, len(self.meta_instr_specific_more) - 1)
                if random.random() < 0.5:
                    zh_num = arabic_to_chinese(num)
                    instruction_meta = self.meta_instr_specific_more[indicator_meta].format(zh_word=zh_word, num=zh_num)
                else:
                    instruction_meta = self.meta_instr_specific_more[indicator_meta].format(zh_word=zh_word, num=num)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "indicator_count_version": indicator_count_version,
                "zh_word": zh_word,
                "num": num
            }
        }


class Rule_ZH_NUM_LOC(Rule):

    def __init__(self):
        self.rule_type = "ZH_NUM_LOC"
        self.meta_instr_equal = [
            '每句话{num}个字',
            '每个句子{num}个字',
            '每句{num}个字',
        ]
        self.meta_instr_less = [
            '每句话不超过{num}个字',
            '每个句子不超过{num}个字',
            '每句不超过{num}个字',
            '每句话不多于{num}个字',
            '每个句子不多于{num}个字',
            '每句不多于{num}个字',
            '每句话小于等于{num}个字',
            '每个句子小于等于{num}个字',
            '每句小于等于{num}个字',
        ]
        self.meta_instr_more = [
            '每句话不少于{num}个字',
            '每个句子不少于{num}个字',
            '每句不少于{num}个字',
            '每句话至少{num}个字',
            '每个句子至少{num}个字',
            '每句至少{num}个字',
            '每句话大于等于{num}个字',
            '每个句子大于等于{num}个字',
            '每句大于等于{num}个字',
        ]

    def check(self, response, slots):
        num = slots["num"]
        indicator_count_version = slots["indicator_count_version"]
        sents = cut_sent(response)
        if indicator_count_version == 'EQUAL':
            for i, sent in enumerate(sents):
                #matches = re.findall(r'[\u4e00-\u9fa5]', sent)
                num_char = count_chinese_chars(sent)
                if not (num_char == num):
                    return False
            return True
        elif indicator_count_version == 'MORE':
            for i, sent in enumerate(sents):
                #matches = re.findall(r'[\u4e00-\u9fa5]', sent)
                num_char = count_chinese_chars(sent)
                if not (num_char >= num):
                    return False
            return True
        elif indicator_count_version == 'LESS':
            for i, sent in enumerate(sents):
                #matches = re.findall(r'[\u4e00-\u9fa5]', sent)
                num_char = count_chinese_chars(sent)
                if not (num_char <= num and num_char >= 1):
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_count_version = random.choice(['EQUAL', 'LESS', 'MORE'])
        num = random.choice([10, 15, 20, 25, 30, 35, 40, 45, 50])

        if indicator_count_version == 'EQUAL':
            indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
            if random.random() < 0.5:
                zh_num = arabic_to_chinese(num)
                instruction_meta = self.meta_instr_equal[indicator_meta].format(num=zh_num)
            else:
                instruction_meta = self.meta_instr_equal[indicator_meta].format(num=num)
        elif indicator_count_version == 'LESS':
            indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
            if random.random() < 0.5:
                zh_num = arabic_to_chinese(num)
                instruction_meta = self.meta_instr_less[indicator_meta].format(num=zh_num)
            else:
                instruction_meta = self.meta_instr_less[indicator_meta].format(num=num)
        elif indicator_count_version == 'MORE':
            indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
            if random.random() < 0.5:
                zh_num = arabic_to_chinese(num)
                instruction_meta = self.meta_instr_more[indicator_meta].format(num=zh_num)
            else:
                instruction_meta = self.meta_instr_more[indicator_meta].format(num=num)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "indicator_count_version": indicator_count_version,
                "num": num
            }
        }


################################################
# 标点类
################################################


class Rule_PUNC_NUM(Rule):

    def __init__(self):
        self.rule_type = "PUNC_NUM"
        self.meta_instr_equal = [
            '在回答内加上{num1}个标点符号',
            '请在回答中加入{num1}个标点',
            '回答中需要包含{num1}个标点',
            '在你的回答里添加{num1}个标点符号',
            '请在答复中加上{num1}个标点符号',
            '回答时请包含{num1}个标点',
            '在回复中插入{num1}个标点符号',
            '你的回答应带有{num1}个标点',
            '请将{num1}个标点添加到回答中',
            '回答需要嵌入{num1}个标点符号',
            '在你的答复中使用{num1}个标点',
            '回答中的标点符号出现{num1}次',
        ]
        self.meta_instr_more = [
            '在回答内加上不少于{num1}个标点符号',
            '请在回答中加入大于等于{num1}个标点',
            '回答中需要包含不少于{num1}个标点',
            '在你的回答里添加大于等于{num1}个标点符号',
            '请在答复中加上不少于{num1}个标点',
            '回答时请包含大于等于{num1}个标点符号',
            '在回复中插入不少于{num1}个标点符号',
            '你的回答应带有大于等于{num1}个标点',
            '请将不少于{num1}个标点添加到回答中',
            '回答需要嵌入大于等于{num1}个标点符号',
            '在你的答复中使用不少于{num1}个标点',
            '回答中的标点不少于{num1}',
        ]
        self.meta_instr_less = [
            '在回答内加上不超过{num1}个标点',
            '请在回答中加入小于等于{num1}个标点符号',
            '回答中需要包含不超过{num1}个标点',
            '在你的回答里添加小于等于{num1}个标点符号',
            '请在答复中加上不超过{num1}个标点',
            '回答时请包含小于等于{num1}个标点符号',
            '在回复中插入不超过{num1}个标点',
            '你的回答应带有小于等于{num1}个标点符号',
            '请将不超过{num1}个标点添加到回答中',
            '回答需要嵌入小于等于{num1}个标点符号',
            '在你的答复中使用不超过{num1}个标点',
            '回答中的标点符号不超过{num1}',
        ]
        self.meta_instr_interval = [
            '标点控制在{num1}到{num2}之间',
            '回答中的标点符号不超过{num2}个，不少于{num1}个',
            '标点个数在{num1}到{num2}之间',
            '标点符号个数在{num1}到{num2}',
            '标点在{num1}到{num2}个',
        ]

    def check(self, response, slots):
        num1 = slots["num1"]
        num2 = slots["num2"]
        interval_type = slots["interval_type"]
        if interval_type == "EQUAL":
            if count_punc(response) == num1:
                return True
            return False
        elif interval_type == "MORE":
            if count_punc(response) >= num1:
                return True
            return False
        elif interval_type == "LESS":
            if count_punc(response) <= num1:
                return True
            return False
        elif interval_type == "INTERVAL":
            if count_punc(response) <= num2 and count_punc(response) >= num1:
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        interval_type = random.choice(["EQUAL", "MORE", "LESS", "INTERVAL"])
        random_nums = random.sample(range(1, 10), 2)
        random_num1, random_num2 = min(random_nums), max(random_nums)

        if interval_type == "EQUAL":
            indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
            instruction_meta = self.meta_instr_equal[indicator_meta].format(num1=random_num1)
            num1 = random_num1
            num2 = ''
        elif interval_type == "MORE":
            indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
            instruction_meta = self.meta_instr_more[indicator_meta].format(num1=random_num1)
            num1 = random_num1
            num2 = ''
        elif interval_type == "LESS":
            indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
            instruction_meta = self.meta_instr_less[indicator_meta].format(num1=random_num1)
            num1 = random_num1
            num2 = ''
        elif interval_type == "INTERVAL":
            indicator_meta = random.randint(0, len(self.meta_instr_interval) - 1)
            instruction_meta = self.meta_instr_interval[indicator_meta].format(num1=random_num1, num2=random_num2)
            num1 = random_num1
            num2 = random_num2

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "interval_type": interval_type,
                "num1": num1,
                "num2": num2
            }
        }


class Rule_PUNC_NUM_LOC(Rule):

    def __init__(self):
        self.rule_type = "PUNC_NUM_LOC"
        self.meta_instr_equal = [
            '第{which}段加上{num1}个标点符号',
            '请在第{which}段加入{num1}个标点',
            '第{which}段需要包含{num1}个标点符号',
            '在第{which}段里添加{num1}个标点',
            '请在第{which}段加上{num1}个标点符号',
            '第{which}段请包含{num1}个标点',
            '在第{which}段插入{num1}个标点符号',
            '第{which}段应带有{num1}个标点',
            '请将{num1}个标点符号添加到第{which}段',
            '第{which}段需要嵌入{num1}个标点',
            '在第{which}段中使用{num1}个标点符号',
            '第{which}段中的标点出现{num1}次',
        ]
        self.meta_instr_more = [
            '第{which}段加上不少于{num1}个标点符号',
            '请在第{which}段加入大于等于{num1}个标点',
            '第{which}段需要包含不少于{num1}个标点符号',
            '在第{which}段里添加大于等于{num1}个标点',
            '请在第{which}段加上不少于{num1}个标点符号',
            '第{which}段请包含大于等于{num1}个标点',
            '第{which}段中插入不少于{num1}个标点符号',
            '第{which}段应带有大于等于{num1}个标点',
            '请将不少于{num1}个标点符号添加到第{which}段',
            '第{which}段需要嵌入大于等于{num1}个标点',
            '在第{which}段使用不少于{num1}个标点符号',
            '第{which}段的标点不少于{num1}',
        ]
        self.meta_instr_less = [
            '在第{which}段加上不超过{num1}个标点符号',
            '请在第{which}段加入小于等于{num1}个标点',
            '第{which}段中需要包含不超过{num1}个标点符号',
            '在第{which}段里添加小于等于{num1}个标点',
            '请在第{which}段加上不超过{num1}个标点符号',
            '第{which}段请包含小于等于{num1}个标点',
            '第{which}段插入不超过{num1}个标点符号',
            '第{which}段应带有小于等于{num1}个标点',
            '请将不超过{num1}个标点符号添加到第{which}段',
            '第{which}段需要嵌入小于等于{num1}个标点',
            '在第{which}段中使用不超过{num1}个标点符号',
            '第{which}段中的标点不超过{num1}',
        ]
        self.meta_instr_interval = [
            '第{which}段标点数量控制在{num1}到{num2}之间',
            '第{which}段中的标点符号不超过{num2}个，不少于{num1}个',
            '第{which}段标点个数在{num1}到{num2}之间',
            '第{which}段标点符号个数在{num1}到{num2}',
            '第{which}段标点符号在{num1}到{num2}个',
        ]

    def check(self, response, slots):
        interval_type = slots["interval_type"]
        num1 = slots["num1"]
        num2 = slots["num2"]
        which = slots["which"]
        paragraphs = re.split(r'\n\s*\n*', response.strip())

        if interval_type == "EQUAL":
            for i, paragraph in enumerate(paragraphs):
                if i + 1 == which and count_punc(paragraph) == num1:
                    return True
            return False
        elif interval_type == "MORE":
            for i, paragraph in enumerate(paragraphs):
                if i + 1 == which and count_punc(paragraph) >= num1:
                    return True
            return False
        elif interval_type == "LESS":
            for i, paragraph in enumerate(paragraphs):
                if i + 1 == which and count_punc(paragraph) <= num1:
                    return True
            return False
        elif interval_type == "INTERVAL":
            for i, paragraph in enumerate(paragraphs):
                if i + 1 == which and count_punc(paragraph) <= num2 and count_punc(paragraph) >= num1:
                    return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        interval_type = random.choice(["EQUAL", "MORE", "LESS", "INTERVAL"])
        random_nums = random.sample(range(1, 10), 2)
        random_num1, random_num2 = min(random_nums), max(random_nums)
        paragraphs = re.split(r'\n\s*\n*', response.strip())
        paragraph_count = len(paragraphs)
        random_which = random.randint(1, min(paragraph_count, 10))

        if interval_type == "EQUAL":
            indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
            instruction_meta = self.meta_instr_equal[indicator_meta].format(which=random_which, num1=random_num1)
            which = random_which
            num1 = random_num1
            num2 = ''
        elif interval_type == "MORE":
            indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
            instruction_meta = self.meta_instr_more[indicator_meta].format(which=random_which, num1=random_num1)
            which = random_which
            num1 = random_num1
            num2 = ''
        elif interval_type == "LESS":
            indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
            instruction_meta = self.meta_instr_less[indicator_meta].format(which=random_which, num1=random_num1)
            which = random_which
            num1 = random_num1
            num2 = ''
        elif interval_type == "INTERVAL":
            indicator_meta = random.randint(0, len(self.meta_instr_interval) - 1)
            instruction_meta = self.meta_instr_interval[indicator_meta].format(which=random_which,
                                                                               num1=random_num1,
                                                                               num2=random_num2)
            which = random_which
            num1 = random_num1
            num2 = random_num2

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "interval_type": interval_type,
                "num1": num1,
                "num2": num2,
                "which": which
            }
        }


class Rule_PUNC_REPLACE(Rule):

    def __init__(self):
        self.rule_type = "PUNC_REPLACE"
        self.meta_instr_specific = [
            '在回复中，用“{punct_new}”替换“{punct_ori}”',
            '在回复中，把“{punct_ori}”换成“{punct_new}”',
            '用“{punct_new}”替换“{punct_ori}”',
            '把“{punct_ori}”换成“{punct_new}”',
            '所有“{punct_ori}”都用“{punct_new}”换掉',
            '把所有“{punct_ori}”换成“{punct_new}”',
        ]
        self.meta_instr_general = [
            '在回复中，用“{punct_new}”替换所有标点符号',
            '在回复中，把所有标点符号换成“{punct_new}”',
            '用“{punct_new}”替换所有标点符号',
            '把所有标点符号换成“{punct_new}”',
            '所有标点符号都用“{punct_new}”换掉',
        ]
        self.meta_instr_zh = [
            '英文标点替换为中文标点',
            '标点符号替换为中文格式',
            '标点替换为中文格式',
        ]
        self.meta_instr_en = [
            '中文标点替换为英文标点',
            '标点符号替换为英文格式',
            '标点替换为英文格式',
        ]

    def check(self, response, slots):
        punc_type = slots['punc_type']
        punct_ori = slots['punct_ori']
        punct_new = slots['punct_new']
        en_punc = [',', '.', ';', ':', '?', '!', '\"', '\'', '(', ')', '[', ']', '<', '>']
        zh_punc = ['，', '。', '；', '：', '？', '！', '“', '”', '‘', '’', '（', '）', '【', '】', '《', '》']

        if punc_type == 'general':
            pattern = f"["
            for s in punctuation:
                pattern += f"{s}"
            for s in string.punctuation:
                pattern += f"{s}"
            pattern += f"]"
            matches = re.findall(pattern, response)
            flag = 1
            for match in matches:
                if match != punct_new:
                    flag = 0
            if flag:
                return True
            return False
        elif punc_type == 'specific':
            pattern = f"[{punct_ori}{punct_new}]"
            matches = re.findall(pattern, response)
            flag = 1
            for match in matches:
                if match != punct_new:
                    flag = 0
            if flag:
                return True
            return False
        elif punc_type == 'zh':
            for punct_ori, punct_new in zip(en_punc, zh_punc):
                pattern = f"[{punct_ori}{punct_new}]"
                matches = re.findall(pattern, response)
                flag = 1
                for match in matches:
                    if match != punct_new:
                        flag = 0
                if flag == 0:
                    return False
            return True
        elif punc_type == 'en':
            for punct_ori, punct_new in zip(zh_punc, en_punc):
                pattern = f"[{punct_ori}{punct_new}]"
                matches = re.findall(pattern, response)
                flag = 1
                for match in matches:
                    if match != punct_new:
                        flag = 0
                if flag == 0:
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        punc_type = random.choice(['general', 'specific', 'zh', 'en'])
        punc_list = [
            "～", "！", "@", "#", "¥", "%", "……", "&", "*", "（", "）", "「", "」", "【", "】", "、", "：", "；", "“", "”", "‘",
            "’", "，", "。", "/", "？", "《", "》"
        ]
        punct_ori = random.choice(punc_list)
        punct_new = random.choice([p for p in list(punctuation) if p != punct_ori])

        if punc_type == 'general':
            indicator_meta = random.randint(0, len(self.meta_instr_general) - 1)
            instruction_meta = self.meta_instr_general[indicator_meta].format(punct_new=punct_new)
        elif punc_type == 'specific':
            indicator_meta = random.randint(0, len(self.meta_instr_specific) - 1)
            instruction_meta = self.meta_instr_specific[indicator_meta].format(punct_new=punct_new, punct_ori=punct_ori)
        elif punc_type == 'zh':
            indicator_meta = random.randint(0, len(self.meta_instr_zh) - 1)
            instruction_meta = self.meta_instr_zh[indicator_meta]
        elif punc_type == 'en':
            indicator_meta = random.randint(0, len(self.meta_instr_en) - 1)
            instruction_meta = self.meta_instr_en[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'punc_type': punc_type,
                'punct_ori': punct_ori,
                'punct_new': punct_new
            }
        }


class Rule_PUNC_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "PUNC_NEGATE"
        self.meta_instr_specific = [
            '忽略所有“{punct_ori}”',
            '省略所有“{punct_ori}”',
            '不要出现“{punct_ori}”',
            '不要输出“{punct_ori}”',
            '去除所有“{punct_ori}”',
            '省略回复中的“{punct_ori}”',
            '去掉所有“{punct_ori}”',
            '不能输出“{punct_ori}”',
            '输出结果不能包含“{punct_ori}”',
            '输出不要带“{punct_ori}”',
            '不能使用“{punct_ori}”',
            '不要包含“{punct_ori}”符号',
            '不能出现“{punct_ori}”符号',
            '整个回复不能使用“{punct_ori}”',
            '输出字符串不包含“{punct_ori}”',
            '不允许出现“{punct_ori}”',
            '删去所有“{punct_ori}”',
            '不要用“{punct_ori}”',
            '绝不加入“{punct_ori}”',
        ]
        self.meta_instr_general = [
            '忽略所有标点符号',
            '省略所有标点',
            '不要出现标点符号',
            '不要输出标点',
            '去除所有标点符号',
            '省略回复中的标点',
            '去掉所有标点符号',
            '不能输出标点',
            '输出结果不能包含标点符号',
            '输出不要带标点',
            '不能使用标点符号',
            '不要包含标点',
            '不能出现标点符号',
            '整个回复不能使用标点',
            '输出字符串不包含标点符号',
            '不允许出现标点',
            '删去所有标点符号',
            '不要用标点',
            '绝不加入标点符号',
            '不用写标点符号',
            '不能有标点符号',
            '不可以出现任何标点符号',
        ]

    def check(self, response, slots):
        punc_type = slots['punc_type']
        punct_ori = slots['punct_ori']
        zh_mapping = {
            '逗号': [',', '，'],
            '句号': ['.', '。'],
            '任何括号': ['(', ')', '（', '）', '[', ']', '【', '】', '「', '」', '{', '}', '《', '》', '<', '>'],
            '空格': [' '],
            '感叹号': ['!', '！'],
            '问号': ['?', '？'],
            '顿号': ['、'],
            '双引号': ['“', '”', '"'],
            '单引号': ["‘", "’", "'"]
        }
        if punc_type == 'general':
            for s in punctuation:
                if s in response:
                    return False
            for s in string.punctuation:
                if s in response:
                    return False
            return True
        elif punc_type == 'specific':
            if punct_ori not in response:
                return True
            return False
        else:
            punc_ori_list = zh_mapping[punct_ori]
            for s in punc_ori_list:
                if s in response:
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        punc_type = random.choice(['specific', 'general', 'zh'])
        zh_punc = ['逗号', '句号', '任何括号', '空格', '感叹号', '问号', '顿号', '双引号', '单引号']
        punc_list = [
            "～", "！", "@", "#", "¥", "%", "……", "&", "*", "（", "）", "「", "」", "【", "】", "、", "：", "；", "“", "”", "‘",
            "’", "，", "。", "/", "？", "《", "》"
        ]

        if punc_type == 'general':
            punct_ori = ''
            indicator_meta = random.randint(0, len(self.meta_instr_general) - 1)
            instruction_meta = self.meta_instr_general[indicator_meta]
        elif punc_type == 'specific':
            punct_ori = random.choice(punc_list)
            indicator_meta = random.randint(0, len(self.meta_instr_specific) - 1)
            instruction_meta = self.meta_instr_specific[indicator_meta].format(punct_ori=punct_ori)
        else:
            punct_ori = random.choice(zh_punc)
            indicator_meta = random.randint(0, len(self.meta_instr_specific) - 1)
            instruction_meta = self.meta_instr_specific[indicator_meta].format(punct_ori=punct_ori).replace('“',
                                                                                                            '').replace(
                                                                                                                '”', '')

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'punc_type': punc_type,
                'punct_ori': punct_ori
            }
        }


class Rule_PUNC_CONTAIN(Rule):

    def __init__(self):
        self.rule_type = "PUNC_CONTAIN"
        self.meta_instr_specific = [
            '禁止忽略“{punct_ori}”',
            '不能省略“{punct_ori}”',
            '要有符号“{punct_ori}”',
            '不要去除“{punct_ori}”',
            '禁止省略回复中的“{punct_ori}”',
            '需要输出符号“{punct_ori}”',
            '输出结果包含“{punct_ori}”',
            '输出带符号“{punct_ori}”',
            '使用符号“{punct_ori}”',
            '包含符号“{punct_ori}”',
            '不能不出现“{punct_ori}”',
            '整个回复要使用“{punct_ori}”',
            '输出字符串包含“{punct_ori}”',
        ]
        self.meta_instr_general = [
            '禁止忽略任何标点符号',
            '不能省略标点',
            '要有标点符号',
            '不要去除标点符号',
            '禁止省略回复中的标点',
            '需要输出标点',
            '输出结果包含标点符号',
            '输出带标点',
            '使用标点符号',
            '包含标点',
            '不能不出现标点符号',
            '整个回复要使用标点',
            '输出字符串包含标点符号',
        ]

    def check(self, response, slots):
        punc_type = slots['punc_type']
        punct_ori = slots['punct_ori']
        zh_mapping = {
            '逗号': [',', '，'],
            '句号': ['.', '。'],
            #   '任何括号':['(',')','（','）','[',']','【','】','「','」','{','}','《','》','<','>'],
            '空格': [' '],
            '感叹号': ['!', '！'],
            '问号': ['?', '？'],
            '顿号': ['、'],
            '双引号': ['“', '”', '"'],
            '单引号': ["‘", "’", "'"]
        }
        if punc_type == 'general':
            for s in punctuation:
                if s in response:
                    return True
            for s in string.punctuation:
                if s in response:
                    return True
            return False
        elif punc_type == 'specific':
            if punct_ori in response:
                return True
            return False
        else:
            punc_ori_list = zh_mapping[punct_ori]
            for s in punc_ori_list:
                if s in response:
                    return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        punc_type = random.choice(['specific', 'general', 'zh'])
        zh_punc = ['逗号', '句号', '空格', '感叹号', '问号', '顿号', '双引号', '单引号']
        punc_list = [
            "～", "！", "@", "#", "¥", "%", "……", "&", "*", "（", "）", "「", "」", "【", "】", "、", "：", "；", "“", "”", "‘",
            "’", "，", "。", "/", "？", "《", "》"
        ]

        if punc_type == 'general':
            punct_ori = ''
            indicator_meta = random.randint(0, len(self.meta_instr_general) - 1)
            instruction_meta = self.meta_instr_general[indicator_meta]
        elif punc_type == 'specific':
            punct_ori = random.choice(punc_list)
            indicator_meta = random.randint(0, len(self.meta_instr_specific) - 1)
            instruction_meta = self.meta_instr_specific[indicator_meta].format(punct_ori=punct_ori)
        else:
            punct_ori = random.choice(zh_punc)
            indicator_meta = random.randint(0, len(self.meta_instr_specific) - 1)
            instruction_meta = self.meta_instr_specific[indicator_meta].format(punct_ori=punct_ori).replace('“',
                                                                                                            '').replace(
                                                                                                                '”', '')

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'punc_type': punc_type,
                'punct_ori': punct_ori
            }
        }


class Rule_PUNC_END(Rule):

    def __init__(self):
        self.rule_type = "PUNC_END"
        self.meta_instr = [
            '回答结尾出现“{punct_ori}”',
            '在回复的末尾使用“{punct_ori}”',
            '结尾加“{punct_ori}”',
            '末尾加入“{punct_ori}”',
            '结尾需要用“{punct_ori}”',
            '回答结尾使用“{punct_ori}”',
            '每次回复用户都要在结尾加“{punct_ori}”',
            '以“{punct_ori}”结尾',
        ]

    def check(self, response, slots):
        punct_ori = slots["punct_ori"]
        punc_type = slots["punc_type"]
        zh_mapping = {
            '逗号': [',', '，'],
            '句号': ['.', '。'],
            #   '任何括号':['(',')','（','）','[',']','【','】','「','」','{','}','《','》','<','>'],
            '空格': [' '],
            '感叹号': ['!', '！'],
            '问号': ['?', '？'],
            '顿号': ['、'],
            '双引号': ['“', '”', '"'],
            '单引号': ["‘", "’", "'"]
        }
        if punc_type == 'specific':
            if response[-1] == punct_ori:
                return True
            return False
        else:
            punc_ori_list = zh_mapping[punct_ori]
            if response[-1] in punc_ori_list:
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        punc_type = random.choice(['specific', 'zh'])
        zh_punc = ['逗号', '句号', '空格', '感叹号', '问号', '顿号', '双引号', '单引号']
        punc_list = [
            "～", "！", "@", "#", "¥", "%", "……", "&", "*", "（", "）", "「", "」", "【", "】", "、", "：", "；", "“", "”", "‘",
            "’", "，", "。", "/", "？", "《", "》"
        ]

        if punc_type == 'specific':
            punct_ori = random.choice(punc_list)
            indicator_meta = random.randint(0, len(self.meta_instr) - 1)
            instruction_meta = self.meta_instr[indicator_meta].format(punct_ori=punct_ori)
        else:
            punct_ori = random.choice(zh_punc)
            indicator_meta = random.randint(0, len(self.meta_instr) - 1)
            instruction_meta = self.meta_instr[indicator_meta].format(punct_ori=punct_ori).replace('“',
                                                                                                   '').replace('”', '')

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "punc_type": punc_type,
                "punct_ori": punct_ori
            }
        }


class Rule_PUNC_END_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "PUNC_END_NEGATE"
        self.meta_instr = [
            '回答结尾不要出现“{punct_ori}”'
            '在回复的末尾不使用“{punct_ori}”'
            '结尾禁止加“{punct_ori}”',
            '末尾不可以加入“{punct_ori}”',
            '结尾不能用“{punct_ori}”',
            '回答结尾不使用“{punct_ori}”',
            '每次回复用户都不要在结尾加“{punct_ori}”',
            '禁止以“{punct_ori}”结尾',
            '最后不要带“{punct_ori}”',
        ]

    def check(self, response, slots):
        punct_ori = slots["punct_ori"]
        punc_type = slots["punc_type"]
        zh_mapping = {
            '逗号': [',', '，'],
            '句号': ['.', '。'],
            '任何括号': ['(', ')', '（', '）', '[', ']', '【', '】', '「', '」', '{', '}', '《', '》', '<', '>'],
            '空格': [' '],
            '感叹号': ['!', '！'],
            '问号': ['?', '？'],
            '顿号': ['、'],
            '双引号': ['“', '”', '"'],
            '单引号': ["‘", "’", "'"]
        }
        if punc_type == 'specific':
            if response[-1] == punct_ori:
                return False
            return True
        else:
            punc_ori_list = zh_mapping[punct_ori]
            if response[-1] in punc_ori_list:
                return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        punc_type = random.choice(['specific', 'zh'])
        zh_punc = ['逗号', '句号', '任何括号', '空格', '感叹号', '问号', '顿号', '双引号', '单引号']
        punc_list = [
            "～", "！", "@", "#", "¥", "%", "……", "&", "*", "（", "）", "「", "」", "【", "】", "、", "：", "；", "“", "”", "‘",
            "’", "，", "。", "/", "？", "《", "》"
        ]

        if punc_type == 'specific':
            punct_ori = random.choice(punc_list)
            indicator_meta = random.randint(0, len(self.meta_instr) - 1)
            instruction_meta = self.meta_instr[indicator_meta].format(punct_ori=punct_ori)
        else:
            punct_ori = random.choice(zh_punc)
            indicator_meta = random.randint(0, len(self.meta_instr) - 1)
            instruction_meta = self.meta_instr[indicator_meta].format(punct_ori=punct_ori).replace('“',
                                                                                                   '').replace('”', '')

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "punc_type": punc_type,
                "punct_ori": punct_ori
            }
        }


class Rule_PUNC_END_LOC(Rule):

    def __init__(self):
        self.rule_type = "PUNC_END_LOC"
        self.meta_instr = [
            '每句话结尾出现“{punct_ori}”'
            '在每个句子的末尾使用“{punct_ori}”'
            '每一句话结尾加“{punct_ori}”',
            '每个句子末尾添加“{punct_ori}”',
            '每句话结尾需要用“{punct_ori}”',
            '每个句子结尾使用“{punct_ori}”',
            '每句话都要在结尾加“{punct_ori}”',
            '每个句子以“{punct_ori}”结尾',
            '每一句话的最后一个标点都是“{punct_ori}”',
        ]

    def check(self, response, slots):
        punct_ori = slots["punct_ori"]
        punc_type = slots["punc_type"]
        zh_mapping = {
            '逗号': [',', '，'],
            '句号': ['.', '。'],
            #   '任何括号':['(',')','（','）','[',']','【','】','「','」','{','}','《','》','<','>'],
            '空格': [' '],
            '感叹号': ['!', '！'],
            '问号': ['?', '？'],
            '顿号': ['、'],
            '双引号': ['“', '”', '"'],
            '单引号': ["‘", "’", "'"]
        }
        sents = cut_sent(response)
        if punc_type == 'specific':
            for i, sent in enumerate(sents):
                if not sent.strip()[-1] == punct_ori:
                    return False
            return True
        else:
            punc_ori_list = zh_mapping[punct_ori]
            for i, sent in enumerate(sents):
                if not sent.strip()[-1] in punc_ori_list:
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        punc_type = random.choice(['specific', 'zh'])
        zh_punc = ['逗号', '句号', '空格', '感叹号', '问号', '顿号', '双引号', '单引号']
        punc_list = [
            "～", "！", "@", "#", "¥", "%", "……", "&", "*", "（", "）", "「", "」", "【", "】", "、", "：", "；", "“", "”", "‘",
            "’", "，", "。", "/", "？", "《", "》"
        ]

        if punc_type == 'specific':
            punct_ori = random.choice(punc_list)
            indicator_meta = random.randint(0, len(self.meta_instr) - 1)
            instruction_meta = self.meta_instr[indicator_meta].format(punct_ori=punct_ori)
        else:
            punct_ori = random.choice(zh_punc)
            indicator_meta = random.randint(0, len(self.meta_instr) - 1)
            instruction_meta = self.meta_instr[indicator_meta].format(punct_ori=punct_ori).replace('“',
                                                                                                   '').replace('”', '')

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "punc_type": punc_type,
                "punct_ori": punct_ori
            }
        }


################################################
# 数字类
################################################


class Rule_NUMBER_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "NUMBER_NEGATE"
        self.meta_instr = [
            '忽略所有阿拉伯数字',
            '省略所有阿拉伯数字',
            '不要出现阿拉伯数字',
            '不要输出阿拉伯数字',
            '去除所有阿拉伯数字',
            '省略回复中的阿拉伯数字',
            '去掉所有阿拉伯数字',
            '不能输出阿拉伯数字',
            '输出结果不能包含阿拉伯数字',
            '输出不要带阿拉伯数字',
            '不能使用阿拉伯数字',
            '不要包含阿拉伯数字',
            '不能出现阿拉伯数字',
            '整个回复不能使用阿拉伯数字',
            '输出字符串不包含阿拉伯数字',
            '不允许出现阿拉伯数字',
            '删去所有阿拉伯数字',
            '不要用阿拉伯数字',
            '绝不加入阿拉伯数字',
            '不用写阿拉伯数字',
            '不能有阿拉伯数字',
            '不可以出现任何阿拉伯数字',
        ]

    def check(self, response, slots):
        for s in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            if s in response:
                return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_NUMBER_CONTAIN(Rule):

    def __init__(self):
        self.rule_type = "NUMBER_CONTAIN"
        self.meta_instr = [
            '禁止忽略任何阿拉伯数字',
            '不能省略阿拉伯数字',
            '要有阿拉伯数字',
            '不要去除阿拉伯数字',
            '禁止省略回复中的阿拉伯数字',
            '需要输出阿拉伯数字',
            '输出结果包含阿拉伯数字',
            '输出带阿拉伯数字',
            '使用阿拉伯数字',
            '包含阿拉伯数字',
            '不能不出现阿拉伯数字',
            '整个回复要使用阿拉伯数字',
            '输出字符串包含阿拉伯数字',
        ]

    def check(self, response, slots):
        for s in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            if s in response:
                return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_NUMBER_START_END(Rule):

    def __init__(self):
        self.rule_type = "NUMBER_START_END"
        self.meta_instr_end = [
            '回答结尾出现阿拉伯数字',
            '在回复的末尾使用阿拉伯数字',
            '结尾加阿拉伯数字',
            '末尾加入阿拉伯数字',
            '结尾需要用阿拉伯数字',
            '回答结尾使用阿拉伯数字',
            '每次回复用户都要在结尾加阿拉伯数字',
            '以阿拉伯数字结尾',
        ]
        self.meta_instr_start = [
            '回答开头出现阿拉伯数字',
            '在回复的开头处使用阿拉伯数字',
            '开头加些阿拉伯数字表情',
            '开头加入一些阿拉伯数字',
            '开头需要包含阿拉伯数字',
            '回复开头要包含阿拉伯数字',
            '回答开头时加入阿拉伯数字',
            '答案开头需包含阿拉伯数字',
            '回复开头加上阿拉伯数字',
            '开头加入阿拉伯数字来回答',
            '回答开头使用一些阿拉伯数字',
            '每次回复用户都要在开头加阿拉伯数字',
        ]

    def check(self, response, slots):
        loc = slots["loc"]
        if loc == 'start':
            if response[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                return True
            return False
        else:
            if response.strip()[-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        loc = random.choice(['start', 'end'])
        self.meta_instr = self.meta_instr_start if loc == 'start' else self.meta_instr_end
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "loc": loc
            }
        }


class Rule_NUMBER_TYPE(Rule):

    def __init__(self):
        self.rule_type = "NUMBER_TYPE"
        self.meta_instr_arab = [
            '所有数字使用阿拉伯数字',
            '数字以阿拉伯形式',
            '用阿拉伯数字',
            '数字采用阿拉伯数字',
        ]
        self.meta_instr_zh = [
            '所有数字使用中文数字',
            '数字以中文形式',
            '用中文数字',
            '数字采用中文数字',
        ]

    def check(self, response, slots):
        number_type = slots["number_type"]
        zh_num = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '零', '十']
        arab_num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '0']
        if number_type == "arab":
            for punct_ori, punct_new in zip(zh_num, arab_num):
                pattern = f"[{punct_ori}{punct_new}]"
                matches = re.findall(pattern, response)
                flag = 1
                for match in matches:
                    if match != punct_new:
                        flag = 0
                if flag == 0:
                    return False
            return True
        else:
            for punct_ori, punct_new in zip(arab_num, zh_num):
                pattern = f"[{punct_ori}{punct_new}]"
                matches = re.findall(pattern, response)
                flag = 1
                for match in matches:
                    if match != punct_new:
                        flag = 0
                if flag == 0:
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        number_type = random.choice(['arab', 'zh'])
        self.meta_instr = self.meta_instr_arab if number_type == "arab" else self.meta_instr_zh
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'number_type': number_type
            }
        }


class Rule_DECIMAL_TYPE(Rule):

    def __init__(self):
        self.rule_type = "DECIMAL_TYPE"
        self.meta_instr_two = [
            '所有数据保留两位小数',
        ]
        self.meta_instr_one = [
            '所有数据保留一位小数',
        ]
        self.meta_instr_no = [
            '所有数据不要有小数',
            '严禁出现任何带小数点的数据',
        ]

    def check(self, response, slots):
        decimal_type = slots["decimal_type"]
        if decimal_type == "two":
            number_regex = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
            matches = number_regex.findall(response)
            return len(matches) > 0 and all(re.fullmatch(r'[-+]?(\d*\.\d{2})', match) for match in matches)
        elif decimal_type == "one":
            number_regex = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
            matches = number_regex.findall(response)
            return len(matches) > 0 and all(re.fullmatch(r'[-+]?\d*\.\d\b', match) for match in matches)
        else:
            decimal_regex = re.compile(r'[-+]?\d*\.\d+')
            return not bool(decimal_regex.search(response))

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        decimal_type = random.choice(['two', 'one', 'no'])
        if decimal_type == "two":
            self.meta_instr = self.meta_instr_two
        elif decimal_type == "one":
            self.meta_instr = self.meta_instr_one
        else:
            self.meta_instr = self.meta_instr_no
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'decimal_type': decimal_type
            }
        }


################################################
# 句子类型
################################################


class Rule_SENT_TYPE_ONLY(Rule):

    def __init__(self):
        self.rule_type = "SENT_TYPE_ONLY"
        self.meta_instr = [
            '只用{sentence_type}回答问题', '回答中只出现{sentence_type}', '回答问题时只能使用{sentence_type}', '仅使用{sentence_type}进行回答',
            '回答时只允许使用{sentence_type}', '请用{sentence_type}作答', '回答必须是{sentence_type}', '问题的回答需要使用{sentence_type}',
            '回答问题时限用{sentence_type}', '答复只能包含{sentence_type}', '内容都用{sentence_type}'
        ]

    def check(self, response, slots):
        sentence_type = slots["sentence_type"]
        sents = cut_sent(response)
        if sentence_type == "问句":
            for sent in sents:
                if not (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                        sent.endswith('?"') or sent.endswith('?\'')):
                    return False
            return True
        elif sentence_type == "感叹句":
            for sent in sents:
                if not (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                        sent.endswith('!"') or sent.endswith('!\'')):
                    return False
            return True
        elif sentence_type == "陈述句":
            for sent in sents:
                if not (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                        sent.endswith('."') or sent.endswith('.\'')):
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        sentence_type = random.choice(["问句", "感叹句", "陈述句"])
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(sentence_type=sentence_type)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "sentence_type": sentence_type
            }
        }


class Rule_SENT_TYPE_CONTAIN(Rule):

    def __init__(self):
        self.rule_type = "SENT_TYPE_CONTAIN"
        self.meta_instr = [
            '回答中要出现{sentence_type}', '至少使用一个{sentence_type}', '答复需要包含{sentence_type}', '内容中要有{sentence_type}'
        ]

    def check(self, response, slots):
        sentence_type = slots["sentence_type"]
        sents = cut_sent(response)
        if sentence_type == "问句":
            for sent in sents:
                if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                        sent.endswith('?"') or sent.endswith('?\'')):
                    return True
            return False
        elif sentence_type == "感叹句":
            for sent in sents:
                if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                        sent.endswith('!"') or sent.endswith('!\'')):
                    return True
            return False
        elif sentence_type == "陈述句":
            for sent in sents:
                if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                        sent.endswith('."') or sent.endswith('.\'')):
                    return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        sentence_type = random.choice(["问句", "感叹句", "陈述句"])
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(sentence_type=sentence_type)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "sentence_type": sentence_type
            }
        }


class Rule_SENT_TYPE_NUM(Rule):

    def __init__(self):
        self.rule_type = "SENT_TYPE_NUM"
        self.meta_instr_equal = [
            '输出{num1}个{sentence_type}',
            '使用{num1}个{sentence_type}',
            '生成{num1}个{sentence_type}',
            '答复需要包含{num1}个{sentence_type}',
            '内容中要有{num1}个{sentence_type}',
            '在回答内加上{num1}个{sentence_type}',
            '请在回答中加入{num1}个{sentence_type}',
            '回答中需要包含{num1}个{sentence_type}',
            '在你的回答里添加{num1}个{sentence_type}',
            '请在答复中加上{num1}个{sentence_type}',
            '回答时请包含{num1}个{sentence_type}',
            '在回复中插入{num1}个{sentence_type}',
            '你的回答应带有{num1}个{sentence_type}',
            '请将{num1}个{sentence_type}添加到回答中',
            '回答需要嵌入{num1}个{sentence_type}',
            '在你的答复中使用{num1}个{sentence_type}',
            '回答中的{sentence_type}出现{num1}次',
        ]
        self.meta_instr_more = [
            '输出不少于{num1}个{sentence_type}',
            '使用至少{num1}个{sentence_type}',
            '生成至少{num1}个{sentence_type}',
            '答复至少需要包含{num1}个{sentence_type}',
            '内容中要有最少{num1}个{sentence_type}',
            '在回答内加上不少于{num1}个{sentence_type}',
            '请在回答中加入至少{num1}个{sentence_type}',
            '回答中需要包含不少于{num1}个{sentence_type}',
            '在你的回答里添加大于等于{num1}个{sentence_type}',
            '请在答复中加上不少于{num1}个{sentence_type}',
            '回答时请包含至少{num1}个{sentence_type}',
            '在回复中插入不少于{num1}个{sentence_type}',
            '你的回答应带有大于等于{num1}个{sentence_type}',
            '请将不少于{num1}个{sentence_type}添加到回答中',
            '回答需要嵌入至少{num1}个{sentence_type}',
            '在你的答复中使用不少于{num1}个{sentence_type}',
            '回答中的{sentence_type}不少于{num1}个',
        ]
        self.meta_instr_less = [
            '输出不超过{num1}个{sentence_type}',
            '使用至多{num1}个{sentence_type}',
            '生成最多{num1}个{sentence_type}',
            '答复最多包含{num1}个{sentence_type}',
            '内容中要有至多{num1}个{sentence_type}',
            '在回答内加上不超过{num1}个{sentence_type}',
            '请在回答中加入最多{num1}个{sentence_type}',
            '回答中需要包含不超过{num1}个{sentence_type}',
            '在你的回答里添加小于等于{num1}个{sentence_type}',
            '请在答复中加上不超过{num1}个{sentence_type}',
            '回答时请包含小于等于{num1}个{sentence_type}',
            '在回复中插入不超过{num1}个{sentence_type}',
            '你的回答应带有至多{num1}个{sentence_type}',
            '请将不超过{num1}个{sentence_type}添加到回答中',
            '回答需要嵌入小于等于{num1}个{sentence_type}',
            '在你的答复中使用不超过{num1}个{sentence_type}',
            '回答中的{sentence_type}不超过{num1}个',
        ]

    def check(self, response, slots):
        num1 = slots["num1"]
        sentence_type = slots["sentence_type"]
        interval_type = slots["interval_type"]
        sents = cut_sent(response)

        if interval_type == "EQUAL":
            if sentence_type == "问句":
                count = 0
                for sent in sents:
                    if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                            sent.endswith('?"') or sent.endswith('?\'')):
                        count += 1
                if count == num1:
                    return True
                return False
            elif sentence_type == "感叹句":
                count = 0
                for sent in sents:
                    if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                            sent.endswith('!"') or sent.endswith('!\'')):
                        count += 1
                if count == num1:
                    return True
                return False
            elif sentence_type == "陈述句":
                count = 0
                for sent in sents:
                    if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                            sent.endswith('."') or sent.endswith('.\'')):
                        count += 1
                if count == num1:
                    return True
                return False
        elif interval_type == "MORE":
            if sentence_type == "问句":
                count = 0
                for sent in sents:
                    if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                            sent.endswith('?"') or sent.endswith('?\'')):
                        count += 1
                if count >= num1:
                    return True
                return False
            elif sentence_type == "感叹句":
                count = 0
                for sent in sents:
                    if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                            sent.endswith('!"') or sent.endswith('!\'')):
                        count += 1
                if count >= num1:
                    return True
                return False
            elif sentence_type == "陈述句":
                count = 0
                for sent in sents:
                    if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                            sent.endswith('."') or sent.endswith('.\'')):
                        count += 1
                if count >= num1:
                    return True
                return False
        elif interval_type == "LESS":
            if sentence_type == "问句":
                count = 0
                for sent in sents:
                    if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                            sent.endswith('?"') or sent.endswith('?\'')):
                        count += 1
                if count <= num1:
                    return True
                return False
            elif sentence_type == "感叹句":
                count = 0
                for sent in sents:
                    if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                            sent.endswith('!"') or sent.endswith('!\'')):
                        count += 1
                if count <= num1:
                    return True
                return False
            elif sentence_type == "陈述句":
                count = 0
                for sent in sents:
                    if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                            sent.endswith('."') or sent.endswith('.\'')):
                        count += 1
                if count <= num1:
                    return True
                return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        sentence_type = random.choice(["问句", "感叹句", "陈述句"])
        interval_type = random.choice(["MORE", "LESS", "EQUAL"])
        num1 = random.randint(1, 10)

        if interval_type == "EQUAL":
            indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
            instruction_meta = self.meta_instr_equal[indicator_meta].format(sentence_type=sentence_type, num1=num1)
        elif interval_type == "MORE":
            indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
            instruction_meta = self.meta_instr_more[indicator_meta].format(sentence_type=sentence_type, num1=num1)
        elif interval_type == "LESS":
            indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
            instruction_meta = self.meta_instr_less[indicator_meta].format(sentence_type=sentence_type, num1=num1)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "interval_type": interval_type,
                "sentence_type": sentence_type,
                "num1": num1
            }
        }


class Rule_SENT_TYPE_START_END(Rule):

    def __init__(self):
        self.rule_type = "SENT_TYPE_START_END"
        self.meta_instr_start = [
            '开头要出现{sentence_type}',
            '开头加一个{sentence_type}',
            '开头用{sentence_type}',
            '开头要有{sentence_type}',
            '以{sentence_type}开头',
        ]
        self.meta_instr_end = [
            '结尾要出现{sentence_type}',
            '结尾加一个{sentence_type}',
            '结尾用{sentence_type}',
            '结尾要有{sentence_type}',
            '以{sentence_type}结尾',
        ]

    def check(self, response, slots):
        loc = slots["loc"]
        sentence_type = slots["sentence_type"]
        sents = cut_sent(response)
        if loc == "start":
            if sentence_type == "问句":
                sent = sents[0]
                if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                        sent.endswith('?"') or sent.endswith('?\'')):
                    return True
                return False
            elif sentence_type == "感叹句":
                sent = sents[0]
                if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                        sent.endswith('!"') or sent.endswith('!\'')):
                    return True
                return False
            elif sentence_type == "陈述句":
                sent = sents[0]
                if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                        sent.endswith('."') or sent.endswith('.\'')):
                    return True
                return False
        else:
            if sentence_type == "问句":
                sent = sents[-1]
                if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                        sent.endswith('?"') or sent.endswith('?\'')):
                    return True
                return False
            elif sentence_type == "感叹句":
                sent = sents[-1]
                if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                        sent.endswith('!"') or sent.endswith('!\'')):
                    return True
                return False
            elif sentence_type == "陈述句":
                sent = sents[-1]
                if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                        sent.endswith('."') or sent.endswith('.\'')):
                    return True
                return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        sentence_type = random.choice(["问句", "感叹句", "陈述句"])
        loc = random.choice(["start", "end"])
        self.meta_instr = self.meta_instr_start if loc == 'start' else self.meta_instr_end
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(sentence_type=sentence_type)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "sentence_type": sentence_type,
                "loc": loc
            }
        }


class Rule_SENT_TYPE_START_END_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "SENT_TYPE_START_END_NEGATE"
        self.meta_instr_start = [
            '开头不要出现{sentence_type}',
            '开头不加{sentence_type}',
            '开头别用{sentence_type}',
            '开头不可以有{sentence_type}',
            '禁止以{sentence_type}开头',
        ]
        self.meta_instr_end = [
            '结尾不要出现{sentence_type}',
            '结尾不加{sentence_type}',
            '结尾别用{sentence_type}',
            '结尾不可以有{sentence_type}',
            '禁止以{sentence_type}结尾',
        ]

    def check(self, response, slots):
        loc = slots["loc"]
        sentence_type = slots["sentence_type"]
        sents = cut_sent(response)
        if loc == "start":
            if sentence_type == "问句":
                sent = sents[0]
                if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                        sent.endswith('?"') or sent.endswith('?\'')):
                    return False
                return True
            elif sentence_type == "感叹句":
                sent = sents[0]
                if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                        sent.endswith('!"') or sent.endswith('!\'')):
                    return False
                return True
            elif sentence_type == "陈述句":
                sent = sents[0]
                if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                        sent.endswith('."') or sent.endswith('.\'')):
                    return False
                return True
        else:
            if sentence_type == "问句":
                sent = sents[-1]
                if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                        sent.endswith('?"') or sent.endswith('?\'')):
                    return False
                return True
            elif sentence_type == "感叹句":
                sent = sents[-1]
                if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                        sent.endswith('!"') or sent.endswith('!\'')):
                    return False
                return True
            elif sentence_type == "陈述句":
                sent = sents[-1]
                if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                        sent.endswith('."') or sent.endswith('.\'')):
                    return False
                return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        sentence_type = random.choice(["问句", "感叹句", "陈述句"])
        loc = random.choice(["start", "end"])
        self.meta_instr = self.meta_instr_start if loc == 'start' else self.meta_instr_end
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(sentence_type=sentence_type)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "sentence_type": sentence_type,
                "loc": loc
            }
        }


class Rule_SENT_TYPE_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "SENT_TYPE_NEGATE"
        self.meta_instr = [
            '不要出现{sentence_type}',
            '不要输出{sentence_type}',
            '去除所有{sentence_type}',
            '去掉{sentence_type}',
            '不能输出{sentence_type}',
            '输出结果不能包含{sentence_type}',
            '输出不要带{sentence_type}',
            '不能使用{sentence_type}',
            '不要包含{sentence_type}',
            '不能出现{sentence_type}',
            '整个回复不能使用{sentence_type}',
            '不允许出现{sentence_type}',
            '删去所有{sentence_type}',
            '不要用{sentence_type}',
            '绝不加入{sentence_type}',
            '不能有{sentence_type}',
            '不可以出现任何{sentence_type}',
        ]

    def check(self, response, slots):
        sentence_type = slots["sentence_type"]
        sents = cut_sent(response)
        if sentence_type == "问句":
            for sent in sents:
                if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                        sent.endswith('?"') or sent.endswith('?\'')):
                    return False
            return True
        elif sentence_type == "感叹句":
            for sent in sents:
                if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                        sent.endswith('!"') or sent.endswith('!\'')):
                    return False
            return True
        elif sentence_type == "陈述句":
            for sent in sents:
                if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                        sent.endswith('."') or sent.endswith('.\'')):
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        sentence_type = random.choice(["问句", "感叹句", "陈述句"])
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(sentence_type=sentence_type)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "sentence_type": sentence_type
            }
        }


class Rule_SENT_TYPE_REPLACE(Rule):

    def __init__(self):
        self.rule_type = "SENT_TYPE_REPLACE"
        self.meta_instr = [
            '将{sent_ori}改为{sent_new}',
            '用“{sent_new}”替换“{sent_ori}”',
            '把“{sent_ori}”换成“{sent_new}”',
            '所有“{sent_ori}”都用“{sent_new}”换掉',
            '把所有“{sent_ori}”换成“{sent_new}”',
        ]

    def check(self, response, slots):
        sent_ori = slots["sent_ori"]
        sent_new = slots["sent_new"]
        sents = cut_sent(response)
        if sent_ori == "问句":
            for sent in sents:
                if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                        sent.endswith('?"') or sent.endswith('?\'')):
                    return False
        elif sent_ori == "感叹句":
            for sent in sents:
                if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                        sent.endswith('!"') or sent.endswith('!\'')):
                    return False
        elif sent_ori == "陈述句":
            for sent in sents:
                if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                        sent.endswith('."') or sent.endswith('.\'')):
                    return False
        if sent_new == "问句":
            for sent in sents:
                if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                        sent.endswith('?"') or sent.endswith('?\'')):
                    return True
        elif sent_new == "感叹句":
            for sent in sents:
                if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                        sent.endswith('!"') or sent.endswith('!\'')):
                    return True
        elif sent_new == "陈述句":
            for sent in sents:
                if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                        sent.endswith('."') or sent.endswith('.\'')):
                    return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        sent_ori = random.choice(["问句", "感叹句", "陈述句"])
        sent_new = random.choice([i for i in ["问句", "感叹句", "陈述句"] if i != sent_ori])
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(sent_ori=sent_ori, sent_new=sent_new)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "sent_ori": sent_ori,
                "sent_new": sent_new
            }
        }


################################################
# 句子数量
################################################


class Rule_SENT_NUM(Rule):

    def __init__(self):
        self.rule_type = "SENT_NUM"
        self.meta_instr_equal = [
            '输出{num1}个句子',
            '生成{num1}个句子',
            '答复需要包含{num1}个句子',
            '回答时请包含{num1}句话',
            '你的回答应带有{num1}个句子',
            '在你的答复中使用{num1}个句子',
            '整合成{num1}句话',
            '用{num1}句话',
            '写成{num1}句话',
            '{num1}句话',
            '每次只能回复{num1}句话',
        ]
        self.meta_instr_more = [
            '输出不少于{num1}个句子',
            '生成至少{num1}个句子',
            '答复至少需要包含{num1}个句子',
            '回答时请包含至少{num1}个句子',
            '你的回答应带有最少{num1}个句子',
            '在你的答复中使用不少于{num1}个句子',
            '回复不能少于{num1}句话',
            '每次回复不能少于{num1}句话',
            '整体不能少于{num1}句话',
            '回复的内容不可以少于{num1}句话',
            '控制在{num1}句话以上',
        ]
        self.meta_instr_less = [
            '输出不超过{num1}个句子', '生成最多{num1}个句子', '答复最多包含{num1}个句子', '回答时请包含小于等于{num1}个句子', '你的回答应带有至多{num1}个句子',
            '在你的答复中使用不超过{num1}个句子', '回复不能超过{num1}句话', '每次回复不能超过{num1}句话', '整体不能超过{num1}句话', '回复的内容不可以超过{num1}句话',
            '控制在{num1}句话以内'
        ]

    def check(self, response, slots):
        num1 = slots["num1"]
        interval_type = slots["interval_type"]
        sents = cut_sent(response)

        if interval_type == "EQUAL":
            if len(sents) == num1:
                return True
            return False
        elif interval_type == "MORE":
            if len(sents) >= num1:
                return True
            return False
        elif interval_type == "LESS":
            if len(sents) <= num1:
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        interval_type = random.choice(["MORE", "LESS", "EQUAL"])
        num1 = random.randint(1, 10)

        if interval_type == "EQUAL":
            indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
            instruction_meta = self.meta_instr_equal[indicator_meta].format(num1=num1)
        elif interval_type == "MORE":
            indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
            instruction_meta = self.meta_instr_more[indicator_meta].format(num1=num1)
        elif interval_type == "LESS":
            indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
            instruction_meta = self.meta_instr_less[indicator_meta].format(num1=num1)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "interval_type": interval_type,
                "num1": num1
            }
        }


class Rule_SENT_NUM_LOC(Rule):

    def __init__(self):
        self.rule_type = "SENT_NUM_LOC"
        self.meta_instr_equal = [
            '第{which}段需要包含{num1}个句子',
            '第{which}段请包含{num1}句话',
            '第{which}段应带有{num1}个句子',
            '在第{which}段中使用{num1}个句子',
            '第{which}段只能有{num1}句话',
        ]
        self.meta_instr_more = [
            '第{which}段需要包含不少于{num1}句话',
            '第{which}段请包含至少{num1}个句子',
            '第{which}段应带有最少{num1}句话',
            '在第{which}段使用不少于{num1}个句子',
            '第{which}段不少于{num1}句话',
        ]
        self.meta_instr_less = [
            '第{which}段中需要包含不超过{num1}句话',
            '第{which}段请包含最多{num1}个句子',
            '第{which}段应带有至多{num1}句话',
            '在第{which}段中使用不超过{num1}个句子',
            '第{which}段不超过{num1}句话',
        ]

    def check(self, response, slots):
        interval_type = slots["interval_type"]
        num1 = slots["num1"]
        which = slots["which"]
        paragraphs = re.split(r'\n\s*\n*', response.strip())

        if interval_type == "EQUAL":
            for i, paragraph in enumerate(paragraphs):
                sents = cut_sent(paragraph)
                if i + 1 == which and len(sents) == num1:
                    return True
            return False
        elif interval_type == "MORE":
            for i, paragraph in enumerate(paragraphs):
                sents = cut_sent(paragraph)
                if i + 1 == which and len(sents) >= num1:
                    return True
            return False
        elif interval_type == "LESS":
            for i, paragraph in enumerate(paragraphs):
                sents = cut_sent(paragraph)
                if i + 1 == which and len(sents) <= num1:
                    return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        interval_type = random.choice(["EQUAL", "MORE", "LESS"])
        num1 = random.randint(1, 10)
        paragraphs = re.split(r'\n\s*\n*', response.strip())
        paragraph_count = len(paragraphs)
        which = random.randint(1, min(paragraph_count, 10))

        if interval_type == "EQUAL":
            indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
            instruction_meta = self.meta_instr_equal[indicator_meta].format(which=which, num1=num1)
        elif interval_type == "MORE":
            indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
            instruction_meta = self.meta_instr_more[indicator_meta].format(which=which, num1=num1)
        elif interval_type == "LESS":
            indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
            instruction_meta = self.meta_instr_less[indicator_meta].format(which=which, num1=num1)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "interval_type": interval_type,
                "num1": num1,
                "which": which
            }
        }


################################################
# 段落数量
################################################


class Rule_PARA_NUM(Rule):

    def __init__(self):
        self.rule_type = "PARA_NUM"
        self.meta_instr_one = [
            '输出一段话',
            '生成一个段落',
            '用一段话',
            '每次只能回复一个段落',
            "整合成一段话",
            "不要分段",
            "回复的内容禁止换行",
            "不可以分行分段，只能用一段话呈现",
        ]
        self.meta_instr_two = [
            "要分段",
            "带换行",
        ]

    def check(self, response, slots):
        para_type = slots["para_type"]
        if para_type == "one":
            paragraphs = re.split(r'\n\s*\n*', response.strip())
            if len(paragraphs) == 1:
                return True
            return False
        else:
            paragraphs = re.split(r'\n\s*\n*', response.strip())
            if len(paragraphs) >= 2:
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        para_type = random.choice(["one", "two"])
        self.meta_instr = self.meta_instr_one if para_type == "one" else self.meta_instr_two
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(para_type=para_type)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "para_type": para_type
            }
        }


################################################
# 格式类
################################################


class Rule_FORMAT_JSON(Rule):

    def __init__(self):
        self.rule_type = "FORMAT_JSON"
        self.meta_instr = [
            '使用json格式输出', '输出格式能被 Python json.loads 解析', '以{"xxx": "xxx"}的格式产出', '以JSON格式呈现结果', '输出采用json格式',
            '输出结果应为JSON格式', '结果使用json格式进行输出', '请以JSON格式提供输出', '只能用json格式输出结构化内容'
        ]

    def check(self, response, slots):
        if response is None or response.strip() == "":
            return False
        try:
            json_obj = json.loads(response)
        except ValueError:
            return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_FORMAT_SEP(Rule):

    def __init__(self):
        self.rule_type = "FORMAT_SEP"
        self.meta_instr_symbol = [
            '段落之间用分隔符“{symbol}”分隔',
            '使用分隔符“{symbol}”来区隔段落',
            '各段落之间用“{symbol}”符号分隔',
            '以“{symbol}”作为段落间的分隔符',
            '用“{symbol}”符号分隔段落',
            '每段之间由“{symbol}”隔开',
            '每段内容用“{symbol}”隔开',
        ]
        self.meta_instr_emoji = [
            '段落之间用emoji分隔',
            '使用表情来区隔段落',
            '各段落之间用emoji符号分隔',
            '以表情作为段落间的分隔符',
            '用emoji符号分隔段落',
            '每段之间由emoji隔开',
            '每段内容用表情隔开',
        ]
        self.meta_instr_newline = [
            '段落之间用一个空行分隔',
            '各段落之间用仅一个空行分隔',
            '每段内容用一个空行隔开',
            '每段之间必须有且只有一个空行',
        ]
        self.SYMBOL_SEP = [
            '###',
            '***',
            '---',
            '===',
            '~~~',
            '####',
            '****',
            '----',
            '====',
            '~~~~',
            '#####',
            '*****',
            '-----',
            '=====',
            '~~~~~',
            '######',
            '******',
            '------',
            '======',
            '~~~~~~',
            '#######',
            '*******',
            '-------',
            '=======',
            '~~~~~~~',
        ]

    def check(self, response, slots):
        symbol = slots["symbol"]
        sep_type = slots["sep_type"]
        if sep_type == "symbol":
            segments = response.strip().split('\n')
            segments = [s for s in segments if s != '']
            if len(segments) < 3:
                return False
            for i in range(1, len(segments), 2):
                # print(i, segments[i])
                if segments[i] != symbol:
                    return False
            return True
        elif sep_type == "emoji":
            segments = response.strip().split('\n')
            segments = [s for s in segments if s != '']
            if len(segments) < 3:
                return False
            for i in range(1, len(segments), 2):
                if segments[i] not in emoji.EMOJI_DATA:
                    return False
            return True
        elif sep_type == "newline":
            segments = response.strip().split('\n')
            segments = [s for s in segments]
            if len(segments) < 3:
                return False
            for i in range(1, len(segments), 2):
                if segments[i] != '':
                    return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        symbol = random.choice(self.SYMBOL_SEP)
        sep_type = random.choice(["symbol", "emoji", "newline"])
        if sep_type == "symbol":
            indicator_meta = random.randint(0, len(self.meta_instr_symbol) - 1)
            instruction_meta = self.meta_instr_symbol[indicator_meta].format(symbol=symbol)
        elif sep_type == "emoji":
            indicator_meta = random.randint(0, len(self.meta_instr_emoji) - 1)
            instruction_meta = self.meta_instr_emoji[indicator_meta]
        elif sep_type == "newline":
            indicator_meta = random.randint(0, len(self.meta_instr_newline) - 1)
            instruction_meta = self.meta_instr_newline[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "sep_type": sep_type,
                "symbol": symbol
            }
        }


class Rule_FORMAT_TAG_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "FORMAT_TAG_NEGATE"
        self.meta_instr = [
            '不要加任何tag',
            '不要出现tag',
            '不要输出tag',
            '去除所有tag',
            '去掉tag',
            '不能输出tag',
            '输出结果不能包含tag',
            '输出不要带tag',
            '不能使用tag',
            '不要包含tag',
            '不能出现tag',
            '整个回复不能使用tag',
            '不允许出现tag',
            '删去所有tag',
            '不要用tag',
            '绝不加入tag',
            '不能有tag',
            '不可以出现任何tag',
        ]

    def check(self, response, slots):
        if '#' in response:
            return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_FORMAT_SEP_PIECE(Rule):

    def __init__(self):
        self.rule_type = "FORMAT_SEP_PIECE"
        self.meta_instr = [
            '输出{num}个词，用逗号分隔',
            '输出{num}个片段，用逗号分隔',
            '输出{num}个片段，用逗号进行分割',
        ]

    def check(self, response, slots):
        num = slots["num"]
        response_sep = re.split(r"，|,", response)
        if len(response_sep) == num:
            return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        num = random.randint(2, 10)
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        if random.random() < 0.5:
            zh_num = arabic_to_chinese(num)
            instruction_meta = self.meta_instr[indicator_meta].format(num=zh_num)
        else:
            instruction_meta = self.meta_instr[indicator_meta].format(num=num)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "num": num
            }
        }


class Rule_FORMAT_SEP_NEWLINE(Rule):

    def __init__(self):
        self.rule_type = "FORMAT_SEP_NEWLINE"
        self.meta_instr = [
            '每个{symbol}后面都换行',
            '每个{symbol}后面都需要换行',
        ]

    def check(self, response, slots):
        symbol = slots["symbol"]
        text = response
        if symbol == '逗号':
            for i in range(len(text) - 1):
                if text[i] in [',', '，']:
                    if text[i + 1] != '\n':
                        return False
            if text[len(text) - 1] in [',', '，']:
                return False
            return True
        elif symbol == '句号':
            for i in range(len(text) - 1):
                if text[i] == '。':
                    if text[i + 1] != '\n':
                        return False
            if text[len(text) - 1] == '。':
                return False
            return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        symbol = random.choice(['逗号', '句号'])
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta].format(symbol=symbol)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "symbol": symbol
            }
        }


class Rule_FORMAT_WRAP(Rule):

    def __init__(self):
        self.rule_type = "FORMAT_WRAP"
        self.meta_instr_word = [
            '“{word}”需要被{bracket}包围起来',
            '“{word}”需要被{bracket}括起来表示',
            '“{word}”需要被“{bracket}”包围起来',
            '“{word}”需要被“{bracket}”括起来表示',
        ]
        self.FORMAT_BRACKET_LIST = [
            '( )',
            '[ ]',
            '< >',
            '<< >>',
            '| |',
            '- -',
            '# #',
            '" "',
            '@ @',
            '$ $',
            '$$ $$',
        ]

    def check(self, response, slots):
        word = slots['word']
        bracket_format = slots['bracket_format']
        left, right = bracket_format.split(' ')
        left_len = len(left)
        right_len = len(right)
        try:
            matches = re.finditer(word, response)
        except:
            return False

        flag = 1
        for match in matches:
            start, end = match.start(), match.end()
            # if (start-1 >=0 and response[start-1].isalpha()) or (end < len(response) and response[end].isalpha()):
            #     continue
            if response[start - left_len:end + right_len] != left + word + right:
                flag = 0
        if flag:
            return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        bracket = random.choice(self.FORMAT_BRACKET_LIST)
        chinese_char_regex = re.compile(r'[\u4E00-\u9FFF]')
        chinese_characters = chinese_char_regex.findall(response)
        if len(chinese_characters) == 0:
            return {'instruction': instruction, 'rule': '', 'output': response, 'type': None, 'slots': {}}
        word = random.choice(chinese_characters)

        indicator_meta = random.randint(0, len(self.meta_instr_word) - 1)
        instruction_meta = self.meta_instr_word[indicator_meta].format(bracket=bracket, word=word)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'word': word,
                'bracket_format': bracket
            }
        }


################################################
# 粗体/斜体类
################################################


class Rule_BOLD_PARA(Rule):

    def __init__(self):
        self.rule_type = "BOLD_PARA"
        self.meta_instr = [
            '加粗第{which}段',
            '第{which}段以粗体的形式呈现',
            '第{which}段话需要加粗',
        ]

    def check(self, response, slots):
        which = slots["which"]
        paragraphs = re.split(r'\n\s*\n*', response.strip())
        bold_regex = re.compile(r'^\s*(\*\*.*\*\*)|(__.*__)\s*$')

        for i, paragraph in enumerate(paragraphs):
            if i + 1 == which and bool(bold_regex.match(paragraph.strip())):
                return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        paragraphs = re.split(r'\n\s*\n*', response.strip())
        paragraph_count = len(paragraphs)
        which = random.randint(1, min(paragraph_count, 10))
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        if random.random() < 0.5:
            zh_num = arabic_to_chinese(which)
            instruction_meta = self.meta_instr[indicator_meta].format(which=zh_num)
        else:
            instruction_meta = self.meta_instr[indicator_meta].format(which=which)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "which": which
            }
        }


class Rule_BOLD_SENT(Rule):

    def __init__(self):
        self.rule_type = "BOLD_SENT"
        self.meta_instr = [
            '加粗第{which}句',
            '第{which}句以粗体的形式呈现',
            '第{which}句话需要加粗',
        ]

    ###TODO: check函数还有点问题
    def check(self, response, slots):
        which = slots["which"]
        sents = cut_sent(response)
        # bold_regex = re.compile(r'^\s*(\*\*.*\*\*)|(__.*__)\s*$')

        for i, sent in enumerate(sents):
            if (i+1 == which and sent.startswith('**') and (sent.endswith('**') or sent+'**' in response)) or \
               (i+1 == which and sent.startswith('__') and (sent.endswith('__') or sent+'__' in response)):#bool(bold_regex.match(sent.strip())):
                return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        sents = cut_sent(response)
        sent_count = len(sents)
        which = random.randint(1, min(sent_count, 10))
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        if random.random() < 0.5:
            zh_num = arabic_to_chinese(which)
            instruction_meta = self.meta_instr[indicator_meta].format(which=zh_num)
        else:
            instruction_meta = self.meta_instr[indicator_meta].format(which=which)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "which": which
            }
        }


class Rule_BOLD_WORD(Rule):

    def __init__(self):
        self.rule_type = "BOLD_WORD"
        self.meta_instr = [
            '加粗第{which}个汉字',
            '第{which}个汉字以粗体的形式呈现',
            '第{which}个汉字需要加粗',
        ]

    def check(self, response, slots):
        which = slots["which"]
        chinese_chars = [char for char in response if '\u4e00' <= char and char <= '\u9fff']
        if which > len(chinese_chars) or which < 1:
            return False
        target_char = chinese_chars[which - 1]
        bold_patterns = [re.compile(re.escape(f"**{target_char}**")), re.compile(re.escape(f"__{target_char}__"))]
        for pattern in bold_patterns:
            if pattern.search(response):
                return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        which = random.randint(1, 10)
        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        if random.random() < 0.5:
            zh_num = arabic_to_chinese(which)
            instruction_meta = self.meta_instr[indicator_meta].format(which=zh_num)
        else:
            instruction_meta = self.meta_instr[indicator_meta].format(which=which)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "which": which
            }
        }


### 先去掉，标题不完全是markdown格式
class Rule_BOLD_HEAD(Rule):

    def __init__(self):
        self.rule_type = "BOLD_HEAD"
        self.meta_instr = [
            '加粗标题',
            '标题以粗体的形式呈现',
            '标题需要加粗',
        ]

    def check(self, response, slots):
        heading_regex = re.compile(r'^(#+)\s+(.*)')
        bold_regex = re.compile(r'^\s*(\*\*.*\*\*)|(__.*__)\s*$')
        text = response

        match = heading_regex.match(text)
        if match:
            content = match.group(2).strip()
            return bool(bold_regex.match(content))
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_BOLD_PARA_FIRST(Rule):

    def __init__(self):
        self.rule_type = "BOLD_PARA_FIRST"
        self.meta_instr = [
            '加粗每段的第一个字符',
            '每段的第一个字符以粗体的形式呈现',
            '每段的第一个字符需要加粗',
        ]

    def check(self, response, slots):

        def is_bold(text, patterns):
            for pattern in patterns:
                if re.match(pattern, text):
                    return True
            return False

        paragraphs = re.split(r'\n\s*\n*', response.strip())
        markdown_bold_patterns = [re.compile(r'^\*\*[^\*]\*\*'), re.compile(r'^__[^\_]+__')]

        for paragraph in paragraphs:
            # 获取第一个字符，并检查是否被加粗标记包裹
            if len(paragraph) > 0:
                if not is_bold(paragraph, markdown_bold_patterns):
                    return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_BOLD_NEGATE(Rule):

    def __init__(self):
        self.rule_type = "BOLD_NEGATE"
        self.meta_instr = [
            '不用加粗',
            '不用加粗任何字符',
            '不要加粗',
            '不要加粗任何字符',
            '不要以加粗的形式呈现',
            '禁止加粗',
            '禁止加粗任何字符',
            '禁止以加粗的形式呈现',
            '不可以加粗',
            '不可以加粗任何字符',
            '不能加粗',
            '不能加粗任何字符',
            '不能以加粗的形式呈现',
        ]

    def check(self, response, slots):
        markdown_bold_regex = re.compile(r'(\*\*[^*]+\*\*)|(__[^_]+__)')
        text = response
        if bool(markdown_bold_regex.search(text)):
            return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_BOLD_DIGIT(Rule):

    def __init__(self):
        self.rule_type = "BOLD_DIGIT"
        self.meta_instr = [
            '加粗所有阿拉伯数字',
            '所有阿拉伯数字以粗体的形式呈现',
            '所有阿拉伯数字需要加粗',
        ]

    def check(self, response, slots):
        text = response
        numbers = re.findall(r'\d+', text)
        for number in numbers:
            bold_patterns = [re.compile(re.escape(f"**{number}**")), re.compile(re.escape(f"__{number}__"))]
            if not any(pattern.search(text) for pattern in bold_patterns):
                return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


### 有问题，区分字母和整句话
class Rule_BOLD_LETTER(Rule):

    def __init__(self):
        self.rule_type = "BOLD_LETTER"
        self.meta_instr = [
            '加粗所有英文字母',
            '所有英文字母以粗体的形式呈现',
            '所有英文字母需要加粗',
        ]

    def check(self, response, slots):
        text = response
        letters = re.findall(r'[a-zA-Z]', text)
        for letter in letters:
            bold_patterns = [re.compile(re.escape(f"**{letter}**")), re.compile(re.escape(f"__{letter}__"))]
            if not any(pattern.search(text) for pattern in bold_patterns):
                return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


### 新增一些规则
class Rule_FORMAT_INDEX(Rule):

    def __init__(self):
        self.rule_type = "FORMAT_INDEX"
        self.meta_instr = [
            '以有序列表形式输出',
            '输出请进行编号',
            '分点列出答案，并对每一项进行编号',
            '请按顺序编号您的答案',
            '请使用数字进行分点列出答案',
        ]

    def check(self, response, slots):
        ordered_list_pattern_arabic = r'^\s*\d+\.\s+'
        ordered_list_pattern_chinese = r'^\s*[一二三四五六七八九十]+\.\s+'
        lines = response.split('\n')
        for line in lines:
            if re.match(ordered_list_pattern_arabic, line) or re.match(ordered_list_pattern_chinese, line):
                return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_FORMAT_TITLE(Rule):

    def __init__(self):
        self.rule_type = "FORMAT_TITLE"
        self.meta_instr = [
            '输出第一行是标题，标题格式：以“#”开头',
            '开头添加一个标题，使用“#”进行标记',
            '回答开头需要有标题，以“#”形式呈现',
        ]

    def check(self, response, slots):
        title_pattern = r'^\s*#\s+.*'
        try:
            first_line = response.split('\n')[0]
        except:
            return False
        if re.match(title_pattern, first_line):
            return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_LANGUAGE_MIX(Rule):

    def __init__(self):
        self.rule_type = "LANGUAGE_MIX"
        self.meta_instr = ['用中文回答，结尾再用英文一句话总结', '请用中文回答问题，并在结尾用一句英文总结', '使用中文回答问题，并在最后一句用英文补充', '回答请使用中文，并在结尾用一句英文解释']

    def check(self, response, slots):
        sentence_end_pattern = r'[。！？]'
        sentences = re.split(sentence_end_pattern, response)
        if len(sentences) < 2:
            return False
        try:
            last_sentence = sentences[-1].strip()
        except:
            return False
        english_sentence_pattern = r'^[0-9a-zA-Z\s’‘“”\'\",.?!;:()\[\]{}\-&]+$'
        if not re.match(english_sentence_pattern, last_sentence):
            return False
        chinese_content_pattern = r'^[0-9\u4e00-\u9fa5\s，。！？；：“”‘’、（）：《》〈〉【】…—－—·]+$'
        for sentence in sentences[:-1]:
            if not re.match(chinese_content_pattern, sentence.strip()):
                return False
        return True

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.meta_instr) - 1)
        instruction_meta = self.meta_instr[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {}
        }


class Rule_SENT_NUM_EXPAND(Rule):

    def __init__(self):
        self.rule_type = "SENT_NUM_EXPAND"
        self.meta_instr_equal = [
            '回答中包含{num1}个句子',
        ]
        self.meta_instr_more = [
            '回答中包含不少于{num1}个句子',
        ]
        self.meta_instr_less = [
            '回答中包含不超过{num1}个句子',
        ]
        self.meta_instr_interval = [
            '回答中包含{num1}到{num2}个句子',
        ]

    def check(self, response, slots):
        num1 = slots["num1"]
        num2 = slots["num2"]
        interval_type = slots["interval_type"]
        sents = cut_sent(response)

        if interval_type == "EQUAL":
            if len(sents) == num1:
                return True
            return False
        elif interval_type == "MORE":
            if len(sents) >= num1:
                return True
            return False
        elif interval_type == "LESS":
            if len(sents) <= num1:
                return True
            return False
        elif interval_type == "INTERVAL":
            if len(sents) <= num2 and len(sents) >= num1:
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        interval_type = random.choice(["MORE", "LESS", "EQUAL", "INTERVAL"])
        num1 = random.randint(1, 10)
        num2 = 0

        if interval_type == "EQUAL":
            indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
            instruction_meta = self.meta_instr_equal[indicator_meta].format(num1=num1)
        elif interval_type == "MORE":
            indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
            instruction_meta = self.meta_instr_more[indicator_meta].format(num1=num1)
        elif interval_type == "LESS":
            indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
            instruction_meta = self.meta_instr_less[indicator_meta].format(num1=num1)
        elif interval_type == "INTERVAL":
            num1 = random.randint(1, 5)
            num2 = random.randint(num1 + 1, 10)
            indicator_meta = random.randint(0, len(self.meta_instr_interval) - 1)
            instruction_meta = self.meta_instr_interval[indicator_meta].format(num1=num1, num2=num2)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "interval_type": interval_type,
                "num1": num1,
                "num2": num2
            }
        }


class Rule_SENT_TYPE_NUM_EXPAND(Rule):

    def __init__(self):
        self.rule_type = "SENT_TYPE_NUM_EXPAND"
        self.meta_instr_equal = [
            '回答中包含正好{num1}个{sentence_type}',
        ]
        self.meta_instr_more = [
            '回答中包含不少于{num1}个{sentence_type}',
        ]
        self.meta_instr_less = [
            '回答中包含不超过{num1}个{sentence_type}',
        ]
        self.meta_instr_interval = [
            '回答中包含{num1}到{num2}个{sentence_type}',
        ]

    def check(self, response, slots):
        num1 = slots["num1"]
        num2 = slots["num2"]
        sentence_type = slots["sentence_type"]
        interval_type = slots["interval_type"]
        sents = cut_sent(response)

        if interval_type == "EQUAL":
            if sentence_type == "问句":
                count = 0
                for sent in sents:
                    if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                            sent.endswith('?"') or sent.endswith('?\'')):
                        count += 1
                if count == num1:
                    return True
                return False
            elif sentence_type == "感叹句":
                count = 0
                for sent in sents:
                    if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                            sent.endswith('!"') or sent.endswith('!\'')):
                        count += 1
                if count == num1:
                    return True
                return False
            elif sentence_type == "陈述句":
                count = 0
                for sent in sents:
                    if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                            sent.endswith('."') or sent.endswith('.\'')):
                        count += 1
                if count == num1:
                    return True
                return False
        elif interval_type == "MORE":
            if sentence_type == "问句":
                count = 0
                for sent in sents:
                    if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                            sent.endswith('?"') or sent.endswith('?\'')):
                        count += 1
                if count >= num1:
                    return True
                return False
            elif sentence_type == "感叹句":
                count = 0
                for sent in sents:
                    if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                            sent.endswith('!"') or sent.endswith('!\'')):
                        count += 1
                if count >= num1:
                    return True
                return False
            elif sentence_type == "陈述句":
                count = 0
                for sent in sents:
                    if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                            sent.endswith('."') or sent.endswith('.\'')):
                        count += 1
                if count >= num1:
                    return True
                return False
        elif interval_type == "LESS":
            if sentence_type == "问句":
                count = 0
                for sent in sents:
                    if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                            sent.endswith('?"') or sent.endswith('?\'')):
                        count += 1
                if count <= num1:
                    return True
                return False
            elif sentence_type == "感叹句":
                count = 0
                for sent in sents:
                    if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                            sent.endswith('!"') or sent.endswith('!\'')):
                        count += 1
                if count <= num1:
                    return True
                return False
            elif sentence_type == "陈述句":
                count = 0
                for sent in sents:
                    if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                            sent.endswith('."') or sent.endswith('.\'')):
                        count += 1
                if count <= num1:
                    return True
                return False
        elif interval_type == "INTERVAL":
            if sentence_type == "问句":
                count = 0
                for sent in sents:
                    if (sent.endswith('？') or sent.endswith('?') or sent.endswith('？”') or sent.endswith('？’') or
                            sent.endswith('?"') or sent.endswith('?\'')):
                        count += 1
                if count <= num2 and count >= num1:
                    return True
                return False
            elif sentence_type == "感叹句":
                count = 0
                for sent in sents:
                    if (sent.endswith('！') or sent.endswith('!') or sent.endswith('！”') or sent.endswith('！’') or
                            sent.endswith('!"') or sent.endswith('!\'')):
                        count += 1
                if count <= num2 and count >= num1:
                    return True
                return False
            elif sentence_type == "陈述句":
                count = 0
                for sent in sents:
                    if (sent.endswith('。') or sent.endswith('.') or sent.endswith('。”') or sent.endswith('。’') or
                            sent.endswith('."') or sent.endswith('.\'')):
                        count += 1
                if count <= num2 and count >= num1:
                    return True
                return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        sentence_type = random.choice(["问句", "感叹句", "陈述句"])
        interval_type = random.choice(["MORE", "LESS", "EQUAL", "INTERVAL"])
        num1 = random.randint(1, 10)
        num2 = 0

        if interval_type == "EQUAL":
            indicator_meta = random.randint(0, len(self.meta_instr_equal) - 1)
            instruction_meta = self.meta_instr_equal[indicator_meta].format(sentence_type=sentence_type, num1=num1)
        elif interval_type == "MORE":
            indicator_meta = random.randint(0, len(self.meta_instr_more) - 1)
            instruction_meta = self.meta_instr_more[indicator_meta].format(sentence_type=sentence_type, num1=num1)
        elif interval_type == "LESS":
            indicator_meta = random.randint(0, len(self.meta_instr_less) - 1)
            instruction_meta = self.meta_instr_less[indicator_meta].format(sentence_type=sentence_type, num1=num1)
        elif interval_type == "INTERVAL":
            num1 = random.randint(1, 5)
            num2 = random.randint(num1 + 1, 10)
            indicator_meta = random.randint(0, len(self.meta_instr_interval) - 1)
            instruction_meta = self.meta_instr_interval[indicator_meta].format(sentence_type=sentence_type,
                                                                               num1=num1,
                                                                               num2=num2)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "interval_type": interval_type,
                "sentence_type": sentence_type,
                "num1": num1,
                "num2": num2
            }
        }


class Rule_ZH_NUM_TOTAL_AROUND(Rule):

    def __init__(self):
        self.rule_type = "ZH_NUM_TOTAL_AROUND"
        self.meta_instr_around = ['回答长度大约{num}字', '你的回答需要在{num}字左右', '请将回答限制在大约{num}字', '回答字数应在{num}字上下']

    def check(self, response, slots):
        num = slots["num"]
        num_char = count_chinese_chars(response)
        if abs(num_char - num) / num <= 0.25:
            return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        num = random.choice([
            10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150, 200, 250, 300, 350, 400, 450, 500,
            550, 600, 650, 700, 750, 800, 850, 900, 950, 1000
        ])
        indicator_meta = random.randint(0, len(self.meta_instr_around) - 1)
        if random.random() < 0.5:
            zh_num = arabic_to_chinese(num)
            instruction_meta = self.meta_instr_around[indicator_meta].format(num=zh_num)
        else:
            instruction_meta = self.meta_instr_around[indicator_meta].format(num=num)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                "num": num
            }
        }


class Rule_PSEUDO_SAME(Rule):

    def __init__(self):
        self.rule_type = "PSEUDO_SAME"

    def check(self, response, slots):
        text = slots["text"]
        return response.strip() == text.strip()


class Rule_PSEUDO_NUM_SENTENCES(Rule):

    def __init__(self):
        self.rule_type = "PSEUDO_NUM_SENTENCES"

    def check(self, response, slots):
        num_sentences = slots["num"]
        sentences = re.split(r'(?<=[.!?]) +', response.strip())
        return len(sentences) == num_sentences


class Rule_PSEUDO_NUM_PLACEHOLDERS(Rule):

    def __init__(self):
        self.rule_type = "PSEUDO_NUM_PLACEHOLDERS"

    def check(self, response, slots):
        num_placeholders = slots["num"]
        placeholders = response.count("[PLACEHOLDER]")
        return placeholders == num_placeholders


class Rule_PSEUDO_NUM_BULLETS(Rule):

    def __init__(self):
        self.rule_type = "PSEUDO_NUM_BULLETS"

    def check(self, response, slots):
        num_bullets = slots["num"]
        bullet_points = re.findall(r'^-\s+', response, flags=re.MULTILINE)
        return len(bullet_points) == num_bullets


class Rule_PSEUDO_STARTER(Rule):

    def __init__(self):
        self.rule_type = "PSEUDO_STARTER"

    def check(self, response, slots):
        starter = slots["starter"]
        return response.startswith(starter)


class Rule_PSEUDO_NUM_HIGHLIGHTS(Rule):

    def __init__(self):
        self.rule_type = "PSEUDO_NUM_HIGHLIGHTS"

    def check(self, response, slots):
        num_highlights = slots["num"]
        highlighted_sentences = re.findall(r'\*\*.*?\*\*', response)
        return len(highlighted_sentences) == num_highlights


class Rule_PSEUDO_NUM_SECTIONS(Rule):

    def __init__(self):
        self.rule_type = "PSEUDO_NUM_SECTIONS"

    def check(self, response, slots):
        num_sections = slots["num"]
        splitter = slots["splitter"]
        sections = re.findall(rf'^{re.escape(splitter)}\s', response, flags=re.MULTILINE)
        return len(sections) == num_sections


class Rule_PSEUDO_NUM_PARAGRAPHS(Rule):

    def __init__(self):
        self.rule_type = "PSEUDO_NUM_PARAGRAPHS"

    def check(self, response, slots):
        num_paragraphs = slots["num"]
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        return len(paragraphs) == num_paragraphs


class Rule_PSEUDO_ENDER(Rule):

    def __init__(self):
        self.rule_type = "PSEUDO_ENDER"

    def check(self, response, slots):
        ender = slots["ender"]
        return response.endswith(ender)


class Rule_PSEUDO_KEYWORDS(Rule):

    def __init__(self):
        self.rule_type = "PSEUDO_KEYWORDS"

    def check(self, response, slots):
        keywords = slots["keywords"]
        return all(keyword in response for keyword in keywords)
