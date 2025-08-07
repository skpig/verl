import json
import logging
import os
# pip install 'volcengine-python-sdk[ark]'
# from volcenginesdkarkruntime import Ark
# import httpx
import random
import re
import time

import openai

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, ExtractAnswerFailed, VerifyResult


def has_chinese(text):
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    if chinese_char_pattern.search(text):
        return 1.0
    else:
        return 0.0


def contains_english_word(text):
    """
    判断字符串是否包含英文单词（定义为：以空格分隔的部分，长度>=2且全为英文字母）
    :param text: 输入字符串
    :return: True（含有英文单词），False（不含）
    """
    for part in text.split(" "):
        if part.isalpha() and len(part) >= 2:
            return 1.0
    return 0.0


logger = logging.getLogger()

VERIFY_TEMPLATE = '''你是一位像计算机程序一样严苛、负责审查语言一致性的AI助手。你的核心任务是进行无情的、逐字逐句的扫描。

### **核心工作流程**
在内心严格模拟以下思考过程，然后仅输出最终结论：
1.  **扫描阶段**: 对“学生的回答”全文进行逐字扫描，列出所有非主要语言的字符、词汇、或疑似乱码的可疑点。
2.  **豁免审查阶段**: 对找到的**每一个**可疑点，逐一检查它是否**明确且完全地**符合下方“合理例外情况”列表中的某一条。
3.  **判定阶段**:
    * 只要**有任何一个**可疑点无法被“合理例外情况”豁免，就**必须**判定为“否”。
    * 只有当全文**没有任何可疑点**，或者**所有可疑点都能被豁免**时，才能判定为“是”。
    
请严格遵守以下规则：

1.  你完全不需要关心回答内容的对错，只需要聚焦于语言的使用是否符合规则。

2.  判断标准 (按顺序三步检查)：

    * 第一步：文本有效性检查 (Sanity Check)
        * 首先，检查“学生的回答”是否为基本可读、有意义的文本。
        * 如果回答包含了明显的乱码、无意义的词汇、或由不同语言字符不合逻辑地随机拼凑而成的内容，应直接判定为“否”。
        * 乱码示例:
            * 例1 (通用乱码): "asjdfhaskl correlational-输出-verbatim text..."
            * 例2 (越南语/中文乱码): "Thị trường lao động rất quan重, nó ảnh hưởng đến nền kinh济."
            * 例3 (俄语/中文乱码): "Это мой答复, содер心жащийслучайныесимволы."
            * 例4 (阿拉伯语/英文/中文乱码): "هذا هو الجواب final answ文r."

    * 第二步：语言纯净度检查 (零容忍原则)
        * 如果回答通过了第一步的有效性检查，再检查其内部是否存在任何不合理的语言混用。
        * 核心原则：即使文本整体可读，但如果在一种主要语言的文本中，出现了任何不符合下方“合理例外情况”的、来自其他语言的单个字符、词汇或短语，也应直接判定为“否”。
        * 典型示例 (这些例子本身是可理解的，但违反了纯净度原则):
            * 例A (中文混英文词汇): "my answer is 北京"
            * 例B (英文混中文短语): "The capital of China is a great city, 我觉得."
            * 例C (俄语混中文词汇): "Это очень хорошая книга, 我推荐."
            * 例D (阿拉伯语混英文短语): "هذا هو جوابي, what do you think?"
            * 例E (越南语混中文短语): "Đây là câu trả lời của tôi, 你觉得呢?"
            * 例F (多语言教学文本中的无关字符): 在一段解释法语的越南语文本中，突然出现单个无关的中文字符“你”。
            * 例G (长文本中藏匿的单个词汇): 在一篇数百词的纯法语分析文章中，深埋着一个不相关的中文词汇“一名”。

    * 第三步：与问题/参考答案的一致性检查
        * **原则**: 如果回答通过了前两步检查，最后判断其主要语言是否与“目标语言”一致。
        * **“目标语言”的确定规则 (按a,b,c,d顺序)**:
            * a. **意图优先**: 首先分析“问题”的**内容含义**。如果问题明确指令了回答的目标语言（例如，包含“翻译成中文”、“translate into Japanese”、“in English please”等），则该被指定的语言（中文、日语、英语）就是“目标语言”。
            * b. **默认语言**: 如果问题没有按a条那样指定目标语言，则“问题”本身所使用的主要语言就是“目标语言”。
            * c. **后备方案**: 如果按a,b条都无法确定“问题”的语言（例如，只有一个“?”），则**必须使用“参考答案”的语言**作为“目标语言”。
            * d. **最终豁免**: 如果上述方法都无法确定“目标语言”，则此步检查无法进行，直接判定为“是”。
        * **进行判断**:
            * 如果“学生的回答”的**主要语言**与“目标语言”一致（并且所有非主要语言部分已根据第二步被成功豁免），判定为“是”。
            * 如果“学生的回答”的**主要语言**与“目标语言”不一致，（并且在“目标语言”的确定过程中未使用a条规则指定特定单一语言时），需要检查是否符合下方“合理例外情况”中的d、e、h或i条（此处的d、e、h、i主要针对回答整体语言与问题语言不一致，或回答中包含辅助性提问语言/引用内容的情况）。若符合则为“是”，否则为“否”。

3.  合理例外情况：
    * 以下情况不视为“文本无效”或“语言混杂”，并且允许回答的主要语言与问题不一致（在第三步判断时）：
        * a. 代码或计算机命令: 例如 `print("Hello")`, `git clone`。
        * b. 数学或科学公式: 例如 `E=mc²`, `H₂O`。
        * c. 广为人知的专有名词、品牌、缩写: 例如 `CPU`, `CEO`, `Python`, `Google`。
        * d. **问题本身明确要求在回答中使用特定外语词汇、短语、或撰写特定段落/部分。** 例如，问题包含“请用英文总结”、“将这句话翻译成德语”、“在你的法语回答中引用这句英文名言”。如果学生按要求使用了指定外语，则该外语部分不视为违规混杂。
        * e. **对问题材料的直接引用、分析、转录或输出（包括双语对照呈现）：**
            *   如果“学生的回答”中出现的外语（或主要语言，若与“目标语言”不同）是直接引用、转述、或作为分析、批改、校对、总结、转录或解释对象自“问题”本身所提供的材料（例如，问题描述中的引文、**选择题的选项文本（即使这些选项是图片形式呈现的）**、图片中包含的文字、被要求检查或解释的文本段落、链接的外部文本等），则该语言部分应被豁免。
            *   **核心原则：** 在此类任务中（例如，校对英文文本、总结法文文章、转录西班牙音频、修正德语作业、OCR识别图片中的多语言文本并呈现、**分析图片中选择题的各个选项并说明理由**），回答的核心内容所使用的语言将由被处理材料的语言决定或任务的呈现要求决定，或者，**在对原文材料进行评论、分析或解释时，允许直接引用原文材料中的外语片段以明确指代。**
            *   **双语或多语言呈现豁免：** 如果问题材料本身是多语言的，或者任务是转录/呈现图片或文档内容，而该内容包含多种语言或包含大量术语，则回答中忠实地以原文和/或“原文(目标语言翻译)”的形式呈现这些内容是合理的。例如，图片中的英文术语后括号加注目标语言翻译。
            *   例如：对图片进行OCR识别出的文字、听写音频得到的内容、**学生在分析题目选项（如图片中的英文选项）时，用目标语言（如阿拉伯语）进行分析，并在分析过程中直接引用英文选项原文以明确所指**、学生在按要求批改或解释特定外语例句时引用这些例句并以该外语输出修正后的版本。
        * f. 表情符号或通用格式化符号: 回答中包含通用的表情符号（如 😊, 👍）或用于排版、强调的格式化符号（如 Markdown 的 `*` `_` `#`，或项目符号 `-` `·`）。
        * g. 语言借用与行业术语: 在非英文文本中，对于一些被广泛接受的英文借用词、行业标准术语、或在特定语境下难以用目标语言精确简练表达的英文词汇（例如，在中文技术文档中提及 'API', 'URL', 'bug'），如果其使用不影响文本主要语言的流畅性和理解，则可视为合理。(此条用于少量、偶发的借用，大量术语的并列呈现参考e或h条)
        * h. **翻译任务的上下文呈现与术语并列：**
            *   当“问题”的主要指令是将内容从语言Y翻译成目标语言X，并且“学生的回答”的核心是提供了语言X的翻译时：
                *   **允许使用语言Y（原文）来提供必要的上下文、引导、说明、或结构化辅助**（例如，引言、总结、表格的原文对照列、表头、注释）。
                *   **允许对原文中的术语、专有名词或关键短语采用“原文Y (目标语言X翻译)”的并列形式呈现，或在目标语言X的翻译文本中保留一些普遍接受或难以精准对应翻译的原文Y术语。** 这种做法旨在提高翻译结果对用户的准确性、清晰度和专业性，特别是在技术、学术、法律等领域。
            *   **条件：** 语言X的翻译必须是回答的明确主体。语言Y的辅助内容或并列呈现的原文术语不应掩盖或削弱语言X的翻译，而是服务于其呈现和理解。回答的整体行文应以目标语言X为主。
            *   **示例：** 问题（越南语）：“请将以下英文段落翻译成越南语并制成表格。” 回答：可以用越南语引导，并提供一个包含“英文原文”和“越南语翻译”两列的表格，或者在越南语翻译段落中，对于一些英文术语采用“English Term (Tiếng Việt dịch)”的形式。只要越南语翻译是核心，这种呈现方式中的英文部分可被豁免。
        * i. **教学或解释性引用：** 当“问题”要求解释特定语言现象（如语法点、词汇用法）或概念时，回答可以使用“问题”的语言（或另一种指定的解释语言）进行阐述，并在此过程中引用目标语言（被解释的语言）的简短示例、词汇或短语。这些引用的目标语言内容应被豁免，前提是它们服务于解释目的且不构成回答的主体篇幅。例如：用中文解释英语语法时，引用英文例句；用越南语解释韩语词汇时，给出韩语词汇及其越南语释义（包括括号内对例句的越南语翻译）。
        * j. **其他合理情况**

4.  **最终输出：**
    * 在给出最终答案前，请在内部进行一次最终复核，确认全文中没有任何一个字符或词语违反了上述规则。
    * 写出你的简洁原因和结论，原因尽量简洁，保持50字以内
    * 输出格式是下面json形式，不输出其他内容: 
        ```json{{"reason": "xxx","result": "是/否"}}```

---

输入结构:

<问题>
{}
</问题>

<学生的回答>
{}
</学生的回答>

<参考答案>
{}
</参考答案>'''


def parse_json_string_to_dict(json_string: str):
    """
    将JSON格式的字符串解析为Python字典。

    参数:
        json_string (str): 要解析的JSON格式字符串。

    返回:
        dict: 解析后的字典，如果解析失败则返回None。
    """
    try:

        json_string = json_string.replace("```json", "").replace("```", "").strip()
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(json_string)
        logging.error(f"解析JSON字符串失败: {e}")
        logging.error(f"无效的JSON字符串: {json_string}")
        return None


# https://ark-cn-beijing.bytedance.net/api/v3
# https://ark.cn-beijing.volces.com/api/v3
#GPT_BASE_URL: https://search.bytedance.net/gpt/openapi/online/multimodal/crawl
#GPT_MODEL_NAME: gpt-4.1-2025-04-14


class LLMArenaLangVerifier(BaseVerifier):

    def __init__(self) -> None:
        super().__init__()
        if not os.environ.get('GPT_BASE_URL', None):
            raise ValueError('GPT_BASE_URL is not set')
        if not os.environ.get('GPT_API_KEY', None):
            raise ValueError('GPT_API_KEY is not set')
        if not os.environ.get('GPT_MODEL_NAME', None):
            raise ValueError('GPT_MODEL_NAME is not set')

        GPT_API_KEY = os.environ.get('GPT_API_KEY', None)
        GPT_API_KEY_BAK = os.environ.get('GPT_API_KEY_BAK', None)
        GPT_BASE_URL = os.environ.get('GPT_BASE_URL', None)
        GPT_MODEL_NAME = os.environ.get('GPT_MODEL_NAME', None)
        self.client = openai.AzureOpenAI(azure_endpoint=GPT_BASE_URL,
                                         api_version="2023-07-01-preview",
                                         api_key=GPT_API_KEY)
        self.client_bak = openai.AzureOpenAI(azure_endpoint=GPT_BASE_URL,
                                             api_version="2023-07-01-preview",
                                             api_key=GPT_API_KEY)
        if GPT_API_KEY_BAK:
            self.client_bak = openai.AzureOpenAI(azure_endpoint=GPT_BASE_URL,
                                                 api_version="2023-07-01-preview",
                                                 api_key=GPT_API_KEY_BAK)

        self.model = GPT_MODEL_NAME

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        problem = verifier_feature_dict['problem']
        answer = verifier_feature_dict['answer']
        lang = verifier_feature_dict["lang"]

        if response == "":
            raise ExtractAnswerFailed
        answer_has_chinese = has_chinese(answer)
        response_has_chinese = has_chinese(response)
        answer_has_en = contains_english_word(answer)
        response_has_en = contains_english_word(response)

        score = 1.0
        if lang == "zh":
            if answer_has_chinese and not response_has_chinese:
                score = 0.0
        elif lang == "en":
            # gt中不含有中文，但是预测答案中含有，判定为0分
            if not answer_has_chinese and response_has_chinese:
                score = 0.0
        else:
            # gt中不含有中文，但是预测答案中含有，判定为0分
            if not answer_has_chinese and response_has_chinese:
                score = 0.0
            # 特定语种下，gt中不含有中文，但是预测答案中含有，判定为0分
            if lang in set(["ar", "fa", "it", "ja", "ko", "mk", "uk", "ru", "vi"]):
                if not answer_has_en and response_has_en:
                    score = 0.0
        # 除以上两种之外，比如gt中不含有中文，但是预测答案中也不含有，但是有可能switch到其他比如韩语，也要打压
        # 用LLM泛化
        if score == 1.0:
            for i in range(3):
                try:
                    prompt = VERIFY_TEMPLATE.format(problem, response, answer)
                    random_number = random.randint(0, 1)
                    if random_number == 0:
                        completion = self.client.chat.completions.create(model=self.model,
                                                                         messages=[
                                                                             {
                                                                                 "role": "user",
                                                                                 "content": prompt
                                                                             },
                                                                         ],
                                                                         timeout=120,
                                                                         max_tokens=200)
                    else:
                        completion = self.client_bak.chat.completions.create(model=self.model,
                                                                             messages=[
                                                                                 {
                                                                                     "role": "user",
                                                                                     "content": prompt
                                                                                 },
                                                                             ],
                                                                             timeout=120,
                                                                             max_tokens=200)

                    judge_response = completion.choices[0].message.content
                    judgement_dict = parse_json_string_to_dict(judge_response)
                    if judgement_dict is None:
                        continue
                    judgement = judgement_dict["result"]
                    if judgement.startswith("否") or judgement.lower().startswith("no"):
                        score = 0.0
                    logging.info(
                        f"judgement_dict: {judgement_dict}\n\nproblem: {problem}\n\nanswer: {answer} \n\nresponse: {response}"
                    )
                    break
                except Exception as ex:
                    import traceback
                    logger.info(traceback.format_exc())
                    time.sleep(random.choice(list(range(10, 25))))
                    continue

        logging.info(
            f"verifier score: {score}\n\nprompt lang:{lang}, answer has chinese: {answer_has_chinese}, response has chinese: {response_has_chinese}, answer_has_en: {answer_has_en}, response_has_en: {response_has_en}\n\nproblem: {problem}\n\nanswer: {answer} \n\nresponse: {response}"
        )

        return VerifyResult(score=score, extracted_answer=response)
