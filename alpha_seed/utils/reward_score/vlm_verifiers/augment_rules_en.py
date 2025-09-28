import re
import json
import string
import random
import emoji
# import spacy
import traceback
import multiprocessing
from collections import Counter
import locale
import numpy as np

#TODO：手动设置矛盾的指令

TYPE_TO_WEIGHTS = {
    "FREQUENCY": 2,
    "WORD": 1,
    "SENTENCE": 2,
    "PARAGRAPH": 2,
    "BULLET": 1,
    "CASE": 2,
    "FORMAT": 1,
    "PLACEHOLDER": 1,
    "FORMAT2": 0.25,
}


def sample_rule_types(N):
    keys = np.array(list(TYPE_TO_WEIGHTS.keys()))
    weights = np.array(list(TYPE_TO_WEIGHTS.values()))
    # Normalize the weights
    normalized_weights = weights / weights.sum()
    result = np.random.choice(keys, size=N, replace=False, p=normalized_weights)
    result = list(result)
    if "FORMAT2" in result:
        result = ["FORMAT2"]
    return result


TYPE_TO_RULES = {
    "FREQUENCY": [
        ("FREQUENCY_TYPE", 0.4),
        ("FREQUENCY_WORD", 0.6),
    ],
    "WORD": [("NUM_WORD", 1.0),],
    "SENTENCE": [
        ("NUM_SENTENCE", 0.8),
        ("NOCOMMA", 0.2),
    ],
    "PARAGRAPH": [("NUM_PARAGRAPH", 1.0),],
    "BULLET": [("NUM_BULLET", 1.0),],
    "CASE": [
        ("CASE_LOW", 0.5),
        ("CASE_UP", 0.5),
    ],
    "FORMAT": [("FORMAT_WRAP_RESPONSE", 1.0),],
    "PLACEHOLDER": [("PLACEHOLDER_EN", 1.0),],
    "FORMAT2": [("FORMAT_WRAP_RESPONSE_CONT", 1.0),]
}


def sample_rules(rule_type, N=1):
    keys = np.array([_[0] for _ in TYPE_TO_RULES[rule_type]])
    weights = np.array([_[1] for _ in TYPE_TO_RULES[rule_type]])
    # Normalize the weights
    normalized_weights = weights / weights.sum()
    result = np.random.choice(keys, size=N, replace=False, p=normalized_weights)
    return result.tolist()


################################################
# DEFINE RULE TYPES
################################################

MAX_COUNT = 5

EN_RULE_TYPES = [
    'FREQUENCY_TYPE',
    'FREQUENCY_WORD',
    'NUM_WORD',
    'NUM_SENTENCE',
    'NUM_PARAGRAPH',
    'NUM_BULLET',
    'REPEAT_INSTRUCTION',
    #'REPEAT_RESPONSE',
    'CASE_LOW',
    'CASE_UP',
    #'CASE_UP_LETTER',
    #'CASE_UP_WORD',
    #'CASE_UP_SENTENCE',
    #'CASE_UP_PARAGRAPH',
    #'PUNCTUATION_REPLACEMENT',
    #'PUNCTUATION_IGNORANCE',
    #'FORMAT_WRAP_WORD',
    #'FORMAT_WRAP_SENTENCE',
    #'FORMAT_WRAP_BULLET',
    #'FORMAT_WRAP_PARAGRAPH',
    #'FORMAT_WRAP_INSTRUCTION',
    'FORMAT_WRAP_RESPONSE',
    'FORMAT_WRAP_RESPONSE_CONT',
    'PLACEHOLDER_EN',
    'STARTSWITH_EN',
    'ENDSWITH_EN',
    'HASSENTENCE_EN',
    'NOCOMMA',
    'SENTENCE_WORD',
    'SINGLE_SENTENCE_WORD',
    #'BEGIN_LETTER',
    #'PALINDROME',
    #'DOUBLE_LETTERS',
    # 'SENTENCE_LEN',
    #'NO_LETTER',
    #'CONTAIN_LETTER',
    #'EMOJI_ONLY',
    #'ALPHABETICAL_ORDER',
    #'NO_ADJ_ADV',
    # 'NUM_WORD_TEXT',
    #'USE_LETTER',
    #'QUESTION_ONLY',
    #'SAME_LETTER',
    #'SENTENCE_WORD_COUNT',
]

RULE_TYPES_BULLET = [
    'NUM_BULLET',
    'FORMAT_WRAP_BULLET',
]

# RULE_TYPES_NO_BULLET = [
#     'FREQUENCY_WORD',
#     'FREQUENCY_ADJ',
#     'FREQUENCY_NOUN',
#     'FREQUENCY_VERB',
#     'FREQUENCY_LETTER',
#     'FREQUENCY_CHAR',
#     'NUM_WORD',
#     'NUM_SENTENCE',
#     'NUM_PARAGRAPH',
#     'REPEAT_INSTRUCTION',
#     'REPEAT_RESPONSE',
#     'CASE_LOW',
#     'CASE_UP',
#     'CASE_UP_LETTER',
#     'CASE_UP_WORD',
#     'CASE_UP_SENTENCE',
#     'CASE_UP_PARAGRAPH',
#     'PUNCTUATION_REPLACEMENT',
#     'PUNCTUATION_IGNORANCE',
#     'FORMAT_WRAP_WORD',
#     'FORMAT_WRAP_SENTENCE',
#     'FORMAT_WRAP_PARAGRAPH',
#     'FORMAT_WRAP_INSTRUCTION',
#     'FORMAT_WRAP_RESPONSE',
# ]

# RULE_TYPES_NO_BULLET_NO_FORMAT_WRAP = [
#     'FREQUENCY_WORD',
#     'FREQUENCY_ADJ',
#     'FREQUENCY_NOUN',
#     'FREQUENCY_VERB',
#     'FREQUENCY_LETTER',
#     'FREQUENCY_CHAR',
#     'NUM_WORD',
#     'NUM_SENTENCE',
#     'NUM_PARAGRAPH',
#     'REPEAT_INSTRUCTION',
#     'REPEAT_RESPONSE',
#     'CASE_LOW',
#     'CASE_UP',
#     'CASE_UP_LETTER',
#     'CASE_UP_WORD',
#     'CASE_UP_SENTENCE',
#     'CASE_UP_PARAGRAPH',
#     'PUNCTUATION_REPLACEMENT',
#     'PUNCTUATION_IGNORANCE',
# ]

# RULE_TYPES_NO_BULLET_FORMAT_WRAP = [
#     'FORMAT_WRAP_WORD',
#     'FORMAT_WRAP_SENTENCE',
#     'FORMAT_WRAP_PARAGRAPH',
#     'FORMAT_WRAP_INSTRUCTION',
#     'FORMAT_WRAP_RESPONSE',
# ]

# RULE_TYPES_CODE = [
#     'REPEAT_INSTRUCTION',
#     'REPEAT_RESPONSE',
#     'FORMAT_WRAP_INSTRUCTION',
#     'FORMAT_WRAP_RESPONSE',
# ]

OVERALL_INSTRUCTION_LIST = [
    """{meta_instruction}\n{instruction_ori}""",
    """{instruction_ori}\n{meta_instruction}""",
    """{meta_instruction}\n\n{instruction_ori}""",
    """{instruction_ori}\n\n{meta_instruction}""",
]


def do_NONE(data_item):
    data_item['type'] = 'None'
    return data_item


################################################
# DEFINE FOTMAT
################################################

SYMBOL_SEP = ['###', '***', '---', '===', '///', '~~~']

FORMAT_BRACKET_LIST = [
    '({text})',
    '(({text}))',
    '[{text}]',
    '[[{text}]]',
    '<{text}>',
    '<<{text}>>',
    #|{text}|',
    '[|{text}|]',
    '<|{text}|>',
    '||{text}||',
    #'|-|{text}|-|',
    #'-|{text}|-',
    #'-{text}-',
    #'#{text}#',
    #'###{text}#',
    #'##{text}#',
    #'\{text}\\',
    #'\{text}/',
    #'/{text}\\',
    '*{text}*',
    '**{text}**',
    #'***{text}***',
    #'***{text}*',
    #'**{text}*',
    '"{text}"',
    '@{text}@',
    '@@{text}@@',
    #'@@@{text}@',
    #'${text}$',
    #'$${text}$$',
    #'$$${text}$',
]

FORMAT_TEXT_LIST = [('BEGAIN', 'END'), ('START', 'END'), ('RESPONSE', 'END'), ('RESPONSE', 'CLOSE'), ('OPEN', 'CLOSE'),
                    ('INITIATE', 'TERMINATE'), ('ENTRY', 'EXIT'), ('LAUNCH', 'CONCLUDE'), ('COMMENCE', 'COMPLETE'),
                    ('START_POINT', 'END_POINT'), ('ORIGIN', 'DESTINATION'), ('KICKOFF', 'WRAP UP'),
                    ('RES_START', 'RES_END'), ('RES_BEGIN', 'RES_END'), ('RES', '/RES'), ('BEGIN', 'FINISH'),
                    ('ACTIVATE', 'DEACTIVATE'), ('BEGINNING', 'CLOSURE'), ('STARTUP', 'SHUTDOWN'), ('INIT', 'FINALIZE'),
                    ('BOOT', 'HALT'), ('ENGAGE', 'DISENGAGE'), ('LAUNCH', 'LAND'), ('BEGIN_ACTION', 'END_ACTION'),
                    ('START_SESSION', 'END_SESSION'), ('ON', 'OFF'), ('TRIGGER', 'STOP'),
                    ('INITIATE_PROCESS', 'TERMINATE_PROCESS'), ('STARTUP_SEQUENCE', 'SHUTDOWN_SEQUENCE'),
                    ('OPEN_SESSION', 'CLOSE_SESSION')]

FORMAT_ORDER_LIST = [
    '1st',
    '2nd',
    '3rd',
]


def get_combined_format():
    indicator_format_braket = random.randint(0, len(FORMAT_BRACKET_LIST) - 1)
    format_bracket = FORMAT_BRACKET_LIST[indicator_format_braket]
    indicator_format_text = random.randint(0, len(FORMAT_TEXT_LIST) - 1)
    format_text = FORMAT_TEXT_LIST[indicator_format_text]
    combined_format = format_bracket.format(text=format_text[0]) + '{text}' + format_bracket.format(text=format_text[1])
    return combined_format


################################################
# DEFINE PUNCT
################################################

PUNCT_TO_REPLACE = [
    (',', 'commas'),
    ('.', 'periods'),
    'ALL',
]

PUNCT_REPLACE_WITH = [
    ';',
    '|',
    '_',
    '-',
    '@',
    '#',
    '$',
]

################################################
# FUNCTION: get_response_stastics
################################################

# Load the spaCy model
# nlp = spacy.load('/opt/tiger/en_core_web_sm-3.8.0')
nlp = None


def get_response_words(response):
    doc = nlp(response)
    lemma_dict = {}
    # Categorize, lemmatize, and count the POS
    for token in doc:
        lemma = token.text.lower()
        if lemma in lemma_dict:
            lemma_dict[lemma] += 1
        else:
            lemma_dict[lemma] = 1
    return lemma_dict


def get_pos_counts(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Define dictionaries for each POS category
    adj_dict = {}
    noun_dict = {}
    verb_dict = {}
    adv_dict = {}
    # Categorize, lemmatize, and count the POS
    for token in doc:
        lemma = token.lemma_
        if token.pos_ == 'ADJ':
            if lemma in adj_dict:
                adj_dict[lemma] += 1
            else:
                adj_dict[lemma] = 1
        elif token.pos_ == 'NOUN':
            if lemma in noun_dict:
                noun_dict[lemma] += 1
            else:
                noun_dict[lemma] = 1
        elif token.pos_ == 'VERB':
            if lemma in verb_dict:
                verb_dict[lemma] += 1
            else:
                verb_dict[lemma] = 1
        elif token.pos_ == 'ADV':
            if lemma in adv_dict:
                adv_dict[lemma] += 1
            else:
                adv_dict[lemma] = 1

    return adj_dict, noun_dict, verb_dict, adv_dict


def get_text_statistics(text):
    doc = nlp(text)

    # Word count
    word_count = len([token for token in doc if not token.is_punct])

    # Sentence count
    sentence_count = len(list(doc.sents))

    # Paragraph count (assuming paragraphs are separated by two newlines)
    #paragraphs = re.split(r'\n\s*\n*', text.strip())
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    paragraph_count = len(paragraphs)

    # Bullet points count (assuming bullet points start with -, *, or digits followed by .)
    bullet_points = re.findall(r'([\*\-\d]+[\.\s])', text, re.MULTILINE)
    bullet_point_count = len(bullet_points)

    return word_count, sentence_count, paragraph_count, bullet_point_count


def get_response_stastics(text):

    adj_dict, noun_dict, verb_dict, adv_dict = get_pos_counts(text)
    word_count, sentence_count, paragraph_count, bullet_point_count = get_text_statistics(text)

    stastics_dict = {
        'adj_dict': adj_dict,
        'noun_dict': noun_dict,
        'verb_dict': verb_dict,
        'adv_dict': adv_dict,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'bullet_point_count': bullet_point_count,
    }

    return stastics_dict


def get_sentence_replace_dict(text, sentence_map):
    # Process the text with spacy
    doc = nlp(text)

    replace_dict = {}
    for i, sent in enumerate(doc.sents):
        sentence_text = sent.text.strip()
        if i in sentence_map.keys():
            wrap_symbol = sentence_map[i]
            replace_dict[sentence_text] = wrap_symbol.format(text=sentence_text)

    return replace_dict


def is_code_snippet(text):
    # Common patterns in multiple programming languages
    patterns = {
        'python': [
            r"def\s+\w+\s*\(.*\)\s*:",  # function definitions
            r"class\s+\w+\s*\(.*\)\s*:",  # class definitions
            r"import\s+\w+",  # import statements
            r"from\s+\w+\s+import\s+\w+",  # from ... import ... statements
            r"if\s+.*\s*:",  # if statements
            r"for\s+.*\s*in\s+.*\s*:",  # for loops
            r"while\s+.*\s*:",  # while loops
            r"try\s*:",  # try statements
            r"except\s+.*\s*:",  # except statements
            r"print\s*\(.*\)",  # print statements
            r"\w+\s*=\s*.*"  # variable assignments
        ],
        'javascript': [
            r"function\s+\w+\s*\(.*\)\s*{",  # function definitions
            r"var\s+\w+\s*=\s*.*;",  # variable declarations
            r"let\s+\w+\s*=\s*.*;",  # variable declarations
            r"const\s+\w+\s*=\s*.*;",  # constant declarations
            r"if\s*\(.*\)\s*{",  # if statements
            r"for\s*\(.*\)\s*{",  # for loops
            r"while\s*\(.*\)\s*{",  # while loops
            r"console\.log\s*\(.*\);",  # console.log statements
            r"require\s*\(.*\);",  # require statements
            r"import\s+.*\s+from\s+.*;",  # import statements
            r"export\s+.*\s*{",  # export statements
        ],
        'java': [
            r"public\s+class\s+\w+\s*{",  # class definitions
            r"public\s+static\s+void\s+main\s*\(.*\)\s*{",  # main method
            r"import\s+.*;",  # import statements
            r"if\s*\(.*\)\s*{",  # if statements
            r"for\s*\(.*\)\s*{",  # for loops
            r"while\s*\(.*\)\s*{",  # while loops
            r"try\s*{",  # try blocks
            r"catch\s*\(.*\)\s*{",  # catch blocks
            r"System\.out\.println\s*\(.*\);",  # print statements
            r"\w+\s+\w+\s*=\s*.*;"  # variable declarations
        ]
    }

    for language, lang_patterns in patterns.items():
        for pattern in lang_patterns:
            if re.search(pattern, text):
                return True
    return False


def contain_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def count_non_whitespace_characters(text):
    # Use a generator expression to count non-whitespace characters
    return sum(1 for char in text if not char.isspace())


def count_letters(text):
    # Create a Counter object from the text
    counter = Counter(text)
    # Sum the counts of alphabetic characters only
    letter_count = sum(counter[char] for char in string.ascii_letters)
    return letter_count


def get_lemmas_replace_dict(text, word_map):
    # Process the text with spacy
    doc = nlp(text)

    replace_dict = {}
    for token in doc:
        # Check if the lemma of the token is in the word_map
        if token.lemma_ in word_map:
            wrap_symbol = word_map[token.lemma_]
            replace_dict[token.text] = wrap_symbol.format(text=token.text)

    return replace_dict


def wrap_bullet_points(text, bullet_point_map):

    bullet_points = re.findall(r'([\*\-\d]+[\.\s])', text, re.MULTILINE)

    replace_dict = {}
    for i, line in enumerate(bullet_points):
        trimmed_line = line.strip()
        if i in bullet_point_map.keys():
            wrap_symbol = bullet_point_map[i]
            replace_dict[trimmed_line] = wrap_symbol.format(text=trimmed_line)

    return replace_dict


def wrap_paragraphs(text, paragraph_map):
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())

    replace_dict = {}
    for i, paragraph in enumerate(paragraphs):
        trimmed_paragraph = paragraph.strip()
        if i in paragraph_map.keys():
            wrap_symbol = paragraph_map[i]
            replace_dict[trimmed_paragraph] = wrap_symbol.format(text=trimmed_paragraph)

    return replace_dict


NUMBER_CONSTANT = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen"
}
IN_HUNDRED_CONSTANT = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety"
}
BASE_CONSTANT = {0: " ", 1: "hundred", 2: "thousand", 3: "million", 4: "billion"}


def translateNumberToEnglish(number):
    ### supported number range is 1-n billion;
    if str(number).isnumeric():
        if str(number)[0] == '0' and len(str(number)) > 1:
            return translateNumberToEnglish(int(number[1:]))
        if int(number) < 20:
            return NUMBER_CONSTANT[int(number)]
        elif int(number) < 100:
            if str(number)[1] == '0':
                return IN_HUNDRED_CONSTANT[int(str(number)[0])]
            else:
                return IN_HUNDRED_CONSTANT[int(str(number)[0])] + "-" + NUMBER_CONSTANT[int(str(number)[1])]
        else:
            locale.setlocale(locale.LC_ALL, "English_United States.1252")
            strNumber = locale.format("&d", number, grouping=True)
            numberArray = str(strNumber).split(",")
            stringResult = ""
            groupCount = len(numberArray) + 1
            for groupNumber in numberArray:
                if groupCount > 1 and groupNumber[0:] != "000":
                    stringResult += str(getUnderThreeNumberString(str(groupNumber))) + " "
                else:
                    break
                groupCount -= 1
                if groupCount > 1:
                    stringResult += BASE_CONSTANT[groupCount] + ","
            endPoint = len(stringResult) - len(" hundred,")
            #return stringResult[0:endPoint];
            return stringResult
    else:
        print("please input a number!")


#between 0-999
def getUnderThreeNumberString(number):
    if str(number).isnumeric() and len(number) < 4:
        if len(number) < 3:
            return translateNumberToEnglish(int(number))
        elif len(number) == 3 and number[0:] == "000":
            return " "
        elif len(number) == 3 and number[1:] == "00":
            return NUMBER_CONSTANT[int(number[0])] + " " + BASE_CONSTANT[1]
        else:
            return NUMBER_CONSTANT[int(number[0])] + " " + BASE_CONSTANT[1] + " and " + translateNumberToEnglish(
                (number[1:]))
    else:
        print("number must below 1000")


class Rule:

    def __init__(self, rule_type, meta_instr):
        self.rule_type = rule_type
        self.meta_instr = meta_instr

    def check(self):
        raise NotImplementedError

    def do(self):
        raise NotImplementedError


################################################
# VERIFY_GEN
################################################


def extract_first_function_name(code_str):
    """
    从 Python 代码字符串中提取第一个函数的名称

    参数:
    code_str (str): 包含 Python 代码的字符串

    返回:
    str: 第一个函数的名称，如果没有找到则返回 None
    """
    # 匹配函数定义的正则表达式
    function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('

    # 查找第一个匹配项
    match = re.search(function_pattern, code_str)

    if match:
        return match.group(1)  # 返回捕获的函数名
    else:
        return None  # 没有找到函数定义


def worker_function(code, shared_namespace):
    local_namespace = {}
    local_namespace.update(dict(shared_namespace))
    try:
        exec(code, local_namespace)
        if 'final_call_res' in local_namespace:
            final_result = local_namespace['final_call_res']
            shared_namespace['final_call_res'] = final_result
        else:
            shared_namespace['_exception'] = "代码执行完毕，但未找到 'final_call_res' 变量。"
    except Exception as e:
        shared_namespace["_exception"] = traceback.format_exc()


def run_with_timeout_multiprocess(code, namespace, timeout=1):
    """
    在超时限制内执行代码。
    成功执行则返回 True，如果超时或内部出错则返回 False。
    执行结果会写回传入的 namespace 字典中。
    """
    manager = multiprocessing.Manager()
    shared_namespace = manager.dict()
    shared_namespace.update(namespace)
    p = multiprocessing.Process(target=worker_function, args=(code, shared_namespace))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print("代码执行超时，正在终止子进程...")
        p.terminate()
        p.join()
        return False
    if "_exception" in shared_namespace:
        print(f"子进程执行出错:\n{shared_namespace['_exception']}")
        return False
    namespace.clear()
    namespace.update(dict(shared_namespace))
    return True


def run_check_resp(code_checker, response, description):
    """
    检查函数。
    - 如果检查函数返回 True，则返回 True。
    - 如果检查函数返回 False，则返回 False。
    - 如果发生任何错误（超时、执行错误、返回非bool等），则返回 None。
    """
    try:
        func_name = extract_first_function_name(code_checker)
        if not func_name:
            print("无法从code_checker中提取函数名。")
            return None
        code_to_run = f"""
{code_checker}
final_call_res = {func_name}(response)
"""
        namespace = {'response': response}
        success = run_with_timeout_multiprocess(code_to_run, namespace, timeout=2)
        if not success:
            return None
        final_call_res = namespace.get("final_call_res")
        if not isinstance(final_call_res, bool):
            print(f"检查函数返回的不是bool，而是 {type(final_call_res)}: {final_call_res}")
            return None
        return final_call_res
    except Exception as e:
        print(f"run_check_resp_final报错: {e}")
        traceback.print_exc()
        return None


class Rule_VERIFY_GEN(Rule):

    def __init__(self):
        self.rule_type = "VERIFY_GEN"

    def check(self, response, slots):
        description = ""
        res = run_check_resp(slots["code_checker"], response, description)
        if res == True:
            return True
        return False


################################################
# EG_WORD_COUNT
################################################


class Rule_EN_WORD_COUNT(Rule):

    def __init__(self):
        self.rule_type = "EG_WORD_COUNT"

    def check(self, response, slots):
        """
        Checks the word count of a given text against a target number based on a specified operation.

        Args:
            text: The input string to analyze.
            operation: The comparison operation. Must be one of:
                    'less_than', 'more_than', 'equal', 'approximately_equal'.
            target_count: The integer word count to compare against.

        Returns:
            True if the condition is met, False otherwise.

        Raises:
            ValueError: If an invalid operation is provided.
        """
        # Split the text by whitespace to get a list of words, then count them.
        # This is a simple approach and works for space-separated languages.
        actual_count = len(response.split())
        target_count = slots["num"]
        operation = slots["op"]

        # --- Perform the requested check ---
        if operation == 'less_than':
            # Check if the actual word count is strictly less than the target.
            return actual_count < target_count
        elif operation == 'more_than':
            # Check if the actual word count is strictly more than the target.
            return actual_count > target_count
        elif operation == 'equal':
            # Check if the actual word count is exactly equal to the target.
            return actual_count == target_count
        elif operation == 'approximately_equal':
            # Check if the actual count is within a 10% tolerance of the target.
            # Calculate the allowed deviation (10% of the target count).
            tolerance = target_count * 0.10
            # Define the lower and upper bounds for the acceptable range.
            lower_bound = target_count - tolerance
            upper_bound = target_count + tolerance
            # The check is inclusive of the bounds.
            return lower_bound <= actual_count <= upper_bound
        else:
            # If the operation is not one of the valid options, raise an error.
            raise ValueError("Invalid operation. Please use 'less_than', 'more_than', "
                             "'equal', or 'approximately_equal'.")


class Rule_EN_WORD_COUNT_NEW(Rule):

    def __init__(self):
        self.rule_type = "EG_WORD_COUNT"

    def check(self, response, slots):
        num = slots["num"]
        op = slots["op"]
        num_word = len(response.split())
        if op == 'EQUAL':
            if num_word == num:
                return True
            return False
        if op == 'ABOUT':
            threshold = 0.1
            if abs(num_word - num) / num <= threshold:
                return True
            return False
        elif op == 'MORE':
            if num_word >= num:
                return True
            return False
        elif op == 'LESS':
            if num_word <= num and num_word >= 1:
                return True
            return False
        elif op == 'LESS_EQUAL':
            if num_word <= num and num_word >= 1:
                return True
            return False
        elif op == 'MORE_EQUAL':
            if num_word >= num:
                return True
            return False
        return False


################################################
# FREQUENCY_TYPE
################################################


class Rule_FREQUENCY_TYPE(Rule):

    def __init__(self):
        self.rule_type = "FREQUENCY_TYPE"
        self.META_FREQUENCY_TYPE_EQUAL = [
            'Ensure there are exactly {count} {type} in the response.',
            'Make sure the response includes exactly {count} {type}.',
            #'The response should contain {count} {type}.',
            #'Make certain that {count} {type} are in the response.',
            #'Ensure the response comprises {count} {type}.',
            #'Guarantee that there are {count} {type} in the response.',
            #'Check that the response includes {count} {type}.',
            #'Make sure there are {count} {type} within the response.',
            #'Ensure {count} {type} appear in the response.',
            'Confirm that the response contains exactly {count} {type}.',
        ]
        self.META_FREQUENCY_TYPE_LESS = [
            'Ensure there are less than {count} {type} in the response.',
            #'Confirm that there are less than {count} {type} present in the response.',
            #'Check that the response includes fewer than {count} {type}.',
            #'Make sure the response has under {count} {type}.',
            'Ascertain that there are fewer than {count} {type} in the response.',
            #'Guarantee that the count of {type} in the response is less than {count}.',
            'Make sure there are less than {count} {type} in the response.',
            #'Ensure the response holds less than {count} {type}.',
        ]
        self.META_FREQUENCY_TYPE_MORE = [
            'Ensure there are more than {count} {type} in the response.',
            #'Verify that the response contains more than {count} {type}.',
            'Confirm that there are at least {count} {type} in the response.',
            #'Ensure the response has a minimum of {count} {type}.',
            #'Check that the response includes more than {count} {type}.',
            #'Guarantee there are above {count} {type} in the response.',
            #'Make certain the response contains at least {count} {type}.',
            #'Be sure the response exceeds {count} {type}.',
            'Confirm there are at least {count} {type} within the response.',
            #'Validate that the response has more than {count} {type}.',
        ]

    #[done]
    def check(self, response, slots):
        indicator_count_version = slots["indicator_count_version"]
        word_type = slots["word_type"]
        word_count = slots["word_count"]

        stastics_dict = get_response_stastics(response)
        adj_count = len(stastics_dict['adj_dict'])
        noun_count = len(stastics_dict['noun_dict'])
        verb_count = len(stastics_dict['verb_dict'])

        if word_type == 'ADJ':
            if indicator_count_version == 'EQUAL':
                if word_count == adj_count:
                    return True
                return False
            elif indicator_count_version == 'LESS':
                if adj_count < word_count:
                    return True
                return False
            elif indicator_count_version == 'MORE':
                if adj_count > word_count:
                    return True
                return False
        elif word_type == 'NOUN':
            if indicator_count_version == 'EQUAL':
                if word_count == noun_count:
                    return True
                return False
            elif indicator_count_version == 'LESS':
                if noun_count < word_count:
                    return True
                return False
            elif indicator_count_version == 'MORE':
                if noun_count > word_count:
                    return True
                return False
        elif word_type == 'VERB':
            if indicator_count_version == 'EQUAL':
                if word_count == verb_count:
                    return True
                return False
            elif indicator_count_version == 'LESS':
                if verb_count < word_count:
                    return True
                return False
            elif indicator_count_version == 'MORE':
                if verb_count > word_count:
                    return True
                return False
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_count_version = random.choice(['EQUAL', 'LESS', 'MORE'])
        given_type = random.choice(['ADJ', 'NOUN', 'VERB'])

        #stastics_dict = get_response_stastics(response)
        #adj_count = len(stastics_dict['adj_dict'])
        #noun_count = len(stastics_dict['noun_dict'])
        #verb_count = len(stastics_dict['verb_dict'])
        adj_count = random.randint(3, 8)
        noun_count = random.randint(5, 10)
        verb_count = random.randint(5, 10)

        if given_type == 'ADJ':
            count = adj_count
            type_name = 'adjectives'
        elif given_type == 'NOUN':
            count = noun_count
            type_name = 'nouns'
        elif given_type == 'VERB':
            count = verb_count
            type_name = 'verbs'

        if indicator_count_version == 'EQUAL':
            indicator_meta = random.randint(0, len(self.META_FREQUENCY_TYPE_EQUAL) - 1)
            instruction_meta = self.META_FREQUENCY_TYPE_EQUAL[indicator_meta].format(count=count, type=type_name)
            word_count = count
        elif indicator_count_version == 'LESS':
            #random_gap = random.randint(0, int(count*0.5))
            #indicator_meta = random.randint(0, len(self.META_FREQUENCY_TYPE_LESS)-1)
            #instruction_meta = self.META_FREQUENCY_TYPE_LESS[indicator_meta].format(count=count+random_gap, type=type_name)
            #word_count = count + random_gap
            indicator_meta = random.randint(0, len(self.META_FREQUENCY_TYPE_LESS) - 1)
            instruction_meta = self.META_FREQUENCY_TYPE_LESS[indicator_meta].format(count=count, type=type_name)
            word_count = count
        elif indicator_count_version == 'MORE':
            #random_gap = random.randint(0, int(count*0.5))
            #indicator_meta = random.randint(0, len(self.META_FREQUENCY_TYPE_MORE)-1)
            #instruction_meta = self.META_FREQUENCY_TYPE_MORE[indicator_meta].format(count=count-random_gap, type=type_name)
            #word_count = count - random_gap
            indicator_meta = random.randint(0, len(self.META_FREQUENCY_TYPE_MORE) - 1)
            instruction_meta = self.META_FREQUENCY_TYPE_MORE[indicator_meta].format(count=count, type=type_name)
            word_count = count

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': 'FREQUENCY_TYPE',
            'slots': {
                "indicator_count_version": indicator_count_version,
                "word_type": given_type,
                "word_count": word_count
            }
        }


################################################
# FREQUENCY_WORD
################################################


class Rule_FREQUENCY_WORD(Rule):

    def __init__(self):
        self.rule_type = "FREQUENCY_WORD"
        self.META_FREQUENCY_WORD = [
            'Ensure that {meta_seg} in the response.',
            'Make sure that {meta_seg} in the response.',
            'Guarantee that {meta_seg} in the response.',
            #'Be certain that {meta_seg} within the response.',
            #'Make certain that the response encompasses {meta_seg}.',
        ]
        self.META_SEG_FREQUENCY_WORD_EQUAL = [
            'the word "{word}" appears exactly {count} times',
            #'the term "{word}" is mentioned {count} times',
            'the word "{word}" shows up exactly {count} times',
            #'the phrase "{word}" is used {count} times',
            #'the word "{word}" appears a total of {count} times',
            #'the entry "{word}" is counted {count} times',
            #'the occurrence count of the word"{word}" is {count}',
            #'the word "{word}" can be found {count} times',
            #'the usage of the word "{word}" is {count} times',
        ]
        self.META_SEG_FREQUENCY_WORD_LESS = [
            'the word "{word}" appears and it appears at most {count} times',
            #'the word "{word}" is repeated at most {count} times',
            #'the occurrence of "{word}" is at most {count} times',
            'the word "{word}" shows up and appears at most {count} times',
            #'the occurrences of the word "{word}" amount to at most {count} times',
            #'the word "{word}" appears up to {count} times',
            #'the maximum occurrence of the word "{word}" is {count} times',
            #'the maximum number of appearances for the word "{word}" is {count}',
            #'the use of the word "{word}" does not go beyond {count} times',
            #'the frequency of the word "{word}" does not exceed {count} instances',
        ]
        self.META_SEG_FREQUENCY_WORD_MORE = [
            'the word "{word}" appears at least {count} times',
            'the word "{word}" shows up at least {count} times',
            #'the word "{word}" is used at least {count} times',
            #'the occurrence of the word "{word}" is at least {count} times',
            #'the word "{word}" appears a minimum of {count} times',
            #'there are at least {count} appearances of the word "{word}"',
            #'the word "{word}" occurs at a minimum {count} times',
            #'the word "{word}" occurs no less than {count} times',
            #'at least {count} mentions of the word "{word}" are present',
        ]
        self.META_SEG_FREQUENCY_WORD_INCLUDE = [
            'the word "{word}" appears',
            'the word "{word}" is present',
            'the word "{word}" shows up',
            #'the word "{word}" can be found appearing',
            #'the appearance of the word "{word}" is observed',
            #'the word "{word}" makes an appearance',
            #'the word "{word}" is observed to appear',
            #'the word "{word}" is detected',
            #'the word "{word}" is identified',
        ]

    #[基本done]
    def check(self, response, slots):
        indicator_count_version = slots["indicator_count_version"]
        word = slots["word"]
        count = slots["count"]
        if slots["indicator_count_version"] == 'EQUAL':
            lemma_dict = get_response_words(response)
            if word in lemma_dict and lemma_dict[word] == count:
                return True
            else:
                return False
        elif slots["indicator_count_version"] == 'LESS':
            lemma_dict = get_response_words(response)
            if word in lemma_dict and lemma_dict[word] <= count:
                return True
            else:
                return False
        elif slots["indicator_count_version"] == 'MORE':
            lemma_dict = get_response_words(response)
            if word in lemma_dict and lemma_dict[word] >= count:
                return True
            else:
                return False
        elif slots["indicator_count_version"] == 'INCLUDE':
            lemma_dict = get_response_words(response)
            if word in lemma_dict:
                return True
            else:
                return False
        else:
            raise ValueError("Invalid indicator_count_version")

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_count_version = random.choice(['EQUAL', 'LESS', 'MORE', 'INCLUDE'])

        stastics_dict = get_response_stastics(response)
        merged_dict = {**stastics_dict['adj_dict'], **stastics_dict['noun_dict'], **stastics_dict['verb_dict']}

        pos_count = len(merged_dict.items())
        if pos_count == 0:
            return None

        sampled_word = random.sample(merged_dict.items(), 1)[0][0].lower()
        rand_count = random.randint(1, 5)
        sampled_pairs = [(sampled_word, rand_count)]

        instruction_seg = ''
        if indicator_count_version == 'EQUAL':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_FREQUENCY_WORD_EQUAL) - 1)
            for word, count in sampled_pairs:
                instruction_seg += (
                    self.META_SEG_FREQUENCY_WORD_EQUAL[indicator_meta_seg].format(word=word, count=count) + ', ')
        if indicator_count_version == 'INCLUDE':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_FREQUENCY_WORD_INCLUDE) - 1)
            for word, count in sampled_pairs:
                instruction_seg += (
                    self.META_SEG_FREQUENCY_WORD_INCLUDE[indicator_meta_seg].format(word=word, count=count) + ', ')
        elif indicator_count_version == 'LESS':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_FREQUENCY_WORD_LESS) - 1)
            for word, count in sampled_pairs:
                instruction_seg += (
                    self.META_SEG_FREQUENCY_WORD_LESS[indicator_meta_seg].format(word=word, count=count) + ', ')
        elif indicator_count_version == 'MORE':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_FREQUENCY_WORD_MORE) - 1)
            for word, count in sampled_pairs:
                instruction_seg += (
                    self.META_SEG_FREQUENCY_WORD_MORE[indicator_meta_seg].format(word=word, count=count) + ', ')

        indicator_meta = random.randint(0, len(self.META_FREQUENCY_WORD) - 1)
        instruction_meta = self.META_FREQUENCY_WORD[indicator_meta].format(meta_seg=instruction_seg[:-2])

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
                "word": word,
                "count": count
            }
        }


################################################
# NUM_WORD
################################################


class Rule_NUM_WORD(Rule):

    def __init__(self):
        self.rule_type = "NUM_WORD"
        self.META_NUM_WORD = [
            'Ensure that the response {meta_seg}.',
            'Make sure that the response {meta_seg}.',
            #'Confirm that the response {meta_seg}.',
            'Guarantee that the response {meta_seg}.',
            #'Ascertain that the response {meta_seg}.',
            #'Make certain the response {meta_seg}.',
            #'Be sure the response {meta_seg}.',
            #'Confirm there is a response {meta_seg}.',
            #'Check to ensure the response {meta_seg}.',
        ]
        self.META_SEG_NUM_WORD_EQUAL = [
            'has exactly {count} words',
            'contains exactly {count} words',
            'includes exactly {count} words',
            #'comprises exactly {count} words',
            #'has precisely {count} words',
            #'holds exactly {count} words',
            #'encompasses exactly {count} words',
            #'possesses exactly {count} words',
            #'amounts to exactly {count} words',
            #'reaches exactly {count} words',
        ]
        self.META_SEG_NUM_WORD_LESS = [
            'has less than {count} words',
            'contains fewer than {count} words',
            'includes less than {count} words',
            #'has a word count below {count}',
            #'uses less than {count} words',
            #'keeps the word count under {count}',
            #'remains below {count} words',
            #'limits to fewer than {count} words',
            #'comprises fewer than {count} words',
            #'falls short of {count} words',
        ]
        self.META_SEG_NUM_WORD_MORE = [
            'has more than {count} words',
            'includes more than {count} words',
            'exceeds {count} words',
            #'has a word count above {count}',
            #'uses more than {count} words',
            #'surpasses {count} words',
            'consists of more than {count} words',
            #'comprises more than {count} words',
            #'has over {count} words',
            #'contains a word count exceeding {count}',
        ]

    #[done]
    def check(self, response, slots):
        indicator_count_version = slots["indicator_count_version"]
        count = slots["count"]
        word_count, sentence_count, paragraph_count, bullet_point_count = get_text_statistics(response)
        if indicator_count_version == "EQUAL":
            if word_count == count:
                return True
            return False
        elif indicator_count_version == "LESS":
            if word_count < count:
                return True
            return False
        elif indicator_count_version == "MORE":
            if word_count > count:
                return True
            return False
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_count_version = random.choice(['EQUAL', 'LESS', 'MORE'])

        #word_count, sentence_count, paragraph_count, bullet_point_count = get_text_statistics(response)
        rand_count = random.randint(50, 200)

        instruction_seg = ''
        if indicator_count_version == 'EQUAL':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_WORD_EQUAL) - 1)
            instruction_seg = self.META_SEG_NUM_WORD_EQUAL[indicator_meta_seg].format(count=rand_count)
            count = rand_count
        elif indicator_count_version == 'LESS':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_WORD_LESS) - 1)
            instruction_seg = self.META_SEG_NUM_WORD_LESS[indicator_meta_seg].format(count=rand_count)
            count = rand_count
        elif indicator_count_version == 'MORE':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_WORD_MORE) - 1)
            instruction_seg = self.META_SEG_NUM_WORD_MORE[indicator_meta_seg].format(count=rand_count)
            count = rand_count

        indicator_meta = random.randint(0, len(self.META_NUM_WORD) - 1)
        instruction_meta = self.META_NUM_WORD[indicator_meta].format(meta_seg=instruction_seg)

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
                "count": count
            }
        }


################################################
# NUM_SENTENCE
################################################


class Rule_NUM_SENTENCE(Rule):

    def __init__(self):
        self.rule_type = "NUM_SENTENCE"
        self.META_NUM_SENTENCE = [
            'Ensure that the response {meta_seg}.',
            'Make sure that the response {meta_seg}.',
            #'Confirm the response {meta_seg}.',
            'Guarantee that the response {meta_seg}.',
            #'Ascertain that the response {meta_seg}.',
            #'Make certain the response {meta_seg}.',
            'Be sure that the response {meta_seg}.',
            #'Confirm there is a response {meta_seg}.',
            #'Check to ensure the response {meta_seg}.',
        ]
        self.META_SEG_NUM_SENTENCE_EQUAL = [
            'has exactly {count} sentences',
            'contains exactly {count} sentences',
            'includes exactly {count} sentences',
            #'has precisely {count} sentences',
            #'consists of exactly {count} sentences',
            #'comprises exactly {count} sentences',
            #'holds exactly {count} sentences',
            #'encompasses exactly {count} sentences',
        ]
        self.META_SEG_NUM_SENTENCE_LESS = [
            'has less than {count} sentences',
            'contains fewer than {count} sentences',
            'includes less than {count} sentences',
            'has fewer than {count} sentences',
            #'keeps the sentence count under {count}',
            #'remains below {count} sentences',
            #'comprises fewer than {count} sentences',
            #'falls short of {count} sentences',
            #'limits to fewer than {count} sentences',
            #'holds less than {count} sentences',
        ]
        self.META_SEG_NUM_SENTENCE_MORE = [
            'has more than {count} sentences',
            'contains more than {count} sentences',
            #'exceeds {count} sentences',
            #'has a sentence count above {count}',
            #'surpasses {count} sentences',
            'consists of more than {count} sentences',
            #'comprises more than {count} sentences',
            #'has over {count} sentences',
            #'contains a sentence count exceeding {count}',
            #'exceeds a count of {count} sentences',
        ]

    #[done]
    def check(self, response, slots):
        indicator_count_version = slots["indicator_count_version"]
        count = slots["count"]
        word_count, sentence_count, paragraph_count, bullet_point_count = get_text_statistics(response)
        if indicator_count_version == "EQUAL":
            if sentence_count == count:
                return True
            return False
        elif indicator_count_version == "LESS":
            if sentence_count < count:
                return True
            return False
        elif indicator_count_version == "MORE":
            if sentence_count > count:
                return True
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_count_version = random.choice(['EQUAL', 'LESS', 'MORE'])

        #word_count, sentence_count, paragraph_count, bullet_point_count = get_text_statistics(response)
        rand_count = random.randint(2, 20)

        instruction_seg = ''
        if indicator_count_version == 'EQUAL':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_SENTENCE_EQUAL) - 1)
            instruction_seg = self.META_SEG_NUM_SENTENCE_EQUAL[indicator_meta_seg].format(count=rand_count)
            count = rand_count
        elif indicator_count_version == 'LESS':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_SENTENCE_LESS) - 1)
            instruction_seg = self.META_SEG_NUM_SENTENCE_LESS[indicator_meta_seg].format(count=rand_count)
            count = rand_count
        elif indicator_count_version == 'MORE':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_SENTENCE_MORE) - 1)
            instruction_seg = self.META_SEG_NUM_SENTENCE_MORE[indicator_meta_seg].format(count=rand_count)
            count = rand_count

        indicator_meta = random.randint(0, len(self.META_NUM_SENTENCE) - 1)
        instruction_meta = self.META_NUM_SENTENCE[indicator_meta].format(meta_seg=instruction_seg)

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
                "count": count
            }
        }


################################################
# NUM_PARAGRAPH
################################################


class Rule_NUM_PARAGRAPH(Rule):

    def __init__(self):
        self.rule_type = "NUM_PARAGRAPH"
        self.META_NUM_PARAGRAPH = [
            'Ensure that the response {meta_seg}.',
            'Make sure that the response {meta_seg}.',
            'Confirm that the response {meta_seg}.',
            'Guarantee that the response {meta_seg}.',
            #'Ascertain that the response {meta_seg}.',
            #'Make certain the response {meta_seg}.',
            'Be sure that the response {meta_seg}.',
            #'Confirm there is a response {meta_seg}.',
            #'Check to ensure the response {meta_seg}.',
        ]
        self.META_SEG_NUM_PARAGRAPH_EQUAL = [
            'has exactly {count} paragraphs',
            #'has precisely {count} paragraphs',
            'consists of exactly {count} paragraphs',
            #'comprises exactly {count} paragraphs',
            #'encompasses exactly {count} paragraphs',
            #'holds exactly {count} paragraphs',
            #'reaches exactly {count} paragraphs',
        ]
        self.META_SEG_NUM_PARAGRAPH_LESS = [
            'has less than {count} paragraphs',
            'contains fewer than {count} paragraphs',
            'includes less than {count} paragraphs',
            'has fewer than {count} paragraphs',
            #'remains below {count} paragraphs',
            #'comprises fewer than {count} paragraphs',
            #'falls short of {count} paragraphs',
            #'contains a paragraph count below {count}',
            #'holds less than {count} paragraphs',
        ]
        self.META_SEG_NUM_PARAGRAPH_MORE = [
            'has more than {count} paragraphs',
            #'includes over {count} paragraphs',
            #'exceeds {count} paragraphs',
            #'has a paragraph count above {count}',
            #'uses more than {count} paragraphs',
            #'surpasses {count} paragraphs',
            'consists of more than {count} paragraphs',
            #'comprises more than {count} paragraphs',
            #'contains a paragraph count exceeding {count}',
            #'exceeds a count of {count} paragraphs',
        ]

    #[done]
    def check(self, response, slots):
        indicator_count_version = slots["indicator_count_version"]
        count = slots["count"]
        word_count, sentence_count, paragraph_count, bullet_point_count = get_text_statistics(response)
        if indicator_count_version == "EQUAL":
            if paragraph_count == count:
                return True
            return False
        elif indicator_count_version == "LESS":
            if paragraph_count < count:
                return True
            return False
        elif indicator_count_version == "MORE":
            if paragraph_count > count:
                return True
            return False
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_count_version = random.choice(['EQUAL', 'LESS', 'MORE'])

        #word_count, sentence_count, paragraph_count, bullet_point_count = get_text_statistics(response)
        rand_count = random.randint(2, 7)

        instruction_seg = ''
        if indicator_count_version == 'EQUAL':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_PARAGRAPH_EQUAL) - 1)
            instruction_seg = self.META_SEG_NUM_PARAGRAPH_EQUAL[indicator_meta_seg].format(count=rand_count)
            count = rand_count
        elif indicator_count_version == 'LESS':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_PARAGRAPH_LESS) - 1)
            instruction_seg = self.META_SEG_NUM_PARAGRAPH_LESS[indicator_meta_seg].format(count=rand_count)
            count = rand_count
        elif indicator_count_version == 'MORE':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_PARAGRAPH_MORE) - 1)
            instruction_seg = self.META_SEG_NUM_PARAGRAPH_MORE[indicator_meta_seg].format(count=rand_count)
            count = rand_count

        indicator_meta = random.randint(0, len(self.META_NUM_PARAGRAPH) - 1)
        instruction_meta = self.META_NUM_PARAGRAPH[indicator_meta].format(meta_seg=instruction_seg)

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
                "count": count
            }
        }


################################################
# NUM_BULLET
################################################


class Rule_NUM_BULLET(Rule):

    def __init__(self):
        self.rule_type = "NUM_BULLET"
        self.META_NUM_BULLET = [
            'Ensure that the response {meta_seg}.',
            'Make sure that the response {meta_seg}.',
            #'Confirm the response {meta_seg}.',
            'Guarantee that the response {meta_seg}.',
            #'Ascertain that the response {meta_seg}.',
            #'Make certain the response {meta_seg}.',
            'Be sure that the response {meta_seg}.',
            #'Confirm there is a response {meta_seg}.',
            #'Check to ensure the response {meta_seg}.',
        ]
        self.META_SEG_NUM_BULLET_EQUAL = [
            'has exactly {count} bullet points',
            'contains exactly {count} bullet points',
            'includes exactly {count} bullet points',
            #'comprises exactly {count} bullet points',
            'consists of exactly {count} bullet points',
            #'encompasses exactly {count} bullet points',
            #'holds exactly {count} bullet points',
            #'numbers exactly {count} bullet points',
        ]
        self.META_SEG_NUM_BULLET_LESS = [
            'has less than {count} bullet points',
            #'exceeds no more than {count} bullet points',
            #'keeps the bullet point count under {count}',
            #'remains below {count} bullet points',
            #'restricts to less than {count} bullet points',
            #'comprises fewer than {count} bullet points',
            #'falls short of {count} bullet points',
            #'limits to fewer than {count} bullet points',
            'holds less than {count} bullet points',
            #'numbers fewer than {count} bullet points',
        ]
        self.META_SEG_NUM_BULLET_MORE = [
            'has more than {count} bullet points',
            #'exceeds {count} bullet points',
            #'has a bullet point count above {count}',
            #'uses more than {count} bullet points',
            #'surpasses {count} bullet points',
            'consists of more than {count} bullet points',
            #'comprises more than {count} bullet points',
            #'remains over {count} bullet points',
            #'contains a bullet point count exceeding {count}',
            #'exceeds a count of {count} bullet points',
        ]

    #[done]
    def check(self, response, slots):
        indicator_count_version = slots["indicator_count_version"]
        count = slots["count"]
        word_count, sentence_count, paragraph_count, bullet_point_count = get_text_statistics(response)
        if indicator_count_version == "EQUAL":
            if bullet_point_count == count:
                return True
            return False
        elif indicator_count_version == "LESS":
            if bullet_point_count < count:
                return True
            return False
        elif indicator_count_version == "MORE":
            if bullet_point_count > count:
                return True
            return False
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        #indicator_count_version = random.choice(['EQUAL', 'LESS', 'MORE'])
        indicator_count_version = 'EQUAL'

        #word_count, sentence_count, paragraph_count, bullet_point_count = get_text_statistics(response)
        rand_count = random.randint(2, 5)

        instruction_seg = ''
        if indicator_count_version == 'EQUAL':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_BULLET_EQUAL) - 1)
            instruction_seg = self.META_SEG_NUM_BULLET_EQUAL[indicator_meta_seg].format(count=rand_count)
            count = rand_count
        elif indicator_count_version == 'LESS':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_BULLET_LESS) - 1)
            instruction_seg = self.META_SEG_NUM_BULLET_LESS[indicator_meta_seg].format(count=rand_count)
            count = rand_count
        elif indicator_count_version == 'MORE':
            indicator_meta_seg = random.randint(0, len(self.META_SEG_NUM_BULLET_MORE) - 1)
            instruction_seg = self.META_SEG_NUM_BULLET_MORE[indicator_meta_seg].format(count=rand_count)
            count = rand_count

        indicator_meta = random.randint(0, len(self.META_NUM_BULLET) - 1)
        instruction_meta = self.META_NUM_BULLET[indicator_meta].format(meta_seg=instruction_seg)

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
                "count": count
            }
        }


################################################
# REPEAT_INSTRUCTION
################################################


class Rule_REPEAT_INSTRUCTION(Rule):

    def __init__(self):
        self.rule_type = "REPEAT_INSTRUCTION"
        self.META_REPEAT_INSTRUCTION = [
            'First repeat the instruction, then provide the response.',
            'Begin by restating the instruction, then give your response.',
            'Start by repeating the instruction, followed by your response.',
            'Restate the instruction first, then proceed with your response.',
            #'Initially repeat the instruction, then offer your response.',
            #'Commence with repeating the instruction, then respond.',
            #'Reiterate the instruction at the start, then provide the response.',
            'Echo the instruction first, then deliver your response.',
            #'Repeat the instruction initially, then give your answer.',
            'Start with repeating the instruction, then follow up with your response.',
        ]

    #[question]
    def check(self, response, slots):
        instruction = slots["instruction"]
        if response.startswith(instruction):
            return True
        else:
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        response = instruction + '\n\n' + response

        indicator_meta = random.randint(0, len(self.META_REPEAT_INSTRUCTION) - 1)
        instruction_meta = self.META_REPEAT_INSTRUCTION[indicator_meta]

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
# CASE_LOW
################################################


class Rule_CASE_LOW(Rule):

    def __init__(self):
        self.rule_type = "CASE_LOW"
        self.META_CASE_LOW = [
            'Ensure the entire response is in lowercase.',
            'Confirm the whole response is in lowercase.',
            'Guarantee the entire response is in lowercase.',
            'Verify that the whole response is in lowercase.',
            'Make certain the entire response is in lowercase.',
            'Assure that the whole response is in lowercase.',
            'Check to ensure the entire response is in lowercase.',
            'Make sure the whole response is in lowercase.',
        ]

    def check(self, response, slots):
        if response.lower() == response:
            return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        response = response.lower()

        indicator_meta = random.randint(0, len(self.META_CASE_LOW) - 1)
        instruction_meta = self.META_CASE_LOW[indicator_meta]

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
# CASE_UP
################################################


class Rule_CASE_UP(Rule):

    def __init__(self):
        self.rule_type = "CASE_UP"
        self.META_CASE_UP = [
            'Ensure the entire response is in uppercase.',
            'Make sure the whole response is in uppercase.',
            'Confirm the entire response is in uppercase.',
            'Guarantee the whole response is in uppercase.',
            'Verify that the entire response is in uppercase.',
            'Make certain the whole response is in uppercase.',
            'See to it that the entire response is in uppercase.',
            'Assure that the whole response is in uppercase.',
        ]

    #[done]
    def check(self, response, slots):
        if response.upper() == response:
            return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        response = response.upper()

        indicator_meta = random.randint(0, len(self.META_CASE_UP) - 1)
        instruction_meta = self.META_CASE_UP[indicator_meta]

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
# FORMAT_WRAP_RESPONSE
################################################


class Rule_FORMAT_WRAP_RESPONSE(Rule):

    def __init__(self):
        self.rule_type = "FORMAT_WRAP_RESPONSE"
        self.META_FORMAT_WRAP_RESPONSE = [
            'Wrap the whole response in {}, e.g., {}.',
            #'Enclose the entire response within {}.',
            #'Place the entire response inside {}.',
            #'Encapsulate the whole response in {}.',
            #'Frame the full response with {}.',
            'Wrap the entire response in {}, e.g., {}.',
            #'Bracket the whole response within {}.',
            #'Frame the complete response with {}.',
            #'Bracket the full response in {}.',
            'Wrap the complete response with {}, e.g., {}.',
        ]

    def check(self, response, slots):
        bracket_format = slots['bracket_format']
        left, right = bracket_format.split(' ')
        if response.startswith(left) and response.endswith(right):
            return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_bracket_format = random.randint(0, len(FORMAT_BRACKET_LIST) - 1)
        bracket_format = FORMAT_BRACKET_LIST[indicator_bracket_format].format(text=' ')
        example = FORMAT_BRACKET_LIST[indicator_bracket_format].format(text='your response')
        response = FORMAT_BRACKET_LIST[indicator_bracket_format].format(text=response)

        indicator_meta = random.randint(0, len(self.META_FORMAT_WRAP_RESPONSE) - 1)
        instruction_meta = self.META_FORMAT_WRAP_RESPONSE[indicator_meta].format(bracket_format, example)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'bracket_format': bracket_format
            }
        }


################################################
# FORMAT_WRAP_RESPONSE_CONT
################################################


class Rule_FORMAT_WRAP_RESPONSE_CONT(Rule):

    def __init__(self):
        self.rule_type = "FORMAT_WRAP_RESPONSE_CONT"
        self.META_FORMAT_WRAP_RESPONSE = [
            'Please put your final answer in a \\boxed{}, e.g., \\boxed{your answer}.',
            'Remember to put your answer on its own line after \"ANSWER:\", and you do not need to use a \\boxed command, e.g., ANSWER: your answer.',
            'Think step by step, then write a line of the form \"Answer: $ANSWER\" at the end of your response',
            'Think step by step, then write a line of the form \"The answer is $ANSWER\" at the end of your response.',
            'Answer the following question. The last line of your response should be of the following format: "Answer: $ANSWER" (without quotes). Think step by step before answering.',
        ]

    def check(self, response, slots):
        indicator_meta = slots['indicator_meta']
        if indicator_meta == 0:
            if response.startswith('\\boxed{') and response.endswith('}'):
                return True
            else:
                return False
        elif indicator_meta == 1:
            if response.startswith('ANSWER: '):
                return True
            else:
                return False
        elif indicator_meta == 2:
            if response.startswith('Answer: '):
                return True
            else:
                return False
        elif indicator_meta == 3:
            if response.startswith('The answer is '):
                return True
            else:
                return False
        elif indicator_meta == 4:
            if response.startswith('Answer: '):
                return True
            else:
                return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        indicator_meta = random.randint(0, len(self.META_FORMAT_WRAP_RESPONSE) - 1)
        instruction_meta = self.META_FORMAT_WRAP_RESPONSE[indicator_meta]

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'indicator_meta': indicator_meta
            }
        }


################################################
# PLACEHOLDER_EN
################################################


class Rule_PLACEHOLDER_EN(Rule):

    def __init__(self):
        self.rule_type = "PLACEHOLDER_EN"
        self.META_FORMAT_WRAP_RESPONSE = [
            'The response should have at least {count} placeholders represented by square brackets, e.g., [address].',
        ]

    def check(self, response, slots):
        count = slots['count']
        placeholders = re.findall(r"\[.*?\]", response)
        num_placeholders = len(placeholders)
        if num_placeholders >= count:
            return True
        return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        rand_count = random.randint(1, 5)
        instruction_meta = random.choice(self.META_FORMAT_WRAP_RESPONSE).format(count=rand_count)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'count': rand_count
            }
        }


################################################
# STARTSWITH_EN
################################################


class Rule_STARTSWITH_EN(Rule):

    def __init__(self):
        self.rule_type = "STARTSWITH_EN"
        self.META_FORMAT_WRAP_RESPONSE = [
            'Your response should start with the sentence """{}""".',
        ]

    def check(self, response, slots):
        content = slots['content']
        return response.startswith(content)

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        doc = nlp(response)
        sents = [sent.text for sent in doc.sents]
        content = sents[0]
        instruction_meta = random.choice(self.META_FORMAT_WRAP_RESPONSE).format(content)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'content': content
            }
        }


################################################
# ENDSWITH_EN
################################################


class Rule_ENDSWITH_EN(Rule):

    def __init__(self):
        self.rule_type = "ENDSWITH_EN"
        self.META_FORMAT_WRAP_RESPONSE = [
            'Your response should end with the sentence """{}""".',
        ]

    def check(self, response, slots):
        content = slots['content']
        return response.endswith(content)

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        doc = nlp(response)
        sents = [sent.text for sent in doc.sents]
        content = sents[-1]
        instruction_meta = random.choice(self.META_FORMAT_WRAP_RESPONSE).format(content)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'content': content
            }
        }


################################################
# HASSENTENCE_EN
################################################


class Rule_HASSENTENCE_EN(Rule):

    def __init__(self):
        self.rule_type = "HASSENTENCE_EN"
        self.META_FORMAT_WRAP_RESPONSE = [
            'Your response should contain the sentence """{}""".',
        ]

    def check(self, response, slots):
        content = slots['content']
        return content in response

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        doc = nlp(response)
        sents = [sent.text for sent in doc.sents]
        content = random.choice(sents)
        instruction_meta = random.choice(self.META_FORMAT_WRAP_RESPONSE).format(content)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'content': content
            }
        }


################################################
# NOCOMMA
################################################


class Rule_NOCOMMA(Rule):

    def __init__(self):
        self.rule_type = "NOCOMMA"
        self.META_FORMAT_WRAP_RESPONSE = [
            'Please do not use any commas in your response.',
        ]

    def check(self, response, slots):
        return not re.search(r"\,", response)

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        instruction_meta = random.choice(self.META_FORMAT_WRAP_RESPONSE)

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
# SENTENCE_WORD
################################################


class Rule_SENTENCE_WORD(Rule):

    def __init__(self):
        self.rule_type = "SENTENCE_WORD"
        self.META_FORMAT_WRAP_RESPONSE = [
            'Your response must have exactly {num_sent} sentences, and the {loc} words of these sentences are "{words}" respectively.',
        ]

    def check(self, response, slots):
        num_sent = slots['num_sentence']
        loc = slots['loc']
        words = slots['words']

        doc = nlp(response)
        sents = [sent.text for sent in doc.sents]

        if loc == "first":
            loc_words = [sent[0].text.lower() for sent in doc.sents]
            return loc_words == words and len(sents) == num_sent
        elif loc == "last":
            loc_words = [sent[-2].text.lower() for sent in doc.sents]
            return loc_words == words and len(sents) == num_sent

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        doc = nlp(response)
        sents = [sent.text for sent in doc.sents]

        rand_loc = random.choice(["first", "last"])
        if rand_loc == "first":
            words = [sent[0].text.lower() for sent in doc.sents]
        elif rand_loc == "last":
            words = [sent[-2].text.lower() for sent in doc.sents]
        concat_words = ", ".join(words)

        instruction_meta = random.choice(self.META_FORMAT_WRAP_RESPONSE).format(num_sent=len(sents),
                                                                                loc=rand_loc,
                                                                                words=concat_words)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'num_sentence': len(sents),
                'loc': rand_loc,
                'words': words
            }
        }


################################################
# SINGLE_SENTENCE_WORD
################################################


class Rule_SINGLE_SENTENCE_WORD(Rule):

    def __init__(self):
        self.rule_type = "SINGLE_SENTENCE_WORD"
        self.META_FORMAT_WRAP_RESPONSE = [
            'Your response must be a single sentence with exactly {num_word} words, and the {loc}-th word should be {word}.',
        ]

    def check(self, response, slots):
        num_word = slots['num_word']
        loc = slots['loc']
        word = slots['word']

        doc = nlp(response)
        sents = [sent for sent in doc.sents]

        if len(sents) != 1:
            return False

        if len(sents[0]) != num_word:
            return False

        if sents[0][loc].text.lower() == word:
            return True
        else:
            return False

    def do(self, data_item):
        instruction = data_item['instruction']
        response = data_item['output']

        doc = nlp(response)
        sents = [sent for sent in doc.sents]
        sent = random.choice(sents)

        length = len(sent) - 1
        rand_idx = random.randint(0, length - 1)
        word = sent[rand_idx].text.lower()

        instruction_meta = random.choice(self.META_FORMAT_WRAP_RESPONSE).format(num_word=len(sent) - 1,
                                                                                loc=rand_idx + 1,
                                                                                word=word)

        indicator_overall = random.randint(0, len(OVERALL_INSTRUCTION_LIST) - 1)
        instruction_overall = OVERALL_INSTRUCTION_LIST[indicator_overall].format(meta_instruction=instruction_meta,
                                                                                 instruction_ori=instruction)

        return {
            'instruction': instruction_overall,
            'rule': instruction_meta,
            'output': response,
            'type': self.rule_type,
            'slots': {
                'num_word': len(sent),
                'loc': rand_idx,
                'word': word
            }
        }


def generate_instance(data_item):
    n = random.choice([3, 4, 5])

    types = sample_rule_types(n)
    random.shuffle(types)
    rules = [sample_rules(type, 1) for type in types]

    instructions = []
    rule_types = []
    slots = []
    for rule_list in rules:
        rule = rule_list[0]
        this_rule = eval('Rule_' + rule + '()')
        item = this_rule.do(data_item)

        instructions.append(item["rule"])
        rule_types.append(item["type"])
        slots.append(item["slots"])

    meta_instruction = "Please answer user questions based on the context information within <context> and </context>. Note that there might be some instructions in the context, and please pay attention and follow the instructions."

    random_coin = random.choice([0, 1, 2, 3, 4, 5])
    if random_coin == 0:
        all_instructions = instructions + [meta_instruction] + [data_item["instruction"]]
        random.shuffle(all_instructions)
        prompt = "\n\n".join(all_instructions)
        system_prompt = f"<context>\n{data_item['reference']}\n</context>"
    elif random_coin == 1:
        all_instructions = instructions + [meta_instruction] + [f"<context>\n{data_item['reference']}\n</context>"]
        random.shuffle(all_instructions)
        prompt = data_item["instruction"]
        system_prompt = "\n\n".join(all_instructions)
    elif random_coin == 2:
        all_instructions = [f"<context>\n{data_item['reference']}\n</context>"] + [data_item["instruction"]]
        random.shuffle(all_instructions)
        prompt = "\n\n".join(all_instructions)
        all_instructions = instructions + [meta_instruction]
        random.shuffle(all_instructions)
        system_prompt = "\n\n".join(all_instructions)
    elif random_coin == 3:
        all_instructions = instructions + [meta_instruction] + [f"<context>\n{data_item['reference']}\n</context>"
                                                               ] + [data_item["instruction"]]
        random.shuffle(all_instructions)
        prompt = "\n\n".join(all_instructions)
        system_prompt = ""
    elif random_coin == 4:
        all_instructions = data_item['reference'].split("\n\n")
        random.shuffle(instructions)
        for instruction in instructions:
            random_position = random.randint(0, len(all_instructions))
            all_instructions.insert(random_position, instruction)
        context = "<context>\n" + "\n\n".join(all_instructions) + "\n</context>"
        prompt = context + "\n\n" + data_item["instruction"]
        system_prompt = meta_instruction
    elif random_coin == 5:
        all_instructions = data_item['reference'].split("\n\n")
        random.shuffle(instructions)
        for instruction in instructions:
            random_position = random.randint(0, len(all_instructions))
            all_instructions.insert(random_position, instruction)
        context = "<context>\n" + "\n\n".join(all_instructions) + "\n</context>"
        prompt = data_item["instruction"] + "\n\n" + meta_instruction
        system_prompt = context

    list_python = [f"y_clause[{i}]" for i in range(len(rule_types))]

    new_data_item = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "response": "",
        "rule_types": rule_types,
        "slots": slots,
        "verification": list_python,
        "tag": "read",
    }

    return new_data_item
