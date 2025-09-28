import re
import json
import string
import random


class ConstrainManager:

    def __init__(self, log=False):
        self.constrain_map = {
            "SentenceNumberConstrain": SentenceNumberConstrain,
            "ParagraphNumberConstrain": ParagraphNumberConstrain,
            "WordNumberConstrain": WordNumberConstrain,
            "CharacterNumberConstrain": CharacterNumberConstrain,
            "ExactWordConstrain": ExactWordConstrain,
            "ExcatSentenceConstrain": ExcatSentenceConstrain,
            "ExactCharacterConstrain": ExactCharacterConstrain,
            "SentenceLastWordConstrain": SentenceLastWordConstrain,
            "PassageLastSentenceConstrain": PassageLastSentenceConstrain
        }
        self.log = log

    def load_constrain(self, constrain_list_json):
        convert_cons_list = []
        for c in json.loads(constrain_list_json):
            c_name = c[0]
            c_info = c[1]
            convert_cons_list.append(self.constrain_map[c_name](c_info))
        return convert_cons_list

    def dump_constrain(self, constrain_list):
        out_list = []
        for c in constrain_list:
            out_list.append((c.__class__.__name__, c.get_info_dict()))
        return json.dumps(out_list, ensure_ascii=False)

    def gerneral_check(self, constrain_list, resp):
        for single_cons in constrain_list:
            if not single_cons.check_resp(resp):
                if self.log:
                    print(str(single_cons) + "\t not valid")

                return False
        return True


def split_sentences_basic(text):
    all_sen = re.split(r'[.!?]\s*', text)
    return [s.strip() for s in all_sen if s.strip()]


def split_words_basic(resp):
    words = re.split(r'(\s+|\n\n)', resp)
    words = [w.strip() for w in words if w.strip()]
    return words


def split_paragraphs_basic(text):
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


class SentenceNumberConstrain:

    def __init__(self, info):
        self.sentence_num = info["sentence_num"]
        self.operate_type = info["operate_type"]
        self.enum_op = [
            "all paragraphs should have at least", "all paragraphs should have at most",
            "all paragraphs should have exactly", "the generated content should contain at least",
            "the generated content should contain at most", "the generated content should contain exactly"
        ]
        assert self.operate_type in self.enum_op

    def get_info_dict(self):
        return {"sentence_num": self.sentence_num, "operate_type": self.operate_type}

    def get_cons_str(self):
        return self.operate_type + " " + str(self.sentence_num) + " sentences."

    def split_paragraphs(self, text):
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def check_resp(self, resp):
        if "paragraphs" in self.operate_type:
            paragraphs = self.split_paragraphs(resp)
            para_sentence = [len(split_sentences_basic(p)) for p in paragraphs]
        else:
            para_sentence = [len(split_sentences_basic(resp))]

        if "at least" in self.operate_type:
            return all([s >= self.sentence_num for s in para_sentence])
        elif "at most" in self.operate_type:
            return all([s <= self.sentence_num for s in para_sentence])
        elif "exactly" in self.operate_type:
            return all([s == self.sentence_num for s in para_sentence])
        else:
            raise NotImplementedError()


class ParagraphNumberConstrain:

    def __init__(self, info):
        self.paragraph_num = info["paragraph_num"]
        self.operate_type = info["operate_type"]
        self.enum_op = [
            "the generated content should contain at least", "the generated content should contain at most",
            "the generated content should contain exactly"
        ]
        assert self.operate_type in self.enum_op

    def get_info_dict(self):
        return {"paragraph_num": self.paragraph_num, "operate_type": self.operate_type}

    def split_paragraphs(self, text):
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def get_cons_str(self):
        return self.operate_type + " " + str(self.paragraph_num) + " paragraphs."

    def check_resp(self, resp):
        paragraphs = self.split_paragraphs(resp)
        if "at least" in self.operate_type:
            return len(paragraphs) >= self.paragraph_num
        elif "at most" in self.operate_type:
            return len(paragraphs) <= self.paragraph_num
        elif "exactly" in self.operate_type:
            return len(paragraphs) == self.paragraph_num
        else:
            raise NotImplementedError()


class WordNumberConstrain:

    def __init__(self, info):
        self.word_num = info["word_num"]
        self.operate_type = info["operate_type"]
        self.enum_op = [
            "the generated content should contain at least", "the generated content should contain at most",
            "the generated content should contain exactly", "all sentences should have at least",
            "all sentences should have at most", "all sentences should have exactly"
        ]
        assert self.operate_type in self.enum_op

    def get_info_dict(self):
        return {"word_num": self.word_num, "operate_type": self.operate_type}

    def get_cons_str(self):
        return self.operate_type + " " + str(self.word_num) + " words."

    def check_resp(self, resp):
        if "sentences" in self.operate_type:
            words = []
            sentences = split_sentences_basic(resp)
            for single_sentence in sentences:
                split_word = re.split(r'(\s+|\n\n)', single_sentence)
                split_word_fileter_empty = []
                for item in split_word:
                    if item.strip() != "":
                        split_word_fileter_empty.append(item)
                words.append(len(split_word_fileter_empty))
        else:
            split_word = re.split(r'(\s+|\n\n)', resp)
            split_word_fileter_empty = []
            for item in split_word:
                if item.strip() != "":
                    split_word_fileter_empty.append(item)
            words = [len(split_word_fileter_empty)]

        if "at least" in self.operate_type:
            return all([w >= self.word_num for w in words])
        elif "at most" in self.operate_type:
            return all([w <= self.word_num for w in words])
        elif "exactly" in self.operate_type:
            return all([w == self.word_num for w in words])
        else:
            raise NotImplementedError()


class CharacterNumberConstrain:

    def __init__(self, info):
        self.character_num = info["character_num"]
        self.operate_type = info["operate_type"]
        self.use_white_space = info["use_white_space"]
        self.enum_op = [
            "the generated content should contain at least", "the generated content should contain at most",
            "the generated content should contain exactly", "all words should have at least",
            "all words should have at most", "all words should have exactly"
        ]
        if "words" in self.operate_type:
            assert self.use_white_space == False
        assert self.operate_type in self.enum_op
        self.char_list = set([
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
            "v", "w", "x", "y", "z"
        ])

    def get_info_dict(self):
        return {
            "character_num": self.character_num,
            "operate_type": self.operate_type,
            "use_white_space": self.use_white_space
        }

    def get_cons_str(self):
        if not self.use_white_space:
            return self.operate_type + " " + str(self.character_num) + " characters."
        else:
            return self.operate_type + " " + str(
                self.character_num) + " characters.(Include whitespace into your character count.)"

    def check_resp(self, resp):
        sentences = split_sentences_basic(resp)
        if "words" in self.operate_type:
            words = []
            for single_sen in sentences:
                split_words = re.split(r'(\s+|\n\n)', single_sen)
                for single_word in split_words:
                    if single_word.strip() != "":
                        words.append(single_word)
        else:
            if not self.use_white_space:
                words = [resp]
            else:
                words = [resp.replace(" ", "a")]

        character_num = [len(w) for w in words]

        if "at least" in self.operate_type:
            return all([c >= self.character_num for c in character_num])
        elif "at most" in self.operate_type:
            return all([c <= self.character_num for c in character_num])
        elif "exactly" in self.operate_type:
            return all([c == self.character_num for c in character_num])
        else:
            raise NotImplementedError()


class ExactWordConstrain:

    def __init__(self, info):
        self.exact_word = info["exact_word"]
        self.positions = info["positions"]
        assert len(info["exact_word"]) == len(info["positions"])

    def get_info_dict(self):
        return {"exact_word": self.exact_word, "positions": self.positions}

    def get_cons_str(self):
        res_str = "the "
        for i in range(len(self.positions)):
            res_str += str(self.positions[i]) + "th, "

        res_str = res_str[:-2]
        res_str += " word should be "
        for i in range(len(self.exact_word)):
            res_str += self.exact_word[i] + ", "

        res_str = res_str[:-2]
        if len(self.exact_word) > 1:
            res_str += " respectively."
        else:
            res_str += "."
        return res_str

    def check_resp(self, resp):
        words = []
        split_words = re.split(r'(\s+|\n\n)', resp)
        for single_word in split_words:
            if single_word.strip() != "":
                words.append(single_word)

        for i in range(len(self.exact_word)):
            if self.positions[i] > len(words):
                return False
            if words[self.positions[i] - 1].replace(",",
                                                    "").replace(".",
                                                                "") != self.exact_word[i].replace(",",
                                                                                                  "").replace(".", ""):
                return False
        return True


class ExcatSentenceConstrain:

    def __init__(self, info):
        self.exact_sentence = info["exact_sentence"]
        self.positions = info["positions"]
        assert len(info["exact_sentence"]) == len(info["positions"])

    def get_info_dict(self):
        return {"exact_sentence": self.exact_sentence, "positions": self.positions}

    def get_cons_str(self):
        res_str = "the "
        for i in range(len(self.positions)):
            res_str += str(self.positions[i]) + "th,"

        res_str = res_str[:-1]
        res_str += " sentence should be \'"
        for i in range(len(self.exact_sentence)):
            res_str += self.exact_sentence[i] + "\',"

        res_str = res_str[:-1]
        res_str += "."
        return res_str

    def check_resp(self, resp):
        sentences = split_sentences_basic(resp)
        for i in range(len(self.exact_sentence)):
            if self.positions[i] > len(sentences):
                return False
            if sentences[self.positions[i] - 1].replace(" ", "") != self.exact_sentence[i].replace(" ", ""):
                return False
        return True


class ExactCharacterConstrain:

    def __init__(self, info):
        self.exact_character = info["exact_character"]
        self.positions = info["positions"]
        assert len(info["exact_character"]) == len(info["positions"])

    def get_info_dict(self):
        return {"exact_character": self.exact_character, "positions": self.positions}

    def get_cons_str(self):
        res_str = "the "
        for i in range(len(self.positions)):
            res_str += str(self.positions[i]) + "th, "
        res_str = res_str[:-2]
        res_str += " character should be "
        for i in range(len(self.exact_character)):
            res_str += self.exact_character[i] + ","
        res_str = res_str[:-1]

        if len(self.exact_character) > 1:
            res_str += " respectively."
        else:
            res_str += "."
        return res_str

    def check_resp(self, resp):
        for i in range(len(self.exact_character)):
            if self.positions[i] > len(resp):
                return False
            if resp[self.positions[i] - 1] != self.exact_character[i]:
                return False
        return True


class SentenceLastWordConstrain:

    def __init__(self, info):
        self.last_word_list = info["last_word_list"]

    def get_info_dict(self):
        return {"last_word_list": self.last_word_list}

    def get_cons_str(self):
        res_str = "the last word of the generated sentences should be "
        for i in range(len(self.last_word_list)):
            res_str += self.last_word_list[i] + ", "
        res_str = res_str[:-2]
        res_str += "."
        return res_str

    def check_resp(self, resp):
        sentences = split_sentences_basic(resp)
        if len(sentences) == 0:
            return False

        last_words = []
        for single_sentence in sentences:
            split_words = split_words_basic(single_sentence)
            last_words.append(split_words[-1])

        if len(last_words) != len(self.last_word_list):
            return False
        for i in range(len(last_words)):
            if last_words[i] != self.last_word_list[i]:
                return False
        return True


class PassageLastSentenceConstrain:

    def __init__(self, info):
        self.last_sentence_list = info["last_sentence_list"]

    def get_info_dict(self):
        return {"last_sentence_list": self.last_sentence_list}

    def get_cons_str(self):
        res_str = "the last sentence of these generated paragraphs should be "
        for i in range(len(self.last_sentence_list)):
            res_str += "\'" + self.last_sentence_list[i] + "\', "
        res_str = res_str[:-2]
        res_str += "."
        return res_str

    def check_resp(self, resp):
        paragraphs = split_paragraphs_basic(resp)
        if len(paragraphs) == 0:
            return False
        last_sentences = []
        for single_paragraph in paragraphs:
            split_sentences = split_sentences_basic(single_paragraph)
            last_sentences.append(split_sentences[-1])

        if len(last_sentences) != len(self.last_sentence_list):
            return False
        for i in range(len(last_sentences)):
            if last_sentences[i].replace(" ", "") != self.last_sentence_list[i].replace(" ", ""):
                return False
        return True


CONSTRAIN_MANAGER = ConstrainManager()
