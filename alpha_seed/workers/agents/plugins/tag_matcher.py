from typing import *
from queue import deque


class TagMatcher:
    """Match pattern <start_tag>...<end_tag>
    start_tag can be '', in that case, match all content before an end_tag
    """

    def __init__(self, start_tag: Union[str, None], end_tag: str):
        assert isinstance(start_tag, str)
        assert isinstance(end_tag, str)
        self.start_tag = start_tag
        self.end_tag = end_tag
        assert len(self.end_tag) > 0
        assert self.start_tag != self.end_tag
        self.start_tag_chars = list(self.start_tag)
        self.end_tag_chars = list(self.end_tag)
        self._reset()

    def _reset(self):
        self.state = 1 if len(self.start_tag_chars) == 0 else 0  # 0 for init, 1 for matching
        self.phase0_buffer = deque()
        self.phase1_buffer = []

    def add_char_match(self, char: str) -> Union[str, None]:
        assert len(char) == 1, "only accept a single char"
        if self.state == 0:
            self.phase0_buffer.append(char)
            # remove all old buffer
            while len(self.phase0_buffer) > len(self.start_tag_chars):
                _ = self.phase0_buffer.popleft()
            if self.phase0_buffer[-1] == self.start_tag_chars[-1]:
                if list(self.phase0_buffer) == self.start_tag_chars:
                    # start tag matched
                    self.state = 1
                    self.phase1_buffer = list(self.phase0_buffer)
        elif self.state == 1:
            self.phase1_buffer.append(char)
            if self.phase1_buffer[-len(self.start_tag_chars):] == self.start_tag_chars:
                # matched a new start tag, reset buffer
                self.phase1_buffer = self.phase1_buffer[-len(self.start_tag_chars):]
            elif (len(self.phase1_buffer) >= len(self.start_tag_chars) +
                  len(self.end_tag_chars)) and (self.phase1_buffer[-len(self.end_tag_chars):] == self.end_tag_chars):
                # matched end tag
                content = "".join(self.phase1_buffer[len(self.start_tag_chars):-len(self.end_tag_chars)])
                self._reset()
                return content
        return None


def test_normal_match():
    start_tag = "<Begin>"
    end_tag = "<End>"

    str_list = [
        "before begin tag",
        "<Begin>",
        "between",
        "two begin tag",
        "<Begin>",
        "content_to_match",
        "<End>",
        "invalid end tags",
        "<End>",
    ]

    string = "".join(str_list)
    matcher = TagMatcher(start_tag, end_tag)
    # expect to print "content to match"
    for c in list(string):
        ret = matcher.add_char_match(c)
        if ret is not None:
            print(ret)


def test_only_end_tag():
    matcher = TagMatcher(start_tag='', end_tag='<End>')
    str_list = [
        "first",
        " block"
        "<End>",
        "second block",
        "<End>",
        "<End>",
    ]
    string = "".join(str_list)
    for c in list(string):
        ret = matcher.add_char_match(c)
        if ret is not None:
            print(ret)


if __name__ == "__main__":
    test_normal_match()
    test_only_end_tag()
