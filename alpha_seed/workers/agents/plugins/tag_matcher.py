from typing import *
from transformers import AutoTokenizer


class TagMatcher:
    """Match pattern <start_tag>...<end_tag>
    start_tag can be '', in that case, match all content before an end_tag
    """

    def __init__(self, start_tag: Union[str, None], end_tag: str):
        if start_tag is None:
            start_tag = ''
        assert isinstance(start_tag, str)
        assert isinstance(end_tag, str)
        self.start_tag = start_tag
        self.end_tag = end_tag
        assert len(self.end_tag) > 0
        assert self.start_tag != self.end_tag
        self.start_tag_chars = list(self.start_tag)
        self.end_tag_chars = list(self.end_tag)

        self.state = 0  # 0 for init, 1 for matching
        self.tag_char_cache = []
        self.matched_tokens = []
        self._reset()

        self.replacement_char = chr(0xFFFD)  # � char, if decode('utf-8') with invalid utf-8 bytes
        self.decode_token_cache = []

    @property
    def is_init_state(self):
        return self.state == 0

    @property
    def is_matching_state(self):
        return self.state == 1

    def _reset(self):
        self.state = 0
        self.tag_char_cache.clear()
        self.matched_tokens.clear()
        if len(self.start_tag) == 0:
            self._to_matching_state()

    def _to_matching_state(self):
        assert self.state == 0
        self.state = 1
        self.tag_char_cache.clear()

    def _decode_token(self, token: str, tokenizer: AutoTokenizer) -> Union[str, None]:
        """Try decode a token to string, if failed, the token is incomplete,
        push into cached_tokens, and try to decode with other tokens later
        """
        self.decode_token_cache.append(token)
        decoded = tokenizer.convert_tokens_to_string(self.decode_token_cache)
        if decoded == self.replacement_char:
            return None
        else:
            self.decode_token_cache.clear()
            return decoded

    def add_token_match(self, token: str, tokenizer: AutoTokenizer) -> Union[str, None]:
        decoded = self._decode_token(token, tokenizer)
        if decoded is None:
            return None

        matched = None
        matched_this_token = []

        max_cache_len = max(len(self.start_tag_chars), len(self.end_tag_chars))
        if len(self.tag_char_cache) > max_cache_len:
            self.tag_char_cache = self.tag_char_cache[-max_cache_len:]
        # scan this token
        for c in list(decoded):
            self.tag_char_cache.append(c)
            if self.is_init_state:
                assert len(self.start_tag_chars) > 0, "empty start_tag should never reach init state"
                if c == self.start_tag_chars[-1] and self.tag_char_cache[-len(self.start_tag_chars
                                                                             ):] == self.start_tag_chars:
                    self._to_matching_state()
            else:
                assert self.is_matching_state
                if len(self.start_tag_chars) > 0 and c == self.start_tag_chars[-1] and self.tag_char_cache[
                        -len(self.start_tag_chars):] == self.start_tag_chars:
                    # matched a new start_tag
                    self._reset()
                    self._to_matching_state()
                    matched_this_token.clear()
                    continue
                matched_this_token.append(c)
                if c == self.end_tag_chars[-1] and self.tag_char_cache[-len(self.end_tag_chars):] == self.end_tag_chars:
                    assert matched is None, f"unexpected: one token have multiple matches"
                    self.matched_tokens.append(''.join(matched_this_token))
                    matched = ''.join(self.matched_tokens)[:-len(self.end_tag)]
                    self._reset()
                    matched_this_token.clear()

        if len(matched_this_token) > 0:
            self.matched_tokens.append(''.join(matched_this_token))

        return matched

    def add_string_match(self, text: str, tokenizer: AutoTokenizer) -> List[str]:
        tokens = tokenizer.tokenize(text)
        all_matched = []
        for token in tokens:
            matched = self.add_token_match(token, tokenizer)
            if matched is not None:
                all_matched.append(matched)
        return all_matched
