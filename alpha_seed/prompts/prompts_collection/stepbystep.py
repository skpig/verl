from alpha_seed.prompts.base import PromptBase


class O1Style(PromptBase):

    def __init__(
        self,
    ):
        super().__init__()
        self.prompt1 = """Please solve the problem step by step, with trail and error.

Problem: {}

End your response with ``Answer:`` 
"""
        self.prob = 1

    def transform(self, question):
        return [{"content": self.prompt1.format(question), 'role': 'user'}]
