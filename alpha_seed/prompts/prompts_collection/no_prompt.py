from alpha_seed.prompts.base import PromptBase


class NoPrompt(PromptBase):
    # for sanity check
    def __init__(
        self,
    ):
        super().__init__()
        self.prompt1 = """
{}
"""

    def transform(self, question):
        return [{"content": self.prompt1.format(question), 'role': 'user'}]
