from alpha_seed.prompts.base import PromptBase


class eotbase(PromptBase):

    def __init__(
        self,
    ):
        super().__init__()
        self.system_prompt = """You are a helpful assistant who provide a monologue style thought before producing your response. So please first provide your thought process in <Begin_of_Thinking><End_of_Thinking>, and then give the final answer in <Begin_of_Response><End_of_Response>.
"""
        self.prompt1 = "{}"

    def transform(self, question):
        return [{
            "content": self.system_prompt,
            "role": "system"
        }, {
            "content": self.prompt1.format(question),
            'role': 'user'
        }]
