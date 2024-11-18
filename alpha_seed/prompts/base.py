class PromptBase:

    def __init__(
        self,
    ):
        self.system_prompt = ""
        self.prompt1 = ""

    def transform(self, question):
        message = self.system_prompt + self.prompt1.format(question)
        return message
