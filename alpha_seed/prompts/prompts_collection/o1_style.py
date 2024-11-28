from alpha_seed.prompts.base import PromptBase


class O1Style(PromptBase):

    def __init__(
        self,
    ):
        super().__init__()
        self.prompt1 = """I'm teacher who is trying to make a lecture video to demonstrate to my students how I solve a difficult math problem via trial and errors. My solution is well structured, but I want to transform it into a transcript so that I can use it in my lecture video. Please help me transform my solution and my thinking process into internal monologue, so that it is raw, organic, and resembles a stream-of-consciousness, which flows seamlessly between ideas, concepts, and knowledge. Remember to keep the trial and errors so that my students will learn from such a thinking process, which will benefit them more than just the correct solution. 'm teacher who is trying to make a lecture video to demonstrate to my students how I solve a difficult math problem via trial and errors. My solution is well structured, but I want to transform it into a transcript so that I can use it in my lecture video. Please help me transform my solution and my thinking process into internal monologue, so that it is raw, organic, and resembles a stream-of-consciousness, which flows seamlessly between ideas, concepts, and knowledge. Remember to keep the trial and errors so that my students will learn from such a thinking process, which will benefit them more than just the correct solution. Please conclude with a final answer after all your monologue, don't leave an open ending!

Problem: {}
"""
        self.prob = 0.5

    def transform(self, question):
        return [{"content": self.prompt1.format(question), 'role': 'user'}]
