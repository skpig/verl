from alpha_seed.prompts.base import PromptBase


class ActionSpaceWMX01(PromptBase):

    def __init__(
        self,
    ):
        super().__init__()
        self.prompt1 = """
Please solve the following complex math problem with a focus on accuracy and clarity. Follow these guidelines to ensure the answer is as precise and well-organized as possible:

	1.	Break down the problem: Start by clearly defining any variables, equations, or conditions given in the problem. List these as separate points if possible.
	2.	Plan your approach: Briefly outline the steps you will take to solve the problem before diving into calculations. This plan should include any formulas, theorems, or methods you intend to use.
	3.	Detailed calculations: Work through each step in detail, explaining your reasoning. Show intermediate steps and calculations, and keep track of units if applicable. Avoid skipping steps, even if they seem straightforward.
	4.	Verify your work: After reaching an answer, review each step to ensure logical consistency and correctness. If there are alternative methods, consider them briefly and confirm that they lead to the same conclusion.
	5.	Summarize the answer: Provide a clear, final answer in the simplest form, and box or highlight it if possible. Include a brief summary explaining why this answer satisfies the conditions of the problem.
	6.	Check assumptions and limitations: State any assumptions made, and note if the answer would change under different conditions or interpretations of the problem.

"""

    def transform(self, question):
        return [{
            "content": self.system_prompt,
            "role": "system"
        }, {
            "content": self.prompt1.format(question),
            'role': 'user'
        }]
