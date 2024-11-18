from alpha_seed.prompts.base import PromptBase


class ActionSpaceSimple(PromptBase):

    def __init__(
        self,
    ):
        super().__init__()
        self.system_prompt = """You are a Socratic scholar with deep thoughtfulness."""
        self.prompt1 = """
Think in a json format.

Note that you are extremely easy to jump some logic. For example (a-b)(c+d), you may directly give the answer, but I REQUIRE YOU to do a(c+d) - b(c+d), then ac + cd -bc-bd. This is for your own safety. 

```json
List of
    - Step: start from 0, increment 1 each step
    - Action: the Action Type
    - MathThought: Do the math concretely with clear logic and dense math.
```

Actions Type:

ReviewProblem: review the problem if you find hard to proceed.
ContinueLastStep: Use this action to continue the previous unfinish action. 
TryAnotherApproach: It might be tempting to try different approach, either to get answer, or to get insight. 
CheckRandomly: Stop frequently, to check 1. calculation, 2. if any logic are not consistant 
CheckJumpStep: Check any steps are jumped over. Jumping step is demaging. Redo from the jumped step because the subsequent thoughts are unreliable.
RethinkTheLogic: rethink the logic of the thought.
HaveIntuition: Pumped up with some intuition
FeelBadSituation: Feels the current process is not going correctly
NoticingInconsistency: Notice some inconsistency between any point in the thought.
ReflectionOnPossibleErrors: Reflection on any possible error. Always tries different potential error location.
ProposingIdeas: I got an idea, or several ideas.
StashAndBackTrack: Make a stash, recall previous thought/result/idea and potential restart there. 
Deduction: The oridinary next step.
SuggestAnswer: Do not left a half-way formula.
AnswerSummary: used in the last step, after SuggestAnswer. No sentence allowed. Be a number, a similied formula,



Question: {}

USE MORE STEPS!!!
USE MORE STEPS!!!
USE MORE STEPS!!!
USE MORE STEPS!!!
USE MORE STEPS!!!
USE MORE STEPS!!!

"""

    def transform(self, question):
        return [{
            "content": self.system_prompt,
            "role": "system"
        }, {
            "content": self.prompt1.format(question),
            'role': 'user'
        }]
