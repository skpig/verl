import re
import json

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import (
    BaseVerifier,
    VerifyResult,
    ExtractAnswerFailed,
)
"""
单选题 json 格式 verifier
- 格式为：```json{"correct_answer": "X"}```，其中 X 为 A/B/C/D

评分规则：
- 答案正确：1.0 分
- 答案错误：0.0 分  
- 格式不符：抛出 ExtractAnswerFailed 异常
"""


class MCQAInstructVerifier(BaseVerifier):
    """JSON format verifier for multiple-choice questions."""

    def _extract_json_answer(self, text: str) -> str:
        """Extract answer letter from JSON format response."""
        # Extract content between ```json and ```
        match = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL | re.I)
        if not match:
            raise ExtractAnswerFailed(f"Failed to extract JSON content: {text}")

        json_content = match.group(1).strip()

        # Try standard JSON parsing first
        try:
            parsed = json.loads(json_content)
            letter = str(parsed['correct_answer']).strip().upper()
            if letter in ['A', 'B', 'C', 'D']:
                return letter
        except:
            raise ExtractAnswerFailed(f"Invalid answer format or letter: {text}")

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        """Verify response against ground truth answer."""
        gt = verifier_feature_dict["answer"].strip().upper()
        letter = self._extract_json_answer(response)
        score = 1.0 if letter == gt else 0.0
        return VerifyResult(score=score, extracted_answer=letter)


def main():
    """Test the verifier with various cases."""
    verifier = MCQAInstructVerifier()

    def run_test(name: str, response: str, answer: str, expect_score=None, expect_error=False):
        try:
            result = verifier.verify(response, {"answer": answer})
            print(f"{name:<30}: score={result.score}, extracted={result.extracted_answer}")
            if expect_score is not None:
                assert result.score == expect_score
        except ExtractAnswerFailed as e:
            if expect_error:
                print(f"{name:<30}: ✓ Expected error")
            else:
                raise

    print("=== Testing JSON Format Verifier ===\n")

    run_test("Valid JSON - Correct", '```json\n{"correct_answer": "C"}\n```', "C", 1.0)
    run_test("Valid JSON - Wrong", '```json\n{"correct_answer": "B"}\n```', "D", 0.0)
    run_test("Double Quotes", '```json\n{"correct_answer": "A"}\n```', "A", 1.0)
    run_test("No Fence", '{"correct_answer": "A"}', "A", expect_error=True)
    run_test("Invalid Letter", '```json\n{"correct_answer": "E"}\n```', "A", expect_error=True)
    run_test("Single Quotes", "```json\n{'correct_answer': 'A'}\n```", "A", expect_error=True)
    run_test("Empty Response", "", "B", expect_error=True)

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
