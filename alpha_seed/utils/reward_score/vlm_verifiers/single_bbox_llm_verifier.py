# -*- coding: utf-8 -*-
"""
Physics answer evaluation utility using LLM-based verification.

Design goals:
  1. ONLY allow exactly one \boxed{...} segment in the model output.
  2. The \boxed{...} must be at the end of the response.
  3. Use LLM to judge if the content inside the box matches the reference answer.

Scoring rules:
  - raise ExtractAnswerFailed: Multiple boxes or box not at the end
  - score = 0: Single box at end but content doesn't match reference
  - score = 1: Single box at end and content matches reference
"""
from __future__ import annotations
import re
import os
import time
import random
import logging

# Add BaseVerifier imports
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, VerifierFailed, ExtractAnswerFailed

# Setup logging
logger = logging.getLogger(__name__)

_BOX_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|{[^{}]*})*)\}")


def extract_box_content(text: str) -> str:
    # Find all \boxed{} patterns
    matches = _BOX_PATTERN.findall(text)

    if len(matches) == 0:
        raise ExtractAnswerFailed("No \\boxed{} found in the text")

    if len(matches) > 1:
        raise ExtractAnswerFailed(f"Multiple \\boxed{{}} found ({len(matches)} instances). Only one is allowed.")

    # Check if the box is at the end (allow up to 10 characters after box)
    # Since we know there's exactly one match, find its position
    match = _BOX_PATTERN.search(text)

    # Check if there's more than 10 characters after the box
    content_after_box = text[match.end():]
    if len(content_after_box) > 10:
        raise ExtractAnswerFailed(
            f"Too much content found after \\boxed{{}}: '{content_after_box[:20]}...'. Box must be at the end with at most 10 characters following."
        )

    box_content = matches[0].strip()
    return box_content


class AnswerVerifier:
    """Uses LLM API to verify if student answer matches reference answer."""

    def __init__(self, volc_ark_key: str, volc_model_name: str):
        if not volc_ark_key:
            raise ValueError('volc_ark_key is required')
        if not volc_model_name:
            raise ValueError('volc_model_name is required')

        base_url = os.environ.get('VOLC_ARK_BASE_URL', "https://ark-cn-beijing.bytedance.net/api/v3")

        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=volc_ark_key, timeout=1800)
        except ImportError:
            raise ImportError("OpenAI package is required for LLM verification")

        self.model = volc_model_name

        # System prompt for answer verification
        self.system_prompt = """你是一个教育评测专家。你需要判断学生的解答是否和标准答案一致。

评判标准：
1. 如果学生的解答包含了标准答案的每一条内容，并且和答案一致，则判定为正确
2. 如果是选择题选项、专有名词或者数字，必须要和答案完全一致
3. 如果是简答题或者论述题，答案可以有一定的表述差异，但必须包含所有关键点且逻辑正确
4. 如果是数学表达式，需要和答案的数学表达式在数学上相等；如果你无法确认数学表达式是否相等，或者在某些情况下不相等，那么认定学生答案是错误的
5. 如果是近似的数值，需要在有效数字位数内和标准答案一致，有效位数太少也算错
6. 如果漏写了单位，也认定为错误
7. 答案内容必须完整，不能遗漏重要部分

请仔细比较学生答案和标准答案，给出你的判断。

输出格式：只输出<correct>或<incorrect>，不要输出其他内容。"""

    def verify_answer(self, student_answer: str, reference_answer: str, problem_text: str) -> bool:
        """
        Use LLM to verify if student answer matches reference answer.
        
        Args:
            student_answer: The student's answer to verify
            reference_answer: The correct reference answer
            problem_text: The original problem/question text for context
            
        Returns:
            True if answers match, False otherwise
        """
        for attempt in range(10):  # Retry up to 10 times
            try:
                # Include problem text in prompt for better context
                prompt = f"""请判断以下学生答案是否和标准答案一致：

题目：
{problem_text}

标准答案：
{reference_answer}

学生答案：
{student_answer}

请根据评判标准仔细比较两个答案。"""

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "system",
                        "content": self.system_prompt
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    timeout=120,
                    temperature=0.1  # Low temperature for consistent results
                )

                response = completion.choices[0].message.content.strip()

                # Extract judgment
                if '<correct>' in response.lower() and '<incorrect>' not in response.lower():
                    return True
                elif '<incorrect>' in response.lower() and '<correct>' not in response.lower():
                    return False
                else:
                    raise ValueError(f"Unexpected LLM response: {response}")

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < 9:  # If not the last attempt
                    time.sleep(random.uniform(60, 180))  # Random wait 60-180 seconds
                    continue
                else:
                    logger.error(f"All API attempts failed")
                    # If all attempts fail, raise VerifierFailed
                    raise VerifierFailed(f"LLM verification failed after 10 attempts: {e}")

        # This should not be reached due to exception above
        raise VerifierFailed("Unexpected end of verification attempts")


class SingleBBoxVerifier(BaseVerifier):

    def __init__(self, volc_ark_key: str | None = None, volc_model_name: str | None = None) -> None:
        if not volc_ark_key or not volc_model_name:
            raise ValueError("Both volc_ark_key and volc_model_name are required for LLM verification")

        self.volc_ark_key = volc_ark_key
        self.volc_model_name = volc_model_name

        # Initialize the answer verifier
        self.answer_verifier = AnswerVerifier(volc_ark_key, volc_model_name)

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        # Extract reference answer
        reference_answer = verifier_feature_dict.get('answer', '')
        if not reference_answer:
            raise VerifierFailed("No reference answer provided in verifier_feature_dict")

        # Extract problem text for context
        problem_text = verifier_feature_dict.get('problem', '')
        box_content = extract_box_content(response)
        content_matches = self.answer_verifier.verify_answer(student_answer=box_content,
                                                             reference_answer=reference_answer,
                                                             problem_text=problem_text)
        score = float(content_matches)

        return VerifyResult(score=score, extracted_answer=box_content)
