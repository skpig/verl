# -*- coding: utf-8 -*-
"""
这段代码完全基于 PHYSICS 这个外部的 benchmark 的逻辑来评测。待评测的解答是一个字符串，参考答案是一个数组。其逻辑是捕捉模型解答中的所有的 bbox，对每一个 bbox，和每一个答案比对，假如一致就记正确。最后输出正确数组的平均值。可以看到，这个 metric 是有一定缺陷的，假如模型反复输出一个小题的正确答案，可能会导致正确率过高。这个 metric 只适用于 PHYSICS 这个 benchmark，不建议用于训练。
https://github.com/yale-nlp/Physics/tree/main/PHYSICS
"""
from __future__ import annotations
import re
import os
import signal
import time
import random
import logging
from openai import OpenAI
from typing import Dict, Any
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, VerifierFailed, ExtractAnswerFailed
from alpha_seed.utils.reward_score.math_v2 import timeout

# 请在选装库里面添加 mpmath==1.3.0 sympy==1.14.0
import sympy  # type: ignore
from sympy import simplify, expand, trigsimp
from sympy.parsing.latex import parse_latex  # may fail if sympy lacks latex parser

# ===========================================================================================
# EXTRACT_BOXED MODULE (INTEGRATED)
# ===========================================================================================


def extract_all_boxed_content(latex_response):
    """
    Extract all \boxed{} content from a LaTeX response, supporting nested {}.

    Args:
        latex_response (str): The LaTeX response text.

    Returns:
        list: Extracted \boxed{} content.
    """
    # Define regex pattern to match nested \boxed{}
    pattern = re.compile(r'\\boxed{((?:[^{}]|{(?:[^{}]|{.*?})*})*)}|\\\\\[boxed{((?:[^{}]|{(?:[^{}]|{.*?})*})*)}\\\\\]',
                         re.DOTALL)
    matches = pattern.findall(latex_response)  # Match all occurrences

    if not matches:
        return []
    # Flatten matches and remove empty strings
    return [match.strip() for sublist in matches for match in sublist if match.strip()]


def extract_final_answer(last_answer):
    """
    Extract the final answer from \boxed{}.

    Args:
        last_answer (str): LaTeX text containing \boxed{}.

    Returns:
        str: Extracted answer.
    """
    match = re.search(r'\\boxed{(.*?)}|\\\\\[boxed{(.*?)}\\\\\]', last_answer)
    if match:
        return next(group for group in match.groups() if group).strip()
    return last_answer


def extract_final_answer_list(last_answer):
    """
    Extract a list of answers from \boxed{} (for multi-part answers).

    Args:
        last_answer (str): LaTeX text containing \boxed{}.

    Returns:
        list: Extracted list of answers.
    """
    matches = re.findall(r'\\boxed{\\\[(.*?)\\\]}|\\\\\[boxed{\\\[(.*?)\\\]}\\\\\]', last_answer)
    if matches:
        result = []
        for sublist in matches:
            for item in sublist:
                if item:
                    for piece in item.split(','):
                        result.append(piece.strip())
        return result
    return [extract_final_answer(last_answer)]


def extract_final_answer_allform(latex_response, answer_type=None):
    """
    General method to extract all final answers.

    Args:
        latex_response (str): LaTeX response text.
        answer_type (str): Type of answer (float, list, math_expression).

    Returns:
        list: Extracted answers.
    """
    boxed_content = extract_all_boxed_content(latex_response)
    if not boxed_content:
        return []

    if answer_type == 'list':
        return [extract_final_answer_list(item) for item in boxed_content]
    return [extract_final_answer(item) for item in boxed_content]


# ===========================================================================================
# EQUATION_EQUIVALENCY MODULE (INTEGRATED)
# ===========================================================================================


def _extract_core_eq(expr: str) -> str:
    """Extract the right-hand side of an equation or implication from a LaTeX expression."""
    if "\\implies" in expr:
        expr = expr.split("\\implies")[-1].strip()
    if "=" in expr:
        expr = expr.split("=")[-1].strip()
    return expr


def _preprocess_latex(string: str) -> str:
    """Preprocess LaTeX to normalize format and separate variables."""
    if not string:
        return ""

    string = re.sub(r"_\{.*?\}", "", string)
    string = re.sub(r"_\\?\w", "", string)
    string = string.replace("\\left", "").replace("\\right", "").replace("\\cdot", "*")
    return string


def _standardize_expr(expr):
    """Standardize a SymPy expression with timeout protection."""
    try:
        with timeout(seconds=10):
            result = simplify(expand(trigsimp(expr)))
            return result
    except TimeoutError:
        raise ValueError("SymPy computation timed out!")
    except Exception as e:
        raise ValueError(f"SymPy error: {e}")


# ===========================================================================================
# LLM-BASED PHYSICS VERIFIER (FOR ROUTER INTEGRATION)
# ===========================================================================================


class PhysicsAnswerVerifier:
    """Uses LLM API to verify if student answer matches reference answer in physics context."""

    def __init__(self, volc_ark_key: str, volc_model_name: str):
        if not volc_ark_key:
            raise ValueError('volc_ark_key is required')
        if not volc_model_name:
            raise ValueError('volc_model_name is required')

        base_url = os.environ.get('VOLC_ARK_BASE_URL', "https://ark-cn-beijing.bytedance.net/api/v3")

        try:
            self.client = OpenAI(base_url=base_url, api_key=volc_ark_key, timeout=1800)
        except Exception as e:
            raise ImportError(f"Failed to initialize OpenAI client: {e}")

        self.model = volc_model_name

        # System prompt for physics answer verification
        self.system_prompt = """You are an assistant that judges equivalence.
Output format: only output <correct> or <incorrect>, do not output anything else."""

    def verify_answer(self, student_answer: str, reference_answer: str) -> bool:
        """
        Use LLM to verify if student answer matches reference answer.
        
        Args:
            student_answer: The student's answer to verify
            reference_answer: The correct reference answer
            problem_text: The original problem/question text for context
            
        Returns:
            True if answers match, False otherwise
        """

        logger = logging.getLogger(__name__)

        for attempt in range(10):  # Retry up to 10 times
            try:
                prompt = f"""Compare the following expressions (focus on mathematical / numeric or final answer tokens).\n\
                        Return <correct> if equivalent else <incorrect>. If multiple choice letters (A-D etc) focus only on letters.\n\n\
                        Expression 1:\n{reference_answer}\n\nExpression 2:\n{student_answer}\n"""

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
                    raise ValueError(f"Unexpected response format: {response}")
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < 9:  # If not the last attempt (0-indexed, so < 9 for 10 attempts)
                    time.sleep(random.uniform(60, 180))  # Random wait 60-180 seconds
                    continue
                else:
                    logger.error(f"All API attempts failed")
                    # If all attempts fail, raise VerifierFailed
                    raise VerifierFailed(f"LLM verification failed after 10 attempts: {e}")

        # This should not be reached due to exception above
        raise VerifierFailed("Unexpected end of verification attempts")


class PhysicsVerifier(BaseVerifier):
    """Physics answer verifier using simple matching and LLM verification."""

    def __init__(self, volc_ark_key: str, volc_model_name: str) -> None:
        self.volc_ark_key = volc_ark_key
        self.volc_model_name = volc_model_name

        # Initialize LLM verifier
        self.llm_verifier = PhysicsAnswerVerifier(volc_ark_key, volc_model_name)

    def verify(self, response: str, verifier_feature_dict: Dict[str, Any]):
        # Extract parameters
        reference_answer = verifier_feature_dict.get('answer', '')
        if not reference_answer:
            raise VerifierFailed("No reference answer provided in verifier_feature_dict")

        # Convert reference answer to appropriate format
        if isinstance(reference_answer, list):
            ref_answers = [str(x) for x in reference_answer]
        else:
            ref_answers = [str(reference_answer)]

        # 提取学生解答
        try:
            llm_final_answers = extract_final_answer_allform(response, answer_type='list')
        except Exception as e:
            raise ExtractAnswerFailed(str(e))

        if not llm_final_answers or len(llm_final_answers) == 0:
            raise ExtractAnswerFailed("No valid answers extracted from response")

        flattened_answers = ([item for sublist in llm_final_answers for item in sublist] if isinstance(
            llm_final_answers[0], list) else llm_final_answers)

        correct_count = 0

        for llm_answer in flattened_answers:
            for dataset_answer in ref_answers:
                equivalency_data = self.is_equiv(llm_answer, dataset_answer, verbose=False)

                if equivalency_data.get("final_result") == True:
                    correct_count += 1
                    break

        total_comparisons = len(flattened_answers)
        accuracy = correct_count / total_comparisons if total_comparisons > 0 else 0.0

        return VerifyResult(score=accuracy, extracted_answer=response)

    def is_equiv(self, expr1: str, expr2: str, verbose: bool = False) -> dict:
        """
        Compare two LaTeX expressions for equivalence and handle errors gracefully.
        """
        result_data = {
            "input_expressions": {
                "expr1": expr1,
                "expr2": expr2
            },
            "preprocessed_expressions": {},
            "sympy_result": None,
            "llm_result": None,
            "final_result": None,
            "error": None,
        }

        # 如果包含文本或者 SymPy 不可用，直接使用 LLM
        if "\\text" in expr1 or "\\text" in expr2:
            result_data["llm_result"] = self.llm_verifier.verify_answer(
                student_answer=expr1,
                reference_answer=expr2,
            )
            result_data["final_result"] = result_data["llm_result"]
            return result_data

        expr1_processed = _preprocess_latex(expr1)
        expr2_processed = _preprocess_latex(expr2)
        expr1_core = _extract_core_eq(expr1_processed)
        expr2_core = _extract_core_eq(expr2_processed)

        try:
            sympy_expr1 = _standardize_expr(parse_latex(expr1_core))
            sympy_expr2 = _standardize_expr(parse_latex(expr2_core))
            result_data["preprocessed_expressions"] = {"expr1": str(sympy_expr1), "expr2": str(sympy_expr2)}

            # Set timeout protection
            with timeout(seconds=10):
                sympy_result = simplify(sympy_expr1 - sympy_expr2) == 0
                sympy_result = sympy_result or sympy_expr1.equals(sympy_expr2)
        except TimeoutError:
            result_data["error"] = "SymPy computation timed out!"
            sympy_result = None
        except Exception as e:
            result_data["error"] = str(e)
            sympy_result = None

        result_data["sympy_result"] = sympy_result

        if sympy_result is not None and sympy_result:
            result_data["final_result"] = sympy_result
            return result_data

        result_data["llm_result"] = self.llm_verifier.verify_answer(expr1, expr2)
        result_data["final_result"] = result_data["llm_result"]
        return result_data
