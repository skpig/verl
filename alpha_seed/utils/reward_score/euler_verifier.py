from typing import *
import os
import json
import dill
from alpha_seed.workers.xperf_rollout.component.query_plugin import EnvStates


class ci_math_verifier:

    def __init__(self):
        self.name = 'ci_math_verifier'

    def verify(self, pred, answer, meta=None):
        # Remove commas for consistency
        answer = answer.replace(',', '')
        pred = pred.replace(',', '')

        # Try to match both integers and decimals
        try:
            # Attempt to match as float first for decimal support
            if float(answer) == float(pred):
                return 1.0
            return -1.0
        except ValueError:
            # If either is not a valid number, return 0
            return -1.0


import re

from .utils import Verifier


class EulerVerifier(Verifier, reward_style="euler_verifier"):

    @staticmethod
    def compute_score(*args, **kwargs) -> float:
        return compute_score(*args, **kwargs)


def compute_score(solution_str, ground_truth, **kwargs):
    ground_truth = json.loads(ground_truth)
    if isinstance(ground_truth['verifiable_answer'], str):
        #This is for Project Euler
        verifier = ci_math_verifier()

        def extract_rightmost_fc_response_corrected(text):
            # Corrected to properly capture only the rightmost substring
            match = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
            return match[-1] if match else 'None'

        extracted_pred = extract_rightmost_fc_response_corrected(solution_str)
        if os.getenv('PRINT_EULER_VERIFIER', '0') == '1':
            print('[euler_verifier] extracted_pred', extracted_pred, 'answer', ground_truth['verifiable_answer'])
        return verifier.verify(pred=extracted_pred, answer=ground_truth['verifiable_answer'])
    elif isinstance(ground_truth['verifiable_answer'], list):
        # This is for ci science
        def extract_response_segment(text):
            """
            Extract the substring between <answer> and </answer>.
            """
            match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""

        def parse_float_list_from_string(s):
            """
            Parse a list of floats from a string like '[1.0, 2.5, 3]'.
            """
            try:
                return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]
            except ValueError:
                raise ValueError(f"Failed to parse float list from: {s}")

        def verify_extracted_prediction_matches_answer(predicted_wrapped_str, answer_list, tol=1e-6):
            """
            Compare the extracted float list from the predicted string with the true answer list.

            Args:
                predicted_wrapped_str (str): The predicted string, containing floats inside markers.
                answer_list (List[float]): The ground-truth list of floats.
                tol (float): Tolerance for float comparison.

            Returns:
                Tuple[bool, str]: (Match status, reason)
            """
            extracted_str = extract_response_segment(predicted_wrapped_str)
            if not extracted_str:
                return -1.0, "No extractable float list found in markers"

            try:
                predicted = parse_float_list_from_string(extracted_str)
            except ValueError as e:
                return -1.0, f"Parsing error: {e}"

            if len(predicted) != len(answer_list):
                return -1.0, f"Length mismatch: {len(predicted)} vs {len(answer_list)}"

            for i, (p, a) in enumerate(zip(predicted, answer_list)):
                if abs(p - a) > tol:
                    return -1.0, f"Value mismatch at index {i}: {p} vs {a}"

            return 1.0, "Match"

        # Test the function
        result, _ = verify_extracted_prediction_matches_answer(solution_str, ground_truth['verifiable_answer'])
        return result
