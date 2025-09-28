from dataclasses import dataclass


class ExtractAnswerFailed(Exception):

    def __init__(self, message: str = ''):
        if message:
            message = ': ' + message
        super().__init__(f'{type(self).__name__}{message}')


class VerifierFailed(Exception):

    def __init__(self, message: str = ''):
        if message:
            message = ': ' + message
        super().__init__(f'{type(self).__name__}{message}')


@dataclass
class VerifyResult:
    score: float  # between 0.0 and 1.0, higher is better, with both 0.0 and 1.0 included
    extracted_answer: str  # e.g., the content within \boxed{} for math, or just the whole response, mainly for logging


class BaseVerifier:

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        raise NotImplementedError
