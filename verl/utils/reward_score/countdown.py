import re
import random
import ast
import operator
from traceback import format_stack


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def get_format_score(solution_str, ground_truth):
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    if equation is None:
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        return 0
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            return 0
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            return 1
        else:
            return 0
    except:
        return 0


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def coutdown_compute_score(data_source, solution_str, ground_truth, extra_info=None, method='strict'):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    msgs = [] if do_print else None

    if do_print:
        msgs.append("=================================")
        msgs.append(f"---- Target: {target} | Numbers: {numbers} ----")
        msgs.append(f"---- Extracted equation: ----\n{equation}")
        msgs.append(f"---- Solution string: ----\n{solution_str}")
        msgs.append(f"---- Result: ----")
    

    if equation is None:
        if do_print:
            msgs.append("No equation found")
            print("\n".join(msgs), flush=True)
        return 0, "No equation found (%)"
    
    format_score = 0.01
    valid_score = 0.1
    score = 1.0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            msgs.append("Invalid equation")
            print("\n".join(msgs), flush=True)
        return format_score, "Invalid equation (%)"
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                msgs.append("Could not evaluate equation")
                print("\n".join(msgs), flush=True)
            return format_score, "Invalid equation (%)"
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                msgs.append(f"Correct equation: {equation} = {result}")
                print("\n".join(msgs), flush=True)
            return score, "Correct answer (%)"
        else:
            if do_print:
                msgs.append(f"Wrong result: equation = {result}, target = {target}")
                print("\n".join(msgs), flush=True)
            return valid_score, "Wrong equation (%)"
    except:
        if do_print:
            msgs.append("Error evaluating equation")
            print("\n".join(msgs), flush=True)
        return format_score, "Invalid equation (%)"


def summarization_compute_score(data_source, solution_str, ground_truth, extra_info=None, method='strict'):
    """The scoring function for summarization task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    # use ROUGE-L for the summarization task
    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(solution_str, ground_truth)
    score = scores[0]['rouge-l']['f']
    return score, "N_A"

def translation_compute_score(data_source, solution_str, ground_truth, extra_info=None, method='strict'):
    """The scoring function for translation task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    # use BLEU for the translation task
    from nltk.translate.bleu_score import sentence_bleu
    reference = [ground_truth.split()]
    candidate = solution_str.split()
    score = sentence_bleu(reference, candidate)
    return score, "N_A"


def compute_score(data_source, solution_str, ground_truth, extra_info=None, method='strict'):
    if data_source == 'countdown':
        return coutdown_compute_score(data_source, solution_str, ground_truth, extra_info, method)
    elif data_source == 'samsum':
        return summarization_compute_score(data_source, solution_str, ground_truth, extra_info, method)
    elif data_source == 'wmt':
        return translation_compute_score(data_source, solution_str, ground_truth, extra_info, method)