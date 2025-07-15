from typing import *
import ast
from urllib.parse import urlparse
import fnmatch


def parse_func_call_kwargs(func_call_str) -> Tuple[str, Dict]:
    tree = ast.parse(func_call_str)
    expr = tree.body[0].value
    assert isinstance(expr, ast.Call), "expr is not a ast.Call"
    func_name = expr.func.id
    kwargs_dict = {}
    for kw in expr.keywords:
        key = kw.arg
        if isinstance(kw.value, ast.Constant):
            value = kw.value.value
        elif isinstance(kw.value, ast.List):
            value = [ast.literal_eval(item) for item in kw.value.elts]
        elif isinstance(kw.value, ast.Dict):
            keys = [ast.literal_eval(k) for k in kw.value.keys]
            values = [ast.literal_eval(v) for v in kw.value.values]
            value = dict(zip(keys, values))
        else:
            value = ast.literal_eval(ast.dump(kw.value))
        kwargs_dict[key] = value
    return func_name, kwargs_dict


def truncate_str_by_tokens(text, max_token_len, tokenizer):
    tokens = tokenizer._batch_encode_plus([text], add_special_tokens=False)
    tokens = tokens.input_ids[0]
    length = len(tokens)
    if length > max_token_len:
        tokens = tokens[:max_token_len]
        text = tokenizer.decode(tokens)
    return text, length


def is_url_blocked(url):
    parsed = urlparse(url)
    domain_path = parsed.netloc + parsed.path
    for pattern in BLOCKLIST:
        if fnmatch.fnmatch(domain_path, pattern):
            return True
    return False


# Blocklist patterns (wildcards allowed)
BLOCKLIST = [
    "projecteuler.net", "www.projecteuler.net", "stephan-brumme.com", "ivl-projecteuler.com",
    "euler.stephan-brumme.com", "euler.synap.co.kr", "mathblog.dk/project-euler", "euler.overclocked.io",
    "github.com/nayuki/Project-Euler-solutions", "github.com/micahyoung324/ProjectEuler",
    "github.com/lucky-bai/ProjectEuler1000", "github.com/*/ProjectEuler*",
    "kaggle.com/datasets/angelorobsonmelo/project-euler-dataset", "kaggle.com/*/project-euler*", "*projecteuler.net*",
    "*projecteuler*solution*", "*euler*.brumme.com*", "github.com/*/ProjectEuler*", "kaggle.com/*/project-euler*",
    "ivl-projecteuler.com/*", "*project*euler*solutions*", "*euler*", "*nayuki*", "*Euler*"
]
