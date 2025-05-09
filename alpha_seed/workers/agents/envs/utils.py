from typing import *
import ast


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
