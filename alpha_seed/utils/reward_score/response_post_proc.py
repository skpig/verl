import re
from pydantic import BaseModel
from typing import List, Dict


def summary_postprocess(input_text, last_response_sep=['<summarize>', '</summarize>'], last_response_strict=True):
    start_tag, end_tag = last_response_sep

    # Find the last occurrence of the start_tag
    start_index = input_text.rfind(start_tag)
    if start_index == -1:
        # Start tag not found
        if last_response_strict:
            return ''
        else:
            return input_text

    # Move the index to the end of the start_tag
    start_index += len(start_tag)

    # Find the end_tag starting from the end of start_tag
    end_index = input_text.rfind(end_tag, start_index)
    if end_index == -1:
        # End tag not found
        if last_response_strict:
            return ''
        else:
            return input_text

    # Extract and return the content between start_tag and end_tag
    return input_text[start_index:end_index].strip()


# Same Code Blocks as https://code.byted.org/seed/code_sandbox/blob/master/sandbox/utils/extraction.py
language_to_aliases = {
    'python': ['python', 'Python', 'py', 'Python3', 'python3', 'PY'],
    'cpp': ['cpp', 'c++', 'C++', 'Cpp', 'CPP'],
    'nodejs': ['javascript', 'Javascript', 'JavaScript', 'JS', 'js'],
    'go': ['go', 'Go'],
    'java': ['java', 'Java'],
    'php': ['php'],
    'csharp': ['csharp', 'c#', 'C#'],
    'bash': ['bash', 'Bash', 'BASH', 'sh', 'shell'],
    'typescript': ['typescript'],
    'rust': ['rust', 'Rust', 'rs'],
    'sql': ['sql', 'SQL', 'Sql'],
    'D': ['D', 'd'],
    'julia': ['julia', 'Julia', 'jl'],
    'lua': ['lua', 'Lua'],
    'php': ['php', 'PHP'],
    'perl': ['perl', 'Perl', 'PERL'],
    'R': ['R', 'r'],
    'ruby': ['ruby', 'Ruby'],
    'rust': ['rust', 'Rust', 'rs'],
    'scala': ['scala', 'Scala'],
    'kotlin': ['kotlin', 'Kotlin'],
    'c': ['c', 'C'],
    'html': ['html', 'Html', 'HTML'],
    'javascript': ['javascript', 'Javascript', 'JavaScript'],
    'verilog': ['verilog', 'Verilog', 'VERILOG'],
    'racket': ['racket'],
    'swift': ['swift'],
    'react': ['tsx'],
}

aliases_to_language_tiled = {v: k for k, vs in language_to_aliases.items() for v in vs}

fenced_code_block_pattern = re.compile(
    # Starting with three backticks and optional language identifier
    r'```([^\n]*)\n'
    r'(.*?)'  # Non-greedy capture of the content
    r'\n\s*```',  # Ending with three backticks
    re.DOTALL | re.MULTILINE)


# code extraction
def extract_fenced_code(completion: str) -> List[Dict]:
    code_matches = re.findall(fenced_code_block_pattern, completion)
    results = []
    for m in code_matches:
        lang = aliases_to_language_tiled.get(m[0].strip(), '')
        if lang != '':
            results.append({
                "lang": lang,
                "code": m[1],
            })
    return results


def last_codeblock_postprocess(input_text, codeblock_seps=['python', 'cpp', 'java'], last_response_strict=True):
    results = extract_fenced_code(input_text)
    if len(results) > 0:
        return f'```{results[-1]["lang"]}\n{results[-1]["code"]}\n```'
    else:
        if last_response_strict:
            return ''
        else:
            return input_text


def test_summary_postprocess():
    input_txt = 'balabala <summarize> sum111 </summarize> heiheihei <summarize> sum222 </summarize>'
    output_txt = summary_postprocess(input_txt)
    print(output_txt.encode())

    input_txt = 'balabala <summarize> sum111 </summarize> heiheihei <summarize>'
    output_txt = summary_postprocess(input_txt)
    print(output_txt.encode())

    input_txt = 'balabala heiheihei '
    output_txt = summary_postprocess(input_txt)
    print(output_txt.encode())


def test_last_codeblock_postprocess():
    input_text = r"""
Some text above.
哈哈哈哈
```python
print("Hello, world!")
```
哈哈哈哈
```python
print("Hello, world222!")
```
哈哈哈哈
```python
print("Hello, world333!")

"""
    output_txt = last_codeblock_postprocess(input_text)
    print(output_txt)


def test_last_codeblock_postprocess2():
    input_text = r"""```python
print("Hello, world!")
```
```test
1+2
```
"""
    output_txt = last_codeblock_postprocess(input_text)
    print(output_txt)


def test_last_codeblock_postprocess3():
    input_text = r"""```python
print("Hello, world!")
```
```cpp
x=1+2
```
"""
    output_txt = last_codeblock_postprocess(input_text)
    print(output_txt)


def test_last_codeblock_postprocess4():
    input_text = r"""```python
print("Hello, world!")
```
```cpp
x=1+2
```

```cpp
x=
```
```python
1+2
"""
    output_txt = last_codeblock_postprocess(input_text)
    print(output_txt)


if __name__ == '__main__':
    test_summary_postprocess()
    test_last_codeblock_postprocess()
    test_last_codeblock_postprocess2()
    test_last_codeblock_postprocess3()
    test_last_codeblock_postprocess4()
