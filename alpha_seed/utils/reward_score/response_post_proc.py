import re


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


import re


def last_codeblock_postprocess(input_text, codeblock_seps=['python', 'cpp', 'java'], last_response_strict=True):
    languages_pattern = '|'.join(map(re.escape, codeblock_seps))
    codeblock_start = f'```({languages_pattern})'
    pattern = re.compile(codeblock_start + r'\n(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
    matches = list(pattern.finditer(input_text))

    if matches:
        last_match = matches[-1]
        language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        return f'```{language}\n{code_content}\n```'
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
