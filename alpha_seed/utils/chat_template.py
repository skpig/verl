# CHATML = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
CHATML = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'tool' %}<|im_start|>tool name=plugin\n{{ message['content'] }}<|im_end|>{% elif message['role'] == 'assistant' %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}{% else %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
CHATML_TOOL = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if tools %}{{ '<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>' }}{%- for tool in tools %}{{- '\n' }}{{ tool | tojson }}{%- endfor %}\n\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n{% endif %}{% for message in messages %}{% if message['role'] == 'tool' %}<|im_start|>user\n<tool_response>\n{{ message['content'] }}\n</tool_response><|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}\n{% else %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""
CHATML_TOOL_V2 = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if tools %}{{ '<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>' }}{%- for tool in tools %}{{- '\n' }}{{ tool | tojson }}{%- endfor %}\n\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n{% endif %}{% for message in messages %}{% if message['role'] == 'tool' %}<|im_start|>tool\n<tool_response>\nname={{ message['name'] }}\n{{ message['content'] }}\n</tool_response><|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}\n{% else %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""
CHATML_TOOL_V3 = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
{% if tools %}
{{ bos_token }}system
System Prompt

你是一个具备很强专业性的智能助手。

在你给出最终答案之前，请先在脑海中进行一步一步的思考和推理。你的推理过程需要包含在<think_never_used_51bce0c785ca2f68081bfa7d91973934></think_never_used_51bce0c785ca2f68081bfa7d91973934>标签中。请使用和用户问题相同的语言进行推理和回答，除非用户有明确要求。

你具备使用多种工具来完成任务的能力。请仔细阅读下面每个工具的功能描述和参数信息。工具不限制调用次数。

工具调用格式 (Tool Call Format)

工具调用需要发生在推理过程之外，即<think_never_used_51bce0c785ca2f68081bfa7d91973934></think_never_used_51bce0c785ca2f68081bfa7d91973934>标识符之外，工具调用放在<|FunctionCallBegin|>和<|FunctionCallEnd|>标签内，以json格式给出name和parameters，你可以一次调用一个或多个工具。

<|FunctionCallBegin|>[{"name": "function_name", "parameters": {"param_name1": "param_value1", "param_name2": "param_value2"}}]<|FunctionCallEnd|>

示例：

调用单个工具：

<|FunctionCallBegin|>[{"name": "Search_Plugin_new", "parameters": {"query": "2025年全球人工智能市场规模预测","result_limits": 15}}]<|FunctionCallEnd|>


调用多个工具：

<|FunctionCallBegin|>[{"name": "LinkReader", "parameters": {"description": "总结这篇文章的核心观点", "url": "http://example.com/ai-report"}}, {"name": "Search_Plugin_new", "parameters": {"query": "文章作者的最新研究","result_limits": 12}}]<|FunctionCallEnd|>

工具清单 (MCP 格式)

你被授权使用以下工具。在执行任何任务前，你必须依据这些工具的描述和参数来决定如何调用它们。

---
{{ eos_token }}

{{ bos_token }}system name=functions
<tool>
{% for tool in tools %}
    {{ tool | tojson }}
{% endfor %}
</tool>
{{ eos_token }}
{% endif %}

{% for message in messages %}
  {% if message['role'] == 'tool' %}
{{ bos_token }}tool {% if message['name'] %}
name={{message['name']}}
{% endif %}
{{ message['content'] }}
{{ eos_token }}
  {% elif message['role'] == 'assistant' %}
{{ bos_token }}assistant
{{ message['content'] }}
  {% else %}
{{ bos_token }}{{ message['role'] }}
{{ message['content'] }}{{ eos_token }}
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ bos_token }}assistant
{% endif %}
"""

CHATML_TOOL_V4 = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
{% if tools %}
{{ bos_token }}system
System Prompt

在你给出最终答案之前，请先在脑海中进行一步一步的思考和推理。你的推理过程需要包含在<think_never_used_51bce0c785ca2f68081bfa7d91973934></think_never_used_51bce0c785ca2f68081bfa7d91973934>标签中。请使用和用户问题相同的语言进行推理和回答，除非用户有明确要求。

{{ eos_token }}

{{ bos_token }}system name=functions

你具备使用多种工具来完成任务的能力。请仔细阅读下面每个工具的功能描述和参数信息。工具不限制调用次数，除非有明确的要求规定。

工具调用格式 (Tool Call Format)

工具调用需要发生在推理过程之外，即<think_never_used_51bce0c785ca2f68081bfa7d91973934></think_never_used_51bce0c785ca2f68081bfa7d91973934>标识符之外，工具调用放在<|FunctionCallBegin|>和<|FunctionCallEnd|>标签内，以json格式给出name和parameters，你可以一次调用一个或多个工具。

<|FunctionCallBegin|>[{"name": "function_name", "parameters": {"param_name1": "param_value1", "param_name2": "param_value2"}}]<|FunctionCallEnd|>

示例：

调用单个工具：

<|FunctionCallBegin|>[{"name": "Search_Plugin_new", "parameters": {"query": "2025年全球人工智能市场规模预测","result_limits": 15}}]<|FunctionCallEnd|>


调用多个工具：

<|FunctionCallBegin|>[{"name": "LinkReader", "parameters": {"description": "总结这篇文章的核心观点", "url": "http://example.com/ai-report"}}, {"name": "Search_Plugin_new", "parameters": {"query": "文章作者的最新研究","result_limits": 12}}]<|FunctionCallEnd|>

工具清单 (MCP 格式)

你被授权使用以下工具。在执行任何任务前，你必须依据这些工具的描述和参数来决定如何调用它们。

<tools>
{{ tools | tojson }}
</tools>
{{ eos_token }}
{% endif %}

{% for message in messages %}
  {% if message['role'] == 'tool' %}
{{ bos_token }}tool {% if message['name'] %}
name={{message['name']}}
{% endif %}
{{ message['content'] }}
{{ eos_token }}
  {% elif message['role'] == 'assistant' %}
{{ bos_token }}assistant
{{ message['content'] }}
  {% else %}
{{ bos_token }}{{ message['role'] }}
{{ message['content'] }}{{ eos_token }}
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ bos_token }}assistant
{% endif %}
"""

CHATML_TOOL_V5 = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
{% if tools %}
{{ bos_token }}system
System Prompt

在你给出最终答案之前，请先在脑海中进行一步一步的思考和推理。你的推理过程需要包含在<think_never_used_51bce0c785ca2f68081bfa7d91973934></think_never_used_51bce0c785ca2f68081bfa7d91973934>标签中。请使用和用户问题相同的语言进行推理和回答，除非用户有明确要求。

{{ eos_token }}

{{ bos_token }}system name=functions

你具备使用多种工具来完成任务的能力。请仔细阅读下面每个工具的功能描述和参数信息。工具不限制调用次数，除非有明确的要求规定。

# 工具调用格式 (Tool Call Format)

工具调用需要发生在推理过程之外，即<think_never_used_51bce0c785ca2f68081bfa7d91973934></think_never_used_51bce0c785ca2f68081bfa7d91973934>标识符之外，工具调用放在<seed:tool_call>和</seed:tool_call>标签内，以xml格式给出name和parameter，你可以一次调用一个或多个工具。

<seed:tool_call>
<function=example_function_name>
<parameter=example_parameter_1>111</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</seed:tool_call>

示例：

调用单个工具：

<seed:tool_call>
<function=Search_Plugin_new>
<parameter=query>2025年全球人工智能市场规模预测</parameter>
<parameter=result_limits>15</parameter>
</function>
</seed:tool_call>

调用多个工具：
<seed:tool_call>
<function=LinkReader>
<parameter=description>总结这篇文章的核心观点</parameter>
<parameter=url>http://example.com/ai-report</parameter>
</function>
<function=Search_Plugin_new>
<parameter=query>文章作者的最新研究</parameter>
<parameter=result_limits>12</parameter>
</function>
</seed:tool_call>

调用复杂参数工具：
<seed:tool_call>
<function=complex_func>
<parameter=complex_para>
{"key1": "value1", "key2": "value2"}
</parameter>
</function>
</seed:tool_call>


# 工具清单
你被授权使用以下工具（以JSON Schema格式描述）。在执行任何任务前，你必须依据这些工具的描述和参数来决定如何调用它们。

<tools>
{{ tools | tojson }}
</tools>
{{ eos_token }}
{% endif %}

{% for message in messages %}
  {% if message['role'] == 'tool' %}
{{ bos_token }}tool {% if message['name'] %}
name={{message['name']}}
{% endif %}
{{ message['content'] }}
{{ eos_token }}
  {% elif message['role'] == 'assistant' %}
{{ bos_token }}assistant
{{ message['content'] }}
  {% else %}
{{ bos_token }}{{ message['role'] }}
{{ message['content'] }}{{ eos_token }}
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ bos_token }}assistant
{% endif %}
"""