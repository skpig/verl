import yaml

from scripts.query_tool.utils import FlowStyleList, represent_flow_list, represent_multiline_str, compact_list_fields

yaml.add_representer(FlowStyleList, represent_flow_list)
yaml.add_representer(str, represent_multiline_str)

obj = {
    "key": "value",
    "list": list(range(12)),
    "string-long": "a long string with multi line\n" * 10,
}

if __name__ == '__main__':
    obj = compact_list_fields(obj)
    # print(obj)
    yaml_str = yaml.dump(obj, default_flow_style=False, allow_unicode=True)
    print(yaml_str)
