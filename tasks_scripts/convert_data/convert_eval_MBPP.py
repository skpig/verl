import json
import pandas as pd


def extract_setup_from_ref_code(ref_code: str):
    import re
    import_pattern = re.compile(r'^\s*import\s+[\w., ]+')
    from_import_pattern = re.compile(r'^\s*from\s+([\w.]+)\s+import\s+([\w, ]+)')
    def_pattern = re.compile(r'^\s*def\s+\w+\s*\([^)]*\)\s*:')

    lines = ref_code.split("\n")

    setup_lines = []
    for line in lines:
        if import_pattern.match(line):
            setup_lines.append(line.strip())
        elif from_import_pattern.match(line):
            setup_lines.append(line.strip())
        elif def_pattern.match(line):
            setup_lines.append(line.strip())
        
    return "\n".join([line + "\n" for line in setup_lines])


def convert_data(path):
    df = pd.read_json(path, lines=True)
    print(df, df.columns)

    df_out = pd.DataFrame(columns=['data_source', 'prompt', 'ability', 'reward_model', 'extra_info',
       'level', 'type', 'answer'])
    
    df_out['data_source'] = ["CODE##MBPP" for _ in range(len(df))]
    def apply_cot(prompt: str, ref_code: str):
        ret = prompt
        ret += "\nYou need first write a step-by-step outline and then complete the following code:\n"
        ret += "```python\n" + extract_setup_from_ref_code(ref_code) + "```\n"
        ret += "You should only print the code in the final answer\n"
        return ret

    df_out['prompt'] = [[{'content': apply_cot(row['prompt'], row['code']), 'role': 'user'},] for i, row in df.iterrows()]
    df_out['ability'] = ["Code" for _ in range(len(df))]
    df_out['reward_model'] = [{'ground_truth': json.dumps({ 'task_id': row['task_id'], 'lang': 'python', 'test_str': '\n'.join(row['test']), 'timeout': 3}), 'style': 'code-localexec'} for i, row in df.iterrows()]
    df_out['extra_info'] = [{'ref_code': code } for code in df['code']]
    df_out['raw_problem'] = [row for row in df['prompt']]
    df_out['level'] = [None for _ in range(len(df))]
    df_out['type'] = [None for _ in range(len(df))]
    df_out['answer'] = [None for _ in range(len(df))]
    print(df_out)
    print(df_out.iloc[0])
    return df_out

if __name__ == "__main__":
    path = f"/opt/tiger/DeepSeek-Coder/Evaluation/MBPP/data/mbpp_test.jsonl"
    df_out = convert_data(path)
    df_out.to_parquet("hdfs://haruna/home/byte_data_seed/ssd_hldy/user/jiangchengquan/rl/datasets/MBPP_eval.parquet")
    