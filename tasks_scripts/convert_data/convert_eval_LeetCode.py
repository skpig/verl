import json
import pandas as pd

def convert_data(path):
    df = pd.read_json(path, lines=True)
    print(df, df.columns)

    df_out = pd.DataFrame(columns=['data_source', 'prompt', 'ability', 'reward_model', 'extra_info',
       'level', 'type', 'answer'])
    
    df_out['data_source'] = ["CODE##LeetCode" for _ in range(len(df))]
    apply_cot = lambda prompt: prompt + "\nYou need first write a step-by-step outline and then write the code."

    df_out['prompt'] = [[{'content': apply_cot(row), 'role': 'user'},] for row in df['prompt_sft']]
    df_out['ability'] = ["Code" for _ in range(len(df))]
    df_out['reward_model'] = [{'ground_truth': json.dumps({ 'task_id': row['task_id'], 'lang': 'python', 'test_str': row['test'], 'timeout': 3}), 'style': 'code-localexec'} for i, row in df.iterrows()]
    df_out['extra_info'] = [{'url': url} for url in df['url']]
    df_out['raw_problem'] = [row for row in df['prompt']]
    df_out['level'] = [None for _ in range(len(df))]
    df_out['type'] = [None for _ in range(len(df))]
    df_out['answer'] = [None for _ in range(len(df))]
    print(df_out)
    print(df_out.iloc[0])
    return df_out

if __name__ == "__main__":
    leetcode_data_version = "20240121-Jul"
    path = f"/data01/home/jiangchengquan/rl/DeepSeek-Coder/Evaluation/LeetCode/data/{leetcode_data_version}.jsonl"
    df_out = convert_data(path)
    df_out.to_parquet("hdfs://haruna/home/byte_data_seed/ssd_hldy/user/jiangchengquan/rl/datasets/LeetCode_20240121-Jul_evals.parquet")
