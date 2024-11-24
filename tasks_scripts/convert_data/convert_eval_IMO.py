
import pandas as pd

def convert_data(path):
    df = pd.read_json(path, lines=True)
    print(df, df.columns)

    df_out = pd.DataFrame(columns=['data_source', 'prompt', 'ability', 'reward_model', 'extra_info',
       'level', 'type', 'answer'])
    
    df_out['data_source'] = ["IMO" for row in df['problem']]
    df_out['prompt'] = [[{'content': row, 'role': 'user'},] for row in df['problem']]
    df_out['ability'] = ["MATH" for _ in range(len(df))]
    df_out['reward_model'] = [{'ground_truth': str(int(row.replace(',',''))), 'style': 'rule-lighteval/MATH_v2'} for row in df['answer']]
    df_out['extra_info'] = [{'index': row} for row in df['id']]
    df_out['level'] = ["" for _ in range(len(df))]
    df_out['type'] = ["" for _ in range(len(df))]
    df_out['answer'] = [str(int(row.replace(',',''))) for row in df['answer']]
    print(df_out)
    print(df_out.iloc[0])
    return df_out

if __name__ == "__main__":
    path = "/opt/tiger/alphaseed_data/data/math_data/evaluation_data/IMO_evals.jsonl"
    df_out = convert_data(path)
    df_out.to_parquet("hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/evaluation_data/IMO_evals.parquet")
