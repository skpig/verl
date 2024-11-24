
import pandas as pd

def convert_data(path, repeat=1):
    df = pd.read_json(path, lines=True)

    df_out = pd.DataFrame(columns=['data_source', 'prompt', 'ability', 'reward_model', 'extra_info',
       'level', 'type', 'answer'])
    
    import numpy as np
    def get_data_source(row):
        if not isinstance(row, dict) and np.isnan(row):
            return "Unknown"
        if 'source' in row:
            row = row['source']
            if '文件' in row:
                row = row['文件']
            if 'contest' in row:
                row = row['contest']
        elif 'contest' in row:
            row = row['contest']
        
        if isinstance(row, dict):
            print(row)
            exit()

        if any(x in row for x in ['大学', 'University']):
            return '大学竞赛'
        if any(x in row for x in ['奥林匹克', '国家队', '国际', '邀请赛', "IMO", '集训队', '罗马尼亚', '捷克波兰斯洛伐克', 'Olympiads', 'China_Team_Selection_Test']):
            return 'IMO_level'
        if any(x in row for x in ['数学联赛', '高中联赛', '二试模拟', '一试', '数学竞赛', 'Purple_Comet_Problems']):
            return '数学联赛'
        if 'AMC' in row:
            return 'AMC'
        if any(x in row for x in ['北大', '清华', '自主招生', '北京大学', '夏令营']):
            return '自主招生'
        if '原创' in row:
            return '原创'
        if 'MATH60' in row:
            return 'MATH60'
        if 'Aops' in row:
            return "Aops"
        if ' AIME' in row:
            return "AIME"
        return 'Others'

    df_out['data_source'] = [get_data_source(row) for row in df['meta']]
    df_out['prompt'] = [[{'content': row, 'role':'user'},] for row in df['problem']]
    df_out['ability'] = ["MATH" for _ in range(len(df))]
    df_out['reward_model'] = [{'ground_truth': str(int(row)), 'style': 'rule-lighteval/MATH_v2'} for row in df['answer']]
    df_out['extra_info'] = [{'index': row} for row in df['id']]
    df_out['level'] = ["" for _ in range(len(df))]
    df_out['type'] = ["" for _ in range(len(df))]
    df_out['answer'] = [str(int(row)) for row in df['answer']]
    if repeat > 1:
        df_out = pd.concat([df_out]*repeat, ignore_index=True)
    print(df_out)
    return df_out

if __name__ == "__main__":
    path = "/opt/tiger/alphaseed_data/data/math_data/training_data/released_data/release_1.3.jsonl"
    df_out = convert_data(path, repeat=10)
    df_out.to_parquet("hdfs://harunava/home/byte_data_seed_azure/seed_research/shengdinghu/rl/datasets/math_data/training_data/released_data/release_1.3.parquet")
