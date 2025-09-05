import wandb
import os
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
import traceback

def extract_one_trial(run):

    USEFUL_COLS = [
        'step', 
        'val-core/aime24/acc/mean@32', 
        'val-core/amc12/acc/mean@16', 
        'perf/global_cumsum_total_dedup_num_response_tokens',
        'perf/global_cumsum_total_dedup_num_prompt_tokens',
        'timing_s/step',
        'timing_s/testing',
        'timing_s/generate_sequences',
    ]

    h = run.history()
    df = pd.DataFrame(h)
    df = df[USEFUL_COLS]
    df['timing_s/testing'] = df['timing_s/testing'].fillna(0)
    df['timing_s/step'] = df['timing_s/step'] - df['timing_s/testing']
    
    USEFUL_COLS.remove('timing_s/testing')

    return df[USEFUL_COLS]

def debug(project_name, trial_name, trial_ids, baselines_dict):
    """处理单个project和trial的任务"""
    project = project_name
    output_path = f'logs/{project_name}_{trial_name}.csv'
    if False:
    # if os.path.exists(output_path):
        print(f"Skip {output_path} because it already exists")
        return f"Skipped {output_path}"

    try:
        api = wandb.TrackingApi()
        df_lst = []
        for trial_id in trial_ids:
            run = api.run(project=project, run_id=trial_id)
            df = extract_one_trial(run)
            df_lst.append(df)
        df = pd.concat(df_lst)
        df = df.sort_values(by='step')
        
        # deduplication
        if trial_name in baselines_dict[project_name]:
            tmp_df = df['val-core/aime24/acc/mean@32'].fillna(2)
            df = df.loc[tmp_df.groupby(df['step']).idxmin()]
        else:
            tmp_df = df['val-core/aime24/acc/mean@32'].fillna(-1)
            df = df.loc[tmp_df.groupby(df['step']).idxmax()]

        # save
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        return f"Successfully processed {output_path}"
    except Exception as e:
        traceback.print_exc()
        error_msg = f"Error processing {project_name}_{trial_name}: {str(e)}"
        print(error_msg)
        return error_msg




all_runs = {
    "debug_hbz": {
        "ppo": ["run_20250825_d75ba81c"], # 43
        "ppo-mopps": ["run_20250825_9c6f7135", "run_20250827_47b5a0c3"], # 44
        "grpo": ["run_20250826_46ba6f99"], # 45
        "grpo-mopps": ["run_20250826_c5e64a3d", "run_20250829_9cd95b6b", "run_20250830_4a833759", "run_20250830_5c8f2b35", "run_20250830_41059af8", "run_20250831_32ba5824"],  # 46
        "84": ["run_20250827_ba477878", "run_20250830_ade4d7b4"],
        "87": ['run_20250829_96f50319'],
        "91": ['run_20250901_4e26f30a'],
    },
}
baselines = {
    "debug_hbz": ["ppo", "ppo-mopps", "grpo", "grpo-mopps"],
    "debug_hbz2": []
}

def process_all_runs_multiprocess(all_runs, baselines, max_workers=4):
    """使用多进程处理所有runs"""
    # 创建任务列表
    tasks = []
    for project_name, trials in all_runs.items():
        for trial_name, trial_ids in trials.items():
            tasks.append((project_name, trial_name, trial_ids, baselines))
    
    print(f"总共需要处理 {len(tasks)} 个任务")
    print(f"使用 {max_workers} 个进程并行处理")
    
    # 使用ProcessPoolExecutor进行多进程处理
    # 在spawn模式下，每个进程都会重新导入模块，所以不需要额外的初始化
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(debug, project_name, trial_name, trial_ids, baselines): (project_name, trial_name)
            for project_name, trial_name, trial_ids, baselines in tasks
        }
        
        # 处理完成的任务
        completed = 0
        for future in as_completed(future_to_task):
            project_name, trial_name = future_to_task[future]
            completed += 1
            try:
                result = future.result()
                print(f"[{completed}/{len(tasks)}] {result}")
            except Exception as e:
                print(f"[{completed}/{len(tasks)}] 任务 {project_name}_{trial_name} 失败: {str(e)}")
    
    print("所有任务处理完成！")

if __name__ == "__main__":
    # 设置多进程启动方式为spawn，确保wandb API的线程安全
    multiprocessing.set_start_method('spawn', force=True)
    process_all_runs_multiprocess(all_runs, baselines, max_workers=10)

    """Single"""
    # for project_name, trial_name in all_runs.items():
    #     for trial_name, trial_ids in trial_name.items():
    #         debug(project_name, trial_name, trial_ids, baselines)


# def doas_get_all(name):
#     proj_id = {
#         "debug1": "project_20250709_aacff54f",
#         "debug2": "project_20250903_e048a9d4",
#     }
#     import requests
#     from ztijwthelper import ZTIJwtHelper
#     r = requests.post('https://ml.bytedance.net/inner/ListTrackingRuns',  json={
#         "ProjectId": proj_id[name],
#         "Keyword": "abc", # optional
#     }, headers={
#         'zti-token': ZTIJwtHelper().get_jwt_svid()
#     })
#     with open(f'logs/{name}.json', 'w') as f:
#         json.dump(r.json(), f)

# doas_get_all("debug1")