from collections import defaultdict
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
import traceback
import numpy as np
from typing import Any, List
import pickle


class TrialGroup:
    def __init__(self, proj_name, group_id: str, trial_id_lst: List[str]):
        self.proj_name = proj_name
        self.trial_id_lst = trial_id_lst
        self.group_id = group_id
        self.groups = [
            Trial(proj_name, trial_id) for id, trial_id in enumerate(trial_id_lst)
        ]
    
    def get_top_aime_mean(self, step_range=-1):
        return np.mean([group.get_top_aime(step_range) for group in self.groups])
    
    def get_top_amc_mean(self, step_range=-1):
        return np.mean([group.get_top_amc(step_range) for group in self.groups])
    
    def get_cumsum_response_tokens_mean(self, step_range=-1):
        return np.mean([group.get_cumsum_response_tokens(step_range) for group in self.groups])
    
    def get_per_step_response_tokens_mean(self, step_range=-1):
        return np.mean([group.get_per_step_response_tokens(step_range) for group in self.groups])
    
    def get_cumsum_timing_mean(self, step_range=-1):
        return np.mean([group.get_cumsum_timing(step_range) for group in self.groups])
    
    def get_per_step_timing_mean(self, step_range=-1):
        return np.mean([group.get_per_step_timing(step_range) for group in self.groups])
    
    def get_top_x_y_dataframe(self, x_col, y_col, step_range=-1):
        all_df = []
        max_value = -1
        max_id = -1
        for i, group in enumerate(self.groups):
            if 'aime' in y_col:
                y_df = group.get_top_aime_dataframe(step_range)
                cur_max_value = y_df['val-core/aime24/acc/mean@32'].max()
            elif 'amc' in y_col:
                y_df = group.get_top_amc_dataframe(step_range)
                cur_max_value = y_df['val-core/amc12/acc/mean@16'].max()
            elif 'length' in y_col:
                y_df = group.get_response_length_dataframe(step_range)
                cur_max_value = -2
            else:
                raise ValueError(f'Invalid y_col: {y_col}')
            
            if cur_max_value > max_value:
                max_value = cur_max_value
                max_id = i
            else:
                all_df.append(y_df) # dummy df
                continue
            
            # print(f"max_id: {max_id}")
            if 'tim' in x_col:
                x_df = group.get_cumsum_timing_dataframe(step_range)
            elif 'token' in x_col:
                x_df = group.get_cumsum_response_tokens_dataframe(step_range)
            else:
                raise ValueError(f'Invalid x_col: {x_col}')
            
            df = pd.merge(x_df, y_df, on='step', how='inner')
            df = df.sort_values(by='step')
            df['seed'] = i

            all_df.append(df)
        return all_df[max_id]
    
    def get_top_amc_dataframe(self, step_range=-1):
        return [group.get_top_amc_dataframe(step_range) for group in self.groups]
    
    def get_cumsum_response_tokens_dataframe(self, step_range=-1):
        return pd.concat([group.get_cumsum_response_tokens_dataframe(step_range) for group in self.groups])
    
    def get_cumsum_timing_dataframe(self, step_range=-1):
        return pd.concat([group.get_cumsum_timing_dataframe(step_range) for group in self.groups])
    
    def get_response_length_dataframe(self, step_range=-1):
        return self.groups[0].get_response_length_dataframe(step_range)

class Trial:
    def __init__(self, proj_name, trial_id):
        if isinstance(trial_id, list):
            self.trial_id = '&'.join(trial_id)
        else:
            self.trial_id = trial_id

        self.proj_name = proj_name

        self.cache_path = f'logs/{proj_name}_{trial_id}.csv'
        if not os.path.exists(self.cache_path):
        # if True:
            api = wandb.TrackingApi()
            if isinstance(trial_id, list):
                df_lst = []
                for trial_id in trial_id:
                    retry = 5
                    while retry > 0:
                        try:
                            run = api.run(project=proj_name, run_id=trial_id)
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                            retry -= 1
                            continue
                    df = extract_one_trial(run)
                    df_lst.append(df)
                df = pd.concat(df_lst)
                # deduplicate
                df = df.drop_duplicates(subset=['step'], keep='last')
                df.to_csv(self.cache_path, index=False)
            else:
                run = api.run(project=proj_name, run_id=trial_id)
                df = extract_one_trial(run)
                df.to_csv(self.cache_path, index=False)
        else:
            df = pd.read_csv(self.cache_path)
        
        if proj_name == "debug_hbz":
            step_range = 400
        else:
            step_range = 300
        self.df = df[df['step'] <= step_range].reset_index(drop=True)
    
    def get_df(self, step_range=-1):
        if step_range == -1:
            return self.df
        else:
            return self.df[self.df['step'] <= step_range]
    
    def get_top_aime(self, step_range=-1):
        df = self.get_df(step_range)
        return df['val-core/aime24/acc/mean@32'].max()

    def get_top_amc(self, step_range=-1):
        df = self.get_df(step_range)
        return df['val-core/amc12/acc/mean@16'].max()
    
    def get_cumsum_response_tokens(self, step_range=-1):
        df = self.get_df(step_range)
        return df['perf/global_cumsum_total_dedup_num_response_tokens'].max()
    
    def get_per_step_response_tokens(self, step_range=-1):
        df = self.get_df(step_range)
        last_step_row = df.loc[df['perf/global_cumsum_total_dedup_num_response_tokens'].idxmax()]
        return last_step_row['perf/global_cumsum_total_dedup_num_response_tokens'] / last_step_row['step']
    
    def get_cumsum_timing(self, step_range=-1):
        df = self.get_df(step_range)
        return df['timing_s/step'].sum()
    
    def get_per_step_timing(self, step_range=-1):
        df = self.get_df(step_range)
        return df['timing_s/step'].mean()
    
    def get_top_aime_dataframe(self, step_range=-1):
        df = self.get_df(step_range)
        df = df[df['val-core/aime24/acc/mean@32'] > 0]
        return df[['step', 'val-core/aime24/acc/mean@32']]
    
    def get_top_amc_dataframe(self, step_range=-1):
        df = self.get_df(step_range)
        df = df[df['val-core/amc12/acc/mean@16'] > 0]
        return df[['step', 'val-core/amc12/acc/mean@16']]
    
    def get_cumsum_response_tokens_dataframe(self, step_range=-1):
        df = self.get_df(step_range)
        return df[['step', 'perf/global_cumsum_total_dedup_num_response_tokens']]
    
    def get_cumsum_timing_dataframe(self, step_range=-1):
        df = self.get_df(step_range)
        df['timing_s/cumsum_step'] = df['timing_s/step'].cumsum()
        return df[['step', 'timing_s/cumsum_step']]
    
    def get_response_length_dataframe(self, step_range=-1):
        df = self.get_df(step_range)
        return df[['step', 'response_length/mean']]
    
    
def extract_one_trial(run):

    USEFUL_COLS = [
        'step', 
        'val-core/aime24/acc/mean@32', 
        'val-core/amc12/acc/mean@16', 
        'perf/global_cumsum_total_dedup_num_response_tokens',
        'perf/global_cumsum_total_dedup_num_prompt_tokens',
        'response_length/mean',
        'actor/entropy',
        'timing_s/step',
        'timing_s/testing',
        'timing_s/generate_sequences',
        # 'perf/global_time',
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


def table_main_result(ids_df):
    # init related groups
    all_trial_groups: defaultdict[Any, dict] = defaultdict(dict)
    for idx, alg, proj in zip(ids_df["id"], ids_df["alg"], ids_df["proj"]):
        all_trial_groups[proj][alg] = TrialGroup(proj, idx, id2trialid[proj][idx])
    
    for proj in all_trial_groups:
        all_results = []
        for alg in all_trial_groups[proj]:

            step_range = -1
            # if proj == 'debug_hbz3' and alg == 'ppo-dynamic':
            #     step_range = 155
            # elif proj == 'debug_hbz' and alg == 'ppo-ours':
            #     step_range = 385
            # else:
            #     step_range = -1

            aime_mean = f"{all_trial_groups[proj][alg].get_top_aime_mean(step_range) * 100:.2f}"
            amc_mean = f"{all_trial_groups[proj][alg].get_top_amc_mean(step_range) * 100:.2f}"
            cumsum_response_tokens_mean = f"{all_trial_groups[proj][alg].get_cumsum_response_tokens_mean(step_range) / 1e9:.2f}"
            cumsum_timing_mean = f"{all_trial_groups[proj][alg].get_cumsum_timing_mean(step_range) / 3600:.2f}"
            per_step_response_tokens_mean = f"{all_trial_groups[proj][alg].get_per_step_response_tokens_mean(step_range) / 1e6:.2f}"
            per_step_timing_mean = f"{all_trial_groups[proj][alg].get_per_step_timing_mean(step_range) / 60 :.2f}"
            all_results.append({
                "alg": alg,
                # "project_name": proj,
                "AIME24": aime_mean,
                "AMC23": amc_mean,
                # "cumsum_response_tokens": cumsum_response_tokens_mean,
                # "cumsum_timing": cumsum_timing_mean,
                "per_step_response_tokens": per_step_response_tokens_mean,
                "per_step_timing": per_step_timing_mean,
            })

        df = pd.DataFrame(all_results)

        latex_code = df.to_latex(
            index=False,           # 不要输出 DataFrame 的索引
            float_format="%.2f",   # 浮点数保留 3 位小数
            column_format="lcccc", # 每列对齐方式：l=左对齐, c=居中, r=右对齐
            # caption="实验结果表", 
            label=f"tab:main_results_{proj}"
        )
        print(f"=== {proj} ===")
        print(latex_code)

# def _plot_raw_and_smooth(ax, df, xcol='timing_s/cumsum_step', ycol='val-core/amc12/acc/mean@16'):
#     # 1) 先画“所有数据点”：更淡、更小，不要重复 legend
#     alg_order = sorted(df['alg'].unique())
#     palette = sns.color_palette(None, n_colors=len(alg_order))
#     pal_map = {alg: palette[i] for i, alg in enumerate(alg_order)}

#     sns.scatterplot(
#         data=df, x=xcol, y=ycol, hue='alg', hue_order=alg_order,
#         palette=pal_map, ax=ax, alpha=0.25, s=12, legend=False
#     )

#     # 2) 每个 alg 分组做 LOWESS 平滑，并叠加“深色”曲线
#     for alg in alg_order:
#         g = df[df['alg'] == alg].sort_values(xcol)
#         if len(g) < 3:
#             continue
#         # frac 控制平滑强度：越大越平滑；可按需调 0.2~0.6
#         z = lowess(g[ycol].to_numpy(), g[xcol].to_numpy(), frac=0.25, it=1, return_sorted=True)
#         ax.plot(z[:, 0], z[:, 1], label=alg, linewidth=2.2, color=pal_map[alg])

#     # 3) 只给平滑曲线加 legend（更干净）
#     ax.legend(title="alg", frameon=False)

# def figure_token_acc():
#     # init related groups（保持你原逻辑）
#     all_trial_groups = defaultdict(dict)
#     for idx, alg, proj in zip(all_ids["id"], all_ids["alg"], all_ids["proj"]):
#         if id2trialid[proj][idx]:
#             all_trial_groups[proj][alg] = TrialGroup(proj, idx, id2trialid[proj][idx])

#     def _subfigure(proj, base_alg='ppo'):
#         x_time_y_aime_lst, x_time_y_amc_lst = [], []
#         x_numtokens_y_aime_lst, x_numtokens_y_amc_lst = [], []
#         for alg in all_trial_groups[proj]:
#             if base_alg not in alg or "mopps" in alg:
#                 print("skip", alg)
#                 continue
#             x_time_y_aime  = all_trial_groups[proj][alg].get_top_x_y_dataframe(x_col='tim',   y_col='aime')
#             x_time_y_amc   = all_trial_groups[proj][alg].get_top_x_y_dataframe(x_col='tim',   y_col='amc')
#             x_ntok_y_aime  = all_trial_groups[proj][alg].get_top_x_y_dataframe(x_col='token', y_col='aime')
#             x_ntok_y_amc   = all_trial_groups[proj][alg].get_top_x_y_dataframe(x_col='token', y_col='amc')

#             for d in (x_time_y_aime, x_time_y_amc, x_ntok_y_aime, x_ntok_y_amc):
#                 d['alg'] = alg

#             x_time_y_aime_lst.append(x_time_y_aime)
#             x_time_y_amc_lst.append(x_time_y_amc)
#             x_numtokens_y_aime_lst.append(x_ntok_y_aime)
#             x_numtokens_y_amc_lst.append(x_ntok_y_amc)

#         x_time_y_aime   = pd.concat(x_time_y_aime_lst)
#         x_time_y_amc    = pd.concat(x_time_y_amc_lst)
#         x_numtokens_y_aime = pd.concat(x_numtokens_y_aime_lst)
#         x_numtokens_y_amc  = pd.concat(x_numtokens_y_amc_lst)

#         # 你的抽样（每 10 步）
#         for d in (x_time_y_aime, x_time_y_amc, x_numtokens_y_aime, x_numtokens_y_amc):
#             d = d[d['step'] % 10 == 0]

#         # 这里假设你内部已经把列重命名成最终使用的列名：
#         # timing_s/cumsum_step & val-core/amc12/acc/mean@16
#         return x_time_y_aime, x_time_y_amc, x_numtokens_y_aime, x_numtokens_y_amc

#     # draw figure
#     fig, ax = plt.subplots(2, 2, figsize=(20, 10))

#     _, df, _, _ = _subfigure("debug_hbz", 'ppo')
#     _plot_raw_and_smooth(ax[0, 0], df)
#     ax[0, 0].set(xlim=(0, 300000), ylim=(0.55, 0.8))
#     ax[0, 0].set_title("ppo on DAPO-Train")

#     _, df, _, _ = _subfigure("debug_hbz3", 'ppo')
#     _plot_raw_and_smooth(ax[0, 1], df)
#     ax[0, 1].set(xlim=(0, 200000), ylim=(0.5, 0.7))
#     ax[0, 1].set_title("ppo on AIME-Old")

#     _, df, _, _ = _subfigure("debug_hbz", 'grpo')
#     _plot_raw_and_smooth(ax[1, 0], df)
#     ax[1, 0].set(xlim=(0, 250000), ylim=(0.65, 0.8))
#     ax[1, 0].set_title("grpo on DAPO-Train")

#     _, df, _, _ = _subfigure("debug_hbz3", 'grpo')
#     _plot_raw_and_smooth(ax[1, 1], df)
#     ax[1, 1].set(xlim=(0, 150000), ylim=(0.5, 0.7))
#     ax[1, 1].set_title("grpo on AIME-Old")

#     plt.savefig('logs/main_figure.png', dpi=600, bbox_inches='tight')
#     plt.savefig('logs/main_figure.pdf', dpi=600, bbox_inches='tight')
#     plt.close()

def figure_token_acc():
    palette = sns.color_palette(
        [
            "#2C67A7", # vanilla
            "#73C37A", # dynamic
            "#76B5AF", # replay
            "#D6E4D2", # prior
            "#FBB463", # ablation
            "#F57F72", # ours
        ]
    )
    # init related groups
    all_trial_groups = defaultdict(dict)
    for idx, alg, proj in zip(all_ids["id"], all_ids["alg"], all_ids["proj"]):
        if id2trialid[proj][idx]:
            all_trial_groups[proj][alg] = TrialGroup(proj, idx, id2trialid[proj][idx])
    
    def _subfigure(proj,base_alg='ppo'):
        x_time_y_aime_lst = []
        x_time_y_amc_lst = []
        x_numtokens_y_aime_lst = []
        x_numtokens_y_amc_lst = []
        for alg in all_trial_groups[proj]:
            if base_alg not in alg or "mopps" in alg:
                print("skip", alg)
                continue

            step_range = -1
            # if proj == 'debug_hbz3' and alg == 'ppo-dynamic':
            #     step_range = 155
            # elif proj == 'debug_hbz' and alg == 'ppo-ours':
            #     step_range = 385
            # else:
            #     step_range = -1


            x_time_y_aime = all_trial_groups[proj][alg].get_top_x_y_dataframe(x_col='tim', y_col='aime', step_range=step_range)
            x_time_y_amc = all_trial_groups[proj][alg].get_top_x_y_dataframe(x_col='tim', y_col='amc', step_range=step_range)
            x_numtokens_y_aime = all_trial_groups[proj][alg].get_top_x_y_dataframe(x_col='token', y_col='aime')
            x_numtokens_y_amc = all_trial_groups[proj][alg].get_top_x_y_dataframe(x_col='token', y_col='amc', step_range=step_range)

            x_time_y_aime['alg'] = alg
            x_time_y_amc['alg'] = alg
            x_numtokens_y_aime['alg'] = alg
            x_numtokens_y_amc['alg'] = alg

            x_time_y_amc['val-core/amc12/acc/mean@16'] = x_time_y_amc['val-core/amc12/acc/mean@16'].cummax()

            x_time_y_aime_lst.append(x_time_y_aime)
            x_time_y_amc_lst.append(x_time_y_amc)
            x_numtokens_y_aime_lst.append(x_numtokens_y_aime)
            x_numtokens_y_amc_lst.append(x_numtokens_y_amc)

        x_time_y_aime = pd.concat(x_time_y_aime_lst)
        x_time_y_amc = pd.concat(x_time_y_amc_lst)
        x_numtokens_y_aime = pd.concat(x_numtokens_y_aime_lst)
        x_numtokens_y_amc = pd.concat(x_numtokens_y_amc_lst)

        # x_time_y_aime = x_time_y_aime[x_time_y_aime['step'] % 10 == 0]
        # x_time_y_amc = x_time_y_amc[x_time_y_amc['step'] % 10 == 0]
        # x_numtokens_y_aime = x_numtokens_y_aime[x_numtokens_y_aime['step'] % 10 == 0]
        # x_numtokens_y_amc = x_numtokens_y_amc[x_numtokens_y_amc['step'] % 10 == 0]


        return x_time_y_aime, x_time_y_amc, x_numtokens_y_aime, x_numtokens_y_amc

    # draw figure
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))

    _, df, _, _ = _subfigure("debug_hbz", 'ppo')
    sns.lineplot(data=df, x='timing_s/cumsum_step', y='val-core/amc12/acc/mean@16', hue='alg', ax=ax[0, 0], drawstyle='steps-post', palette=palette)
    # sns.regplot(data=df, x='timing_s/cumsum_step', y='val-core/amc12/acc/mean@16', hue='alg', ax=ax[0, 0], lowess=True)
    ax[0,0].set(xlim=(0, 300000), ylim=(0.55, 0.8))
    ax[0,0].set_title("ppo on DAPO-Train")

    _, df, _, _ = _subfigure("debug_hbz3", 'ppo')
    # df['val-core/amc12/acc/mean@16'] = df['val-core/amc12/acc/mean@16'].cummax()
    sns.lineplot(data=df, x='timing_s/cumsum_step', y='val-core/amc12/acc/mean@16', hue='alg', ax=ax[0, 1], drawstyle='steps-post', palette=palette)
    # sns.regplot(data=df, x='timing_s/cumsum_step', y='val-core/amc12/acc/mean@16', hue='alg', ax=ax[0, 1], lowess=True)
    ax[0,1].set(xlim=(0, 200000), ylim=(0.5, 0.7))
    ax[0,1].set_title("ppo on AIME-Old")

    _, df, _, _ = _subfigure("debug_hbz", 'grpo')
    # df['val-core/amc12/acc/mean@16'] = df['val-core/amc12/acc/mean@16'].cummax()
    sns.lineplot(data=df, x='timing_s/cumsum_step', y='val-core/amc12/acc/mean@16', hue='alg', ax=ax[1, 0], drawstyle='steps-post', palette=palette)
    # sns.regplot(data=df, x='timing_s/cumsum_step', y='val-core/amc12/acc/mean@16', hue='alg', ax=ax[1, 0], lowess=True)
    ax[1,0].set(xlim=(0, 250000), ylim=(0.65, 0.8))
    ax[1,0].set_title("grpo on DAPO-Train")
    
    _, df, _, _ = _subfigure("debug_hbz3", 'grpo')
    # df['val-core/amc12/acc/mean@16'] = df['val-core/amc12/acc/mean@16'].cummax()
    sns.lineplot(data=df, x='timing_s/cumsum_step', y='val-core/amc12/acc/mean@16', hue='alg', ax=ax[1, 1], drawstyle='steps-post', palette=palette)
    # sns.regplot(data=df, x='timing_s/cumsum_step', y='val-core/amc12/acc/mean@16', hue='alg', ax=ax[1, 1], lowess=True)
    ax[1,1].set(xlim=(0, 150000), ylim=(0.5, 0.7))
    ax[1,1].set_title("grpo on AIME-Old")

    fig.savefig(f'logs/main_figure.png', dpi=600, bbox_inches='tight')
    fig.savefig(f'logs/main_figure.pdf', dpi=600, bbox_inches='tight')
    plt.close()


def figure_length_scaling():
    palette = sns.color_palette(
        [
            "#2C67A7", # vanilla
            "#73C37A", # dynamic
            "#76B5AF", # replay
            "#D6E4D2", # prior
            # "#FBB463", # ablation
            "#F57F72", # ours
        ]
    )
    all_trial_groups = defaultdict(dict)
    for idx, alg, proj in zip(all_ids["id"], all_ids["alg"], all_ids["proj"]):
        if id2trialid[proj][idx]:
            all_trial_groups[proj][alg] = TrialGroup(proj, idx, id2trialid[proj][idx])
    
    def _subfigure(proj, base_alg='ppo'):
        x_step_y_length_lst = []
        for alg in all_trial_groups[proj]:
            if base_alg not in alg or "mopps" in alg or 'abl' in alg:
                print("skip", alg)
                continue

            step_range = -1
            # if proj == 'debug_hbz3' and alg == 'ppo-dynamic':
            #     step_range = 155
            # elif proj == 'debug_hbz' and alg == 'ppo-ours':
            #     step_range = 385
            # else:
            #     step_range = -1

            x_step_y_length = all_trial_groups[proj][alg].get_response_length_dataframe(step_range)
            x_step_y_length['alg'] = alg
            x_step_y_length_lst.append(x_step_y_length)
        x_step_y_length = pd.concat(x_step_y_length_lst)
        x_step_y_length = x_step_y_length[x_step_y_length['step'] % 10 == 0]
        return x_step_y_length
    
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))

    df = _subfigure("debug_hbz", 'ppo')
    sns.lineplot(data=df, x='step', y='response_length/mean', hue='alg', ax=ax[0, 0], palette=palette)
    ax[0,0].set_title("ppo on DAPO-Train")

    df = _subfigure("debug_hbz3", 'ppo')
    sns.lineplot(data=df, x='step', y='response_length/mean', hue='alg', ax=ax[0, 1], palette=palette)
    ax[0,1].set_title("ppo on AIME-Old")

    df = _subfigure("debug_hbz", 'grpo')
    sns.lineplot(data=df, x='step', y='response_length/mean', hue='alg', ax=ax[1, 0], palette=palette)
    ax[1,0].set_title("grpo on DAPO-Train")
    
    df = _subfigure("debug_hbz3", 'grpo')
    sns.lineplot(data=df, x='step', y='response_length/mean', hue='alg', ax=ax[1, 1], palette=palette)
    ax[1,1].set_title("grpo on AIME-Old")

    plt.savefig('logs/length_scaling.png', dpi=600, bbox_inches='tight')
    plt.savefig('logs/length_scaling.pdf', dpi=600, bbox_inches='tight')
    plt.close()



    
def figure_prelim():
    # data =  [
    #     {"max_length": 2048, "global_time": 2296.503740604967, "global_gen_time": 1338.4544192207977},
    #     {"max_length": 4096, "global_time": 5725.167300617322, "global_gen_time": 3701.101067681797},
    #     {"max_length": 6144, "global_time": 9088.044254933484, "global_gen_time": 6313.591441338882},
    #     {"max_length": 8192, "global_time": 14610.272450559773, "global_gen_time": 10530.684233290143},
    #     {"max_length": 10240, "global_time": 20099.040543206036, "global_gen_time": 15168.782776040025},
    # ]
    # df = pd.DataFrame(data)
    # df['gen_share_fraction'] = df['global_gen_time'] / df['global_time']
    # fig, ax = plt.subplots(figsize=(10, 5))
    # sns.lineplot(data=df, x='max_length', y='gen_share_fraction', ax=ax)
    # plt.savefig('logs/prelim1.png', dpi=600, bbox_inches='tight')
    # plt.savefig('logs/prelim1.pdf', dpi=600, bbox_inches='tight')
    # plt.close()
    scale2run = {
        10:"run_20250913_85ceddfd",
        8:"run_20250914_6b0c2db7",
        6:"run_20250914_ba1958ac",
        4:"run_20250914_bfc47a7c",
        2:"run_20250913_c48c7ff2"
    }

    api = wandb.TrackingApi()
    all_data = []
    for scale, run_id in scale2run.items():
        if os.path.exists(f'logs/prelim1_{run_id}.csv'):
            df = pd.read_csv(f'logs/prelim1_{run_id}.csv')
        else:
            run = api.run(project="debug_hbz2", run_id=run_id)
            h = run.history()
            df = pd.DataFrame(h)
            df.to_csv(f'logs/prelim1_{run_id}.csv', index=False)
        records = df.to_dict(orient="records")
        for record in records:
            all_data.append({
                "scale": scale,
                "global_time": record["timing_s/step"],
                "gen_time": record["timing_s/generate_sequences"],
                "gen_fraction": record["timing_s/generate_sequences"] / record["timing_s/step"],
            })
    df = pd.DataFrame(all_data)
    fig, ax = plt.subplots(1,3,figsize=(15, 5))
    sns.violinplot(data=df, x='scale', y='gen_fraction', ax=ax[0])
    # plt.savefig('logs/prelim1.png', dpi=600, bbox_inches='tight')
    # plt.savefig('logs/prelim1.pdf', dpi=600, bbox_inches='tight')
    # plt.close()


    dir = "/mnt/hdfs/huangbaizhou_wl/tmp/ckpt/debug_hbz2/DAPO_ID80_bon128/"
    with open(os.path.join(dir, "case_emb_sim_mean.pkl"), "rb") as f:
        results = pickle.load(f)
    with open(os.path.join(dir, "case_edit_results.pkl"), "rb") as f:
        edit_results = pickle.load(f)
    with open(f"{dir}/case_rougel_results.pkl", "rb") as f:
        rouge_results = pickle.load(f)
    
    # listofdict_cossim = [{"idx": result[0], "length": result[1], "cossim": result[2]} for result in results]
    listofdict_editdist = [{"idx": result[0], "length": result[1], "norm_editdist": result[3] / result[1]} for result in edit_results]
    listofdict_rougel = [{"idx": result[0], "length": result[1], "rougel": result[2]} for result in rouge_results]
    
    # cossim_df = pd.DataFrame(listofdict_cossim)
    editdist_df = pd.DataFrame(listofdict_editdist)
    rougel_df = pd.DataFrame(listofdict_rougel)

    # plt.figure(figsize=(10, 5))
    # sns.violinplot(x="length", y="cossim", data=cossim_df)
    # plt.savefig(f"case_emb_sim_mean.png")
    # plt.close()
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True)
    sns.violinplot(x="length", y="norm_editdist", data=editdist_df, ax=ax[1])
    # sns.violinplot(x="length", y="selfbleu", data=editdist_df, ax=ax2)
    sns.violinplot(x="length", y="rougel", data=rougel_df, ax=ax[2])
    plt.savefig(f"logs/prelim.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"logs/prelim.pdf", dpi=600, bbox_inches='tight')
    plt.close()

def figure_sampler():
    runs= {
        (0.3, 0.99): "run_20250908_0d850204",
        (0.1, 0.99): "run_20250920_f7922ba3",
        (0.2, 0.99): "run_20250918_b9f682a9",
        (0.4, 0.99): "run_20250919_9bc4e8e5",
        (0.5, 0.99): "run_20250918_0707b433",
        (0.3, 0.9): "run_20250918_bcfc9e33",
        (0.3, 0.95): "run_20250919_edec02d6",
        (0.3, 0.995): "run_20250918_c1746896",
    }

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    api = wandb.TrackingApi()
    all_df = []
    for (sigma, lambda_value), run_id in runs.items():
        if os.path.exists(f'logs/sampler_{run_id}.csv'):
            df = pd.read_csv(f'logs/sampler_{run_id}.csv')
        else:
            run = api.run(project="debug_hbz3", run_id=run_id)
            h = run.history()
            df = pd.DataFrame(h)
            df = df[['step', 'sampler/pg_correlation', 'sampler/pg_error']]
            df.to_csv(f'logs/sampler_{run_id}.csv', index=False)
        df['sigma'] = sigma
        df['lambda'] = lambda_value
        all_df.append(df)
    df = pd.concat(all_df)
    
    sns.lineplot(data=df, x='step', y='sampler/pg_correlation', hue='sigma', ax=axs[0])
    sns.lineplot(data=df, x='step', y='sampler/pg_error', hue='sigma', ax=axs[1])

    
    plt.savefig(f"logs/sampler.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"logs/sampler.pdf", dpi=600, bbox_inches='tight')
    plt.close()







id2trialid = {
    "debug_hbz": {
        "43": ["run_20250825_d75ba81c"], #, "run_20250914_094b03f3"], # ppo, run_20250910_2b916b5d
        "44": ["run_20250901_d4ee6ed5"], #, ["run_20250825_9c6f7135", "run_20250827_47b5a0c3"]], # ppo-mopps  ,
        "47": ["run_20250909_49bb1ae6"], # ppo-dynamic, run_20250914_75e3cd39
        "49": ["run_20250912_de88cc71"], # ppo-replay, another run left
        "53": ["run_20250914_2b3af110"], # ppo-prior
        "51": ["run_20250911_8956cc9f"], # ppo-ablation, run_20250914_78ae334d
        "109": ["run_20250911_5c97fa50"], #, "run_20250911_5c97fa50"], # ppo-prost
        "45": ["run_20250902_086dc19c"],# "run_20250910_53457396"], # grpo
        "46": ["run_20250826_c5e64a3d"],# "run_20250901_1f64b57e"],  # grpo-mopps
        "48": ["run_20250914_f8c4a6f8"], # grpo-dynamic, run_20250914_f8c4a6f8, run_20250910_d1ebb807
        "50": ["run_20250911_500dd081"], # grpo-replay, run_20250913_0352e438
        "54": ["run_20250914_d4cdde86"], # grpo-prior
        "52": ["run_20250911_b45b98e3"], # grpo-ablation, run_20250914_293b77af
        "110": ["run_20250908_7f6a8186"], #, "run_20250910_5523b7f5"],
        # "84": ["run_20250827_ba477878", "run_20250830_ade4d7b4"],
        # "87": ['run_20250829_96f50319'],
        # "91": ['run_20250901_4e26f30a'],
    },
    "debug_hbz3": {
        "4": ["run_20250910_d26b10d8"], #, "run_20250910_d26b10d8"], # ppo
        "5": ["run_20250909_c2ede604"], # "run_20250914_44220a21"], # ppo-mopps
        "8": ["run_20250917_43061166"], # ppo-dynamic, ["run_20250911_a6b42635"] not good
        "10": ["run_20250911_76482ee4"], #ppo-replay, ["run_20250911_76482ee4"] not finished
        "14": ["run_20250913_85374dab"], # ppo-prior, ["run_20250913_85374dab"] not finished
        "12": ["run_20250913_18e79f96"], #ppo-ablation ["run_20250913_18e79f96"] not finished
        "25": ["run_20250908_9baae94d"], # run_20250909_6bf8a0f8"], # "run_20250908_9baae94d", "run_20250909_f1264f19"], # ppo-prost
        "6": ["run_20250910_199b7e6f"], # "run_20250912_6605481d"], # grpo, "run_20250908_9d2d86b8"
        "7": ["run_20250909_050f6407"], # "run_20250910_3845177e"], # grpo-mopps
        "9": ["run_20250911_21082c38"], # grpo-dynamic, ["run_20250911_21082c38"] not finished
        "11": ["run_20250912_3d20de87"], # "run_20250912_3d20de87"], # grpo-replay
        "15": ["run_20250913_c8c23d45"], # "run_20250915_2ba7a109"], # grpo-prior
        "13": ["run_20250913_76ad6fa3"], # grpo-ablation
        "26": ["run_20250908_0d850204"], # "run_20250909_98c5b567", "run_20250909_47695421"] # grpo-prost
    },
    "debug_hbz2": {
        "43": ["run_20250918_f476cc80"], # ppo
        "47": ["run_20250918_d692a272"], # ppo-dynamic
        "49": ["run_20250918_d9a61e4c"], # ppo-replay
        "53": ["run_20250918_bb76b3ab"], # ppo-prior
        "109": ["run_20250918_ffa2ae2e"], # ppo-prost
        # ""
    }
}
# baselines = {
#     "debug_hbz": ["ppo", "ppo-mopps", "grpo", "grpo-mopps"],
#     "debug_hbz2": []
# }
all_ids = {
    "id": ["43", "44", "47", "49", "53", "51", "109", "45", "46", "48", "50", "54", "52", "110"] + ["4", "5", "8", "10", "14", "12", "25", "6", "7", "9", "11", "15", "13", "26"],
    "alg": ["ppo", "ppo-mopps", "ppo-dynamic", "ppo-replay", "ppo-prior", "ppo-ablation", "ppo-ours", "grpo", "grpo-mopps", "grpo-dynamic", "grpo-replay", "grpo-prior", "grpo-ablation", "grpo-ours"] * 2,
    "proj": ['debug_hbz'] * 14 + ['debug_hbz3'] * 14,
}
_ = pd.DataFrame(all_ids)
assert(len(_) == 28)
all_ids = pd.DataFrame(all_ids)

small_ids = {
    "id": ["43", "47", "49", "53", "109"],
    "alg": ["ppo", "ppo-dynamic", "ppo-replay", "ppo-prior", "ppo-prost"],
    "proj": ['debug_hbz2'] * 5,
}
# all_ids.to_csv("logs/all_ids.csv", index=False)

def process_all_runs_multiprocess(max_workers=4):
    """使用多进程处理所有runs"""
    print(f"使用 {max_workers} 个进程并行处理")
    
    # 使用ProcessPoolExecutor进行多进程处理
    # 在spawn模式下，每个进程都会重新导入模块，所以不需要额外的初始化
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = []

        # all trial_groups
        for idx, alg, proj in zip(all_ids["id"], all_ids["alg"], all_ids["proj"]):
            # TrialGroup(proj, id2trialid[proj][id])
            future_to_task.append(executor.submit(TrialGroup, proj, idx, id2trialid[proj][idx]))
        
        # 处理完成的任务
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            trial_group = future.result()
            print(f"[{completed}/{len(future_to_task)}] {trial_group.proj_name}_{trial_group.group_id}")
    print("所有任务处理完成！")

if __name__ == "__main__":
    # 设置多进程启动方式为spawn，确保wandb API的线程安全
    # multiprocessing.set_start_method('spawn', force=True)
    # process_all_runs_multiprocess(max_workers=1)
    sns.set_theme(style="whitegrid")

    # table_main_result(all_ids)
    # figure_token_acc()
    # figure_prelim()
    # figure_length_scaling()
    figure_sampler()

    # small model
    # table_main_result(small_ids)

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