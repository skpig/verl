import wandb
from tracking_out import all_runs


# wandb.init(
#     project="test",
#     anonymous="must")

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

def log_run(proj_name, run_name):
    pass
    


if __name__ == "__main__":
    for proj_name, run_name in all_runs.items():
        for run_name in run_name.items():
            log_run(proj_name, run_name)


