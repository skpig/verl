RUN_ID=12
WANDB_VERSION=bwandb
# one node
FORWARD_RATIO=8
BACKWARD_RATIO=3

# Model settings
PROMPT_ID=3
ROLLOUT_N=4
OVERLONG_BUFFER_LEN=$((1024 * 1))
MAX_PROMPT_LEN=$((1024 * 1))
MAX_RESPONSE_LEN=$((1024 * 5 + OVERLONG_BUFFER_LEN))
BATCH_SIZE=2048
MINI_BSZ=32

# Performance tuning
N_NODES=${ARNOLD_WORKER_NUM:-1}
N_GPUS=${ARNOLD_WORKER_GPU:-16}
ROLLOUT_TP_SIZE=1
OFFLOAD=True
# SP_SIZE=4 # TODO:
FORWARD_BSZ=16 # no use
BACKWARD_BSZ=2 # no use
TOTAL_EPOCHS=1000
FORWARD_MAX_TOKEN_LEN=$((FORWARD_RATIO * (MAX_PROMPT_LEN + MAX_RESPONSE_LEN))) # 12 for 40GB
BACKWARD_MAX_TOKEN_LEN=$((BACKWARD_RATIO * (MAX_PROMPT_LEN + MAX_RESPONSE_LEN)))  # 4 for 40GB


MY_CKPT_DIR=/mnt/hdfs/huangbaizhou/tmp/ckpt
BASE_MODEL=${MY_MODEL_DIR}Qwen/Qwen2.5-3B-Instruct
TEMPLATE_TYPE=chat
TRAIN_FILE="${MY_DATA_DIR}DAPO-Math-17k/train.parquet"
TEST_FILES="['${MY_DATA_DIR}DAPO-Math-17k/test.parquet', '${MY_DATA_DIR}MATH-500/test.parquet', '${MY_DATA_DIR}aimo-validation-amc/test.parquet']"

# BASE_MODEL=/tmp/pretrain/Qwen/Qwen2.5-3B-Instruct
# TEMPLATE_TYPE=chat # or chat# TRAIN_FILE="${MY_DATA_DIR}Eurus-2-RL-Data/train.parquet"
# gsm8k_train_path=$HOME/data/gsm8k/train.parquet
# gsm8k_test_path=$HOME/data/gsm8k/test.parquet
# train_files="['$gsm8k_train_path']"
# test_files="['$gsm8k_test_path']"


PROJ_NAME="debug_hbz"
MODEL_NAME=$(basename $BASE_MODEL)
DATA_NAME=DAPOMATH
EXPERIMENT_NAME="ID${RUN_ID}_${DATA_NAME}_ppo_${MODEL_NAME}_prompt${PROMPT_ID}_n${ROLLOUT_N}_resplen${MAX_RESPONSE_LEN}_bsz${BATCH_SIZE}-${MINI_BSZ}"

python3 examples/data_preprocess/custom.py \
    --resume


# set -x
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="."

# 定义要执行的命令
CMD="python3 -m verl.trainer.main_ppo \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    algorithm.adv_estimator=gae \
    algorithm.variable_lambda_scalar=0.05 \
    algorithm.critic_lam=1.0 \
    data.prompt_id=$PROMPT_ID \
    data.train_files=$TRAIN_FILE \
    data.val_files=\"$TEST_FILES\" \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BSZ \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$BACKWARD_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$FORWARD_MAX_TOKEN_LEN \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$FORWARD_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(($MAX_PROMPT_LEN + $MAX_RESPONSE_LEN)) \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.fsdp_config.param_offload=$OFFLOAD \
    critic.model.fsdp_config.optimizer_offload=$OFFLOAD \
    critic.use_dynamic_bsz=True \
    critic.ppo_max_token_len_per_gpu=$BACKWARD_MAX_TOKEN_LEN \
    critic.forward_max_token_len_per_gpu=$FORWARD_MAX_TOKEN_LEN \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward_model.launch_reward_fn_async=True \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
    trainer.critic_warmup=10 \
    trainer.logger=['console','$WANDB_VERSION'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir=$MY_CKPT_DIR/$PROJ_NAME/$EXPERIMENT_NAME"


# 打印要执行的命令
echo "即将执行的命令：\n$CMD"

# 执行命令
eval $CMD
