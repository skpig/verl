# BASE_MODEL=${MY_MODEL_DIR}Qwen/Qwen2.5-3B
# TEMPLATE_TYPE=base # or chat
BASE_MODEL=${MY_MODEL_DIR}Qwen/Qwen2.5-3B-Instruct
TEMPLATE_TYPE=chat # or chat
TRAIN_FILE="${MY_DATA_DIR}Eurus-2-RL-Data/train.parquet"
TEST_FILES="['${MY_DATA_DIR}Eurus-2-RL-Data/test.parquet', '${MY_DATA_DIR}MATH-500/test.parquet', '${MY_DATA_DIR}aimo-validation-amc/test.parquet']"

RUN_ID=$1

# Model settings
ROLLOUT_N=16
MAX_PROMPT_LEN=$((1024 * 1))
MAX_RESPONSE_LEN=$((1024 * 3))
BATCH_SIZE=512
MINI_BSZ=64

# Performance tuning
N_GPUS=4
ROLLOUT_TP_SIZE=1
OFFLOAD=True
# SP_SIZE=4 # TODO:
FORWARD_BSZ=16
BACKWARD_BSZ=8
TOTAL_EPOCHS=1
FORWARD_MAX_TOKEN_LEN=$((8 * (MAX_PROMPT_LEN + MAX_RESPONSE_LEN)))
BACKWARD_MAX_TOKEN_LEN=$((4 * (MAX_PROMPT_LEN + MAX_RESPONSE_LEN)))

PROJ_NAME="TinyMATH"
MODEL_NAME=$(basename $BASE_MODEL)
DATA_NAME=NUMINA
EXPERIMENT_NAME="ID${RUN_ID}_${DATA_NAME}_grpo_${MODEL_NAME}_n${ROLLOUT_N}_resplen${MAX_RESPONSE_LEN}_bsz${BATCH_SIZE}-${MINI_BSZ}"

python3 examples/data_preprocess/custom.py \
    --resume


# set -x
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# 定义要执行的命令
CMD="python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=\"$TEST_FILES\" \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=1312 \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$FORWARD_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS"

# 获取vllm版本号
verl_version=$(conda list | grep 'vllm' | awk '{print $2}')

# 定义比较函数
function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$(printf '%s\n' "$@" | sort -V | tail -n 1)"; }

# 条件判断分支语句
if version_gt "$verl_version" "0.8"; then
    echo "vllm版本大于0.8"
    CMD="${CMD} \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False "
else
    echo "vllm版本小于等于0.8"
    export VLLM_ATTENTION_BACKEND=XFORMERS
fi


# 打印要执行的命令
echo "即将执行的命令：\n$CMD"

# 执行命令
eval $CMD
