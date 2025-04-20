# BASE_MODEL=${MY_MODEL_DIR}Qwen/Qwen2.5-3B
# TEMPLATE_TYPE=base # or chat
BASE_MODEL=${MY_MODEL_DIR}Qwen/Qwen2.5-3B-Instruct
TEMPLATE_TYPE=chat # or chat
DATA_DIR=${MY_DATA_DIR}countdown
REWARD_FILE=/home/huangbz/verl/verl/utils/reward_score/countdown.py 
REWARD_NAME=compute_score
RUN_ID=$1

# Model settings
ROLLOUT_N=5
MAX_PROMPT_LEN=256
MAX_RESPONSE_LEN=1024
BATCH_SIZE=512
MINI_BSZ=64

# Performance tuning
N_GPUS=4
ROLLOUT_TP_SIZE=1
FORWARD_BSZ=16
BACKWARD_BSZ=8
TOTAL_EPOCHS=1
FORWARD_MAX_TOKEN_LEN=$((16 * (MAX_PROMPT_LEN + MAX_RESPONSE_LEN)))
BACKWARD_MAX_TOKEN_LEN=$((8 * MAX_PROMPT_LEN + MAX_RESPONSE_LEN))

PROJ_NAME="TinyZero"
MODEL_NAME=$(basename $BASE_MODEL)
DATA_NAME=$(basename $DATA_DIR)
EXPERIMENT_NAME="ID${RUN_ID}_${DATA_NAME}_onlinerft_${MODEL_NAME}_n${ROLLOUT_N}_resplen${MAX_RESPONSE_LEN}_bsz${BATCH_SIZE}-${MINI_BSZ}"

python3 data_preprocess/countdown.py \
  --local_dir $DATA_DIR \
  --template_type $TEMPLATE_TYPE

DATA_DIR="${DATA_DIR}_${TEMPLATE_TYPE}"

export VLLM_ATTENTION_BACKEND=XFORMERS

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_online_rft \
 custom_reward_function.path=$REWARD_FILE \
 custom_reward_function.name=$REWARD_NAME \
 data.train_files=$DATA_DIR/train.parquet \
 data.val_files=$DATA_DIR/test.parquet \
 data.train_batch_size=$BATCH_SIZE \
 data.val_batch_size=1312 \
 data.max_prompt_length=$MAX_PROMPT_LEN \
 data.max_response_length=$MAX_RESPONSE_LEN \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BSZ \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$BACKWARD_MAX_TOKEN_LEN \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$FORWARD_MAX_TOKEN_LEN \
 actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 actor_rollout_ref.rollout.temperature=1.2 \
 data.template_type=$TEMPLATE_TYPE \
 trainer.logger=['wandb'] \
 trainer.val_before_train=True \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=$N_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.project_name=$PROJ_NAME \
 trainer.experiment_name=$EXPERIMENT_NAME \
 trainer.total_epochs=$TOTAL_EPOCHS \
 trainer.reward_scale_fn=max_only