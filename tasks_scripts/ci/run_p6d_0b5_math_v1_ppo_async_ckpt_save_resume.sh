set -x

NUM_STEPS="${NUM_STEPS:-240}"
echo $NUM_STEPS

SAVE_FREQ="${SAVE_FREQ:--1}"
echo "SAVE_FREQ: $SAVE_FREQ"

RESUME_STEPS="${RESUME_STEPS:-disable}"
echo "RESUME_STEPS: $RESUME_STEPS"

CKPT_VERSION="${CKPT_VERSION:-v1}"
echo "CKPT_VERSION: $CKPT_VERSION"

TEST_NAME="${TEST_NAME:-ckpt_save_resume}"
echo "TEST_NAME: $TEST_NAME"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"

N_GPUS_PER_NODE_STREAMING=$((N_GPUS_PER_NODE/2))

echo $N_GPUS_PER_NODE_STREAMING

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/models/p6dense-0.5B-Instruct
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/models/p6dense-0.5B-Instruct_rm
TRAIN_FILE0=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet
TRAIN_FILE1=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/train_with_ref_ans.parquet
TRAIN_FILE="[$TRAIN_FILE0,$TRAIN_FILE1]"
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/test_with_ref_ans_top_100.parquet
default_hdfs_dir="/opt/tiger/p6d_0b5_math_v1_ppo_async/$TEST_NAME"
mkdir -p ${default_hdfs_dir}

# 训练长度
max_prompt_length=512 # 16384
max_response_length=512 # 16384
# batch size && 训练epoch
train_batch_size=64
val_batch_size=5000
ppo_mini_batch_size=8
ppo_micro_batch_size=4
total_epochs=100
test_freq=5
# 算法相关的参数
actor_lr=1e-6
critic_lr=1e-5
lr_warmup_steps_ratio=0.0003 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.001
entropy_coeff=0.0001
use_last_response=False
use_ref_answer=False
gae_gamma=1.0
gae_lam=0.95
kl_penalty=low_var_kl
# 工程参数
actor_sp_size=2
critic_sp_size=2
ref_sp_size=1
reward_sp_size=1

# tracking实验名
project_name='verl_example_math_ci'
experiment_name='p6_400m_math-v1'
export PYTHONPATH=$PYTHONPATH:/opt/tiger/mono_rl:/opt/tiger/verl:/opt/tiger/seed_models:/opt/tiger/verifiable_tasks:/opt/tiger/bpex_triton
python3 tasks/main_ppo.py \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.use_ref_answer=${use_ref_answer} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.truncation='left' \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    critic.ulysses_sequence_parallel_size=${critic_sp_size} \
    reward_model.ulysses_sequence_parallel_size=${reward_sp_size} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${ppo_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.rollout.micro_batch_size=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=256 \
    +actor_rollout_ref.rollout.complete_ratio=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=512 \
    actor_rollout_ref.ref.ema=0.99 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=${ppo_micro_batch_size} \
    critic.model.fsdp_config.param_offload=False \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    reward_model.enable=False \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.micro_batch_size=512 \
    reward_model.mean=0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    reward_model.reward_0_for_overlong_rsp=True \
    reward_model.punish_no_answer=v0 \
    +reward_model.use_rmpad=True \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.kl_penalty=${kl_penalty} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=False \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps=${RESUME_STEPS} \
    trainer.set_fake_attention_mask=False \
    trainer.fake_seqlen_ratio=0.5 \
    streaming_rollout.nnodes=1 \
    streaming_rollout.n_gpus_per_node=${N_GPUS_PER_NODE_STREAMING} \
    streaming_rollout.warmup_step=0 \
    streaming_rollout.force_eos=True \
    streaming_validator.nnodes=1 \
    streaming_validator.n_gpus_per_node=${N_GPUS_PER_NODE_STREAMING} \
    trainer.total_steps=${NUM_STEPS} \
    trainer.ckpt_version=${CKPT_VERSION}