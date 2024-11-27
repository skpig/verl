set -x

# ckpt和路径
# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/ct128kv2_baseline_sft32k_v27_lr2e5_epoch4_rope1000_hf
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/alphaseed/20241107/3b3p60905_137k_revisedonly_scalingexp_5xsample_bsz1600_lr5e6_tp4pp5_hf
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/hard60_format_repeat10.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/math_500.parquet

# 实验参数
num_bon=16
max_response_length=2048
reward_overlong=True
train_batch_size=1024
ppo_mini_batch_size=128
actor_lr=2e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=1.0
lr_warmup_steps_ratio=0.00003 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.0001
actor_entropy_coeff=0.001

# tracking log
model_size="3b3"  # in 400m, 3b3, 12B
project_name='alphaseed_bon'
experiment_name=${model_size}'_bo1_speed_2k_streaming_1024'
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/ssd_hldy/evals_pipeline/user/liulingjun.godzilla/20241109bo16

# 工程参数
# 工程参数
if [ "${model_size}" = "400m" ]; then
    gen_micro_batch_size=1024
    infer_micro_batch_size=256
    train_micro_batch_size=64
    ulysses_sequence_parallel_size=1
    rollout_tensor_model_parallel_size=1 
elif [ "${model_size}" = "3b3" ]; then
    gen_micro_batch_size=128
    infer_micro_batch_size=512
    train_micro_batch_size=64
    ulysses_sequence_parallel_size=1
    rollout_tensor_model_parallel_size=4
elif [ "${model_size}" = "12B" ]; then
    gen_micro_batch_size=512
    infer_micro_batch_size=512
    train_micro_batch_size=64
    ulysses_sequence_parallel_size=1
    rollout_tensor_model_parallel_size=4
else
    echo "Model size is not recognized, taking default actions."
    exit 1
fi

# 固定参数
max_prompt_length=2048
val_batch_size=448
total_epochs=5000
test_freq=5
save_freq=50
use_last_response=False
use_ref_answer=True


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
    +data.chat_template=seed \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_coeff=${actor_entropy_coeff} \
    actor_rollout_ref.rollout.micro_batch_size=${gen_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.bon_strategy=best_worst \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.buffer_dtype=bf16 \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=${train_micro_batch_size} \
    critic.infer_micro_batch_size=${infer_micro_batch_size} \
    critic.model.fsdp_config.param_offload=False \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    reward_model.enable=False \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.micro_batch_size=${infer_micro_batch_size} \
    reward_model.mean=0.0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    +reward_model.use_rmpad=True \
    reward_model.reward_0_for_overlong_rsp=${reward_overlong} \
    reward_model.punish_no_answer=v0 \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=False \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.save_cases_to_hdfs=False\
    streaming_rollout.nnodes=2 \
    streaming_rollout.n_gpus_per_node=8\
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8 \
    +actor_rollout_ref.rollout.complete_ratio=0.8 