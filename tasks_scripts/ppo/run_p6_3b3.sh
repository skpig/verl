set -x

# ckpt和路径
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/yueyu/model/rl/alpha_seed/3.3b_refl_sft/checkpoints/global_epoch_4/p6_to_models/3b3.sft27.M.CNEN.reflect.v0_hf
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/caizhao/3b3_release/rm_p6_moe_3b3_0812_sftv27_stage2_fix_order_aux_32k_v2/checkpoints/global_epoch_1/p6_to_models/rm_p6_moe_3.3m_0716_sftv27_stage2_hf
TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/gsm8k/multi_turn_data_reflection_penalty.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/gsm8k/multi_turn_data_reflection_penalty_test.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/yueyu/model/rl/alpha_seed/3.3B_refl_rl
# 训练长度
max_prompt_length=1024
max_response_length=4096
# batch size && 训练epoch
train_batch_size=1024
val_batch_size=1024
ppo_mini_batch_size=128
total_epochs=500
test_freq=10
save_freq=50
# 算法相关的参数
actor_lr=2e-6
critic_lr=3e-6
lr_warmup_steps_ratio=0.005 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.003
use_last_response=True
use_ref_answer=True
gae_gamma=1.0
gae_lam=0.95
# tracking实验名
project_name='alpha_seed_exp'
experiment_name='p6_3.3b_refl_add_reward'
# 工程参数
gen_micro_batch_size=1024
infer_micro_batch_size=256
train_micro_batch_size=16
ulysses_sequence_parallel_size=4


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
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.micro_batch_size=${gen_micro_batch_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.buffer_dtype=bf16 \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps_ratio=${lr_warmup_steps_ratio} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size=${train_micro_batch_size} \
    critic.infer_micro_batch_size=${infer_micro_batch_size} \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    reward_model.enable=True \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.micro_batch_size=${infer_micro_batch_size} \
    reward_model.mean=0.0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    +reward_model.use_rmpad=True \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs}