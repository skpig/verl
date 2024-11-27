set -x

ray stop --force

# ckpt和路径

# 12B
# SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/P6.1_12B_32k_SFT29_Fix_RoPE_Base_hf
# RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/rm_p6_dense_12b_phase2_exp1_hf

# 70B
SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/70bdense_P61_D7_wd01_sft29_1022_2e_64gpu_hf
RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhiqi.0/rlhf/p6dense_70b_rm

TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/hard60_format_repeat10.parquet
TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/math_500.parquet
default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhiqi.0/alpha-seed/experiments/p6d-70b/

# 训练长度
max_prompt_length=2048
max_response_length=16384
# batch size && 训练epoch
train_batch_size=4096
ppo_mini_batch_size=1024
val_batch_size=500
total_epochs=5000
test_freq=5
save_freq=10
# 算法相关的参数
actor_lr=1.5e-6
critic_lr=2e-6
lr_warmup_steps=10
kl_coef=0.0001
use_last_response=False
use_ref_answer=True
gae_gamma=1.0
gae_lam=0.95
force_append_eos=True
upgo_loss_weight=0.2
upgo_loss_version=1
clip_ratio2=2.0
# tracking实验名
project_name='verl_example_math'
experiment_name=p6d_70b_tp${xperf_tp_size}_fsdp${fsdp_size}
# 工程参数
use_dynamic_bsz=True
actor_ppo_max_token_len=61440
critic_ppo_max_token_len=122880
infer_ppo_max_token_len=24576
actor_sp_size=4
critic_sp_size=4
ref_sp_size=1
reward_sp_size=1
fsdp_size=16
xperf_tp_size=4
offload=True
offload_train_memory=True

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
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.max_token_len=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_token_len=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    +actor_rollout_ref.model.use_rmpad=True \
    actor_rollout_ref.model.external_lib=seed_models \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=512 \
    actor_rollout_ref.actor.scale_pg_by_kl=True \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    critic.use_dynamic_bsz=${use_dynamic_bsz} \
    critic.ppo_max_token_len=${critic_ppo_max_token_len} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=64 \
    critic.infer_micro_batch_size=512 \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    +critic.use_rmpad=True \
    critic.model.external_lib=seed_models \
    reward_model.enable=False \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.micro_batch_size=512 \
    reward_model.mean=0.0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    +reward_model.use_rmpad=True \
    reward_model.reward_0_for_overlong_rsp=False \
    reward_model.punish_no_answer=v0 \
    reward_model.use_dynamic_bsz=${use_dynamic_bsz} \
    reward_model.max_token_len=${infer_ppo_max_token_len} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    algorithm.force_append_eos=${force_append_eos} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=False \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl \
    trainer.resume_steps="disable" \
    actor_rollout_ref.actor.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_size=${fsdp_size} \
    critic.fsdp_size=${fsdp_size} \
    reward_model.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.fsdp_config.param_offload=${offload} \
    reward_model.model.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    critic.ulysses_sequence_parallel_size=${critic_sp_size} \
    reward_model.ulysses_sequence_parallel_size=${reward_sp_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${xperf_tp_size} \
    +actor_rollout_ref.rollout.use_vllm=True \
    actor_rollout_ref.rollout.micro_batch_size=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=512 \
    trainer.offload_train_memory=${offload_train_memory} \
    critic.profile.enable=True \
    critic.profile.upload_to_mlx=True \
    critic.profile.filename=actor.tp${xperf_tp_size}.fsdp${fsdp_size} \
    actor_rollout_ref.actor.profile.enable=True \
    actor_rollout_ref.actor.profile.upload_to_mlx=True \
    actor_rollout_ref.actor.profile.filename=actor.tp${xperf_tp_size}.fsdp${fsdp_size}
