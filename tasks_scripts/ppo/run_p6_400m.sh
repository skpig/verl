set -x

if [[ "${ARNOLD_REGION}" == "US" ]]; then
    echo "Running in US Region"
    # ckpt格式和master冲突，需要重新同步
    SFT_MODEL_PATH=hdfs://harunava/home/byte_data_seed_azure/seed_rlhf/user/zhangchi.usc1992/alpha-seed/models/400m.sft27.baseline
    RM_MODEL_PATH=hdfs://harunava/home/byte_data_seed_azure/seed_rlhf/user/zhangchi.usc1992/alpha-seed/models/rm_p6_moe_400m_baseline
    TRAIN_FILE=hdfs://harunava/home/byte_data_seed_azure/seed_rlhf/user/zhangchi.usc1992/alpha-seed/data/math/hard60_format.parquet
    TEST_FILE=hdfs://harunava/home/byte_data_seed_azure/seed_rlhf/user/zhangchi.usc1992/alpha-seed/data/math/math_500.parquet
    default_hdfs_dir=hdfs://harunava/home/byte_data_seed_azure/seed_rlhf/user/zhangchi.usc1992/alpha-seed/experiments/test

else
    echo "Running in CN Region"
    # ckpt和路径
    SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/p6_400m_moe_4T_sft_v27_bs128_lr4e-4_master_dyn_epoch4_hf
    RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/seed_rl/models/rm_p6_moe_400m_0716_sftv27_stage2_hf
    TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/hard60_format.parquet
    TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/zhangchi.usc1992/data/rlhf/math/math_500.parquet
    default_hdfs_dir=hdfs://haruna/home/byte_data_seed/lf_lq/user/yueyu/model/rl/alpha_seed/test5
fi

# 训练长度
max_prompt_length=2048 # 16384
max_response_length=8192 # 16384
# batch size && 训练epoch
train_batch_size=1024
val_batch_size=500
ppo_mini_batch_size=128
ppo_micro_batch_size=64
total_epochs=5000
test_freq=2
save_freq=50
# 算法相关的参数
actor_lr=1e-5
critic_lr=2e-5
lr_warmup_steps=1 # 10 / (train_size * total_epochs / train_batch_size)
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
experiment_name='p6_400m_verifier_1024_32'
export PYTHONPATH=$PYTHONPATH:/opt/tiger/verl:/opt/tiger/seed_models:/opt/tiger/verifiable_tasks:/opt/tiger/bpex_triton
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
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${ppo_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.clip_ratio2=${clip_ratio2} \
    actor_rollout_ref.rollout.micro_batch_size=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=xperf_gpt \
    +actor_rollout_ref.rollout.use_vllm=False \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=1024 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=512 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.scale_pg_by_kl=True \
    actor_rollout_ref.actor.upgo_loss_weight=${upgo_loss_weight} \
    actor_rollout_ref.actor.upgo_loss_version=${upgo_loss_version} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size=${ppo_micro_batch_size} \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
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
    algorithm.force_append_eos=${force_append_eos} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tracking'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.eval_before_training=True \
    trainer.val_only=False \
    trainer.val_epoch=1 \
    trainer.need_log=False \
    trainer.log_file=/opt/tiger/alpha-seed/log.jsonl